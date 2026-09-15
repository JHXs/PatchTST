"""Run the preregistered Round 14 cross-city generalization experiment.

Real Guangzhou data are accessible only in ``--formal`` mode.  ``--smoke`` uses
generated data and is never eligible for a gate decision.  The module exposes
small, pure helpers so that the leakage and pairing invariants can be tested
without opening any station CSV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ST_PatchTST_model import ST_PatchTST


B0_STATIONS = (9047, 9030, 9025, 9028, 9024, 9017, 9040, 9021)
B1_STATIONS = (9026, 9039, 9033, 9046, 9020, 9038, 9032, 9045)
FORBIDDEN_STATIONS = frozenset(
    (9022, 9027, 9029, 9041, 9023, 9031, 9019, 9037, 9034,
     9018, 9044, 9035, 9058, 9043, 9042, 9036)
)
FORMAL_SEEDS = (7001, 7002, 7003, 7004, 7005)
ARMS = ("degraded_patchtst", "st_sparse_station_bias_delta_forecast")
TASKS = ((24, 1), (168, 6))
DATA_DIR = "tsai/data/stations_data_Guangzhou"
IMPLEMENTATION_FILES = (
    "run_cross_city_generalization.py",
    "summarize_cross_city_generalization.py",
    "test_cross_city_generalization.py",
)


class ProtocolViolation(RuntimeError):
    """Raised before data access when a frozen protocol boundary is violated."""


class InsufficientNeighbors(RuntimeError):
    """The C-fit segment contains fewer than five eligible candidates."""


@dataclass(frozen=True)
class RunConfig:
    history: int
    horizon: int
    batch_size: int
    epochs: int
    patience: int
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    neighbor_hidden_dim: int = 32
    forecast_alpha_max: float = 0.5
    forecast_alpha_init: float = 0.1
    spatial_pool_bins: int = 4
    sparse_neighbor_top_k: int = 5
    n_layers: int = 3
    n_heads: int = 4
    d_model: int = 16
    d_ff: int = 128
    dropout: float = 0.2
    patch_len: int = 4
    stride: int = 2


def formal_config(history: int, horizon: int) -> RunConfig:
    if (history, horizon) == (24, 1):
        return RunConfig(history, horizon, batch_size=256, epochs=40, patience=8)
    if (history, horizon) == (168, 6):
        return RunConfig(history, horizon, batch_size=512, epochs=30, patience=6)
    raise ProtocolViolation(f"未预注册任务: {history}→{horizon}")


def authorized_station_ids(center_station_id: int) -> tuple[int, ...]:
    if center_station_id not in B1_STATIONS:
        raise ProtocolViolation(f"正式中心站 {center_station_id} 不属于显式 b1 白名单")
    candidates = tuple(s for s in B1_STATIONS if s != center_station_id) + B0_STATIONS
    return (center_station_id,) + candidates


def validate_authorized_station_set(
    center_station_id: int, station_ids: tuple[int, ...] | list[int]
) -> tuple[int, ...]:
    requested = tuple(int(value) for value in station_ids)
    expected = authorized_station_ids(center_station_id)
    if len(requested) != len(set(requested)):
        raise ProtocolViolation("授权站点集合包含重复站号")
    if set(requested) != set(expected):
        raise ProtocolViolation(
            f"授权集合必须精确等于中心站及其 15 个候选: requested={sorted(requested)}"
        )
    overlap = set(requested) & FORBIDDEN_STATIONS
    if overlap:
        raise ProtocolViolation(f"授权集合触及已消费站或 b2/b3: {sorted(overlap)}")
    return requested


def guarded_read_station(
    data_dir: str | Path,
    station_id: int,
    authorized_ids: tuple[int, ...],
    opened_station_ids: set[int],
) -> pd.Series:
    """Perform the final fail-closed checks immediately before one CSV read."""
    station_id = int(station_id)
    authorized = set(authorized_ids)
    if station_id not in authorized or station_id in FORBIDDEN_STATIONS:
        raise ProtocolViolation(f"拒绝读取未授权站点 {station_id}")
    if not opened_station_ids <= authorized or opened_station_ids & FORBIDDEN_STATIONS:
        raise ProtocolViolation("累计实际打开集合越过授权边界")
    path = Path(data_dir) / f"df_station_{station_id}.csv"
    frame = pd.read_csv(path, usecols=["time", "PM25_Concentration"])
    opened_station_ids.add(station_id)
    frame["time"] = pd.to_datetime(frame["time"], errors="raise")
    frame = frame.sort_values("time").drop_duplicates("time", keep="last")
    return frame.set_index("time")["PM25_Concentration"].rename(station_id)


def load_center_candidate_frame(
    center_station_id: int, data_dir: str | Path = DATA_DIR
) -> tuple[pd.DataFrame, tuple[int, ...]]:
    authorized = validate_authorized_station_set(
        center_station_id, authorized_station_ids(center_station_id)
    )
    opened: set[int] = set()
    series = [guarded_read_station(data_dir, sid, authorized, opened) for sid in authorized]
    if opened != set(authorized) or opened & FORBIDDEN_STATIONS:
        raise ProtocolViolation(
            f"实际打开集合必须精确等于授权集合: opened={sorted(opened)}"
        )
    aligned = pd.concat(series, axis=1, join="inner").sort_index()
    if aligned.index.has_duplicates or not aligned.index.is_monotonic_increasing:
        raise ProtocolViolation("共同时间轴必须严格按时间升序且无重复")
    return aligned, tuple(sorted(opened))


def causal_forward_fill(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill only from observations earlier in the same supplied segment."""
    if not frame.index.is_monotonic_increasing:
        raise ValueError("因果填充前时间轴必须升序")
    return frame.ffill()


def segment_bounds(row_count: int) -> dict[str, tuple[int, int]]:
    fit_end = math.floor(row_count * 0.70)
    val_end = math.floor(row_count * 0.80)
    if not 0 < fit_end < val_end < row_count:
        raise ValueError("共同时间轴过短，无法建立 C-fit/C-val/C-confirm")
    return {
        "fit": (0, fit_end),
        "val": (fit_end, val_end),
        "confirm": (val_end, row_count),
    }


def select_top5_neighbors(
    fit_frame: pd.DataFrame, center_station_id: int, candidate_ids: list[int] | tuple[int, ...]
) -> tuple[list[int], dict[int, float]]:
    """Select solely from a pre-cut C-fit frame using finite common rows."""
    filled = causal_forward_fill(fit_frame[[center_station_id, *candidate_ids]])
    common = filled.to_numpy(dtype=np.float64)
    common = common[np.isfinite(common).all(axis=1)]
    correlations: dict[int, float] = {}
    if len(common) < 2 or np.std(common[:, 0]) == 0:
        raise InsufficientNeighbors("C-fit 的 16 站共同有限时间戳不足或中心站零方差")
    for column, station_id in enumerate(candidate_ids, start=1):
        if np.std(common[:, column]) == 0:
            continue
        value = float(np.corrcoef(common[:, 0], common[:, column])[0, 1])
        if np.isfinite(value):
            correlations[int(station_id)] = value
    ranked = sorted(correlations, key=lambda sid: (-correlations[sid], sid))
    if len(ranked) < 5:
        raise InsufficientNeighbors(
            f"C-fit 合格候选仅 {len(ranked)} 个，少于固定 Top-5"
        )
    return ranked[:5], correlations


def confirmation_block_map(timestamps: pd.Index, block_count: int = 5) -> dict[pd.Timestamp, int]:
    ordered = pd.DatetimeIndex(timestamps)
    if ordered.has_duplicates or not ordered.is_monotonic_increasing:
        raise ValueError("确认时间戳必须升序且唯一")
    quotient, remainder = divmod(len(ordered), block_count)
    sizes = [quotient + (1 if block < remainder else 0) for block in range(block_count)]
    mapping: dict[pd.Timestamp, int] = {}
    offset = 0
    for block, size in enumerate(sizes, start=1):
        for stamp in ordered[offset:offset + size]:
            mapping[pd.Timestamp(stamp)] = block
        offset += size
    return mapping


def build_segment_sample_indices(
    values: np.ndarray,
    start: int,
    stop: int,
    history: int,
    horizon: int,
) -> np.ndarray:
    starts = []
    first = start
    last_exclusive = stop - history - horizon + 1
    for window_start in range(first, max(first, last_exclusive)):
        target_start = window_start + history
        target_stop = target_start + horizon
        window = values[window_start:target_stop]
        if target_start >= start and target_stop <= stop and np.isfinite(window).all():
            starts.append(window_start)
    return np.asarray(starts, dtype=np.int64)


class SharedWindowDataset(Dataset):
    def __init__(self, values, indices, history, horizon, center_idx):
        self.values = values
        self.indices = indices
        self.history = history
        self.horizon = horizon
        self.center_idx = center_idx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        start = int(self.indices[item])
        target = start + self.history
        x = self.values[start:target].T
        y = self.values[target:target + self.horizon, self.center_idx][None, :]
        return torch.from_numpy(x), torch.from_numpy(y)


def prepare_shared_data(frame: pd.DataFrame, center_station_id: int, config: RunConfig):
    bounds = segment_bounds(len(frame))
    candidates = [int(sid) for sid in frame.columns if int(sid) != center_station_id]
    fit_start, fit_stop = bounds["fit"]
    top5, correlations = select_top5_neighbors(
        frame.iloc[fit_start:fit_stop], center_station_id, candidates
    )
    station_ids = [center_station_id, *top5]

    # Each segment is filled separately, so no value can cross a temporal boundary.
    processed = frame[station_ids].copy()
    for start, stop in bounds.values():
        processed.iloc[start:stop] = causal_forward_fill(processed.iloc[start:stop])

    fit_values = processed.iloc[fit_start:fit_stop].to_numpy(dtype=np.float64)
    means = np.nanmean(fit_values, axis=0)
    stds = np.nanstd(fit_values, axis=0)
    if not np.isfinite(means).all():
        raise InsufficientNeighbors("C-fit 缩放统计非有限")
    stds = np.where(np.isfinite(stds) & (stds >= 1e-6), stds, 1.0)
    raw_values = processed.to_numpy(dtype=np.float64)
    normalized = ((raw_values - means) / stds).astype(np.float32)
    indices = {
        name: build_segment_sample_indices(
            normalized, start, stop, config.history, config.horizon
        )
        for name, (start, stop) in bounds.items()
    }
    if any(len(indices[name]) == 0 for name in ("fit", "val", "confirm")):
        raise InsufficientNeighbors("至少一个时间子段没有完整、有限的预测窗口")
    block_map = confirmation_block_map(frame.index[bounds["confirm"][0]:bounds["confirm"][1]])
    confirm_timestamps = pd.DatetimeIndex(
        frame.index[bounds["confirm"][0]:bounds["confirm"][1]]
    )
    confirm_block_labels = [block_map[pd.Timestamp(stamp)] for stamp in confirm_timestamps]
    confirm_scalar_blocks = []
    confirm_scalar_timestamps = []
    confirm_scalar_leads = []
    for start in indices["confirm"]:
        target = int(start) + config.history
        for lead in range(config.horizon):
            stamp = pd.Timestamp(frame.index[target + lead])
            confirm_scalar_timestamps.append(stamp)
            confirm_scalar_blocks.append(block_map[stamp])
            confirm_scalar_leads.append(lead + 1)
    fit_center = fit_values[:, 0]
    fit_center = fit_center[np.isfinite(fit_center)]
    quantile_edges = np.quantile(fit_center, [0.2, 0.4, 0.6, 0.8]).tolist()
    metadata = {
        "station_ids": station_ids,
        "center_station_idx": 0,
        "selected_top5": top5,
        "candidate_correlations": {str(k): float(v) for k, v in correlations.items()},
        "bounds": {k: list(v) for k, v in bounds.items()},
        "sample_indices": {k: v.tolist() for k, v in indices.items()},
        "station_means": means.tolist(),
        "station_stds": stds.tolist(),
        "center_mean": float(means[0]),
        "center_std": float(stds[0]),
        "peak_threshold_ugm3": float(np.quantile(fit_center, 0.9)),
        "fit_quantile_edges_ugm3": quantile_edges,
        "confirm_timestamps_ns": [int(value.value) for value in confirm_timestamps],
        "confirm_block_labels": confirm_block_labels,
        "confirm_scalar_block_labels": confirm_scalar_blocks,
        "confirm_scalar_timestamps_ns": [int(value.value) for value in confirm_scalar_timestamps],
        "confirm_scalar_leads": confirm_scalar_leads,
    }
    datasets = {
        name: SharedWindowDataset(normalized, split_indices, config.history, config.horizon, 0)
        for name, split_indices in indices.items()
    }
    return datasets, metadata


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(config: RunConfig, arm: str) -> ST_PatchTST:
    if arm not in ARMS:
        raise ValueError(f"未知实验臂: {arm}")
    degraded = arm == "degraded_patchtst"
    return ST_PatchTST(
        c_in=6,
        c_out=1,
        seq_len=config.history,
        pred_dim=config.horizon,
        num_stations=6,
        feat_size=1,
        center_station_idx=0,
        neighbor_hidden_dim=config.neighbor_hidden_dim,
        neighbor_dropout=config.dropout,
        alpha_max=0.0 if degraded else config.forecast_alpha_max,
        alpha_init=0.0 if degraded else config.forecast_alpha_init,
        gate_mode="pairwise_summary",
        use_null_neighbor=True,
        neighbor_value_mode="difference",
        fusion_stage="forecast",
        spatial_pool_bins=config.spatial_pool_bins,
        use_station_gate_bias=not degraded,
        neighbor_top_k=None if degraded else 5,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_model=config.d_model,
        d_ff=config.d_ff,
        dropout=config.dropout,
        patch_len=config.patch_len,
        stride=config.stride,
        padding_patch=True,
    )


def make_loader(dataset, config: RunConfig, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    predictions, targets = [], []
    for x, y in loader:
        predictions.append(model(x.to(device)).cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(predictions), np.concatenate(targets)


def train_arm(
    config: RunConfig,
    datasets: dict[str, Dataset],
    arm: str,
    seed: int,
    device: torch.device,
    checkpoint_path: Path,
    initialization_checkpoint: Path | None = None,
) -> tuple[ST_PatchTST, list[dict], int, float]:
    set_seed(seed)
    model = build_model(config, arm).to(device)
    if initialization_checkpoint is not None:
        state = torch.load(initialization_checkpoint, map_location=device, weights_only=True)
        incompatible = model.load_state_dict(state, strict=False)
        missing = [key for key in incompatible.missing_keys if key != "alpha_logit"]
        if missing or incompatible.unexpected_keys:
            raise RuntimeError(f"同种子退化初始化不兼容: missing={missing}, unexpected={incompatible.unexpected_keys}")
        for parameter in model.patch_tst.parameters():
            parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    fit_loader = make_loader(datasets["fit"], config, True, seed)
    val_loader = make_loader(datasets["val"], config, False, seed)
    loss_fn = nn.MSELoss()
    best_loss, best_epoch, stale = math.inf, 0, 0
    history_rows = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for x, y in fit_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{arm} seed={seed} 训练损失非有限")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss) * len(x)
            count += len(x)
        val_prediction, val_target = predict(model, val_loader, device)
        val_loss = float(np.mean((val_prediction - val_target) ** 2))
        scheduler.step(val_loss)
        history_rows.append({"epoch": epoch, "train_loss": total / count, "valid_loss": val_loss})
        if val_loss < best_loss - 1e-7:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale += 1
        if stale >= config.patience:
            break
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    return model, history_rows, best_epoch, best_loss


def build_frozen_run_payload(metadata: dict) -> dict:
    """Build the complete, canonical support/config payload covered by the hash."""
    correlations = metadata["candidate_correlations"]
    sorted_correlations = [
        {"station_id": int(station_id), "pearson": float(value)}
        for station_id, value in sorted(
            ((int(key), float(value)) for key, value in correlations.items()),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    return {
        "protocol": "round14-cross-city-generalization-v1",
        "config": metadata["config"],
        "center_station_id": int(metadata["center_station_id"]),
        "seeds": [int(value) for value in metadata["seeds"]],
        "arms": list(metadata["arms"]),
        "opened_station_ids": [int(value) for value in metadata["opened_station_ids"]],
        "smoke_test": bool(metadata["smoke_test"]),
        "code_commit": str(metadata["code_commit"]),
        "device": str(metadata["device"]),
        "station_ids": [int(value) for value in metadata["station_ids"]],
        "selected_top5": [int(value) for value in metadata["selected_top5"]],
        "sorted_candidate_correlations": sorted_correlations,
        "bounds": metadata["bounds"],
        "sample_indices": metadata["sample_indices"],
        "station_means": metadata["station_means"],
        "station_stds": metadata["station_stds"],
        "center_mean": metadata["center_mean"],
        "center_std": metadata["center_std"],
        "peak_threshold_ugm3": metadata["peak_threshold_ugm3"],
        "fit_quantile_edges_ugm3": metadata["fit_quantile_edges_ugm3"],
        "confirm_timestamps_ns": metadata["confirm_timestamps_ns"],
        "confirm_block_labels": metadata["confirm_block_labels"],
        "confirm_scalar_timestamps_ns": metadata["confirm_scalar_timestamps_ns"],
        "confirm_scalar_leads": metadata["confirm_scalar_leads"],
        "confirm_scalar_block_labels": metadata["confirm_scalar_block_labels"],
    }


def config_fingerprint(config_payload: dict) -> str:
    encoded = json.dumps(
        config_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def verify_formal_git_state() -> str:
    """Require HEAD to identify exactly the code that a formal run would execute."""
    head = git_commit()
    if re.fullmatch(r"[0-9a-f]{40}", head) is None:
        raise ProtocolViolation("正式运行要求 HEAD 为 40 位小写十六进制 commit")
    for relative_path in IMPLEMENTATION_FILES:
        tracked = subprocess.run(
            ["git", "cat-file", "-e", f"HEAD:{relative_path}"],
            check=False,
            capture_output=True,
            text=True,
        )
        if tracked.returncode != 0:
            raise ProtocolViolation(f"正式运行要求实现文件已被 HEAD 跟踪: {relative_path}")
    status = subprocess.run(
        ["git", "status", "--porcelain"], check=False, capture_output=True, text=True
    )
    if status.returncode != 0 or status.stdout.strip():
        raise ProtocolViolation("正式运行要求 git status --porcelain 为空")
    return head


def scalarize_predictions(prediction, target, metadata, horizon):
    prediction_ugm3 = prediction * metadata["center_std"] + metadata["center_mean"]
    target_ugm3 = target * metadata["center_std"] + metadata["center_mean"]
    flat_prediction = prediction_ugm3.reshape(-1).astype(np.float64)
    flat_target = target_ugm3.reshape(-1).astype(np.float64)
    timestamps = np.asarray(metadata["confirm_scalar_timestamps_ns"], dtype=np.int64)
    blocks = np.asarray(metadata["confirm_scalar_block_labels"], dtype=np.int16)
    leads = np.tile(np.arange(1, horizon + 1, dtype=np.int16), len(prediction))
    if not (len(flat_prediction) == len(flat_target) == len(timestamps) == len(blocks) == len(leads)):
        raise RuntimeError("逐标量预测与共享时间/块标签长度不一致")
    return flat_prediction, flat_target, timestamps, leads, blocks


def save_prediction_artifact(path: Path, prediction, target, timestamps, leads, blocks):
    errors = prediction - target
    sse = np.asarray([np.square(errors[blocks == b]).sum() for b in range(1, 6)], dtype=np.float64)
    counts = np.asarray([(blocks == b).sum() for b in range(1, 6)], dtype=np.int64)
    np.savez_compressed(
        path,
        prediction_ugm3=prediction,
        target_ugm3=target,
        target_timestamp_ns=timestamps,
        lead=leads,
        block_label=blocks,
        sse_by_block=sse,
        element_count_by_block=counts,
    )


def run_station_task(
    frame: pd.DataFrame,
    opened_station_ids: tuple[int, ...],
    center_station_id: int,
    config: RunConfig,
    seeds: tuple[int, ...],
    output_dir: Path,
    device: torch.device,
    smoke: bool,
):
    datasets, metadata = prepare_shared_data(frame, center_station_id, config)
    run_identity = {
        "config": asdict(config),
        "center_station_id": center_station_id,
        "seeds": list(seeds),
        "arms": list(ARMS),
        "opened_station_ids": list(opened_station_ids),
        "smoke_test": smoke,
        "code_commit": git_commit(),
        "device": str(device),
    }
    task_dir = output_dir / f"{config.history}h_{config.horizon}h" / f"station_{center_station_id}"
    (task_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
    (task_dir / "predictions").mkdir()
    (task_dir / "training_logs").mkdir()
    metadata.update(run_identity)
    frozen_run_payload = build_frozen_run_payload(metadata)
    fingerprint = config_fingerprint(frozen_run_payload)
    metadata["frozen_run_payload"] = frozen_run_payload
    metadata["config_fingerprint"] = fingerprint
    (task_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest_rows = []
    for seed in seeds:
        baseline_checkpoint = task_dir / "checkpoints" / f"degraded_patchtst_seed{seed}.pt"
        for arm in ARMS:
            checkpoint = task_dir / "checkpoints" / f"{arm}_seed{seed}.pt"
            initialization = baseline_checkpoint if arm != "degraded_patchtst" else None
            if initialization is not None and not initialization.is_file():
                raise RuntimeError("ST 臂必须在同种子退化基线之后训练")
            started = time.perf_counter()
            model, history_rows, best_epoch, best_valid_loss = train_arm(
                config, datasets, arm, seed, device, checkpoint, initialization
            )
            confirm_loader = make_loader(datasets["confirm"], config, False, seed)
            prediction, target = predict(model, confirm_loader, device)
            scalar = scalarize_predictions(prediction, target, metadata, config.horizon)
            artifact_name = f"{arm}_seed{seed}.npz"
            save_prediction_artifact(task_dir / "predictions" / artifact_name, *scalar)
            pd.DataFrame(history_rows).to_csv(
                task_dir / "training_logs" / f"{arm}_seed{seed}.csv", index=False
            )
            manifest_rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "artifact": f"predictions/{artifact_name}",
                    "best_epoch": best_epoch,
                    "best_valid_loss": best_valid_loss,
                    "trained_epochs": len(history_rows),
                    "training_seconds": time.perf_counter() - started,
                    "initialized_from_degraded": initialization is not None,
                    "backbone_frozen": arm != "degraded_patchtst",
                    "config_fingerprint": fingerprint,
                }
            )
    pd.DataFrame(manifest_rows).to_csv(task_dir / "run_manifest.csv", index=False)


def synthetic_frame(length: int = 700, seed: int = 1400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=length, freq="h")
    common = np.sin(np.arange(length) / 11.0) * 12 + np.arange(length) * 0.02
    data = {100: common + rng.normal(0, 0.8, length)}
    for offset, station_id in enumerate((101, 102, 103, 104, 105, 106), start=1):
        data[station_id] = common * (1 - offset * 0.01) + rng.normal(0, 1 + offset * 0.05, length)
    frame = pd.DataFrame(data, index=index)
    frame.iloc[2:5, 3] = np.nan
    return frame


def write_stop(output_dir: Path, reason: str, context: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "STOP.json").write_text(
        json.dumps({"status": "STOP", "reason": reason, **context}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Validate and record the frozen access plan; open no CSV.")
    mode.add_argument("--smoke", action="store_true", help="Run one tiny synthetic-only, non-evidentiary task.")
    mode.add_argument("--formal", action="store_true", help="Run the complete preregistered matrix.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def ensure_new_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"输出目录必须不存在或为空，拒绝混入旧结果: {path}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    # Validate every formal access set before creating any output path.
    access_plan = {str(center): list(authorized_station_ids(center)) for center in B1_STATIONS}
    for center, ids in access_plan.items():
        validate_authorized_station_set(int(center), ids)
    print(json.dumps({"planned_station_access": access_plan, "forbidden": sorted(FORBIDDEN_STATIONS)}))
    ensure_new_output_dir(output_dir)
    if args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "access_preflight.json").write_text(
            json.dumps(
                {"planned_station_access": access_plan, "actual_opened_station_ids": [], "csv_reads": 0},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return
    device = torch.device(args.device)
    if args.smoke:
        config = replace(
            formal_config(24, 1),
            batch_size=32,
            epochs=1,
            patience=1,
            neighbor_hidden_dim=8,
            n_layers=1,
            d_model=8,
            n_heads=2,
            d_ff=16,
            dropout=0.0,
        )
        frame = synthetic_frame()
        run_station_task(frame, (), 100, config, (14001,), output_dir, device, smoke=True)
        print(f"合成冒烟产物: {output_dir.resolve()}")
        return

    verify_formal_git_state()
    output_dir.mkdir(parents=True, exist_ok=True)
    for center_station_id in B1_STATIONS:
        try:
            frame, opened = load_center_candidate_frame(center_station_id)
            for history, horizon in TASKS:
                run_station_task(
                    frame, opened, center_station_id, formal_config(history, horizon),
                    FORMAL_SEEDS, output_dir, device, smoke=False,
                )
        except InsufficientNeighbors as exc:
            write_stop(
                output_dir / "stops" / f"station_{center_station_id}", str(exc),
                {"center_station_id": center_station_id},
            )
    print(f"正式矩阵产物: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
