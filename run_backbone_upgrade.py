"""Run the preregistered two-stage backbone-upgrade experiment.

P1 selects an already-trained center-only backbone using ``best_valid_loss``
only.  P2 freezes that backbone and trains the existing Top-5 + station-bias +
bounded forecast-residual branch.  Rows and artifacts are flushed after every
seed, making the matrix safely resumable.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

import run_beijing_leakfree_coverage as beijing
import run_cross_city_generalization as guangzhou
import run_st_patchtst_ablation as legacy
from backbone_candidates import (
    candidate_in_protocol_coverage,
    expected_candidates,
    load_candidate_adapter,
)


MAIN_CHECKOUT = Path("/home/hansel/Documents/ITProject/Python/PatchTST")
BASELINE_ROOT = MAIN_CHECKOUT / "experiments/results/baselines"
BEIJING_ST_ASSET_ROOTS = (
    Path("experiments/results/beijing_leakfree_coverage"),
    MAIN_CHECKOUT / "experiments/results/beijing_leakfree_coverage",
    Path("/home/hansel/.herdr/worktrees/PatchTST/experiment-beijing-leakfree-coverage-ablation/experiments/results/beijing_leakfree_coverage"),
)
GUANGZHOU_ST_ASSET_ROOTS = (
    Path("experiments/results/guangzhou_horizon_coverage"),
    MAIN_CHECKOUT / "experiments/results/guangzhou_horizon_coverage",
)
BEIJING_SEEDS = (2047, 2048, 2049)
BEIJING_HEADLINE_SEEDS = (2047, 2048, 2049, 2050, 2051)
GUANGZHOU_SEEDS = (7001, 7002, 7003)
GUANGZHOU_STATIONS = tuple(guangzhou.B1_STATIONS)
SPATIAL_VARIANT = "st_sparse_station_bias_delta_forecast"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", choices=("beijing", "guangzhou", "all"), default="all")
    parser.add_argument("--configs", default=None, help="逗号分隔 LxH")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--stations", default=None, help="广州中心站，逗号分隔")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-root", default="experiments/results/backbone_upgrade")
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="北京24→1/seed2047，两轮训练")
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _first_existing(paths) -> Path | None:
    return next((Path(path) for path in paths if Path(path).is_file()), None)


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {}
    return json.loads(str(value))


def _baseline_dir(city: str, history: int, horizon: int, station_id: int) -> Path:
    root = BASELINE_ROOT / city / f"{history}h_{horizon}h"
    return root if city == "beijing" else root / f"station_{station_id}"


def _degraded_sources(
    city: str, history: int, horizon: int, station_id: int, seed: int
) -> tuple[Path | None, Path | None]:
    config_name = f"{history}h_{horizon}h"
    if city == "beijing":
        dirs = [root / config_name for root in BEIJING_ST_ASSET_ROOTS]
        metric = _first_existing(path / "raw_metrics.csv" for path in dirs)
        checkpoint = _first_existing(
            path / "checkpoints" / f"degraded_patchtst_seed{seed}.pt" for path in dirs
        )
        return metric, checkpoint
    dirs = [root / config_name / f"station_{station_id}" for root in GUANGZHOU_ST_ASSET_ROOTS]
    metric = _first_existing(path / "run_manifest.csv" for path in dirs)
    checkpoint = _first_existing(
        path / "checkpoints" / f"degraded_patchtst_seed{seed}.pt" for path in dirs
    )
    return metric, checkpoint


def collect_candidate_rows(
    city: str,
    history: int,
    horizon: int,
    station_id: int,
    seed: int,
    *,
    probe_context: tuple[Any, dict[str, Any], dict[str, Any], torch.device] | None = None,
) -> list[dict[str, Any]]:
    """Expand the expected universe and annotate every candidate's availability."""
    baseline_dir = _baseline_dir(city, history, horizon, station_id)
    baseline_metrics = _read_csv(baseline_dir / "raw_metrics.csv")
    rows: list[dict[str, Any]] = []
    for spec in expected_candidates(city, history, horizon):
        row = {
            "city": city,
            "history": history,
            "horizon": horizon,
            "station_id": station_id,
            "seed": seed,
            "arm": spec.arm,
            "capacity": spec.capacity,
            "variant": spec.variant,
            "best_valid_loss": float("nan"),
            "checkpoint_path": "",
            "metric_source": "",
            "hyperparameters": "{}",
            "candidate_status": "metric_missing",
            "candidate_reason": "",
            "selected": False,
        }
        if not candidate_in_protocol_coverage(city, history, horizon, spec):
            row["candidate_status"] = "not_in_protocol_coverage"
            rows.append(row)
            continue
        if spec.arm != "degraded_patchtst":
            matches = baseline_metrics[
                (baseline_metrics.get("variant", pd.Series(dtype=str)).astype(str) == spec.variant)
                & (baseline_metrics.get("seed", pd.Series(dtype=str)).astype(str) == str(seed))
            ]
            checkpoint = baseline_dir / "checkpoints" / f"{spec.variant}_seed{seed}.pt"
            row["checkpoint_path"] = str(checkpoint)
            row["metric_source"] = str(baseline_dir / "raw_metrics.csv")
            if not matches.empty:
                source = matches.iloc[-1]
                row["best_valid_loss"] = float(source["best_valid_loss"])
                row["hyperparameters"] = str(source.get("hyperparameters", "{}"))
                row["candidate_status"] = "eligible"
                if str(source.get("status", "completed")) != "completed":
                    row["candidate_status"] = f"source_{source.get('status')}"
            if not checkpoint.is_file() and row["candidate_status"] == "eligible":
                row["candidate_status"] = "checkpoint_missing"
        else:
            metric_path, checkpoint = _degraded_sources(
                city, history, horizon, station_id, seed
            )
            row["checkpoint_path"] = "" if checkpoint is None else str(checkpoint)
            row["metric_source"] = "" if metric_path is None else str(metric_path)
            if metric_path is not None:
                metrics = _read_csv(metric_path)
                arm_column = "variant" if "variant" in metrics else "arm"
                matches = metrics[
                    (metrics[arm_column].astype(str) == "degraded_patchtst")
                    & (metrics["seed"].astype(str) == str(seed))
                ]
                if not matches.empty:
                    row["best_valid_loss"] = float(matches.iloc[-1]["best_valid_loss"])
                    row["candidate_status"] = "eligible"
            if checkpoint is None and row["candidate_status"] == "eligible":
                row["candidate_status"] = "checkpoint_missing"
        if row["candidate_status"] == "eligible" and not math.isfinite(row["best_valid_loss"]):
            row["candidate_status"] = "invalid_validation"
        if row["candidate_status"] == "eligible" and probe_context is not None:
            config, datasets, metadata, device = probe_context
            usable, reason = probe_candidate_backbone(
                row, config, datasets["valid"], metadata, device
            )
            if not usable:
                row["candidate_status"] = "backbone_nonfinite"
                row["candidate_reason"] = reason
        rows.append(row)
    return select_by_validation(rows)


def select_by_validation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select strictly from status and validation loss; other metrics are ignored."""
    eligible = [
        row for row in rows
        if row.get("candidate_status") == "eligible"
        and math.isfinite(float(row["best_valid_loss"]))
    ]
    if not eligible:
        return rows
    winner = min(eligible, key=lambda row: (float(row["best_valid_loss"]), row["variant"]))
    winner_id = winner["variant"]
    for row in rows:
        row["selected"] = row["variant"] == winner_id and row["candidate_status"] == "eligible"
    return rows


def selected_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    winners = [row for row in rows if bool(row.get("selected"))]
    if not winners:
        return None
    if len(winners) != 1:
        raise RuntimeError(f"验证集主干选择至多一个胜者，实际 {len(winners)}")
    return winners[0]


def prepare_data(city: str, config, station_id: int):
    if city == "beijing":
        return beijing.prepare_datasets_leakfree(config)
    frame, opened = guangzhou.load_center_candidate_frame(station_id)
    cross_config = guangzhou.RunConfig(
        history=config.history,
        horizon=config.horizon,
        batch_size=config.batch_size,
        epochs=config.epochs,
        patience=config.patience,
    )
    source, metadata = guangzhou.prepare_shared_data(frame, station_id, cross_config)
    datasets = {"train": source["fit"], "valid": source["val"], "test": source["confirm"]}
    metadata = {
        **metadata,
        "center_station_id": station_id,
        "opened_station_ids": list(opened),
        "split_sizes": {name: len(dataset) for name, dataset in datasets.items()},
        "station_correlations": metadata["candidate_correlations"],
    }
    return datasets, metadata


def build_upgraded_model(config, metadata: dict, winner: dict, device: torch.device):
    model = legacy.build_model(
        config,
        SPATIAL_VARIANT,
        num_stations=len(metadata["station_ids"]),
        center_idx=metadata["center_station_idx"],
    )
    adapter = load_candidate_adapter(
        winner["arm"],
        config.history,
        config.horizon,
        winner["checkpoint_path"],
        _json_dict(winner.get("hyperparameters")),
        map_location=device,
    )
    model.patch_tst = adapter
    for parameter in model.patch_tst.parameters():
        parameter.requires_grad = False
    return model.to(device)


@torch.no_grad()
def probe_candidate_backbone(
    candidate: dict[str, Any], config, dataset, metadata: dict, device: torch.device
) -> tuple[bool, str]:
    """Load one P1 checkpoint and require a finite real validation forward."""
    try:
        adapter = load_candidate_adapter(
            candidate["arm"],
            config.history,
            config.horizon,
            candidate["checkpoint_path"],
            _json_dict(candidate.get("hyperparameters")),
            map_location=device,
        ).to(device)
        loader = legacy.make_loader(dataset, config, False, 0)
        x, _ = next(iter(loader))
        center_idx = metadata["center_station_idx"]
        center = x[:, center_idx:center_idx + 1].to(device)
        adapter.eval()
        output = adapter(center)
        if torch.isfinite(output).all():
            return True, "finite_validation_forward"
        nonfinite_count = int((~torch.isfinite(output)).sum().item())
        return False, f"validation_forward_nonfinite(count={nonfinite_count})"
    except Exception as error:  # checkpoint load/forward failures are unusable too
        return False, f"validation_forward_failed({type(error).__name__}: {error})"


@torch.no_grad()
def initial_forward_audit(model, dataset, config, metadata, device) -> dict[str, Any]:
    loader = legacy.make_loader(dataset, config, False, 0)
    x, _ = next(iter(loader))
    x = x.to(device)
    center = x[:, metadata["center_station_idx"]:metadata["center_station_idx"] + 1]
    model.eval()
    backbone_output = model.patch_tst(center)
    mounted_output = model(x)
    return {
        "backbone_output_finite": bool(torch.isfinite(backbone_output).all().item()),
        "mounted_output_finite": bool(torch.isfinite(mounted_output).all().item()),
        "backbone_nonfinite_count": int((~torch.isfinite(backbone_output)).sum().item()),
        "mounted_nonfinite_count": int((~torch.isfinite(mounted_output)).sum().item()),
        "reference_scale": float(backbone_output.abs().max().item()),
        "zero_init_max_abs": float((mounted_output - backbone_output).abs().max().item()),
    }


def initial_equivalence_max_abs(model, dataset, config, metadata, device) -> float:
    """Compatibility wrapper retained for the protocol's original T4 check."""
    return float(
        initial_forward_audit(model, dataset, config, metadata, device)[
            "zero_init_max_abs"
        ]
    )


def _backbone_nonfinite_result(
    winner: dict[str, Any], reason: str, audit: dict[str, Any] | None = None
) -> dict[str, Any]:
    audit = audit or {}
    return {
        "status": "backbone_nonfinite",
        "failure_reason": reason,
        "selected_variant": winner.get("variant", ""),
        "selected_arm": winner.get("arm", ""),
        "selected_capacity": winner.get("capacity", ""),
        "selected_best_valid_loss": winner.get("best_valid_loss", math.nan),
        "source_checkpoint": winner.get("checkpoint_path", ""),
        "zero_init_max_abs": audit.get("zero_init_max_abs", math.nan),
        "zero_init_tolerance": math.nan,
        "zero_init_ok": False,
        "backbone_output_finite": audit.get("backbone_output_finite", False),
        "mounted_output_finite": audit.get("mounted_output_finite", False),
        "backbone_nonfinite_count": audit.get("backbone_nonfinite_count", 0),
        "mounted_nonfinite_count": audit.get("mounted_nonfinite_count", 0),
    }


def _prediction_path(output_root: Path, identity: str) -> Path:
    return output_root / "predictions" / f"{identity}.npz"


def train_one_upgrade(
    config,
    datasets,
    metadata: dict,
    winner: dict,
    seed: int,
    output_root: Path,
    identity: str,
    device: torch.device,
) -> dict[str, Any]:
    legacy.set_seed(seed)
    model = build_upgraded_model(config, metadata, winner, device)
    if any(parameter.requires_grad for parameter in model.patch_tst.parameters()):
        raise AssertionError("主干冻结失败")
    audit = initial_forward_audit(
        model, datasets["valid"], config, metadata, device
    )
    zero_init_max_abs = audit["zero_init_max_abs"]
    if not audit["backbone_output_finite"]:
        return _backbone_nonfinite_result(
            winner,
            "主干验证前向包含 NaN/Inf "
            f"(count={audit['backbone_nonfinite_count']})",
            audit,
        )
    if not audit["mounted_output_finite"] or not math.isfinite(zero_init_max_abs):
        return _backbone_nonfinite_result(
            winner,
            "挂载模型输出或零初始化等价差值为 NaN/Inf "
            f"(mounted_nonfinite_count={audit['mounted_nonfinite_count']})",
            audit,
        )
    # 该检查的科学目的是"确认空间残差在起点处≈0"（预测端线性层零初始化）。
    # 实测：个别主干在"整批切片调用"与"单独中心通道调用"之间存在 GEMM 路径差异，
    # 会产生 ~1e-7 量级的浮点差（完成行的记录值恰为 0.0，故非逻辑错误）。
    # 因此改为**相对容差 + 记录字段、不中断运行**：容差取 1e-6 × 预测幅度量级。
    reference_scale = audit["reference_scale"]
    if not math.isfinite(reference_scale):
        return _backbone_nonfinite_result(
            winner, "主干预测幅度为 NaN/Inf", audit
        )
    zero_init_tolerance = 1e-6 * max(reference_scale, 1.0)
    zero_init_ok = bool(zero_init_max_abs <= zero_init_tolerance)
    if not zero_init_ok:
        raise AssertionError(
            f"零初始化不等价: max_abs={zero_init_max_abs}  tolerance={zero_init_tolerance}"
        )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    loss_fn = nn.MSELoss()
    train_loader = legacy.make_loader(datasets["train"], config, True, seed)
    valid_loader = legacy.make_loader(datasets["valid"], config, False, seed)
    test_loader = legacy.make_loader(datasets["test"], config, False, seed)
    checkpoint = output_root / "checkpoints" / f"{identity}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_loss, best_epoch, stale = math.inf, 0, 0
    history_rows = []
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{identity} 出现非有限训练损失")
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total += float(loss.item()) * len(x)
            count += len(x)
        valid_prediction, valid_target, _ = legacy.predict(
            model, valid_loader, device, metadata["center_station_idx"]
        )
        valid_loss = float(np.mean((valid_prediction - valid_target) ** 2))
        scheduler.step(valid_loss)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": total / count,
                "valid_loss": valid_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if valid_loss < best_loss - 1e-7:
            best_loss, best_epoch, stale = valid_loss, epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
        if stale >= config.patience:
            break
    training_seconds = time.perf_counter() - started
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    prediction, target, inference_seconds = legacy.predict(
        model, test_loader, device, metadata["center_station_idx"]
    )
    backbone_prediction, _, _ = legacy.predict(
        model, test_loader, device, metadata["center_station_idx"], neighbor_mode="disable"
    )
    if not np.isfinite(backbone_prediction).all():
        return _backbone_nonfinite_result(
            winner,
            "主干完整测试前向包含 NaN/Inf "
            f"(count={int((~np.isfinite(backbone_prediction)).sum())})",
            audit,
        )
    if not np.isfinite(prediction).all():
        return _backbone_nonfinite_result(
            winner,
            "挂载模型完整测试前向包含 NaN/Inf "
            f"(count={int((~np.isfinite(prediction)).sum())})",
            audit,
        )
    metrics = legacy.regression_metrics(
        target, prediction, metadata["center_mean"], metadata["center_std"]
    )
    backbone_metrics = legacy.regression_metrics(
        target, backbone_prediction, metadata["center_mean"], metadata["center_std"]
    )
    try:
        diagnostics = legacy.collect_spatial_diagnostics(model, test_loader, device)
    except (ValueError, OverflowError) as error:
        raise FloatingPointError(
            f"{identity} 空间诊断出现非有限数值: {error}"
        ) from error
    prediction_path = _prediction_path(output_root, identity)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prediction_path,
        prediction_scaled=prediction,
        backbone_prediction_scaled=backbone_prediction,
        target_scaled=target,
        prediction_ugm3=prediction * metadata["center_std"] + metadata["center_mean"],
        backbone_prediction_ugm3=(
            backbone_prediction * metadata["center_std"] + metadata["center_mean"]
        ),
        target_ugm3=target * metadata["center_std"] + metadata["center_mean"],
    )
    logs = output_root / "training_logs"
    logs.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history_rows).to_csv(logs / f"{identity}.csv", index=False)
    return {
        "status": "completed",
        "selected_variant": winner["variant"],
        "selected_arm": winner["arm"],
        "selected_capacity": winner["capacity"],
        "selected_best_valid_loss": winner["best_valid_loss"],
        "source_checkpoint": winner["checkpoint_path"],
        "best_epoch": best_epoch,
        "best_valid_loss": best_loss,
        "training_seconds": training_seconds,
        "test_inference_seconds": inference_seconds,
        "prediction_file": str(prediction_path.relative_to(output_root)),
        "checkpoint_file": str(checkpoint.relative_to(output_root)),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "backbone_parameter_count": sum(p.numel() for p in model.patch_tst.parameters()),
        "backbone_frozen": not any(p.requires_grad for p in model.patch_tst.parameters()),
        "zero_init_max_abs": zero_init_max_abs,
        "zero_init_tolerance": zero_init_tolerance,
        "zero_init_ok": zero_init_ok,
        "backbone_output_finite": audit["backbone_output_finite"],
        "mounted_output_finite": audit["mounted_output_finite"],
        "backbone_nonfinite_count": audit["backbone_nonfinite_count"],
        "mounted_nonfinite_count": audit["mounted_nonfinite_count"],
        **metrics,
        **{f"backbone_{key}": value for key, value in backbone_metrics.items()},
        **diagnostics,
    }


def _identity(city: str, history: int, horizon: int, station_id: int, seed: int) -> str:
    return f"{city}_{history}h_{horizon}h_station{station_id}_seed{seed}"


def _save_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _upsert(rows: list[dict], new_row: dict, keys: tuple[str, ...]) -> None:
    identity = tuple(new_row.get(key) for key in keys)
    rows[:] = [row for row in rows if tuple(row.get(key) for key in keys) != identity]
    rows.append(new_row)


def _is_oom(error: BaseException, device: torch.device) -> bool:
    message = str(error).lower()
    return device.type == "cuda" and (
        isinstance(error, torch.OutOfMemoryError)
        or "out of memory" in message
        or "hip error out of memory" in message
    )


def config_for(history: int, horizon: int, smoke: bool):
    config = beijing.config_for(history, horizon)
    if smoke:
        config = replace(config, epochs=2, patience=2)
    return config


def run_spec(args: argparse.Namespace):
    if args.smoke:
        return (("beijing", 24, 1, 1013, 2047),)
    cities = ("beijing", "guangzhou") if args.city == "all" else (args.city,)
    requested_tasks = None
    if args.configs:
        requested_tasks = tuple(
            tuple(int(part) for part in item.lower().split("x"))
            for item in args.configs.split(",")
        )
    requested_seeds = (
        tuple(int(item) for item in args.seeds.split(",")) if args.seeds else None
    )
    requested_stations = (
        tuple(int(item) for item in args.stations.split(",")) if args.stations else None
    )
    rows = []
    for city in cities:
        tasks = requested_tasks or (
            tuple(beijing.TASK_GRID) if city == "beijing" else tuple(guangzhou.TASKS)
        )
        stations = (1013,) if city == "beijing" else (requested_stations or GUANGZHOU_STATIONS)
        for history, horizon in tasks:
            default_seeds = (
                (BEIJING_HEADLINE_SEEDS if (history, horizon) in ((24, 1), (168, 6)) else BEIJING_SEEDS)
                if city == "beijing" else GUANGZHOU_SEEDS
            )
            for station_id in stations:
                for seed in requested_seeds or default_seeds:
                    rows.append((city, history, horizon, station_id, seed))
    return tuple(rows)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root) / ("smoke" if args.smoke else "formal")
    output_root.mkdir(parents=True, exist_ok=True)
    selection_path = output_root / "selection.csv"
    raw_path = output_root / "raw_metrics.csv"
    selection_rows = _read_csv(selection_path).to_dict("records")
    raw_rows = _read_csv(raw_path).to_dict("records")
    device = torch.device(args.device)
    prepared_key = None
    config = datasets = metadata = None
    for city, history, horizon, station_id, seed in run_spec(args):
        current_key = (city, history, horizon, station_id)
        if current_key != prepared_key:
            config = config_for(history, horizon, args.smoke)
            datasets, metadata = prepare_data(city, config, station_id)
            prepared_key = current_key
        candidate_rows = collect_candidate_rows(
            city,
            history,
            horizon,
            station_id,
            seed,
            probe_context=(config, datasets, metadata, device),
        )
        for row in candidate_rows:
            _upsert(
                selection_rows,
                row,
                ("city", "history", "horizon", "station_id", "seed", "variant"),
            )
        _save_rows(selection_rows, selection_path)
        identity = _identity(city, history, horizon, station_id, seed)
        prior = [row for row in raw_rows if row.get("run_id") == identity]
        if prior and str(prior[-1].get("status")) in {
            "completed", "infeasible_oom", "nonfinite", "backbone_nonfinite",
            "no_usable_backbone",
        }:
            print(f"[跳过] {identity}: {prior[-1]['status']}")
            continue
        winner = selected_candidate(candidate_rows)
        if winner is None:
            unusable = [
                row for row in candidate_rows
                if row.get("candidate_status") == "backbone_nonfinite"
            ]
            result = {
                "status": "no_usable_backbone",
                "failure_reason": (
                    "P1 无可用主干；backbone_nonfinite="
                    f"{len(unusable)}, total_candidates={len(candidate_rows)}"
                ),
                "selected_variant": "",
                "selected_arm": "",
                "selected_capacity": "",
                "source_checkpoint": "",
            }
            row = {
                "run_id": identity,
                "city": city,
                "history": history,
                "horizon": horizon,
                "station_id": station_id,
                "seed": seed,
                "smoke_test": args.smoke,
                **result,
            }
            _upsert(raw_rows, row, ("run_id",))
            _save_rows(raw_rows, raw_path)
            print(f"[登记] {identity}: no_usable_backbone")
            continue
        if args.selection_only:
            print(f"[P1] {identity}: {winner['variant']} ({winner['best_valid_loss']:.6f})")
            continue
        base = {
            "run_id": identity,
            "city": city,
            "history": history,
            "horizon": horizon,
            "station_id": station_id,
            "seed": seed,
            "smoke_test": args.smoke,
        }
        try:
            result = train_one_upgrade(
                config, datasets, metadata, winner, seed, output_root, identity, device
            )
            result["learning_rate_retry"] = False
        except FloatingPointError as first_error:
            print(f"[{identity}] 非有限损失，按协议以 lr/10 重试一次")
            retry_config = replace(config, learning_rate=config.learning_rate / 10.0)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            try:
                result = train_one_upgrade(
                    retry_config, datasets, metadata, winner, seed, output_root, identity, device
                )
                result["learning_rate_retry"] = True
                result["retry_reason"] = str(first_error)
            except FloatingPointError as retry_error:
                result = {
                    "status": "nonfinite",
                    "failure_reason": str(retry_error),
                    "learning_rate_retry": True,
                    "selected_variant": winner["variant"],
                }
        except (RuntimeError, torch.OutOfMemoryError) as error:
            if not _is_oom(error, device):
                raise
            result = {
                "status": "infeasible_oom",
                "failure_reason": str(error),
                "selected_variant": winner["variant"],
            }
            if device.type == "cuda":
                torch.cuda.empty_cache()
        row = {**base, **result}
        _upsert(raw_rows, row, ("run_id",))
        _save_rows(raw_rows, raw_path)
        (output_root / "metadata").mkdir(exist_ok=True)
        (output_root / "metadata" / f"{identity}.json").write_text(
            json.dumps(
                {
                    "config": asdict(config),
                    "dataset": metadata,
                    "winner": winner,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        if result["status"] == "completed":
            change = 100 * (result["rmse_ugm3"] / result["backbone_rmse_ugm3"] - 1)
            print(
                f"[完成] {identity}: {winner['variant']}，"
                f"ST={result['rmse_ugm3']:.4f}，主干={result['backbone_rmse_ugm3']:.4f}，"
                f"相对变化={change:.3f}%"
            )
        else:
            print(f"[登记] {identity}: {result['status']}")


if __name__ == "__main__":
    main()
