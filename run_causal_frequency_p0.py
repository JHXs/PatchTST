"""Run the P0 leak-safe bridge for the causal frequency research branch.

P0 intentionally contains no frequency-domain module.  It rebuilds the
degraded PatchTST and the locked Top-5 ST model under FREQ-PROTOCOL-v1, using
validation data only for the 2052--2054 selection seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from causal_frequency_data import prepare_causal_datasets
from run_st_patchtst_ablation import (
    ExperimentConfig,
    collect_spatial_diagnostics,
    make_loader,
    predict,
    regression_metrics,
    set_seed,
)
from ST_PatchTST_model import ST_PatchTST


P0_VARIANTS = ("degraded_patchtst_clean", "locked_st_clean")
SELECTION_SEEDS = (2052, 2053, 2054)
MODEL_SOURCE_FILES = (
    "causal_frequency_data.py",
    "run_causal_frequency_p0.py",
    "ST_PatchTST_model.py",
    "run_st_patchtst_ablation.py",
)


@dataclass(frozen=True)
class P0ExperimentConfig(ExperimentConfig):
    """Locked P0 configuration layered on the established ST hyperparameters."""

    history: int = 168
    horizon: int = 6
    batch_size: int = 512
    epochs: int = 30
    patience: int = 6
    sparse_neighbor_top_k: int = 5
    evaluation_split: str = "valid"
    initialize_from_degraded: bool = True
    freeze_backbone: bool = True
    protocol_version: str = "FREQ-PROTOCOL-v1"
    stage: str = "p0_bridge"
    evidence_label: str = "selection"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=int, default=168)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--seeds", default="2052,2053,2054")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Defaults to experiments/results/causal_frequency_ablation/"
            "p0_bridge/<history>h_<horizon>h_<evidence_label>"
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run one seed for at most two epochs and keep Gate F1 pending.",
    )
    return parser.parse_args()


def build_clean_model(
    config: P0ExperimentConfig,
    variant: str,
    num_stations: int,
    center_idx: int,
) -> ST_PatchTST:
    """Build clean variants with identical locked structure except spatial alpha."""

    if variant not in P0_VARIANTS:
        raise ValueError(f"未知 P0 变体: {variant}")
    if config.sparse_neighbor_top_k != 5:
        raise ValueError("P0 锁定结构要求 sparse_neighbor_top_k=5")
    if num_stations - 1 < config.sparse_neighbor_top_k:
        raise ValueError(
            f"Top-5 需要至少 5 个邻站，当前只有 {num_stations - 1} 个"
        )
    is_degraded = variant == "degraded_patchtst_clean"
    return ST_PatchTST(
        c_in=num_stations,
        c_out=1,
        seq_len=config.history,
        pred_dim=config.horizon,
        num_stations=num_stations,
        feat_size=1,
        center_station_idx=center_idx,
        neighbor_hidden_dim=config.neighbor_hidden_dim,
        neighbor_dropout=config.dropout,
        alpha_max=0.0 if is_degraded else config.forecast_alpha_max,
        alpha_init=0.0 if is_degraded else config.forecast_alpha_init,
        gate_mode="pairwise_summary",
        use_null_neighbor=True,
        neighbor_value_mode="difference",
        fusion_stage="forecast",
        spatial_pool_bins=config.spatial_pool_bins,
        use_station_gate_bias=True,
        neighbor_top_k=5,
        use_forecast_confidence=False,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_model=config.d_model,
        d_ff=config.d_ff,
        dropout=config.dropout,
        patch_len=config.patch_len,
        stride=config.stride,
        padding_patch=True,
    )


def _state_sha256(state_dict: dict[str, torch.Tensor], prefix: str = "") -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        if prefix and not name.startswith(prefix):
            continue
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype="<i8").tobytes())
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _source_files_sha256(paths: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for path_text in sorted(paths):
        path = Path(path_text)
        digest.update(path_text.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_metadata() -> dict:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {
        "git_commit": commit,
        "git_worktree_dirty": dirty,
        "source_files": list(MODEL_SOURCE_FILES),
        "source_files_sha256": _source_files_sha256(MODEL_SOURCE_FILES),
    }


def _model_signature(config: P0ExperimentConfig, metadata: dict) -> dict:
    signature = {
        "protocol_version": config.protocol_version,
        "history": config.history,
        "horizon": config.horizon,
        "station_ids": metadata["station_ids"],
        "center_station_idx": metadata["center_station_idx"],
        "time_index_sha256": metadata["time_index_sha256"],
        "n_layers": config.n_layers,
        "n_heads": config.n_heads,
        "d_model": config.d_model,
        "d_ff": config.d_ff,
        "dropout": config.dropout,
        "patch_len": config.patch_len,
        "stride": config.stride,
        "neighbor_hidden_dim": config.neighbor_hidden_dim,
        "spatial_pool_bins": config.spatial_pool_bins,
        "sparse_neighbor_top_k": config.sparse_neighbor_top_k,
    }
    payload = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    return {**signature, "sha256": hashlib.sha256(payload.encode()).hexdigest()}


def _save_checkpoint(
    path: Path,
    model: ST_PatchTST,
    signature: dict,
    variant: str,
    seed: int,
    epoch: int,
    valid_loss: float,
) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "signature": signature,
            "variant": variant,
            "seed": seed,
            "epoch": epoch,
            "valid_loss": valid_loss,
            "backbone_sha256": _state_sha256(
                model.state_dict(), prefix="patch_tst."
            ),
        },
        path,
    )


def _load_checkpoint(
    path: Path,
    model: ST_PatchTST,
    expected_signature: dict,
    expected_seed: int,
    allow_missing_alpha: bool,
) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("signature") != expected_signature:
        raise RuntimeError(f"检查点配置签名不匹配: {path}")
    if int(payload.get("seed", -1)) != expected_seed:
        raise RuntimeError(f"检查点种子不匹配: {path}")
    incompatible = model.load_state_dict(payload["state_dict"], strict=False)
    allowed_missing = {"alpha_logit"} if allow_missing_alpha else set()
    if set(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "检查点参数不兼容: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return payload


def _per_horizon_rows(
    variant: str,
    seed: int,
    target_scaled: np.ndarray,
    prediction_scaled: np.ndarray,
    center_mean: float,
    center_std: float,
) -> list[dict]:
    rows = []
    for step in range(target_scaled.shape[-1]):
        metrics = regression_metrics(
            target_scaled[..., step:step + 1],
            prediction_scaled[..., step:step + 1],
            center_mean,
            center_std,
        )
        rows.append(
            {
                "variant": variant,
                "seed": seed,
                "horizon_step": step + 1,
                **metrics,
            }
        )
    return rows


def _peak_metrics_row(
    variant: str,
    seed: int,
    target_scaled: np.ndarray,
    prediction_scaled: np.ndarray,
    center_mean: float,
    center_std: float,
    q90_ugm3: float,
) -> dict:
    target = target_scaled.reshape(-1) * center_std + center_mean
    prediction = prediction_scaled.reshape(-1) * center_std + center_mean
    error = prediction - target
    peak_mask = target > q90_ugm3
    non_peak_mask = ~peak_mask
    predicted_peak = prediction > q90_ugm3
    true_positive = int(np.sum(peak_mask & predicted_peak))
    false_positive = int(np.sum(non_peak_mask & predicted_peak))
    false_negative = int(np.sum(peak_mask & ~predicted_peak))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    def group_metrics(mask: np.ndarray) -> tuple[float, float]:
        if not mask.any():
            return math.nan, math.nan
        group_error = error[mask]
        return (
            float(np.sqrt(np.mean(group_error ** 2))),
            float(np.mean(np.abs(group_error))),
        )

    peak_rmse, peak_mae = group_metrics(peak_mask)
    non_peak_rmse, non_peak_mae = group_metrics(non_peak_mask)
    return {
        "variant": variant,
        "seed": seed,
        "q90_ugm3": q90_ugm3,
        "peak_element_count": int(peak_mask.sum()),
        "non_peak_element_count": int(non_peak_mask.sum()),
        "peak_rmse_ugm3": peak_rmse,
        "peak_mae_ugm3": peak_mae,
        "non_peak_rmse_ugm3": non_peak_rmse,
        "non_peak_mae_ugm3": non_peak_mae,
        "threshold_precision": precision,
        "threshold_recall": recall,
        "threshold_f1": f1,
    }


def freeze_patchtst_backbone(model: ST_PatchTST) -> None:
    """Freeze PatchTST parameters and running-state updates."""

    for parameter in model.patch_tst.parameters():
        parameter.requires_grad = False
    model.patch_tst.eval()


def train_p0_run(
    config: P0ExperimentConfig,
    datasets: dict,
    metadata: dict,
    variant: str,
    seed: int,
    output_dir: Path,
    device: torch.device,
    initialization_checkpoint: Path | None = None,
) -> tuple[dict, list[dict], dict]:
    """Train one P0 run and return raw, per-horizon and peak metrics."""

    set_seed(seed)
    model = build_clean_model(
        config,
        variant,
        num_stations=len(metadata["station_ids"]),
        center_idx=metadata["center_station_idx"],
    )
    signature = _model_signature(config, metadata)
    initialized_backbone_sha256 = None
    if initialization_checkpoint is not None:
        payload = _load_checkpoint(
            initialization_checkpoint,
            model,
            signature,
            seed,
            allow_missing_alpha=True,
        )
        initialized_backbone_sha256 = payload["backbone_sha256"]
        loaded_backbone_sha256 = _state_sha256(
            model.state_dict(), prefix="patch_tst."
        )
        if loaded_backbone_sha256 != initialized_backbone_sha256:
            raise AssertionError("退化检查点加载后 PatchTST 哈希不一致")

    model = model.to(device)
    is_locked = variant == "locked_st_clean"
    if is_locked:
        if initialization_checkpoint is None:
            raise ValueError("locked_st_clean 必须由同种子退化检查点初始化")
        freeze_patchtst_backbone(model)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("没有可训练参数")
    frozen_parameter_ids = {
        id(parameter) for parameter in model.patch_tst.parameters()
    }
    if is_locked and any(
        id(parameter) in frozen_parameter_ids for parameter in trainable_parameters
    ):
        raise AssertionError("冻结主干参数仍进入优化器")

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    loss_fn = nn.MSELoss()
    train_loader = make_loader(datasets["train"], config, True, seed)
    valid_loader = make_loader(datasets["valid"], config, False, seed)

    checkpoint_dir = output_dir / "checkpoints"
    logs_dir = output_dir / "training_logs"
    predictions_dir = output_dir / "predictions"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{variant}_seed{seed}.pt"

    best_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history_rows = []
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        if is_locked:
            # model.train() recursively toggles the frozen PatchTST; restore eval
            # so BatchNorm/dropout state cannot change behind requires_grad=False.
            model.patch_tst.eval()
        train_loss_sum = 0.0
        train_count = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = loss_fn(prediction, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"{variant} seed={seed} 出现非有限训练损失"
                )
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(x)
            train_count += len(x)

        valid_prediction, valid_target, _ = predict(
            model, valid_loader, device, metadata["center_station_idx"]
        )
        train_loss = train_loss_sum / train_count
        valid_loss = float(np.mean((valid_prediction - valid_target) ** 2))
        scheduler.step(valid_loss)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if valid_loss < best_loss - 1e-7:
            best_loss = valid_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            _save_checkpoint(
                checkpoint_path,
                model,
                signature,
                variant,
                seed,
                epoch,
                valid_loss,
            )
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.patience:
            break

    training_seconds = time.perf_counter() - started
    _load_checkpoint(
        checkpoint_path,
        model,
        signature,
        seed,
        allow_missing_alpha=False,
    )
    model = model.to(device)
    final_backbone_sha256 = _state_sha256(
        model.state_dict(), prefix="patch_tst."
    )
    backbone_frozen_unchanged = (
        None
        if not is_locked
        else final_backbone_sha256 == initialized_backbone_sha256
    )
    if is_locked and not backbone_frozen_unchanged:
        raise AssertionError("locked_st_clean 训练后 PatchTST 参数发生变化")

    prediction, target, inference_seconds = predict(
        model, valid_loader, device, metadata["center_station_idx"]
    )
    metrics = regression_metrics(
        target, prediction, metadata["center_mean"], metadata["center_std"]
    )
    diagnostics = collect_spatial_diagnostics(model, valid_loader, device)
    disabled_prediction, _, _ = predict(
        model,
        valid_loader,
        device,
        metadata["center_station_idx"],
        neighbor_mode="disable",
    )
    disabled_metrics = regression_metrics(
        target,
        disabled_prediction,
        metadata["center_mean"],
        metadata["center_std"],
    )
    metrics["disable_spatial_rmse_ugm3"] = disabled_metrics["rmse_ugm3"]
    metrics["disable_spatial_mae_ugm3"] = disabled_metrics["mae_ugm3"]
    metrics["disable_spatial_max_abs_prediction_difference_scaled"] = float(
        np.max(np.abs(prediction - disabled_prediction))
    )

    pd.DataFrame(history_rows).to_csv(
        logs_dir / f"{variant}_seed{seed}.csv", index=False
    )
    np.savez_compressed(
        predictions_dir / f"{variant}_seed{seed}.npz",
        prediction_scaled=prediction,
        target_scaled=target,
        prediction_ugm3=prediction * metadata["center_std"]
        + metadata["center_mean"],
        target_ugm3=target * metadata["center_std"] + metadata["center_mean"],
        target_origins=datasets["valid"].target_origins,
        target_timestamp_matrix_ns=datasets["valid"].target_timestamp_matrix_ns(),
    )

    row = {
        "variant": variant,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_valid_loss": best_loss,
        "training_seconds": training_seconds,
        "evaluation_inference_seconds": inference_seconds,
        "evaluation_split": "valid",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "initialized_from_degraded": initialization_checkpoint is not None,
        "backbone_initialized_sha256": initialized_backbone_sha256,
        "backbone_final_sha256": final_backbone_sha256,
        "backbone_frozen_unchanged": backbone_frozen_unchanged,
        **metrics,
        **diagnostics,
    }
    per_horizon = _per_horizon_rows(
        variant,
        seed,
        target,
        prediction,
        metadata["center_mean"],
        metadata["center_std"],
    )
    peak_row = _peak_metrics_row(
        variant,
        seed,
        target,
        prediction,
        metadata["center_mean"],
        metadata["center_std"],
        metadata["peak_q90_ugm3"],
    )
    print(
        f"[{variant} seed={seed}] valid RMSE={metrics['rmse_ugm3']:.4f} μg/m³, "
        f"MAE={metrics['mae_ugm3']:.4f}, best_epoch={best_epoch}"
    )
    return row, per_horizon, peak_row


def aggregate_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "rmse_ugm3",
        "mae_ugm3",
        "smape_percent",
        "best_valid_loss",
        "training_seconds",
        "evaluation_inference_seconds",
        "alpha",
        "spatial_residual_rms_ratio",
        "gate_normalized_entropy",
        "null_neighbor_weight_mean",
        "gate_sample_std_mean",
        "disable_spatial_rmse_ugm3",
    ]
    rows = []
    for variant, group in raw_df.groupby("variant", sort=False):
        row = {"variant": variant, "runs": len(group)}
        for column in metric_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = (
                float(group[column].std(ddof=1)) if len(group) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def paired_differences(raw_df: pd.DataFrame) -> pd.DataFrame:
    baseline = raw_df[
        raw_df["variant"] == "degraded_patchtst_clean"
    ].set_index("seed")
    candidate = raw_df[raw_df["variant"] == "locked_st_clean"].set_index("seed")
    rows = []
    for seed in sorted(set(baseline.index) & set(candidate.index)):
        baseline_rmse = float(baseline.loc[seed, "rmse_ugm3"])
        candidate_rmse = float(candidate.loc[seed, "rmse_ugm3"])
        rows.append(
            {
                "variant": "locked_st_clean",
                "baseline_variant": "degraded_patchtst_clean",
                "seed": int(seed),
                "rmse_difference_ugm3": candidate_rmse - baseline_rmse,
                "mae_difference_ugm3": float(
                    candidate.loc[seed, "mae_ugm3"]
                    - baseline.loc[seed, "mae_ugm3"]
                ),
                "rmse_relative_change_percent": 100
                * (candidate_rmse / baseline_rmse - 1),
                "rmse_relative_improvement_percent": 100
                * (baseline_rmse - candidate_rmse)
                / baseline_rmse,
            }
        )
    return pd.DataFrame(rows)


def evaluate_f1_gate(
    config: P0ExperimentConfig,
    metadata: dict,
    raw_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    smoke: bool,
) -> dict:
    base = {
        "gate": "F1",
        "protocol_version": config.protocol_version,
        "evidence_label": config.evidence_label,
        "task": f"{config.history}h_{config.horizon}h",
        "required_primary_task": "168h_6h",
    }
    if smoke:
        return {
            **base,
            "status": "pending",
            "reason": "smoke run cannot evaluate Gate F1",
        }
    if (config.history, config.horizon) != (168, 6):
        return {
            **base,
            "status": "pending",
            "reason": "auxiliary task is reported but does not decide Gate F1",
        }

    expected_rows = len(SELECTION_SEEDS) * len(P0_VARIANTS)
    improvements = paired_df["rmse_relative_improvement_percent"].to_numpy()
    checks = {
        "complete_seed_variant_grid": len(raw_df) == expected_rows,
        "selection_seeds_exact": sorted(raw_df["seed"].unique().tolist())
        == list(SELECTION_SEEDS),
        "evaluation_split_is_valid": set(raw_df["evaluation_split"]) == {"valid"},
        "target_timestamps_disjoint": bool(
            metadata["target_timestamp_intersections_empty"]
        ),
        "top5_available": len(metadata["station_ids"]) - 1 >= 5,
        "locked_backbone_unchanged": bool(
            raw_df.loc[
                raw_df["variant"] == "locked_st_clean",
                "backbone_frozen_unchanged",
            ].all()
        ),
        "three_of_three_rmse_improved": len(improvements) == 3
        and bool(np.all(improvements > 0)),
        "mean_rmse_improvement_at_least_0_5_percent": len(improvements) == 3
        and float(np.mean(improvements)) >= 0.5,
        "all_core_metrics_finite": bool(
            np.isfinite(raw_df[["rmse_ugm3", "mae_ugm3", "best_valid_loss"]]).all().all()
        ),
    }
    return {
        **base,
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "improved_seed_count": int(np.sum(improvements > 0)),
        "paired_seed_count": int(len(improvements)),
        "mean_rmse_relative_improvement_percent": (
            float(np.mean(improvements)) if len(improvements) else None
        ),
        "minimum_rmse_relative_improvement_percent": (
            float(np.min(improvements)) if len(improvements) else None
        ),
    }


def write_auto_report(
    output_dir: Path,
    config: P0ExperimentConfig,
    metadata: dict,
    summary_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    gate_status: dict,
) -> None:
    lines = [
        "# 频域分支 P0 自动实验摘要",
        "",
        (
            "> SMOKE TEST DATA - DO NOT USE AS THESIS EVIDENCE."
            if config.evidence_label == "smoke"
            else "> SELECTION DATA - validation only; do not report as final confirmation."
        ),
        "",
        f"- 任务：{config.history}→{config.horizon}",
        f"- 评估划分：{config.evaluation_split}",
        f"- 训练期筛选站点数：{len(metadata['station_ids'])}",
        f"- 目标时间交集为空：{metadata['target_timestamp_intersections_empty']}",
        f"- Gate F1 状态：{gate_status['status']}",
        "",
        "## 验证集聚合结果",
        "",
        "| 变体 | RMSE (μg/m³) | MAE (μg/m³) | SMAPE (%) |",
        "|---|---:|---:|---:|",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['variant']} | {row['rmse_ugm3_mean']:.4f} ± "
            f"{row['rmse_ugm3_std']:.4f} | {row['mae_ugm3_mean']:.4f} ± "
            f"{row['mae_ugm3_std']:.4f} | {row['smape_percent_mean']:.3f} ± "
            f"{row['smape_percent_std']:.3f} |"
        )
    if not paired_df.empty:
        lines.extend(
            [
                "",
                "## 锁定 ST 相对退化基线",
                "",
                f"- 改善种子数：{int((paired_df['rmse_difference_ugm3'] < 0).sum())}/"
                f"{len(paired_df)}",
                "- 平均 RMSE 相对改善："
                f"{paired_df['rmse_relative_improvement_percent'].mean():.4f}%",
            ]
        )
    (output_dir / "auto_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if args.quick:
        seeds = seeds[:1]
        args.epochs = min(args.epochs, 2)
        args.patience = min(args.patience, 2)
        evidence_label = "smoke"
    else:
        if seeds != list(SELECTION_SEEDS):
            raise ValueError(
                f"正式 selection 种子必须严格为 {list(SELECTION_SEEDS)}"
            )
        evidence_label = "selection"

    config = P0ExperimentConfig(
        history=args.history,
        horizon=args.horizon,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        evidence_label=evidence_label,
    )
    output_dir = Path(
        args.output_dir
        or (
            "experiments/results/causal_frequency_ablation/p0_bridge/"
            f"{config.history}h_{config.horizon}h_{evidence_label}"
        )
    )
    if args.quick and not output_dir.name.endswith("_smoke"):
        output_dir = output_dir.with_name(output_dir.name + "_smoke")
    if "causal_frequency_ablation" not in output_dir.parts:
        raise ValueError("P0 输出必须位于 causal_frequency_ablation 目录")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    datasets, metadata = prepare_causal_datasets(config)
    experiment_config = {
        **asdict(config),
        "seeds": seeds,
        "variants": list(P0_VARIANTS),
        "device": str(device),
        "smoke_test": args.quick,
        "evaluation_data_used": ["train", "valid"],
        "test_metrics_computed": False,
        **_git_metadata(),
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(experiment_config, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output_dir / "dataset_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    result_rows = []
    per_horizon_rows = []
    peak_rows = []
    for seed in seeds:
        degraded_checkpoint = None
        for variant in P0_VARIANTS:
            row, horizon_rows, peak_row = train_p0_run(
                config,
                datasets,
                metadata,
                variant,
                seed,
                output_dir,
                device,
                initialization_checkpoint=degraded_checkpoint,
            )
            result_rows.append(row)
            per_horizon_rows.extend(horizon_rows)
            peak_rows.append(peak_row)
            pd.DataFrame(result_rows).to_csv(
                output_dir / "raw_metrics.csv", index=False
            )
            degraded_checkpoint = (
                output_dir / "checkpoints" / f"degraded_patchtst_clean_seed{seed}.pt"
            )

    raw_df = pd.DataFrame(result_rows)
    summary_df = aggregate_results(raw_df)
    paired_df = paired_differences(raw_df)
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)
    paired_df.to_csv(output_dir / "paired_differences.csv", index=False)
    pd.DataFrame(per_horizon_rows).to_csv(
        output_dir / "per_horizon_metrics.csv", index=False
    )
    pd.DataFrame(peak_rows).to_csv(output_dir / "peak_metrics.csv", index=False)
    (output_dir / "frequency_diagnostics").mkdir(exist_ok=True)
    gate_status = evaluate_f1_gate(
        config, metadata, raw_df, paired_df, args.quick
    )
    (output_dir / "stage_status.json").write_text(
        json.dumps(gate_status, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    write_auto_report(
        output_dir, config, metadata, summary_df, paired_df, gate_status
    )
    print(f"P0 产物已写入: {output_dir.resolve()}")
    print(f"Gate F1: {gate_status['status']}")


if __name__ == "__main__":
    main()
