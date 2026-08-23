"""Run the P1 fixed-frequency MVP and capacity-matched time control."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from causal_frequency_data import prepare_causal_datasets
from frequency_residual_adapter import (
    FixedFrequencyResidualAdapter,
    FrozenSTForecastResidual,
    TimeResidualAdapter,
    trainable_parameter_count,
)
from run_causal_frequency_p0 import (
    P0ExperimentConfig,
    SELECTION_SEEDS,
    _load_checkpoint,
    _model_signature,
    _peak_metrics_row,
    _per_horizon_rows,
    _state_sha256,
    build_clean_model,
)
from run_st_patchtst_ablation import make_loader, regression_metrics, set_seed


P1_VARIANTS = (
    "locked_st_clean",
    "st_time_residual_adapter",
    "st_fixed_frequency_residual",
)
P1_SOURCE_FILES = (
    "causal_frequency_data.py",
    "frequency_residual_adapter.py",
    "run_causal_frequency_p0.py",
    "run_causal_frequency_p1.py",
    "ST_PatchTST_model.py",
)


@dataclass(frozen=True)
class P1ExperimentConfig(P0ExperimentConfig):
    stage: str = "p1_fixed_selection"
    adapter_hidden_dim: int = 32
    adapter_pool_bins: int = 4
    adapter_num_streams: int = 3
    beta_max: float = 0.5
    beta_init: float = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=int, default=168)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--seeds", default="2052,2053,2054")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--adapter-hidden-dim", type=int, default=32)
    parser.add_argument("--adapter-pool-bins", type=int, default=4)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--p0-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def _source_hash() -> str:
    digest = hashlib.sha256()
    for name in sorted(P1_SOURCE_FILES):
        digest.update(name.encode("utf-8"))
        digest.update(Path(name).read_bytes())
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
        "source_files": list(P1_SOURCE_FILES),
        "source_files_sha256": _source_hash(),
    }


def load_locked_base(
    config: P1ExperimentConfig,
    metadata: dict,
    seed: int,
    checkpoint_path: Path,
    device: torch.device,
):
    model = build_clean_model(
        config,
        "locked_st_clean",
        len(metadata["station_ids"]),
        metadata["center_station_idx"],
    )
    signature = _model_signature(config, metadata)
    payload = _load_checkpoint(
        checkpoint_path,
        model,
        signature,
        seed,
        allow_missing_alpha=False,
    )
    model = model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    loaded_hash = _state_sha256(model.state_dict())
    return model, payload, loaded_hash


@torch.no_grad()
def predict_model(
    model: nn.Module,
    loader,
    device: torch.device,
    disable_frequency: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    predictions = []
    targets = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        if disable_frequency:
            prediction = model.forward_components(
                x, disable_frequency=True
            )["prediction"]
        else:
            prediction = model(x)
        predictions.append(prediction.cpu())
        targets.append(y)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return torch.cat(predictions).numpy(), torch.cat(targets).numpy(), elapsed


def evaluate_locked_base(
    config: P1ExperimentConfig,
    datasets: dict,
    metadata: dict,
    seed: int,
    checkpoint_path: Path,
    output_dir: Path,
    device: torch.device,
) -> tuple[dict, list[dict], dict]:
    model, payload, state_hash = load_locked_base(
        config, metadata, seed, checkpoint_path, device
    )
    valid_loader = make_loader(datasets["valid"], config, False, seed)
    prediction, target, inference_seconds = predict_model(
        model, valid_loader, device
    )
    metrics = regression_metrics(
        target, prediction, metadata["center_mean"], metadata["center_std"]
    )
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        predictions_dir / f"locked_st_clean_seed{seed}.npz",
        prediction_scaled=prediction,
        target_scaled=target,
        prediction_ugm3=prediction * metadata["center_std"]
        + metadata["center_mean"],
        target_ugm3=target * metadata["center_std"] + metadata["center_mean"],
        target_origins=datasets["valid"].target_origins,
        target_timestamp_matrix_ns=datasets["valid"].target_timestamp_matrix_ns(),
    )
    row = {
        "variant": "locked_st_clean",
        "seed": seed,
        "best_epoch": int(payload["epoch"]),
        "best_valid_loss": float(payload["valid_loss"]),
        "training_seconds": 0.0,
        "evaluation_inference_seconds": inference_seconds,
        "evaluation_split": "valid",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": 0,
        "base_state_initial_sha256": state_hash,
        "base_state_final_sha256": state_hash,
        "base_state_unchanged": True,
        "beta": 0.0,
        "forecast_residual_rms_ratio": 0.0,
        "disable_frequency_rmse_ugm3": metrics["rmse_ugm3"],
        "disable_frequency_max_abs_difference_scaled": 0.0,
        "stream_1_energy_ratio": math.nan,
        "stream_2_energy_ratio": math.nan,
        "stream_3_energy_ratio": math.nan,
        "adapter_output_weight_norm": 0.0,
        "initial_exact_base": True,
        "capacity_difference_percent_vs_time": 0.0,
        **metrics,
    }
    per_horizon = _per_horizon_rows(
        "locked_st_clean",
        seed,
        target,
        prediction,
        metadata["center_mean"],
        metadata["center_std"],
    )
    peak = _peak_metrics_row(
        "locked_st_clean",
        seed,
        target,
        prediction,
        metadata["center_mean"],
        metadata["center_std"],
        metadata["peak_q90_ugm3"],
    )
    return row, per_horizon, peak


def build_adapter(config: P1ExperimentConfig, variant: str) -> nn.Module:
    common = {
        "seq_len": config.history,
        "pred_len": config.horizon,
        "hidden_dim": config.adapter_hidden_dim,
        "pool_bins": config.adapter_pool_bins,
        "dropout": config.dropout,
        "num_streams": config.adapter_num_streams,
    }
    if variant == "st_time_residual_adapter":
        return TimeResidualAdapter(**common)
    if variant == "st_fixed_frequency_residual":
        return FixedFrequencyResidualAdapter(**common)
    raise ValueError(f"未知 P1 适配器变体: {variant}")


def _adapter_signature(
    config: P1ExperimentConfig,
    metadata: dict,
    base_state_sha256: str,
    variant: str,
) -> dict:
    signature = {
        "protocol_version": config.protocol_version,
        "variant": variant,
        "history": config.history,
        "horizon": config.horizon,
        "station_ids": metadata["station_ids"],
        "time_index_sha256": metadata["time_index_sha256"],
        "base_state_sha256": base_state_sha256,
        "adapter_hidden_dim": config.adapter_hidden_dim,
        "adapter_pool_bins": config.adapter_pool_bins,
        "adapter_num_streams": config.adapter_num_streams,
        "beta_max": config.beta_max,
        "beta_init": config.beta_init,
    }
    encoded = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    return {**signature, "sha256": hashlib.sha256(encoded.encode()).hexdigest()}


def _save_adapter_checkpoint(
    path: Path,
    model: FrozenSTForecastResidual,
    signature: dict,
    seed: int,
    epoch: int,
    valid_loss: float,
) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "signature": signature,
            "seed": seed,
            "epoch": epoch,
            "valid_loss": valid_loss,
        },
        path,
    )


def _load_adapter_checkpoint(
    path: Path,
    model: FrozenSTForecastResidual,
    signature: dict,
    seed: int,
) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["signature"] != signature or int(payload["seed"]) != seed:
        raise RuntimeError(f"P1 检查点签名或种子不匹配: {path}")
    model.load_state_dict(payload["state_dict"], strict=True)
    return payload


@torch.no_grad()
def collect_adapter_diagnostics(
    model: FrozenSTForecastResidual,
    loader,
    device: torch.device,
) -> dict:
    base_square = 0.0
    residual_square = 0.0
    element_count = 0
    stream_squares = None
    stream_count = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        components = model.forward_components(x)
        base = components["base_prediction"]
        residual = components["forecast_residual"]
        streams = components["streams"]
        base_square += float(base.square().sum().item())
        residual_square += float(residual.square().sum().item())
        element_count += base.numel()
        batch_stream_squares = streams.square().sum(dim=(0, 2)).cpu()
        stream_squares = (
            batch_stream_squares
            if stream_squares is None
            else stream_squares + batch_stream_squares
        )
        stream_count += streams.shape[0] * streams.shape[-1]
    stream_energy = stream_squares / max(stream_count, 1)
    total_stream_energy = float(stream_energy.sum().item())
    stream_ratios = stream_energy / max(total_stream_energy, 1e-12)
    return {
        "beta": float(
            (model.max_beta * torch.sigmoid(model.beta_logit)).item()
        ),
        "forecast_residual_rms_ratio": math.sqrt(
            residual_square / max(element_count, 1)
        )
        / max(math.sqrt(base_square / max(element_count, 1)), 1e-12),
        "stream_1_energy_ratio": float(stream_ratios[0].item()),
        "stream_2_energy_ratio": float(stream_ratios[1].item()),
        "stream_3_energy_ratio": float(stream_ratios[2].item()),
        "adapter_output_weight_norm": float(
            model.adapter.forecast_out.weight.norm().item()
        ),
    }


def train_adapter_run(
    config: P1ExperimentConfig,
    datasets: dict,
    metadata: dict,
    variant: str,
    seed: int,
    base_checkpoint: Path,
    output_dir: Path,
    device: torch.device,
    time_trainable_parameter_count: int,
) -> tuple[dict, list[dict], dict, dict]:
    base_model, _, base_state_hash = load_locked_base(
        config, metadata, seed, base_checkpoint, device
    )
    # Reset after constructing/loading the base so time and frequency heads have
    # identical trainable initialization under the same seed.
    set_seed(seed)
    adapter = build_adapter(config, variant)
    model = FrozenSTForecastResidual(
        base_model,
        adapter,
        beta_max=config.beta_max,
        beta_init=config.beta_init,
    ).to(device)
    trainable_count = trainable_parameter_count(model)
    capacity_difference_percent = 100 * abs(
        trainable_count - time_trainable_parameter_count
    ) / max(time_trainable_parameter_count, 1)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    base_parameter_ids = {id(parameter) for parameter in model.base_model.parameters()}
    if any(id(parameter) in base_parameter_ids for parameter in trainable_parameters):
        raise AssertionError("锁定 ST 参数进入 P1 优化器")

    train_loader = make_loader(datasets["train"], config, True, seed)
    valid_loader = make_loader(datasets["valid"], config, False, seed)
    first_x, _ = next(iter(valid_loader))
    first_x = first_x.to(device)
    model.eval()
    with torch.no_grad():
        initial = model.forward_components(first_x)
    initial_max_difference = float(
        (initial["prediction"] - initial["base_prediction"]).abs().max().item()
    )
    initial_exact_base = initial_max_difference == 0.0
    if not initial_exact_base:
        raise AssertionError("P1 零初始化未严格退化为锁定 ST")

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    loss_fn = nn.MSELoss()
    checkpoint_dir = output_dir / "checkpoints"
    logs_dir = output_dir / "training_logs"
    predictions_dir = output_dir / "predictions"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{variant}_seed{seed}.pt"
    signature = _adapter_signature(
        config, metadata, base_state_hash, variant
    )

    best_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history_rows = []
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
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
                    f"{variant} seed={seed} 出现非有限损失"
                )
            loss.backward()
            if any(
                parameter.grad is not None
                and not torch.isfinite(parameter.grad).all()
                for parameter in trainable_parameters
            ):
                raise FloatingPointError(
                    f"{variant} seed={seed} 出现非有限梯度"
                )
            nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(x)
            train_count += len(x)

        valid_prediction, valid_target, _ = predict_model(
            model, valid_loader, device
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
            _save_adapter_checkpoint(
                checkpoint_path,
                model,
                signature,
                seed,
                epoch,
                valid_loss,
            )
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.patience:
            break

    training_seconds = time.perf_counter() - started
    payload = _load_adapter_checkpoint(
        checkpoint_path, model, signature, seed
    )
    model = model.to(device)
    final_base_state_hash = _state_sha256(model.base_model.state_dict())
    base_state_unchanged = final_base_state_hash == base_state_hash
    if not base_state_unchanged:
        raise AssertionError("P1 训练改变了锁定 ST state")

    prediction, target, inference_seconds = predict_model(
        model, valid_loader, device
    )
    disabled_prediction, _, _ = predict_model(
        model, valid_loader, device, disable_frequency=True
    )
    metrics = regression_metrics(
        target, prediction, metadata["center_mean"], metadata["center_std"]
    )
    disabled_metrics = regression_metrics(
        target,
        disabled_prediction,
        metadata["center_mean"],
        metadata["center_std"],
    )
    diagnostics = collect_adapter_diagnostics(model, valid_loader, device)
    disabled_max_difference = float(
        np.max(
            np.abs(
                disabled_prediction
                - np.load(
                    output_dir / "predictions" / f"locked_st_clean_seed{seed}.npz"
                )["prediction_scaled"]
            )
        )
    )
    if disabled_max_difference != 0.0:
        raise AssertionError("禁用 P1 残差后未精确恢复同种子锁定 ST")

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
        "best_epoch": int(payload["epoch"]),
        "best_valid_loss": float(payload["valid_loss"]),
        "training_seconds": training_seconds,
        "evaluation_inference_seconds": inference_seconds,
        "evaluation_split": "valid",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": trainable_count,
        "base_state_initial_sha256": base_state_hash,
        "base_state_final_sha256": final_base_state_hash,
        "base_state_unchanged": base_state_unchanged,
        "disable_frequency_rmse_ugm3": disabled_metrics["rmse_ugm3"],
        "disable_frequency_max_abs_difference_scaled": disabled_max_difference,
        "initial_exact_base": initial_exact_base,
        "capacity_difference_percent_vs_time": capacity_difference_percent,
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
    peak = _peak_metrics_row(
        variant,
        seed,
        target,
        prediction,
        metadata["center_mean"],
        metadata["center_std"],
        metadata["peak_q90_ugm3"],
    )
    runtime = {
        "variant": variant,
        "seed": seed,
        "initial_exact_base": initial_exact_base,
        "base_state_unchanged": base_state_unchanged,
        "disable_frequency_exact_base": disabled_max_difference == 0.0,
        "all_metrics_finite": bool(
            np.isfinite([metrics["rmse_ugm3"], metrics["mae_ugm3"], best_loss]).all()
        ),
        "capacity_difference_percent_vs_time": capacity_difference_percent,
    }
    print(
        f"[{variant} seed={seed}] valid RMSE={metrics['rmse_ugm3']:.4f} μg/m³, "
        f"MAE={metrics['mae_ugm3']:.4f}, best_epoch={best_epoch}, "
        f"beta={diagnostics['beta']:.4f}"
    )
    return row, per_horizon, peak, runtime


def aggregate_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "rmse_ugm3",
        "mae_ugm3",
        "smape_percent",
        "best_valid_loss",
        "training_seconds",
        "evaluation_inference_seconds",
        "beta",
        "forecast_residual_rms_ratio",
        "stream_1_energy_ratio",
        "stream_2_energy_ratio",
        "stream_3_energy_ratio",
        "adapter_output_weight_norm",
        "disable_frequency_rmse_ugm3",
    ]
    rows = []
    for variant, group in raw_df.groupby("variant", sort=False):
        row = {"variant": variant, "runs": len(group)}
        for column in columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = (
                float(group[column].std(ddof=1)) if len(group) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def paired_differences(raw_df: pd.DataFrame) -> pd.DataFrame:
    indexed = {
        variant: raw_df[raw_df["variant"] == variant].set_index("seed")
        for variant in P1_VARIANTS
    }
    comparisons = (
        ("st_time_residual_adapter", "locked_st_clean"),
        ("st_fixed_frequency_residual", "locked_st_clean"),
        ("st_fixed_frequency_residual", "st_time_residual_adapter"),
    )
    rows = []
    for candidate_name, baseline_name in comparisons:
        candidate = indexed[candidate_name]
        baseline = indexed[baseline_name]
        for seed in sorted(set(candidate.index) & set(baseline.index)):
            candidate_rmse = float(candidate.loc[seed, "rmse_ugm3"])
            baseline_rmse = float(baseline.loc[seed, "rmse_ugm3"])
            rows.append(
                {
                    "comparison": f"{candidate_name}_vs_{baseline_name}",
                    "candidate_variant": candidate_name,
                    "baseline_variant": baseline_name,
                    "seed": int(seed),
                    "rmse_difference_ugm3": candidate_rmse - baseline_rmse,
                    "mae_difference_ugm3": float(
                        candidate.loc[seed, "mae_ugm3"]
                        - baseline.loc[seed, "mae_ugm3"]
                    ),
                    "rmse_relative_improvement_percent": 100
                    * (baseline_rmse - candidate_rmse)
                    / baseline_rmse,
                }
            )
    return pd.DataFrame(rows)


def evaluate_stage(
    config: P1ExperimentConfig,
    raw_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    runtime_checks: list[dict],
    smoke: bool,
) -> dict:
    runtime_pass = all(
        item["initial_exact_base"]
        and item["base_state_unchanged"]
        and item["disable_frequency_exact_base"]
        and item["all_metrics_finite"]
        and item["capacity_difference_percent_vs_time"] <= 5
        for item in runtime_checks
    )
    if smoke:
        return {
            "gate": "F2",
            "status": "passed" if runtime_pass else "failed",
            "evidence_label": "smoke",
            "runtime_checks_passed": runtime_pass,
            "note": "F2 also requires the separately recorded unit-test suite.",
        }

    fixed_vs_st = paired_df[
        paired_df["comparison"]
        == "st_fixed_frequency_residual_vs_locked_st_clean"
    ]
    fixed = raw_df[raw_df["variant"] == "st_fixed_frequency_residual"]
    time_control = raw_df[raw_df["variant"] == "st_time_residual_adapter"]
    improvements = fixed_vs_st["rmse_relative_improvement_percent"].to_numpy()
    checks = {
        "runtime_checks_passed": runtime_pass,
        "complete_seed_variant_grid": len(raw_df)
        == len(SELECTION_SEEDS) * len(P1_VARIANTS),
        "selection_seeds_exact": sorted(raw_df["seed"].unique().tolist())
        == list(SELECTION_SEEDS),
        "fixed_three_of_three_better_than_st": len(improvements) == 3
        and bool(np.all(improvements > 0)),
        "fixed_mean_improvement_at_least_0_5_percent": len(improvements) == 3
        and float(np.mean(improvements)) >= 0.5,
        "fixed_mean_rmse_better_than_time_control": float(
            fixed["rmse_ugm3"].mean()
        )
        < float(time_control["rmse_ugm3"].mean()),
        "capacity_difference_at_most_5_percent": float(
            fixed["capacity_difference_percent_vs_time"].max()
        )
        <= 5,
    }
    return {
        "gate": "F3",
        "status": "passed" if all(checks.values()) else "failed",
        "evidence_label": "selection",
        "checks": checks,
        "fixed_vs_st_improved_seed_count": int(np.sum(improvements > 0)),
        "fixed_vs_st_mean_rmse_improvement_percent": (
            float(np.mean(improvements)) if len(improvements) else None
        ),
        "fixed_rmse_mean": float(fixed["rmse_ugm3"].mean()),
        "time_control_rmse_mean": float(time_control["rmse_ugm3"].mean()),
        "allowed_next_gate": "F4" if all(checks.values()) else None,
    }


def write_band_masks(output_dir: Path, config: P1ExperimentConfig) -> None:
    adapter = FixedFrequencyResidualAdapter(
        seq_len=config.history,
        pred_len=config.horizon,
        hidden_dim=config.adapter_hidden_dim,
        pool_bins=config.adapter_pool_bins,
        dropout=config.dropout,
        num_streams=config.adapter_num_streams,
    )
    frequencies = adapter.decomposition.frequencies.cpu().numpy()
    masks = adapter.decomposition.masks.cpu().numpy()
    period = np.divide(
        1.0,
        frequencies,
        out=np.full_like(frequencies, np.nan),
        where=frequencies > 0,
    )
    diagnostics_dir = output_dir / "frequency_diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "frequency_cycles_per_hour": frequencies,
            "period_hours": period,
            "low_mask": masks[0],
            "mid_mask": masks[1],
            "high_mask": masks[2],
            "mask_sum": masks.sum(axis=0),
        }
    ).to_csv(diagnostics_dir / "fixed_band_masks.csv", index=False)


def write_report(
    output_dir: Path,
    config: P1ExperimentConfig,
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    stage: dict,
) -> None:
    lines = [
        "# P1 固定频带与容量控制自动摘要",
        "",
        (
            "> SMOKE TEST DATA - DO NOT USE AS THESIS EVIDENCE."
            if config.evidence_label == "smoke"
            else "> SELECTION DATA - validation only; test metrics were not computed."
        ),
        "",
        f"- 任务：{config.history}→{config.horizon}",
        f"- 当前 Gate：{stage['gate']} {stage['status']}",
        "",
        "| 变体 | RMSE (μg/m³) | MAE (μg/m³) | 残差RMS比 |",
        "|---|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['variant']} | {row['rmse_ugm3_mean']:.4f} ± "
            f"{row['rmse_ugm3_std']:.4f} | {row['mae_ugm3_mean']:.4f} ± "
            f"{row['mae_ugm3_std']:.4f} | "
            f"{row['forecast_residual_rms_ratio_mean']:.6f} |"
        )
    if not paired.empty:
        lines.extend(
            [
                "",
                "完整配对差值见 `paired_differences.csv`。F3 只有在固定频带同时超过锁定 ST 和等容量时域控制时才通过。",
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
    config = P1ExperimentConfig(
        history=args.history,
        horizon=args.horizon,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        adapter_hidden_dim=args.adapter_hidden_dim,
        adapter_pool_bins=args.adapter_pool_bins,
        evidence_label=evidence_label,
    )
    p0_dir = Path(
        args.p0_dir
        or (
            "experiments/results/causal_frequency_ablation/p0_bridge/"
            f"{config.history}h_{config.horizon}h_selection"
        )
    )
    output_dir = Path(
        args.output_dir
        or (
            "experiments/results/causal_frequency_ablation/p1_fixed_selection/"
            f"{config.history}h_{config.horizon}h_{evidence_label}"
        )
    )
    if args.quick and not output_dir.name.endswith("_smoke"):
        output_dir = output_dir.with_name(output_dir.name + "_smoke")
    if "causal_frequency_ablation" not in output_dir.parts:
        raise ValueError("P1 输出必须位于 causal_frequency_ablation 目录")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    datasets, metadata = prepare_causal_datasets(config)
    p0_metadata = json.loads(
        (p0_dir / "dataset_metadata.json").read_text(encoding="utf-8")
    )
    for key in (
        "time_index_sha256",
        "station_ids",
        "center_station_idx",
        "station_means",
        "station_stds",
    ):
        if metadata[key] != p0_metadata[key]:
            raise RuntimeError(f"P1 数据契约与 P0 不一致: {key}")

    time_probe = TimeResidualAdapter(
        seq_len=config.history,
        pred_len=config.horizon,
        hidden_dim=config.adapter_hidden_dim,
        pool_bins=config.adapter_pool_bins,
        dropout=config.dropout,
        num_streams=config.adapter_num_streams,
    )
    time_trainable_count = trainable_parameter_count(time_probe) + 1
    fixed_probe = FixedFrequencyResidualAdapter(
        seq_len=config.history,
        pred_len=config.horizon,
        hidden_dim=config.adapter_hidden_dim,
        pool_bins=config.adapter_pool_bins,
        dropout=config.dropout,
        num_streams=config.adapter_num_streams,
    )
    fixed_trainable_count = trainable_parameter_count(fixed_probe) + 1
    capacity_difference_percent = 100 * abs(
        fixed_trainable_count - time_trainable_count
    ) / max(time_trainable_count, 1)
    experiment_config = {
        **asdict(config),
        "seeds": seeds,
        "variants": list(P1_VARIANTS),
        "device": str(device),
        "smoke_test": args.quick,
        "evaluation_data_used": ["train", "valid"],
        "test_metrics_computed": False,
        "p0_checkpoint_dir": str(p0_dir),
        "time_adapter_trainable_parameters": time_trainable_count,
        "fixed_frequency_trainable_parameters": fixed_trainable_count,
        "capacity_difference_percent": capacity_difference_percent,
        "fixed_band_period_boundaries_hours": [48.0, 12.0],
        "explicit_band_gate": "uniform_1_over_3",
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
    write_band_masks(output_dir, config)

    raw_rows = []
    per_horizon_rows = []
    peak_rows = []
    runtime_checks = []
    for seed in seeds:
        base_checkpoint = p0_dir / "checkpoints" / f"locked_st_clean_seed{seed}.pt"
        if not base_checkpoint.is_file():
            raise FileNotFoundError(f"缺少 P0 锁定 ST 检查点: {base_checkpoint}")
        base_row, base_horizon, base_peak = evaluate_locked_base(
            config,
            datasets,
            metadata,
            seed,
            base_checkpoint,
            output_dir,
            device,
        )
        raw_rows.append(base_row)
        per_horizon_rows.extend(base_horizon)
        peak_rows.append(base_peak)
        for variant in P1_VARIANTS[1:]:
            row, horizon_rows, peak, runtime = train_adapter_run(
                config,
                datasets,
                metadata,
                variant,
                seed,
                base_checkpoint,
                output_dir,
                device,
                time_trainable_count,
            )
            raw_rows.append(row)
            per_horizon_rows.extend(horizon_rows)
            peak_rows.append(peak)
            runtime_checks.append(runtime)
        pd.DataFrame(raw_rows).to_csv(output_dir / "raw_metrics.csv", index=False)

    raw_df = pd.DataFrame(raw_rows)
    summary_df = aggregate_results(raw_df)
    paired_df = paired_differences(raw_df)
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)
    paired_df.to_csv(output_dir / "paired_differences.csv", index=False)
    pd.DataFrame(per_horizon_rows).to_csv(
        output_dir / "per_horizon_metrics.csv", index=False
    )
    pd.DataFrame(peak_rows).to_csv(output_dir / "peak_metrics.csv", index=False)
    stage_status = evaluate_stage(
        config, raw_df, paired_df, runtime_checks, args.quick
    )
    (output_dir / "stage_status.json").write_text(
        json.dumps(stage_status, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output_dir / "runtime_checks.json").write_text(
        json.dumps(runtime_checks, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    write_report(output_dir, config, summary_df, paired_df, stage_status)
    print(f"P1 产物已写入: {output_dir.resolve()}")
    print(f"{stage_status['gate']}: {stage_status['status']}")


if __name__ == "__main__":
    main()
