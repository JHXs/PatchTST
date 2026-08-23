"""Run the sole pre-registered causal local-filterbank alternative (FA1)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from causal_frequency_data import prepare_causal_datasets
from causal_local_filterbank_adapter import CausalFilterBankResidualAdapter
from frequency_residual_adapter import FrozenSTForecastResidual, trainable_parameter_count
from run_causal_frequency_p0 import (
    SELECTION_SEEDS,
    _peak_metrics_row,
    _per_horizon_rows,
    _state_sha256,
)
from run_causal_frequency_p1 import (
    P1ExperimentConfig,
    _adapter_signature,
    _load_adapter_checkpoint,
    _save_adapter_checkpoint,
    aggregate_results,
    collect_adapter_diagnostics,
    load_locked_base,
    predict_model,
)
from run_st_patchtst_ablation import make_loader, regression_metrics, set_seed


ALTERNATIVE_VARIANT = "st_causal_filterbank_residual"
CONTROL_VARIANTS = ("locked_st_clean", "st_time_residual_adapter")
ALL_VARIANTS = (*CONTROL_VARIANTS, ALTERNATIVE_VARIANT)
SOURCE_FILES = (
    "causal_frequency_data.py",
    "causal_local_filterbank_adapter.py",
    "frequency_residual_adapter.py",
    "run_causal_frequency_p0.py",
    "run_causal_frequency_p1.py",
    "run_causal_frequency_p1a.py",
    "ST_PatchTST_model.py",
)


@dataclass(frozen=True)
class P1AExperimentConfig(P1ExperimentConfig):
    stage: str = "p1_alternative_selection"
    short_scale_hours: int = 12
    long_scale_hours: int = 48


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
    parser.add_argument("--p0-dir", default=None)
    parser.add_argument("--control-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def _files_sha256(paths: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for name in sorted(paths):
        digest.update(name.encode("utf-8"))
        digest.update(Path(name).read_bytes())
    return digest.hexdigest()


def _path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
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
        "source_files": list(SOURCE_FILES),
        "source_files_sha256": _files_sha256(SOURCE_FILES),
    }


def build_alternative_adapter(config: P1AExperimentConfig) -> nn.Module:
    return CausalFilterBankResidualAdapter(
        seq_len=config.history,
        pred_len=config.horizon,
        hidden_dim=config.adapter_hidden_dim,
        pool_bins=config.adapter_pool_bins,
        dropout=config.dropout,
        num_streams=config.adapter_num_streams,
    )


def _alternative_signature(
    config: P1AExperimentConfig,
    metadata: dict,
    base_state_hash: str,
) -> dict:
    signature = _adapter_signature(
        config, metadata, base_state_hash, ALTERNATIVE_VARIANT
    )
    signature.update(
        {
            "short_scale_hours": config.short_scale_hours,
            "long_scale_hours": config.long_scale_hours,
        }
    )
    return signature


def train_alternative_run(
    config: P1AExperimentConfig,
    datasets: dict,
    metadata: dict,
    seed: int,
    base_checkpoint: Path,
    locked_prediction_path: Path,
    output_dir: Path,
    device: torch.device,
    expected_trainable_count: int,
) -> tuple[dict, list[dict], dict, dict]:
    base_model, _, base_state_hash = load_locked_base(
        config, metadata, seed, base_checkpoint, device
    )
    set_seed(seed)
    adapter = build_alternative_adapter(config)
    model = FrozenSTForecastResidual(
        base_model,
        adapter,
        beta_max=config.beta_max,
        beta_init=config.beta_init,
    ).to(device)
    trainable_count = trainable_parameter_count(model)
    capacity_difference_percent = 100 * abs(
        trainable_count - expected_trainable_count
    ) / max(expected_trainable_count, 1)
    if capacity_difference_percent > 5:
        raise AssertionError("FA1与时域控制参数差超过5%")
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    base_parameter_ids = {id(parameter) for parameter in model.base_model.parameters()}
    if any(id(parameter) in base_parameter_ids for parameter in trainable_parameters):
        raise AssertionError("锁定ST参数进入FA1优化器")

    train_loader = make_loader(datasets["train"], config, True, seed)
    valid_loader = make_loader(datasets["valid"], config, False, seed)
    first_x, _ = next(iter(valid_loader))
    first_x = first_x.to(device)
    model.eval()
    with torch.no_grad():
        initial = model.forward_components(first_x)
        reconstruction_error = float(
            (
                initial["streams"].sum(dim=1, keepdim=True)
                - first_x[
                    :,
                    metadata["center_station_idx"]:metadata["center_station_idx"] + 1,
                ]
            )
            .abs()
            .max()
            .item()
        )
    initial_max_difference = float(
        (initial["prediction"] - initial["base_prediction"]).abs().max().item()
    )
    if initial_max_difference != 0.0 or reconstruction_error > 1e-5:
        raise AssertionError("FA1初始退化或滤波器组重构失败")

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
    checkpoint_path = checkpoint_dir / f"{ALTERNATIVE_VARIANT}_seed{seed}.pt"
    signature = _alternative_signature(config, metadata, base_state_hash)

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
                raise FloatingPointError(f"FA1 seed={seed} 出现非有限损失")
            loss.backward()
            if any(
                parameter.grad is not None
                and not torch.isfinite(parameter.grad).all()
                for parameter in trainable_parameters
            ):
                raise FloatingPointError(f"FA1 seed={seed} 出现非有限梯度")
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
    payload = _load_adapter_checkpoint(checkpoint_path, model, signature, seed)
    model = model.to(device)
    final_base_state_hash = _state_sha256(model.base_model.state_dict())
    base_state_unchanged = final_base_state_hash == base_state_hash
    if not base_state_unchanged:
        raise AssertionError("FA1训练改变了锁定ST state")

    prediction, target, inference_seconds = predict_model(
        model, valid_loader, device
    )
    disabled_prediction, _, _ = predict_model(
        model, valid_loader, device, disable_frequency=True
    )
    locked_prediction = np.load(locked_prediction_path)["prediction_scaled"]
    disabled_max_difference = float(
        np.max(np.abs(disabled_prediction - locked_prediction))
    )
    if disabled_max_difference != 0.0:
        raise AssertionError("禁用FA1后未精确恢复锁定ST")
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

    pd.DataFrame(history_rows).to_csv(
        logs_dir / f"{ALTERNATIVE_VARIANT}_seed{seed}.csv", index=False
    )
    np.savez_compressed(
        predictions_dir / f"{ALTERNATIVE_VARIANT}_seed{seed}.npz",
        prediction_scaled=prediction,
        target_scaled=target,
        prediction_ugm3=prediction * metadata["center_std"]
        + metadata["center_mean"],
        target_ugm3=target * metadata["center_std"] + metadata["center_mean"],
        target_origins=datasets["valid"].target_origins,
        target_timestamp_matrix_ns=datasets["valid"].target_timestamp_matrix_ns(),
    )
    row = {
        "variant": ALTERNATIVE_VARIANT,
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
        "initial_exact_base": True,
        "capacity_difference_percent_vs_time": capacity_difference_percent,
        **metrics,
        **diagnostics,
    }
    per_horizon = _per_horizon_rows(
        ALTERNATIVE_VARIANT,
        seed,
        target,
        prediction,
        metadata["center_mean"],
        metadata["center_std"],
    )
    peak = _peak_metrics_row(
        ALTERNATIVE_VARIANT,
        seed,
        target,
        prediction,
        metadata["center_mean"],
        metadata["center_std"],
        metadata["peak_q90_ugm3"],
    )
    runtime = {
        "variant": ALTERNATIVE_VARIANT,
        "seed": seed,
        "initial_exact_base": True,
        "base_state_unchanged": base_state_unchanged,
        "disable_frequency_exact_base": disabled_max_difference == 0.0,
        "reconstruction_max_abs_error": reconstruction_error,
        "all_metrics_finite": bool(
            np.isfinite([metrics["rmse_ugm3"], metrics["mae_ugm3"], best_loss]).all()
        ),
        "capacity_difference_percent_vs_time": capacity_difference_percent,
    }
    print(
        f"[{ALTERNATIVE_VARIANT} seed={seed}] valid RMSE="
        f"{metrics['rmse_ugm3']:.4f} μg/m³, MAE={metrics['mae_ugm3']:.4f}, "
        f"best_epoch={best_epoch}, beta={diagnostics['beta']:.4f}"
    )
    return row, per_horizon, peak, runtime


def paired_differences(raw_df: pd.DataFrame) -> pd.DataFrame:
    indexed = {
        variant: raw_df[raw_df["variant"] == variant].set_index("seed")
        for variant in ALL_VARIANTS
    }
    comparisons = (
        ("st_time_residual_adapter", "locked_st_clean"),
        (ALTERNATIVE_VARIANT, "locked_st_clean"),
        (ALTERNATIVE_VARIANT, "st_time_residual_adapter"),
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
    raw_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    runtime_checks: list[dict],
    smoke: bool,
) -> dict:
    runtime_passed = all(
        item["initial_exact_base"]
        and item["base_state_unchanged"]
        and item["disable_frequency_exact_base"]
        and item["reconstruction_max_abs_error"] <= 1e-5
        and item["all_metrics_finite"]
        and item["capacity_difference_percent_vs_time"] <= 5
        for item in runtime_checks
    )
    if smoke:
        return {
            "gate": "FA1",
            "status": "in_progress" if runtime_passed else "failed",
            "evidence_label": "smoke",
            "runtime_checks_passed": runtime_passed,
            "note": "Smoke validates correctness but cannot decide FA1.",
        }
    alt_vs_st = paired_df[
        paired_df["comparison"]
        == f"{ALTERNATIVE_VARIANT}_vs_locked_st_clean"
    ]
    alt = raw_df[raw_df["variant"] == ALTERNATIVE_VARIANT]
    time_control = raw_df[raw_df["variant"] == "st_time_residual_adapter"]
    improvements = alt_vs_st["rmse_relative_improvement_percent"].to_numpy()
    checks = {
        "runtime_checks_passed": runtime_passed,
        "complete_seed_variant_grid": len(raw_df)
        == len(SELECTION_SEEDS) * len(ALL_VARIANTS),
        "selection_seeds_exact": sorted(raw_df["seed"].unique().tolist())
        == list(SELECTION_SEEDS),
        "alternative_three_of_three_better_than_st": len(improvements) == 3
        and bool(np.all(improvements > 0)),
        "alternative_mean_improvement_at_least_0_5_percent": len(improvements) == 3
        and float(np.mean(improvements)) >= 0.5,
        "alternative_mean_rmse_better_than_time_control": float(
            alt["rmse_ugm3"].mean()
        )
        < float(time_control["rmse_ugm3"].mean()),
        "capacity_difference_at_most_5_percent": float(
            alt["capacity_difference_percent_vs_time"].max()
        )
        <= 5,
    }
    passed = all(checks.values())
    return {
        "gate": "FA1",
        "status": "passed" if passed else "failed",
        "evidence_label": "selection",
        "checks": checks,
        "alternative_vs_st_improved_seed_count": int(np.sum(improvements > 0)),
        "alternative_vs_st_mean_rmse_improvement_percent": float(
            np.mean(improvements)
        ),
        "alternative_rmse_mean": float(alt["rmse_ugm3"].mean()),
        "time_control_rmse_mean": float(time_control["rmse_ugm3"].mean()),
        "allowed_next_gate": None,
        "failure_action": (
            None
            if passed
            else "stop current frequency/multiscale direction and redesign innovation 2"
        ),
    }


def write_report(
    output_dir: Path,
    config: P1AExperimentConfig,
    summary: pd.DataFrame,
    stage: dict,
) -> None:
    lines = [
        "# FA1 因果局部滤波器组自动摘要",
        "",
        (
            "> SMOKE TEST DATA - DO NOT USE AS THESIS EVIDENCE."
            if config.evidence_label == "smoke"
            else "> SELECTION DATA - the sole pre-registered alternative after F3 failure."
        ),
        "",
        f"- 任务：{config.history}→{config.horizon}",
        f"- Gate FA1：{stage['status']}",
        "- 分解：low=MA48, mid=MA12-MA48, high=x-MA12",
        "",
        "| 变体 | RMSE (μg/m³) | MAE (μg/m³) |",
        "|---|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['variant']} | {row['rmse_ugm3_mean']:.4f} ± "
            f"{row['rmse_ugm3_std']:.4f} | {row['mae_ugm3_mean']:.4f} ± "
            f"{row['mae_ugm3_std']:.4f} |"
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
                f"正式selection种子必须严格为{list(SELECTION_SEEDS)}"
            )
        evidence_label = "selection"
    config = P1AExperimentConfig(
        history=args.history,
        horizon=args.horizon,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        evidence_label=evidence_label,
    )
    if (config.history, config.horizon) != (168, 6):
        raise ValueError("FA1只允许预注册的168→6主任务")
    p0_dir = Path(
        args.p0_dir
        or "experiments/results/causal_frequency_ablation/p0_bridge/168h_6h_selection"
    )
    control_dir = Path(
        args.control_dir
        or "experiments/results/causal_frequency_ablation/p1_fixed_selection/168h_6h_selection"
    )
    output_dir = Path(
        args.output_dir
        or (
            "experiments/results/causal_frequency_ablation/"
            f"p1_alternative_selection/168h_6h_{evidence_label}"
        )
    )
    if args.quick and not output_dir.name.endswith("_smoke"):
        output_dir = output_dir.with_name(output_dir.name + "_smoke")
    if "causal_frequency_ablation" not in output_dir.parts:
        raise ValueError("FA1输出必须位于causal_frequency_ablation目录")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    datasets, metadata = prepare_causal_datasets(config)
    control_config = json.loads(
        (control_dir / "experiment_config.json").read_text(encoding="utf-8")
    )
    control_metadata = json.loads(
        (control_dir / "dataset_metadata.json").read_text(encoding="utf-8")
    )
    for key in (
        "time_index_sha256",
        "station_ids",
        "center_station_idx",
        "station_means",
        "station_stds",
    ):
        if metadata[key] != control_metadata[key]:
            raise RuntimeError(f"FA1数据与P1控制不一致: {key}")
    if control_config["test_metrics_computed"] is not False:
        raise RuntimeError("P1控制目录包含测试指标")

    control_raw = pd.read_csv(control_dir / "raw_metrics.csv")
    control_horizon = pd.read_csv(control_dir / "per_horizon_metrics.csv")
    control_peak = pd.read_csv(control_dir / "peak_metrics.csv")
    expected_trainable_count = int(
        control_config["time_adapter_trainable_parameters"]
    )
    probe = build_alternative_adapter(config)
    alternative_trainable_count = trainable_parameter_count(probe) + 1
    capacity_difference_percent = 100 * abs(
        alternative_trainable_count - expected_trainable_count
    ) / max(expected_trainable_count, 1)
    experiment_config = {
        **asdict(config),
        "seeds": seeds,
        "variants": list(ALL_VARIANTS),
        "device": str(device),
        "smoke_test": args.quick,
        "evaluation_data_used": ["train", "valid"],
        "test_metrics_computed": False,
        "p0_checkpoint_dir": str(p0_dir),
        "reused_control_dir": str(control_dir),
        "reused_control_raw_metrics_sha256": _path_sha256(
            control_dir / "raw_metrics.csv"
        ),
        "time_adapter_trainable_parameters": expected_trainable_count,
        "alternative_trainable_parameters": alternative_trainable_count,
        "capacity_difference_percent": capacity_difference_percent,
        "filterbank_definition": {
            "left_padding": "replicate_first_observed_value_only",
            "short_trailing_mean_hours": 12,
            "long_trailing_mean_hours": 48,
            "streams": ["MA48", "MA12-MA48", "x-MA12"],
            "learnable_decomposition_parameters": 0,
        },
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
    diagnostics_dir = output_dir / "frequency_diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    (diagnostics_dir / "causal_filterbank_definition.json").write_text(
        json.dumps(
            experiment_config["filterbank_definition"],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    predictions_dir = output_dir / "predictions"
    logs_dir = output_dir / "training_logs"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = []
    per_horizon_rows = []
    peak_rows = []
    runtime_checks = []
    for seed in seeds:
        for variant in CONTROL_VARIANTS:
            row = control_raw[
                (control_raw["variant"] == variant)
                & (control_raw["seed"] == seed)
            ]
            if len(row) != 1:
                raise RuntimeError(f"缺少唯一控制结果: {variant} seed={seed}")
            raw_rows.append(row.iloc[0].to_dict())
            per_horizon_rows.extend(
                control_horizon[
                    (control_horizon["variant"] == variant)
                    & (control_horizon["seed"] == seed)
                ].to_dict("records")
            )
            peak_rows.extend(
                control_peak[
                    (control_peak["variant"] == variant)
                    & (control_peak["seed"] == seed)
                ].to_dict("records")
            )
            source_prediction = (
                control_dir / "predictions" / f"{variant}_seed{seed}.npz"
            )
            shutil.copy2(
                source_prediction,
                predictions_dir / source_prediction.name,
            )
            source_log = control_dir / "training_logs" / f"{variant}_seed{seed}.csv"
            if source_log.is_file():
                shutil.copy2(source_log, logs_dir / source_log.name)

        base_checkpoint = p0_dir / "checkpoints" / f"locked_st_clean_seed{seed}.pt"
        row, horizon_rows, peak, runtime = train_alternative_run(
            config,
            datasets,
            metadata,
            seed,
            base_checkpoint,
            predictions_dir / f"locked_st_clean_seed{seed}.npz",
            output_dir,
            device,
            expected_trainable_count,
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
    stage = evaluate_stage(raw_df, paired_df, runtime_checks, args.quick)
    (output_dir / "stage_status.json").write_text(
        json.dumps(stage, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output_dir / "runtime_checks.json").write_text(
        json.dumps(runtime_checks, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    write_report(output_dir, config, summary_df, stage)
    print(f"FA1产物已写入: {output_dir.resolve()}")
    print(f"Gate FA1: {stage['status']}")


if __name__ == "__main__":
    main()
