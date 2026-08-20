"""Run reproducible ST-PatchTST ablations on multi-station PM2.5 data.

The degraded PatchTST baseline is the exact same ST_PatchTST class with alpha fixed
to zero. Consequently, it shares data, initialization order, backbone, optimizer,
and training loop with the spatial variants while ignoring every neighbor input.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ST_PatchTST_model import ST_PatchTST


MODEL_VARIANTS = (
    "degraded_patchtst",
    "st_center_only",
    "st_pairwise",
    "st_pairwise_delta",
)


@dataclass(frozen=True)
class ExperimentConfig:
    data_dir: str = "tsai/data/stations_data"
    center_station_id: int = 1013
    station_start: int = 1001
    station_end: int = 1035
    correlation_threshold: float = 0.85
    history: int = 24
    horizon: int = 1
    train_fraction: float = 0.7
    valid_fraction: float = 0.1
    batch_size: int = 256
    epochs: int = 40
    patience: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    neighbor_hidden_dim: int = 32
    alpha_max: float = 0.5
    alpha_init: float = 0.05
    n_layers: int = 3
    n_heads: int = 4
    d_model: int = 16
    d_ff: int = 128
    dropout: float = 0.2
    patch_len: int = 4
    stride: int = 2


class ForecastWindowDataset(Dataset):
    """Lazy chronological windows over a normalized [time, station] array."""

    def __init__(self, values, sample_indices, history, horizon, center_idx):
        self.values = values
        self.sample_indices = np.asarray(sample_indices, dtype=np.int64)
        self.history = int(history)
        self.horizon = int(horizon)
        self.center_idx = int(center_idx)

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, item):
        start = int(self.sample_indices[item])
        target_start = start + self.history
        x = self.values[start:target_start].T
        y = self.values[
            target_start:target_start + self.horizon, self.center_idx
        ][None, :]
        return torch.from_numpy(x), torch.from_numpy(y)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--seeds", default="2024,2025,2026")
    parser.add_argument("--variants", default=",".join(MODEL_VARIANTS))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to experiments/results/st_patchtst_ablation/<history>h_<horizon>h",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Two-epoch, one-seed smoke run; results are labeled as smoke-test data.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_station_data(config):
    frames = []
    station_ids = list(range(config.station_start, config.station_end + 1))
    for station_id in station_ids:
        path = Path(config.data_dir) / f"df_station_{station_id}.csv"
        df = pd.read_csv(path, usecols=["time", "PM25_Concentration"])
        df["time"] = pd.to_datetime(df["time"])
        series = df.drop_duplicates("time").set_index("time")["PM25_Concentration"]
        frames.append(series.rename(station_id))

    aligned = pd.concat(frames, axis=1, join="inner").sort_index()
    aligned = aligned.ffill().bfill()
    correlations = aligned.corr()[config.center_station_id]
    selected_ids = [
        station_id
        for station_id in station_ids
        if correlations.loc[station_id] > config.correlation_threshold
    ]
    if config.center_station_id not in selected_ids:
        selected_ids.append(config.center_station_id)
        selected_ids.sort()
    selected = aligned[selected_ids]
    return selected, correlations.loc[selected_ids]


def prepare_datasets(config):
    frame, correlations = load_station_data(config)
    raw_values = frame.to_numpy(dtype=np.float32)
    num_samples = len(frame) - config.history - config.horizon + 1
    if num_samples < 3:
        raise ValueError("时间序列太短，无法构造训练、验证和测试窗口")

    train_end = int(num_samples * config.train_fraction)
    valid_end = int(num_samples * (config.train_fraction + config.valid_fraction))
    split_indices = {
        "train": np.arange(0, train_end),
        "valid": np.arange(train_end, valid_end),
        "test": np.arange(valid_end, num_samples),
    }

    # Fit statistics on raw rows touched by training windows and their targets.
    train_row_stop = train_end + config.history + config.horizon - 1
    train_values = raw_values[:train_row_stop]
    means = train_values.mean(axis=0)
    stds = train_values.std(axis=0)
    stds = np.where(stds < 1e-6, 1.0, stds)
    normalized = ((raw_values - means) / stds).astype(np.float32)

    station_ids = [int(value) for value in frame.columns]
    center_idx = station_ids.index(config.center_station_id)
    datasets = {
        split: ForecastWindowDataset(
            normalized,
            indices,
            config.history,
            config.horizon,
            center_idx,
        )
        for split, indices in split_indices.items()
    }
    metadata = {
        "rows": len(frame),
        "start_time": str(frame.index.min()),
        "end_time": str(frame.index.max()),
        "num_samples": num_samples,
        "split_sizes": {key: len(value) for key, value in datasets.items()},
        "station_ids": station_ids,
        "station_correlations": {
            str(key): float(value) for key, value in correlations.items()
        },
        "center_station_idx": center_idx,
        "center_mean": float(means[center_idx]),
        "center_std": float(stds[center_idx]),
        "station_means": means.tolist(),
        "station_stds": stds.tolist(),
        "station_selection_note": (
            "Per user instruction, correlations were computed on the complete aligned "
            "series; leakage correction is outside this experiment's scope."
        ),
    }
    return datasets, metadata


def build_model(config, variant, num_stations, center_idx):
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"未知模型变体: {variant}")
    is_degraded = variant == "degraded_patchtst"
    gate_mode = "center_only" if variant == "st_center_only" else "pairwise_summary"
    uses_delta = variant == "st_pairwise_delta"
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
        alpha_max=0.0 if is_degraded else (0.2 if uses_delta else config.alpha_max),
        alpha_init=0.0 if is_degraded else (0.02 if uses_delta else config.alpha_init),
        gate_mode=gate_mode,
        use_null_neighbor=True,
        neighbor_value_mode="difference" if uses_delta else "raw",
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_model=config.d_model,
        d_ff=config.d_ff,
        dropout=config.dropout,
        patch_len=config.patch_len,
        stride=config.stride,
        padding_patch=True,
    )


def make_loader(dataset, config, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def predict(model, loader, device, center_idx, neighbor_mode="normal"):
    model.eval()
    predictions = []
    targets = []
    started = time.perf_counter()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        if neighbor_mode == "zero":
            mask = torch.ones(x.shape[1], dtype=torch.bool, device=device)
            mask[center_idx] = False
            x[:, mask] = 0
        elif neighbor_mode == "shuffle":
            mask = torch.ones(x.shape[1], dtype=torch.bool, device=device)
            mask[center_idx] = False
            x[:, mask] = torch.roll(x[:, mask], shifts=1, dims=0)
        elif neighbor_mode == "center_copy":
            mask = torch.ones(x.shape[1], dtype=torch.bool, device=device)
            mask[center_idx] = False
            x[:, mask] = x[:, center_idx:center_idx + 1]
        predictions.append(model(x).cpu())
        targets.append(y)
    elapsed = time.perf_counter() - started
    return torch.cat(predictions).numpy(), torch.cat(targets).numpy(), elapsed


def regression_metrics(y_true, y_pred, center_mean, center_std):
    error = y_pred - y_true
    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(error)))
    physical_true = y_true * center_std + center_mean
    physical_pred = y_pred * center_std + center_mean
    physical_error = physical_pred - physical_true
    physical_rmse = float(np.sqrt(np.mean(physical_error ** 2)))
    physical_mae = float(np.mean(np.abs(physical_error)))
    denominator = np.abs(physical_true) + np.abs(physical_pred)
    smape = float(200 * np.mean(np.abs(physical_error) / np.maximum(denominator, 1e-6)))
    return {
        "mse_scaled": mse,
        "rmse_scaled": rmse,
        "mae_scaled": mae,
        "rmse_ugm3": physical_rmse,
        "mae_ugm3": physical_mae,
        "smape_percent": smape,
    }


@torch.no_grad()
def collect_spatial_diagnostics(model, loader, device):
    model.eval()
    center_sq = 0.0
    residual_sq = 0.0
    element_count = 0
    gate_batches = []
    null_batches = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        components = model.spatial_components(x)
        center = components["center_x"]
        residual = components["alpha"] * components["neighbor_context"]
        center_sq += float(center.square().sum().item())
        residual_sq += float(residual.square().sum().item())
        element_count += center.numel()
        gate_batches.append(components["gate_weights"].cpu())
        null_batches.append(components["null_weight"].cpu())

    gates = torch.cat(gate_batches)
    nulls = torch.cat(null_batches).unsqueeze(1)
    all_weights = torch.cat([gates, nulls], dim=1)
    entropy = -(all_weights * all_weights.clamp_min(1e-12).log()).sum(dim=1)
    normalized_entropy = entropy / math.log(all_weights.shape[1])
    rms_ratio = math.sqrt(residual_sq / max(element_count, 1)) / max(
        math.sqrt(center_sq / max(element_count, 1)), 1e-12
    )
    alpha = 0.0 if model.alpha_logit is None else float(
        (model.max_alpha * torch.sigmoid(model.alpha_logit)).item()
    )
    return {
        "alpha": alpha,
        "spatial_residual_rms_ratio": rms_ratio,
        "gate_normalized_entropy": float(normalized_entropy.mean().item()),
        "null_neighbor_weight_mean": float(nulls.mean().item()),
        "gate_sample_std_mean": float(all_weights.std(dim=0, unbiased=False).mean().item()),
        "neighbor_out_weight_norm": float(model.neighbor_out.weight.norm().item()),
    }


def train_one_run(config, datasets, metadata, variant, seed, output_dir, device):
    set_seed(seed)
    model = build_model(
        config,
        variant,
        num_stations=len(metadata["station_ids"]),
        center_idx=metadata["center_station_idx"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    loss_fn = nn.MSELoss()
    train_loader = make_loader(datasets["train"], config, True, seed)
    valid_loader = make_loader(datasets["valid"], config, False, seed)
    test_loader = make_loader(datasets["test"], config, False, seed)

    best_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history_rows = []
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{variant}_seed{seed}.pt"
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
                raise FloatingPointError(f"{variant} seed={seed} 出现非有限训练损失")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(x)
            train_count += len(x)

        valid_pred, valid_true, _ = predict(
            model, valid_loader, device, metadata["center_station_idx"]
        )
        train_loss = train_loss_sum / train_count
        valid_loss = float(np.mean((valid_pred - valid_true) ** 2))
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
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.patience:
            break

    training_seconds = time.perf_counter() - started
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    prediction, target, inference_seconds = predict(
        model, test_loader, device, metadata["center_station_idx"]
    )
    metrics = regression_metrics(
        target, prediction, metadata["center_mean"], metadata["center_std"]
    )
    diagnostics = collect_spatial_diagnostics(model, test_loader, device)

    for intervention in ("zero", "shuffle", "center_copy"):
        perturbed_prediction, _, _ = predict(
            model,
            test_loader,
            device,
            metadata["center_station_idx"],
            neighbor_mode=intervention,
        )
        perturbed = regression_metrics(
            target,
            perturbed_prediction,
            metadata["center_mean"],
            metadata["center_std"],
        )
        metrics[f"{intervention}_neighbor_rmse_ugm3"] = perturbed["rmse_ugm3"]
        metrics[f"{intervention}_neighbor_mae_ugm3"] = perturbed["mae_ugm3"]

    logs_dir = output_dir / "training_logs"
    predictions_dir = output_dir / "predictions"
    logs_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history_rows).to_csv(
        logs_dir / f"{variant}_seed{seed}.csv", index=False
    )
    np.savez_compressed(
        predictions_dir / f"{variant}_seed{seed}.npz",
        prediction_scaled=prediction,
        target_scaled=target,
        prediction_ugm3=prediction * metadata["center_std"] + metadata["center_mean"],
        target_ugm3=target * metadata["center_std"] + metadata["center_mean"],
    )

    row = {
        "variant": variant,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_valid_loss": best_loss,
        "training_seconds": training_seconds,
        "test_inference_seconds": inference_seconds,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        **metrics,
        **diagnostics,
    }
    print(
        f"[{variant} seed={seed}] RMSE={metrics['rmse_ugm3']:.4f} μg/m³, "
        f"MAE={metrics['mae_ugm3']:.4f}, alpha={diagnostics['alpha']:.5f}, "
        f"residual_ratio={diagnostics['spatial_residual_rms_ratio']:.5f}"
    )
    return row


def aggregate_results(raw_df):
    metric_columns = [
        "rmse_ugm3",
        "mae_ugm3",
        "smape_percent",
        "best_valid_loss",
        "training_seconds",
        "test_inference_seconds",
        "alpha",
        "spatial_residual_rms_ratio",
        "gate_normalized_entropy",
        "null_neighbor_weight_mean",
        "gate_sample_std_mean",
        "neighbor_out_weight_norm",
        "zero_neighbor_rmse_ugm3",
        "shuffle_neighbor_rmse_ugm3",
        "center_copy_neighbor_rmse_ugm3",
    ]
    rows = []
    for variant, group in raw_df.groupby("variant", sort=False):
        row = {"variant": variant, "runs": len(group)}
        for column in metric_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def paired_differences(raw_df):
    baseline = raw_df[raw_df["variant"] == "degraded_patchtst"].set_index("seed")
    rows = []
    for variant in ("st_center_only", "st_pairwise", "st_pairwise_delta"):
        candidate = raw_df[raw_df["variant"] == variant].set_index("seed")
        for seed in sorted(set(baseline.index) & set(candidate.index)):
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "rmse_difference_ugm3": (
                        candidate.loc[seed, "rmse_ugm3"] - baseline.loc[seed, "rmse_ugm3"]
                    ),
                    "mae_difference_ugm3": (
                        candidate.loc[seed, "mae_ugm3"] - baseline.loc[seed, "mae_ugm3"]
                    ),
                    "rmse_relative_change_percent": 100
                    * (
                        candidate.loc[seed, "rmse_ugm3"]
                        / baseline.loc[seed, "rmse_ugm3"]
                        - 1
                    ),
                }
            )
    return pd.DataFrame(rows)


def write_machine_report(output_dir, config, metadata, raw_df, summary_df, paired_df, smoke):
    lines = [
        "# ST-PatchTST 自动实验摘要",
        "",
        "> SMOKE TEST DATA - DO NOT USE AS FINAL THESIS EVIDENCE."
        if smoke
        else "> REAL EXPERIMENT DATA - generated from the recorded runs.",
        "",
        f"- 任务：{config.history} 小时历史预测未来 {config.horizon} 小时 PM2.5",
        f"- 中心站：{config.center_station_id}",
        f"- 站点数：{len(metadata['station_ids'])}",
        f"- 样本划分：{metadata['split_sizes']}",
        "- 已知边界：按用户要求，本轮站点相关性使用完整序列计算。",
        "",
        "## 聚合结果",
        "",
        "| 模型 | RMSE (μg/m³) | MAE (μg/m³) | SMAPE (%) | 空间残差比 |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['variant']} | {row['rmse_ugm3_mean']:.4f} ± {row['rmse_ugm3_std']:.4f} "
            f"| {row['mae_ugm3_mean']:.4f} ± {row['mae_ugm3_std']:.4f} "
            f"| {row['smape_percent_mean']:.3f} ± {row['smape_percent_std']:.3f} "
            f"| {row['spatial_residual_rms_ratio_mean']:.6f} |"
        )
    if not paired_df.empty:
        lines.extend(
            [
                "",
                "## 相对退化基线的配对变化",
                "",
                "负值表示误差下降。逐种子原始差值见 `paired_differences.csv`。",
            ]
        )
    (output_dir / "auto_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    if args.quick:
        seeds = seeds[:1]
        args.epochs = min(args.epochs, 2)
        args.patience = min(args.patience, 2)
    config = ExperimentConfig(
        history=args.history,
        horizon=args.horizon,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
    )
    output_dir = Path(args.output_dir or (
        f"experiments/results/st_patchtst_ablation/{config.history}h_{config.horizon}h"
    ))
    if args.quick:
        output_dir = output_dir.with_name(output_dir.name + "_smoke")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    datasets, metadata = prepare_datasets(config)
    (output_dir / "experiment_config.json").write_text(
        json.dumps(
            {
                **asdict(config),
                "seeds": seeds,
                "variants": variants,
                "device": str(device),
                "smoke_test": args.quick,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "dataset_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    result_rows = []
    for seed in seeds:
        for variant in variants:
            result_rows.append(
                train_one_run(
                    config, datasets, metadata, variant, seed, output_dir, device
                )
            )
            pd.DataFrame(result_rows).to_csv(output_dir / "raw_metrics.csv", index=False)

    raw_df = pd.DataFrame(result_rows)
    summary_df = aggregate_results(raw_df)
    paired_df = paired_differences(raw_df)
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)
    paired_df.to_csv(output_dir / "paired_differences.csv", index=False)
    write_machine_report(
        output_dir, config, metadata, raw_df, summary_df, paired_df, args.quick
    )
    print(f"实验产物已写入: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
