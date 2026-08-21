"""Build thesis-ready round-two summaries from recorded ST-PatchTST artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_st_patchtst_ablation import (
    ExperimentConfig,
    build_model,
    make_loader,
    prepare_datasets,
)


DEFAULT_ROOT = Path("experiments/results/st_patchtst_ablation")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebuild thesis-ready summaries from explicit experiment directories."
    )
    parser.add_argument(
        "--forecast-24-dir",
        type=Path,
        default=DEFAULT_ROOT / "round2_forecast_24h_1h",
    )
    parser.add_argument(
        "--station-bias-24-dir",
        type=Path,
        default=DEFAULT_ROOT / "round2_station_bias_24h_1h",
    )
    parser.add_argument(
        "--forecast-168-dir",
        type=Path,
        default=DEFAULT_ROOT / "round2_forecast_168h_6h",
    )
    parser.add_argument(
        "--station-bias-168-dir",
        type=Path,
        default=DEFAULT_ROOT / "round2_station_bias_168h_6h",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_ROOT / "round2_combined_summary.csv",
    )
    return parser.parse_args()


def load_config(directory):
    raw = json.loads((directory / "experiment_config.json").read_text(encoding="utf-8"))
    valid_fields = {field.name for field in fields(ExperimentConfig)}
    return ExperimentConfig(**{key: value for key, value in raw.items() if key in valid_fields})


def build_combined_summary(formal_experiments, output_path):
    rows = []
    for task, directory in formal_experiments.items():
        raw = pd.read_csv(directory / "raw_metrics.csv")
        baseline = raw[raw["variant"] == "degraded_patchtst"].set_index("seed")
        for variant, group in raw.groupby("variant", sort=False):
            row = {"task": task, "variant": variant, "runs": len(group)}
            for metric in ("rmse_ugm3", "mae_ugm3", "smape_percent"):
                row[f"{metric}_mean"] = group[metric].mean()
                row[f"{metric}_std"] = group[metric].std(ddof=1)
            if variant == "degraded_patchtst":
                row["paired_rmse_relative_change_percent_mean"] = 0.0
                row["rmse_improved_seed_count"] = 0
            else:
                candidate = group.set_index("seed")
                relative = (candidate["rmse_ugm3"] / baseline["rmse_ugm3"] - 1) * 100
                row["paired_rmse_relative_change_percent_mean"] = relative.mean()
                row["rmse_improved_seed_count"] = int((relative < 0).sum())
                row["disable_rmse_increase_mean"] = (
                    group["disable_neighbor_rmse_ugm3"] - group["rmse_ugm3"]
                ).mean()
                row["shuffle_rmse_increase_mean"] = (
                    group["shuffle_neighbor_rmse_ugm3"] - group["rmse_ugm3"]
                ).mean()
            rows.append(row)
    output = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output


def build_per_horizon_summary(directory):
    raw = pd.read_csv(directory / "raw_metrics.csv")
    rows = []
    for _, metric_row in raw.iterrows():
        variant = metric_row["variant"]
        seed = int(metric_row["seed"])
        prediction_file = directory / "predictions" / f"{variant}_seed{seed}.npz"
        arrays = np.load(prediction_file)
        error = arrays["prediction_ugm3"] - arrays["target_ugm3"]
        for horizon_idx in range(error.shape[-1]):
            step_error = error[..., horizon_idx]
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "horizon_step": horizon_idx + 1,
                    "rmse_ugm3": float(np.sqrt(np.mean(step_error ** 2))),
                    "mae_ugm3": float(np.mean(np.abs(step_error))),
                }
            )
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(directory / "per_horizon_metrics.csv", index=False)
    summary = (
        per_seed.groupby(["variant", "horizon_step"], sort=False)
        .agg(
            rmse_ugm3_mean=("rmse_ugm3", "mean"),
            rmse_ugm3_std=("rmse_ugm3", "std"),
            mae_ugm3_mean=("mae_ugm3", "mean"),
            mae_ugm3_std=("mae_ugm3", "std"),
        )
        .reset_index()
    )
    summary.to_csv(directory / "per_horizon_summary.csv", index=False)
    baseline = summary[summary["variant"] == "degraded_patchtst"].set_index(
        "horizon_step"
    )
    relative_rows = []
    for _, row in summary.iterrows():
        baseline_row = baseline.loc[row["horizon_step"]]
        relative_rows.append(
            {
                **row.to_dict(),
                "rmse_relative_change_percent": (
                    row["rmse_ugm3_mean"] / baseline_row["rmse_ugm3_mean"] - 1
                )
                * 100,
                "mae_relative_change_percent": (
                    row["mae_ugm3_mean"] / baseline_row["mae_ugm3_mean"] - 1
                )
                * 100,
            }
        )
    pd.DataFrame(relative_rows).to_csv(
        directory / "per_horizon_relative_summary.csv", index=False
    )


@torch.no_grad()
def build_station_gate_summary(directory):
    config = load_config(directory)
    datasets, metadata = prepare_datasets(config)
    station_ids = metadata["station_ids"]
    center_idx = metadata["center_station_idx"]
    neighbor_ids = [
        station_id for idx, station_id in enumerate(station_ids) if idx != center_idx
    ]
    raw = pd.read_csv(directory / "raw_metrics.csv")
    variants = [
        variant for variant in raw["variant"].unique() if variant != "degraded_patchtst"
    ]
    rows = []
    for variant in variants:
        for seed in sorted(raw[raw["variant"] == variant]["seed"].astype(int)):
            model = build_model(config, variant, len(station_ids), center_idx)
            checkpoint = directory / "checkpoints" / f"{variant}_seed{seed}.pt"
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            model.eval()
            loader = make_loader(datasets["test"], config, False, seed)
            gate_batches = []
            for x, _ in loader:
                gate_batches.append(model.spatial_components(x)["gate_weights"])
            gates = torch.cat(gate_batches)
            biases = model.station_gate_bias.detach().cpu().numpy()
            for station_idx, station_id in enumerate(neighbor_ids):
                rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "station_id": station_id,
                        "station_bias": float(biases[station_idx]),
                        "gate_weight_mean": float(gates[:, station_idx].mean()),
                        "gate_weight_std": float(gates[:, station_idx].std(unbiased=False)),
                    }
                )
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(directory / "station_gate_summary.csv", index=False)
    aggregate = (
        per_seed.groupby(["variant", "station_id"], sort=False)
        .agg(
            station_bias_mean=("station_bias", "mean"),
            station_bias_std=("station_bias", "std"),
            gate_weight_mean=("gate_weight_mean", "mean"),
            gate_weight_between_seed_std=("gate_weight_mean", "std"),
            gate_weight_within_sample_std_mean=("gate_weight_std", "mean"),
        )
        .reset_index()
        .sort_values("gate_weight_mean", ascending=False)
    )
    aggregate.to_csv(directory / "station_gate_aggregate.csv", index=False)


def main():
    args = parse_args()
    formal_experiments = {
        "24h_to_1h_forecast": args.forecast_24_dir,
        "24h_to_1h_station_bias": args.station_bias_24_dir,
        "168h_to_6h_forecast": args.forecast_168_dir,
        "168h_to_6h_station_bias": args.station_bias_168_dir,
    }
    summary = build_combined_summary(formal_experiments, args.summary_output)
    for directory in (args.station_bias_24_dir, args.station_bias_168_dir):
        build_per_horizon_summary(directory)
        build_station_gate_summary(directory)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
