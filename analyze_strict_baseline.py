"""Compare strict center-station PatchTST with the completed alpha ablation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent
BASELINE_PATH = ROOT / "tsai/models/patchtst_strict_baseline/baseline_results.csv"
ABLATION_PATH = ROOT / "tsai/models/alpha_ablation/ablation_results.csv"
DATA_DIR = ROOT / "figures/data"
CONDITIONS = [
    "strict_patchtst",
    "alpha_zero",
    "uniform_alpha",
    "per_pollutant_alpha",
    "learnable_alpha",
]
SPATIAL_CONDITIONS = [
    "uniform_alpha",
    "per_pollutant_alpha",
    "learnable_alpha",
]
METRICS = ["valid_mse", "valid_rmse", "valid_mae"]
POLLUTANTS = ["PM2.5", "PM10", "NO2", "CO", "O3", "SO2"]


def holm_adjust(p_values):
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def main():
    baseline = pd.read_csv(BASELINE_PATH)
    ablation = pd.read_csv(ABLATION_PATH)
    data = pd.concat([baseline, ablation], ignore_index=True, sort=False)
    expected_seeds = {20260809, 20260810, 20260811}
    for condition in CONDITIONS:
        group = data[data["condition"] == condition]
        if set(group["seed"]) != expected_seeds or len(group) != 3:
            raise RuntimeError(f"{condition} 的三 seed 数据不完整")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    selected = data[data["condition"].isin(CONDITIONS)].copy()
    selected.to_csv(DATA_DIR / "strict_patchtst_seed_results.csv", index=False)

    strict = baseline.set_index("seed").sort_index()
    t_critical = stats.t.ppf(0.975, df=2)
    summary_rows = []
    for condition in CONDITIONS:
        group = selected[selected["condition"] == condition].sort_values("seed")
        row = {
            "condition": condition,
            "n": len(group),
            "elapsed_minutes_mean": group["elapsed_seconds"].mean() / 60,
            "total_params": int(group["total_params"].iloc[0]),
            "trainable_params": int(group["trainable_params"].iloc[0]),
        }
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            mean = values.mean()
            std = values.std(ddof=1)
            half_width = t_critical * std / np.sqrt(len(values))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci_low"] = mean - half_width
            row[f"{metric}_ci_high"] = mean + half_width
        if condition == "strict_patchtst":
            row["mse_improvement_vs_strict_percent"] = 0.0
        else:
            row["mse_improvement_vs_strict_percent"] = 100 * (
                strict["valid_mse"].mean() - row["valid_mse_mean"]
            ) / strict["valid_mse"].mean()
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(DATA_DIR / "strict_patchtst_summary.csv", index=False)

    pairwise_rows = []
    for condition in ["alpha_zero", *SPATIAL_CONDITIONS]:
        comparison = selected[selected["condition"] == condition].set_index("seed").sort_index()
        difference = strict["valid_mse"] - comparison["valid_mse"]
        percentages = 100 * difference / strict["valid_mse"]
        statistic, p_value = stats.ttest_rel(
            strict["valid_mse"], comparison["valid_mse"]
        )
        std = difference.std(ddof=1)
        pairwise_rows.append({
            "comparison": f"strict_patchtst - {condition}",
            "mean_mse_reduction": difference.mean(),
            "mean_improvement_percent": percentages.mean(),
            "min_improvement_percent": percentages.min(),
            "max_improvement_percent": percentages.max(),
            "paired_t": statistic,
            "df": 2,
            "p_raw": p_value,
            "cohen_dz": np.nan if std == 0 else difference.mean() / std,
        })
    pairwise = pd.DataFrame(pairwise_rows)
    pairwise["p_holm"] = holm_adjust(pairwise["p_raw"].to_numpy())
    pairwise.to_csv(DATA_DIR / "strict_patchtst_pairwise.csv", index=False)

    strict_pollutants = {
        pollutant: np.mean([
            json.loads(value)[pollutant] for value in baseline["pollutant_mse"]
        ])
        for pollutant in POLLUTANTS
    }
    pollutant_rows = []
    for condition in SPATIAL_CONDITIONS:
        group = selected[selected["condition"] == condition]
        for pollutant in POLLUTANTS:
            values = np.asarray([
                json.loads(value)[pollutant] for value in group["pollutant_mse"]
            ])
            mean = values.mean()
            pollutant_rows.append({
                "condition": condition,
                "pollutant": pollutant,
                "strict_mse_mean": strict_pollutants[pollutant],
                "method_mse_mean": mean,
                "improvement_percent": 100 * (
                    strict_pollutants[pollutant] - mean
                ) / strict_pollutants[pollutant],
            })
    pd.DataFrame(pollutant_rows).to_csv(
        DATA_DIR / "strict_patchtst_pollutant_improvement.csv", index=False
    )

    print(summary.to_string(index=False))
    print("\nPaired strict-baseline comparisons:")
    print(pairwise.to_string(index=False))


if __name__ == "__main__":
    main()
