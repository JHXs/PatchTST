"""Aggregate the completed 3-seed alpha ablation without accessing test data."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent
RAW_PATH = ROOT / "tsai/models/alpha_ablation/ablation_results.csv"
DATA_DIR = ROOT / "figures/data"
CONDITION_ORDER = [
    "alpha_zero",
    "uniform_alpha",
    "per_pollutant_alpha",
    "learnable_alpha",
]
METRICS = ["valid_mse", "valid_rmse", "valid_mae"]
POLLUTANTS = ["PM2.5", "PM10", "NO2", "CO", "O3", "SO2"]


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def main():
    df = pd.read_csv(RAW_PATH)
    expected = {(condition, seed) for condition in CONDITION_ORDER
                for seed in (20260809, 20260810, 20260811)}
    actual = set(zip(df["condition"], df["seed"]))
    if actual != expected or len(df) != len(expected):
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(f"实验矩阵不完整: missing={missing}, extra={extra}, rows={len(df)}")

    df["condition"] = pd.Categorical(
        df["condition"], categories=CONDITION_ORDER, ordered=True
    )
    df = df.sort_values(["condition", "seed"]).reset_index(drop=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_DIR / "alpha_ablation_seed_results.csv", index=False)

    summary_rows = []
    t_critical = stats.t.ppf(0.975, df=2)
    for condition in CONDITION_ORDER:
        group = df[df["condition"] == condition]
        row = {
            "condition": condition,
            "n": len(group),
            "epochs_mean": group["epochs_completed"].mean(),
            "elapsed_minutes_mean": group["elapsed_seconds"].mean() / 60,
            "elapsed_minutes_std": group["elapsed_seconds"].std(ddof=1) / 60,
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
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(DATA_DIR / "alpha_ablation_summary.csv", index=False)

    indexed = {
        condition: df[df["condition"] == condition].set_index("seed")
        for condition in CONDITION_ORDER
    }
    pairwise_rows = []
    comparisons = [
        ("alpha_zero", "uniform_alpha"),
        ("alpha_zero", "per_pollutant_alpha"),
        ("alpha_zero", "learnable_alpha"),
        ("uniform_alpha", "per_pollutant_alpha"),
        ("learnable_alpha", "per_pollutant_alpha"),
    ]
    for left_name, right_name in comparisons:
        left = indexed[left_name]
        right = indexed[right_name]
        differences = left.loc[right.index, "valid_mse"] - right["valid_mse"]
        statistic, p_value = stats.ttest_rel(
            left.loc[right.index, "valid_mse"], right["valid_mse"]
        )
        std = differences.std(ddof=1)
        effect_dz = np.nan if std == 0 else differences.mean() / std
        pairwise_rows.append({
            "comparison": f"{left_name} - {right_name}",
            "mean_mse_difference": differences.mean(),
            "std_mse_difference": std,
            "relative_difference_percent": 100 * differences.mean() / left["valid_mse"].mean(),
            "paired_t": statistic,
            "df": 2,
            "p_raw": p_value,
            "cohen_dz": effect_dz,
        })
    pairwise = pd.DataFrame(pairwise_rows)
    pairwise["p_holm"] = holm_adjust(pairwise["p_raw"].to_numpy())
    pairwise.to_csv(DATA_DIR / "alpha_ablation_pairwise.csv", index=False)

    alpha_rows = []
    for condition in CONDITION_ORDER:
        values = np.vstack([
            np.asarray(ast.literal_eval(value), dtype=float)
            for value in df.loc[df["condition"] == condition, "selected_alpha"]
        ])
        for idx, pollutant in enumerate(POLLUTANTS):
            alpha_rows.append({
                "condition": condition,
                "pollutant": pollutant,
                "alpha_mean": values[:, idx].mean(),
                "alpha_std": values[:, idx].std(ddof=1),
                "alpha_min": values[:, idx].min(),
                "alpha_max": values[:, idx].max(),
            })
    pd.DataFrame(alpha_rows).to_csv(
        DATA_DIR / "alpha_ablation_selected_alpha.csv", index=False
    )

    pollutant_rows = []
    pollutant_values = {
        condition: {pollutant: [] for pollutant in POLLUTANTS}
        for condition in CONDITION_ORDER
    }
    for _, row in df.iterrows():
        values = json.loads(row["pollutant_mse"])
        for pollutant in POLLUTANTS:
            pollutant_values[str(row["condition"])][pollutant].append(values[pollutant])
    baseline_means = {
        pollutant: np.mean(pollutant_values["alpha_zero"][pollutant])
        for pollutant in POLLUTANTS
    }
    for condition in CONDITION_ORDER:
        for pollutant in POLLUTANTS:
            values = np.asarray(pollutant_values[condition][pollutant], dtype=float)
            mean = values.mean()
            pollutant_rows.append({
                "condition": condition,
                "pollutant": pollutant,
                "mse_mean": mean,
                "mse_std": values.std(ddof=1),
                "improvement_vs_alpha_zero_percent": (
                    100 * (baseline_means[pollutant] - mean) / baseline_means[pollutant]
                ),
            })
    pd.DataFrame(pollutant_rows).to_csv(
        DATA_DIR / "alpha_ablation_pollutant_summary.csv", index=False
    )

    print(summary.to_string(index=False))
    print("\nPaired comparisons (positive difference favors the right-hand strategy):")
    print(pairwise.to_string(index=False))


if __name__ == "__main__":
    main()
