"""Summarize pre-registered ST-PatchTST stability confirmation attempts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


ROOT = Path("experiments/results/st_patchtst_ablation")
ATTEMPTS = {
    "joint_2027_2031": {
        "24h_to_1h": ("stability_confirmation_24h_1h", "st_station_bias_delta_forecast"),
        "168h_to_6h": ("stability_confirmation_168h_6h", "st_station_bias_delta_forecast"),
    },
    "frozen_2032_2036": {
        "24h_to_1h": ("stability_confirmation_frozen_24h_1h", "st_station_bias_delta_forecast"),
        "168h_to_6h": ("stability_confirmation_frozen_168h_6h", "st_station_bias_delta_forecast"),
    },
    "anchored_2037_2041": {
        "24h_to_1h": ("stability_confirmation_anchored_24h_1h", "st_station_bias_delta_forecast"),
        "168h_to_6h": ("stability_confirmation_anchored_168h_6h", "st_station_bias_delta_forecast"),
    },
    "frozen_sparse_station_2042_2046": {
        "24h_to_1h": (
            "stability_confirmation_sparse_station_24h_1h",
            "st_sparse_station_bias_delta_forecast",
        ),
        "168h_to_6h": (
            "stability_confirmation_sparse_station_168h_6h",
            "st_sparse_station_bias_delta_forecast",
        ),
    },
    "frozen_topk5_2047_2051": {
        "24h_to_1h": (
            "stability_confirmation_topk5_24h_1h",
            "st_sparse_station_bias_delta_forecast",
        ),
        "168h_to_6h": (
            "stability_confirmation_topk5_168h_6h",
            "st_sparse_station_bias_delta_forecast",
        ),
    },
}
PRACTICAL_THRESHOLDS = {"24h_to_1h": 1.0, "168h_to_6h": 0.5}
FINAL_ATTEMPT = "frozen_topk5_2047_2051"
BOOTSTRAP_SAMPLES = 100_000
EXPECTED_SEEDS = {
    "joint_2027_2031": list(range(2027, 2032)),
    "frozen_2032_2036": list(range(2032, 2037)),
    "anchored_2037_2041": list(range(2037, 2042)),
    "frozen_sparse_station_2042_2046": list(range(2042, 2047)),
    "frozen_topk5_2047_2051": list(range(2047, 2052)),
}


def bootstrap_mean_ci(values, seed):
    generator = np.random.default_rng(seed)
    samples = generator.choice(
        values, size=(BOOTSTRAP_SAMPLES, len(values)), replace=True
    ).mean(axis=1)
    return np.quantile(samples, [0.025, 0.975])


def summarize_attempt(attempt, task, directory_name, candidate_variant, seed_offset):
    directory = ROOT / directory_name
    raw = pd.read_csv(directory / "raw_metrics.csv")
    if set(raw["evaluation_split"]) != {"test"}:
        raise ValueError(f"{attempt} {task} 不是纯测试集确认结果")
    baseline = raw[raw["variant"] == "degraded_patchtst"].set_index("seed")
    candidate = raw[raw["variant"] == candidate_variant].set_index("seed")
    seeds = sorted(set(baseline.index) & set(candidate.index))
    if seeds != EXPECTED_SEEDS[attempt] or len(baseline) != 5 or len(candidate) != 5:
        raise ValueError(
            f"{attempt} {task} 配对种子不完整: baseline={list(baseline.index)}, "
            f"candidate={list(candidate.index)}"
        )
    if attempt in {
        "frozen_sparse_station_2042_2046",
        "frozen_topk5_2047_2051",
    }:
        config = json.loads((directory / "experiment_config.json").read_text())
        expected_top_k = 5 if attempt == FINAL_ATTEMPT else 7
        configured_top_k = int(config.get("sparse_neighbor_top_k", 7))
        if configured_top_k != expected_top_k:
            raise ValueError(
                f"{attempt} {task} Top-k配置错误: {configured_top_k} != {expected_top_k}"
            )
    rows = []
    for seed in seeds:
        baseline_row = baseline.loc[seed]
        candidate_row = candidate.loc[seed]
        rows.append(
            {
                "attempt": attempt,
                "task": task,
                "seed": int(seed),
                "candidate_variant": candidate_variant,
                "baseline_rmse_ugm3": baseline_row["rmse_ugm3"],
                "candidate_rmse_ugm3": candidate_row["rmse_ugm3"],
                "rmse_reduction_percent": (
                    baseline_row["rmse_ugm3"] - candidate_row["rmse_ugm3"]
                )
                / baseline_row["rmse_ugm3"]
                * 100,
                "baseline_mae_ugm3": baseline_row["mae_ugm3"],
                "candidate_mae_ugm3": candidate_row["mae_ugm3"],
                "mae_reduction_percent": (
                    baseline_row["mae_ugm3"] - candidate_row["mae_ugm3"]
                )
                / baseline_row["mae_ugm3"]
                * 100,
            }
        )
    paired = pd.DataFrame(rows)
    reductions = paired["rmse_reduction_percent"].to_numpy()
    ci_low, ci_high = bootstrap_mean_ci(reductions, 20260820 + seed_offset)
    improved_count = int((reductions > 0).sum())
    sign_p = binomtest(
        improved_count, len(reductions), p=0.5, alternative="greater"
    ).pvalue
    threshold = PRACTICAL_THRESHOLDS[task]
    summary = {
        "attempt": attempt,
        "task": task,
        "runs": len(paired),
        "baseline_rmse_ugm3_mean": paired["baseline_rmse_ugm3"].mean(),
        "candidate_rmse_ugm3_mean": paired["candidate_rmse_ugm3"].mean(),
        "rmse_reduction_percent_mean": reductions.mean(),
        "rmse_reduction_percent_std": reductions.std(ddof=1),
        "rmse_reduction_bootstrap_ci95_low": ci_low,
        "rmse_reduction_bootstrap_ci95_high": ci_high,
        "rmse_improved_seed_count": improved_count,
        "one_sided_exact_sign_p": sign_p,
        "practical_threshold_percent": threshold,
        "direction_gate_pass": improved_count == len(paired),
        "practical_gate_pass": reductions.mean() >= threshold,
        "strict_task_gate_pass": (
            improved_count == len(paired)
            and sign_p <= 0.05
            and reductions.mean() >= threshold
        ),
        "baseline_mae_ugm3_mean": paired["baseline_mae_ugm3"].mean(),
        "candidate_mae_ugm3_mean": paired["candidate_mae_ugm3"].mean(),
        "mae_improved_seed_count": int(
            (paired["mae_reduction_percent"] > 0).sum()
        ),
    }
    return paired, summary


def main():
    paired_frames = []
    summary_rows = []
    seed_offset = 0
    for attempt, tasks in ATTEMPTS.items():
        for task, (directory_name, candidate_variant) in tasks.items():
            paired, summary = summarize_attempt(
                attempt, task, directory_name, candidate_variant, seed_offset
            )
            seed_offset += 1
            paired_frames.append(paired)
            summary_rows.append(summary)
    paired_output = pd.concat(paired_frames, ignore_index=True)
    summary_output = pd.DataFrame(summary_rows)
    paired_output.to_csv(ROOT / "stability_confirmation_paired.csv", index=False)
    summary_output.to_csv(ROOT / "stability_confirmation_summary.csv", index=False)

    final_rows = summary_output[summary_output["attempt"] == FINAL_ATTEMPT]
    strict_release_pass = bool(final_rows["strict_task_gate_pass"].all())
    status = {
        "final_attempt": FINAL_ATTEMPT,
        "strict_release_gate_pass": strict_release_pass,
        "direction_consistency_pass": bool(final_rows["direction_gate_pass"].all()),
        "task_status": final_rows.set_index("task")[
            [
                "rmse_reduction_percent_mean",
                "rmse_improved_seed_count",
                "one_sided_exact_sign_p",
                "practical_threshold_percent",
                "strict_task_gate_pass",
            ]
        ].to_dict(orient="index"),
        "interpretation": (
            "The final attempt passed the pre-registered seed-direction, exact-sign, "
            "and practical-effect gates for both tasks. Scope remains limited to the "
            "current city, center station, pollutant, and horizons."
            if strict_release_pass
            else "Direction-consistent RMSE improvement may be reported only with the "
            "pre-registered practical-effect caveat when strict_release_gate_pass is false."
        ),
    }
    (ROOT / "stability_release_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary_output.to_string(index=False))
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
