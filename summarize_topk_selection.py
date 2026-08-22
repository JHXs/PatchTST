"""Apply the pre-registered validation-only Top-k selection rule."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path("experiments/results/st_patchtst_ablation")
CANDIDATE = "st_sparse_station_bias_delta_forecast"
TASKS = {
    "24h_to_1h": (1.0, "24h_1h"),
    "168h_to_6h": (0.5, "168h_6h"),
}
TOP_K_VALUES = (3, 5, 7, 9, 12)


def directory_for(top_k: int, suffix: str) -> Path:
    if top_k == 7:
        return ROOT / f"round5_selection_frozen_sparse_station_{suffix}"
    return ROOT / f"round6_selection_topk{top_k}_{suffix}"


def task_summary(top_k: int, task: str, threshold: float, suffix: str) -> dict:
    directory = directory_for(top_k, suffix)
    config = json.loads((directory / "experiment_config.json").read_text())
    configured_top_k = int(config.get("sparse_neighbor_top_k", 7))
    if configured_top_k != top_k:
        raise ValueError(
            f"目录标注k={top_k}，但实验配置为k={configured_top_k}: {directory}"
        )
    raw = pd.read_csv(directory / "raw_metrics.csv")
    if set(raw["evaluation_split"]) != {"valid"}:
        raise ValueError(f"k={top_k} {task} 不是纯验证集结果")
    baseline = raw[raw["variant"] == "degraded_patchtst"].set_index("seed")
    candidate = raw[raw["variant"] == CANDIDATE].set_index("seed")
    seeds = sorted(set(baseline.index) & set(candidate.index))
    if seeds != [2024, 2025, 2026]:
        raise ValueError(f"k={top_k} {task} 种子不完整: {seeds}")
    reductions = (
        (baseline.loc[seeds, "rmse_ugm3"] - candidate.loc[seeds, "rmse_ugm3"])
        / baseline.loc[seeds, "rmse_ugm3"]
        * 100
    )
    mean_reduction = float(reductions.mean())
    return {
        "top_k": top_k,
        "task": task,
        "mean_rmse_reduction_percent": mean_reduction,
        "improved_seed_count": int((reductions > 0).sum()),
        "practical_threshold_percent": threshold,
        "normalized_margin": mean_reduction / threshold,
        "task_gate_pass": bool((reductions > 0).all() and mean_reduction >= threshold),
    }


def main() -> None:
    rows = [
        task_summary(top_k, task, threshold, suffix)
        for top_k in TOP_K_VALUES
        for task, (threshold, suffix) in TASKS.items()
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(ROOT / "topk_validation_selection.csv", index=False)

    candidate_scores = []
    for top_k, group in summary.groupby("top_k"):
        candidate_scores.append(
            {
                "top_k": int(top_k),
                "all_task_gates_pass": bool(group["task_gate_pass"].all()),
                "selection_score": float(group["normalized_margin"].min()),
            }
        )
    scores = pd.DataFrame(candidate_scores).sort_values(
        ["all_task_gates_pass", "selection_score", "top_k"],
        ascending=[False, False, True],
    )
    scores.to_csv(ROOT / "topk_validation_scores.csv", index=False)
    selected = scores.iloc[0]
    top7_score = float(scores.loc[scores["top_k"] == 7, "selection_score"].iloc[0])
    status = {
        "selected_top_k": int(selected["top_k"]),
        "selected_score": float(selected["selection_score"]),
        "top7_score": top7_score,
        "all_task_gates_pass": bool(selected["all_task_gates_pass"]),
        "eligible_for_fresh_seed_confirmation": bool(
            selected["all_task_gates_pass"]
            and int(selected["top_k"]) != 7
            and float(selected["selection_score"]) > top7_score
        ),
    }
    (ROOT / "topk_validation_selection.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(scores.to_string(index=False))
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
