"""Recompute the combined P0 bridge summary and overall Gate F1 status."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_causal_frequency_p0 import P0_VARIANTS, SELECTION_SEEDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary-dir",
        default=(
            "experiments/results/causal_frequency_ablation/p0_bridge/"
            "168h_6h_selection"
        ),
    )
    parser.add_argument(
        "--auxiliary-dir",
        default=(
            "experiments/results/causal_frequency_ablation/p0_bridge/"
            "24h_1h_selection"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/results/causal_frequency_ablation/p0_bridge",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def recompute_task(task_dir: Path, expected_task: tuple[int, int]) -> dict:
    """Load one P0 task and independently recompute its paired RMSE effects."""

    required = (
        "experiment_config.json",
        "dataset_metadata.json",
        "raw_metrics.csv",
        "summary_metrics.csv",
        "paired_differences.csv",
        "per_horizon_metrics.csv",
        "peak_metrics.csv",
        "stage_status.json",
        "auto_report.md",
    )
    missing = [name for name in required if not (task_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{task_dir} 缺少产物: {missing}")

    config = _load_json(task_dir / "experiment_config.json")
    metadata = _load_json(task_dir / "dataset_metadata.json")
    recorded_stage = _load_json(task_dir / "stage_status.json")
    raw = pd.read_csv(task_dir / "raw_metrics.csv")
    recorded_paired = pd.read_csv(task_dir / "paired_differences.csv")
    history, horizon = expected_task
    if (config["history"], config["horizon"]) != expected_task:
        raise ValueError(f"任务目录与配置不符: {task_dir}")

    baseline = raw[
        raw["variant"] == "degraded_patchtst_clean"
    ].set_index("seed")
    locked = raw[raw["variant"] == "locked_st_clean"].set_index("seed")
    common_seeds = sorted(set(baseline.index) & set(locked.index))
    paired_rows = []
    for seed in common_seeds:
        baseline_rmse = float(baseline.loc[seed, "rmse_ugm3"])
        locked_rmse = float(locked.loc[seed, "rmse_ugm3"])
        paired_rows.append(
            {
                "seed": int(seed),
                "rmse_difference_ugm3": locked_rmse - baseline_rmse,
                "rmse_relative_improvement_percent": 100
                * (baseline_rmse - locked_rmse)
                / baseline_rmse,
            }
        )
    paired = pd.DataFrame(paired_rows)
    recorded = recorded_paired.sort_values("seed").reset_index(drop=True)
    recomputed = paired.sort_values("seed").reset_index(drop=True)
    paired_matches_recorded = (
        recorded["seed"].tolist() == recomputed["seed"].tolist()
        and np.allclose(
            recorded["rmse_difference_ugm3"],
            recomputed["rmse_difference_ugm3"],
            rtol=0,
            atol=1e-10,
        )
        and np.allclose(
            recorded["rmse_relative_improvement_percent"],
            recomputed["rmse_relative_improvement_percent"],
            rtol=0,
            atol=1e-10,
        )
    )
    improvements = recomputed["rmse_relative_improvement_percent"].to_numpy()

    summary_rows = []
    for variant in P0_VARIANTS:
        group = raw[raw["variant"] == variant]
        summary_rows.append(
            {
                "task": f"{history}h_{horizon}h",
                "role": "primary" if expected_task == (168, 6) else "auxiliary",
                "variant": variant,
                "runs": len(group),
                "rmse_ugm3_mean": float(group["rmse_ugm3"].mean()),
                "rmse_ugm3_std": float(group["rmse_ugm3"].std(ddof=1)),
                "mae_ugm3_mean": float(group["mae_ugm3"].mean()),
                "mae_ugm3_std": float(group["mae_ugm3"].std(ddof=1)),
                "smape_percent_mean": float(group["smape_percent"].mean()),
                "smape_percent_std": float(group["smape_percent"].std(ddof=1)),
                "paired_rmse_improved_seeds": (
                    int(np.sum(improvements > 0))
                    if variant == "locked_st_clean"
                    else None
                ),
                "paired_rmse_improvement_percent_mean": (
                    float(np.mean(improvements))
                    if variant == "locked_st_clean"
                    else None
                ),
            }
        )

    checks = {
        "complete_seed_variant_grid": len(raw)
        == len(SELECTION_SEEDS) * len(P0_VARIANTS),
        "selection_seeds_exact": sorted(raw["seed"].unique().tolist())
        == list(SELECTION_SEEDS),
        "variant_ids_exact": sorted(raw["variant"].unique().tolist())
        == sorted(P0_VARIANTS),
        "evidence_label_selection": config["evidence_label"] == "selection",
        "evaluation_split_valid": set(raw["evaluation_split"]) == {"valid"},
        "test_metrics_not_computed": config["test_metrics_computed"] is False,
        "target_timestamps_disjoint": bool(
            metadata["target_timestamp_intersections_empty"]
        ),
        "paired_values_match_recorded": bool(paired_matches_recorded),
        "locked_backbone_unchanged": bool(
            locked["backbone_frozen_unchanged"].all()
        ),
        "core_metrics_finite": bool(
            np.isfinite(raw[["rmse_ugm3", "mae_ugm3", "best_valid_loss"]])
            .all()
            .all()
        ),
    }
    return {
        "task": f"{history}h_{horizon}h",
        "task_dir": str(task_dir),
        "summary_rows": summary_rows,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "improved_seed_count": int(np.sum(improvements > 0)),
        "paired_seed_count": len(improvements),
        "mean_rmse_relative_improvement_percent": float(np.mean(improvements)),
        "minimum_rmse_relative_improvement_percent": float(np.min(improvements)),
        "recorded_task_gate_status": recorded_stage["status"],
    }


def build_overall_gate(primary: dict, auxiliary: dict) -> dict:
    checks = {
        "primary_artifacts_and_invariants_passed": primary["all_checks_passed"],
        "auxiliary_artifacts_and_invariants_passed": auxiliary[
            "all_checks_passed"
        ],
        "primary_three_of_three_rmse_improved": primary["improved_seed_count"]
        == 3,
        "primary_mean_rmse_improvement_at_least_0_5_percent": primary[
            "mean_rmse_relative_improvement_percent"
        ]
        >= 0.5,
        "primary_recorded_gate_passed": primary["recorded_task_gate_status"]
        == "passed",
        "auxiliary_fully_reported": auxiliary["paired_seed_count"] == 3,
    }
    return {
        "gate": "F1",
        "protocol_version": "FREQ-PROTOCOL-v1",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "primary": {
            key: primary[key]
            for key in (
                "task",
                "improved_seed_count",
                "paired_seed_count",
                "mean_rmse_relative_improvement_percent",
                "minimum_rmse_relative_improvement_percent",
            )
        },
        "auxiliary": {
            key: auxiliary[key]
            for key in (
                "task",
                "improved_seed_count",
                "paired_seed_count",
                "mean_rmse_relative_improvement_percent",
                "minimum_rmse_relative_improvement_percent",
            )
        },
        "allowed_next_gate": "F2" if all(checks.values()) else None,
    }


def write_report(output_dir: Path, combined: pd.DataFrame, gate: dict) -> None:
    lines = [
        "# 频域分支 P0 无泄漏过渡汇总",
        "",
        "> SELECTION DATA - validation only; test metrics were not computed.",
        "",
        f"- Gate F1：{gate['status']}",
        "- 主任务：168→6",
        "- 辅助任务：24→1",
        "",
        "## 配对 RMSE 改善",
        "",
        "| 任务 | 改善种子 | 平均相对改善 | 最小相对改善 |",
        "|---|---:|---:|---:|",
    ]
    for role in ("primary", "auxiliary"):
        item = gate[role]
        lines.append(
            f"| {item['task']} | {item['improved_seed_count']}/"
            f"{item['paired_seed_count']} | "
            f"{item['mean_rmse_relative_improvement_percent']:.4f}% | "
            f"{item['minimum_rmse_relative_improvement_percent']:.4f}% |"
        )
    lines.extend(
        [
            "",
            "Gate F1 只确认无泄漏协议下的锁定 ST 过渡成立，不是频域模型结果。",
        ]
    )
    (output_dir / "p0_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    primary = recompute_task(Path(args.primary_dir), (168, 6))
    auxiliary = recompute_task(Path(args.auxiliary_dir), (24, 1))
    gate = build_overall_gate(primary, auxiliary)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.DataFrame(primary["summary_rows"] + auxiliary["summary_rows"])
    combined.to_csv(output_dir / "p0_combined_summary.csv", index=False)
    (output_dir / "p0_gate_status.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    write_report(output_dir, combined, gate)
    print(f"P0 汇总已写入: {output_dir.resolve()}")
    print(f"Gate F1: {gate['status']}")


if __name__ == "__main__":
    main()
