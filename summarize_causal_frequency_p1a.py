"""Independently verify FA1 artifacts and reproduce its gate decision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_causal_frequency_p0 import SELECTION_SEEDS
from run_causal_frequency_p1a import ALTERNATIVE_VARIANT, ALL_VARIANTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        default=(
            "experiments/results/causal_frequency_ablation/"
            "p1_alternative_selection/168h_6h_selection"
        ),
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha256(source_files: list[str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(source_files):
        digest.update(name.encode("utf-8"))
        digest.update(Path(name).read_bytes())
    return digest.hexdigest()


def verify(result_dir: Path) -> dict:
    required = (
        "experiment_config.json",
        "dataset_metadata.json",
        "raw_metrics.csv",
        "summary_metrics.csv",
        "paired_differences.csv",
        "per_horizon_metrics.csv",
        "peak_metrics.csv",
        "stage_status.json",
        "runtime_checks.json",
        "auto_report.md",
    )
    missing = [name for name in required if not (result_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{result_dir} 缺少产物: {missing}")

    config = _load_json(result_dir / "experiment_config.json")
    metadata = _load_json(result_dir / "dataset_metadata.json")
    recorded_stage = _load_json(result_dir / "stage_status.json")
    runtime = _load_json(result_dir / "runtime_checks.json")
    raw = pd.read_csv(result_dir / "raw_metrics.csv")
    recorded_paired = pd.read_csv(result_dir / "paired_differences.csv")

    prediction_rmse_errors = []
    for row in raw.itertuples():
        prediction_path = (
            result_dir / "predictions" / f"{row.variant}_seed{row.seed}.npz"
        )
        payload = np.load(prediction_path)
        prediction = payload["prediction_ugm3"]
        target = payload["target_ugm3"]
        recomputed_rmse = float(np.sqrt(np.mean((prediction - target) ** 2)))
        prediction_rmse_errors.append(abs(recomputed_rmse - row.rmse_ugm3))
        if payload["target_timestamp_matrix_ns"].shape != target.shape[:1] + (
            config["horizon"],
        ):
            raise AssertionError(f"时间戳矩阵形状不符: {prediction_path}")

    indexed = {
        variant: raw[raw["variant"] == variant].set_index("seed")
        for variant in ALL_VARIANTS
    }
    alt_vs_st = []
    alt_vs_time = []
    recomputed_pair_rows = []
    for candidate, baseline in (
        ("st_time_residual_adapter", "locked_st_clean"),
        (ALTERNATIVE_VARIANT, "locked_st_clean"),
        (ALTERNATIVE_VARIANT, "st_time_residual_adapter"),
    ):
        for seed in SELECTION_SEEDS:
            candidate_rmse = float(indexed[candidate].loc[seed, "rmse_ugm3"])
            baseline_rmse = float(indexed[baseline].loc[seed, "rmse_ugm3"])
            improvement = 100 * (baseline_rmse - candidate_rmse) / baseline_rmse
            recomputed_pair_rows.append(
                {
                    "comparison": f"{candidate}_vs_{baseline}",
                    "seed": seed,
                    "rmse_difference_ugm3": candidate_rmse - baseline_rmse,
                    "rmse_relative_improvement_percent": improvement,
                }
            )
            if candidate == ALTERNATIVE_VARIANT and baseline == "locked_st_clean":
                alt_vs_st.append(improvement)
            if (
                candidate == ALTERNATIVE_VARIANT
                and baseline == "st_time_residual_adapter"
            ):
                alt_vs_time.append(improvement)

    recomputed_pairs = pd.DataFrame(recomputed_pair_rows).sort_values(
        ["comparison", "seed"]
    )
    recorded_pairs = recorded_paired.sort_values(["comparison", "seed"])
    paired_values_match = (
        recomputed_pairs[["comparison", "seed"]].reset_index(drop=True).equals(
            recorded_pairs[["comparison", "seed"]].reset_index(drop=True)
        )
        and np.allclose(
            recomputed_pairs["rmse_difference_ugm3"],
            recorded_pairs["rmse_difference_ugm3"],
            rtol=0,
            atol=1e-12,
        )
        and np.allclose(
            recomputed_pairs["rmse_relative_improvement_percent"],
            recorded_pairs["rmse_relative_improvement_percent"],
            rtol=0,
            atol=1e-12,
        )
    )
    alt_mean = float(indexed[ALTERNATIVE_VARIANT]["rmse_ugm3"].mean())
    time_mean = float(indexed["st_time_residual_adapter"]["rmse_ugm3"].mean())
    gate_checks = {
        "runtime_checks_passed": all(
            row["initial_exact_base"]
            and row["base_state_unchanged"]
            and row["disable_frequency_exact_base"]
            and row["reconstruction_max_abs_error"] <= 1e-5
            and row["all_metrics_finite"]
            and row["capacity_difference_percent_vs_time"] <= 5
            for row in runtime
        ),
        "complete_seed_variant_grid": len(raw)
        == len(SELECTION_SEEDS) * len(ALL_VARIANTS),
        "selection_seeds_exact": sorted(raw["seed"].unique().tolist())
        == list(SELECTION_SEEDS),
        "alternative_three_of_three_better_than_st": bool(
            np.all(np.asarray(alt_vs_st) > 0)
        ),
        "alternative_mean_improvement_at_least_0_5_percent": float(
            np.mean(alt_vs_st)
        )
        >= 0.5,
        "alternative_mean_rmse_better_than_time_control": alt_mean < time_mean,
        "capacity_difference_at_most_5_percent": float(
            indexed[ALTERNATIVE_VARIANT][
                "capacity_difference_percent_vs_time"
            ].max()
        )
        <= 5,
    }
    recomputed_status = "passed" if all(gate_checks.values()) else "failed"
    control_raw = Path(config["reused_control_dir"]) / "raw_metrics.csv"
    integrity_checks = {
        "task_is_168_to_6": (config["history"], config["horizon"]) == (168, 6),
        "evidence_label_selection": config["evidence_label"] == "selection",
        "test_metrics_not_computed": config["test_metrics_computed"] is False,
        "target_timestamps_disjoint": bool(
            metadata["target_timestamp_intersections_empty"]
        ),
        "source_hash_matches": _source_sha256(config["source_files"])
        == config["source_files_sha256"],
        "control_raw_hash_matches": _path_sha256(control_raw)
        == config["reused_control_raw_metrics_sha256"],
        "prediction_rmse_matches_raw": max(prediction_rmse_errors) == 0.0,
        "paired_values_match_recorded": bool(paired_values_match),
        "recorded_gate_checks_match": gate_checks == recorded_stage["checks"],
        "recorded_gate_status_matches": recomputed_status
        == recorded_stage["status"],
    }
    return {
        "gate": "FA1",
        "recomputed_status": recomputed_status,
        "recorded_status": recorded_stage["status"],
        "integrity_checks": integrity_checks,
        "all_integrity_checks_passed": all(integrity_checks.values()),
        "gate_checks": gate_checks,
        "alternative_vs_st_improvements_percent": alt_vs_st,
        "alternative_vs_st_mean_improvement_percent": float(
            np.mean(alt_vs_st)
        ),
        "alternative_vs_time_improvements_percent": alt_vs_time,
        "alternative_rmse_mean": alt_mean,
        "time_control_rmse_mean": time_mean,
        "allowed_next_gate": None,
    }


def write_report(result_dir: Path, verification: dict) -> None:
    lines = [
        "# FA1 独立复算报告",
        "",
        "> SELECTION DATA - validation only; test metrics were not computed.",
        "",
        f"- 记录状态：{verification['recorded_status']}",
        f"- 复算状态：{verification['recomputed_status']}",
        f"- 完整性检查：{verification['all_integrity_checks_passed']}",
        "- 相对锁定ST的逐种子RMSE改善："
        + ", ".join(
            f"{value:.4f}%"
            for value in verification[
                "alternative_vs_st_improvements_percent"
            ]
        ),
        "- 相对锁定ST的平均RMSE改善："
        f"{verification['alternative_vs_st_mean_improvement_percent']:.4f}%",
        f"- 替代平均RMSE：{verification['alternative_rmse_mean']:.4f} μg/m³",
        f"- 时域控制平均RMSE：{verification['time_control_rmse_mean']:.4f} μg/m³",
        "",
        "FA1失败表示当前预注册局部分解未超过等容量时域控制；失败结果不开放后续Gate。",
    ]
    (result_dir / "independent_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    verification = verify(result_dir)
    (result_dir / "independent_verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    write_report(result_dir, verification)
    if not verification["all_integrity_checks_passed"]:
        raise AssertionError("FA1独立完整性复算失败")
    print(f"FA1独立复算已写入: {result_dir.resolve()}")
    print(f"Gate FA1: {verification['recomputed_status']}")


if __name__ == "__main__":
    main()
