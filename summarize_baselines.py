"""Independently recompute baseline metrics and emit protocol tables/figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_NPZ_KEYS = (
    "prediction_scaled",
    "target_scaled",
    "prediction_ugm3",
    "target_ugm3",
)
METRIC_COLUMNS = (
    "mse_scaled",
    "rmse_scaled",
    "mae_scaled",
    "rmse_ugm3",
    "mae_ugm3",
    "smape_percent",
)


def recompute_prediction_file(path: str | Path) -> dict:
    """Compute every reported metric directly from one persisted NPZ."""
    path = Path(path)
    with np.load(path) as payload:
        missing = set(REQUIRED_NPZ_KEYS) - set(payload.files)
        if missing:
            raise ValueError(f"{path} missing keys: {sorted(missing)}")
        prediction_scaled = payload["prediction_scaled"]
        target_scaled = payload["target_scaled"]
        prediction = payload["prediction_ugm3"]
        target = payload["target_ugm3"]
    scaled_error = prediction_scaled - target_scaled
    error = prediction - target
    denominator = np.abs(target) + np.abs(prediction)
    mse_scaled = float(np.mean(scaled_error ** 2))
    return {
        "mse_scaled": mse_scaled,
        # Match the legacy two-step scalar conversion exactly: its sqrt sees a
        # Python float (float64), while physical RMSE takes sqrt of a numpy scalar.
        "rmse_scaled": float(np.sqrt(mse_scaled)),
        "mae_scaled": float(np.mean(np.abs(scaled_error))),
        "rmse_ugm3": float(np.sqrt(np.mean(error ** 2))),
        "mae_ugm3": float(np.mean(np.abs(error))),
        "smape_percent": float(
            200 * np.mean(np.abs(error) / np.maximum(denominator, 1e-6))
        ),
    }


def _relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(abs(right), 1e-30)


def recompute_directory(result_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match NPZ files to runner rows, then verify without reusing runner math."""
    result_root = Path(result_root)
    rows = []
    comparisons = []
    for raw_path in sorted(result_root.rglob("raw_metrics.csv")):
        raw = pd.read_csv(raw_path)
        config_path = raw_path.parent / "experiment_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
        for _, recorded in raw.iterrows():
            status = str(recorded.get("status", "completed"))
            if status == "infeasible_oom":
                row = recorded.to_dict()
                row["result_dir"] = str(raw_path.parent)
                row["history"] = config.get("history")
                row["horizon"] = config.get("horizon")
                row["smoke"] = bool(config.get("smoke", False))
                row["independent_recomputed"] = False
                rows.append(row)
                continue
            relative_path = Path(str(recorded["prediction_file"]))
            prediction_path = raw_path.parent / relative_path
            if not prediction_path.is_file():
                raise FileNotFoundError(prediction_path)
            metrics = recompute_prediction_file(prediction_path)
            row = recorded.to_dict()
            row.update(metrics)
            row["result_dir"] = str(raw_path.parent)
            row["history"] = config.get("history")
            row["horizon"] = config.get("horizon")
            row["smoke"] = bool(config.get("smoke", False))
            row["independent_recomputed"] = True
            rows.append(row)
            for metric in METRIC_COLUMNS:
                relative = _relative_difference(metrics[metric], float(recorded[metric]))
                comparisons.append(
                    {
                        "result_dir": str(raw_path.parent),
                        "variant": recorded["variant"],
                        "seed": recorded["seed"],
                        "metric": metric,
                        "recorded": float(recorded[metric]),
                        "recomputed": metrics[metric],
                        "relative_difference": relative,
                        "pass": relative <= 1e-9,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(comparisons)


def compliance_checks(result_root: Path, rows: pd.DataFrame, comparisons: pd.DataFrame):
    checks = []

    def add(check: str, passed: bool, detail: str):
        checks.append({"check": check, "pass": bool(passed), "detail": detail})

    statuses = rows.get("status", pd.Series("completed", index=rows.index)).fillna("completed")
    feasible = rows[statuses != "infeasible_oom"]
    infeasible_count = int((statuses == "infeasible_oom").sum())
    recomputed_ok = bool(feasible["independent_recomputed"].all()) if len(feasible) else False
    add(
        "predictions independently recomputed",
        recomputed_ok,
        f"feasible_rows={len(feasible)}, infeasible_oom_rows={infeasible_count}",
    )
    worst = float(comparisons["relative_difference"].max()) if len(comparisons) else float("inf")
    add("runner metrics agree within 1e-9", bool(comparisons["pass"].all()) if len(comparisons) else False, f"max={worst:.3e}")
    add(
        "evaluation split fixed to test",
        bool((rows["evaluation_split"] == "test").all()) if len(rows) else False,
        "all rows must be test",
    )
    neural = rows[rows["layer"].isin(["B", "C"])] if len(rows) else rows
    add(
        "neural parameter counts registered",
        bool((neural["parameter_count"] > 0).all()) if len(neural) else False,
        f"neural_rows={len(neural)}",
    )
    matched = neural[neural["requested_capacity"] == "matched"]
    statuses = set(matched["capacity_status"].astype(str)) if len(matched) else set()
    add(
        "matched exceptions explicitly labelled",
        statuses <= {"matched", "matched_nearest"},
        f"statuses={sorted(statuses)}",
    )
    metadata_files = list(result_root.rglob("run_metadata.json"))
    stable = True
    for path in metadata_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        stable &= payload.get("code_state_before") == payload.get("code_state_after")
    add("code state stable during runs", stable and bool(metadata_files), f"runs={len(metadata_files)}")
    add(
        "traditional fitting policy registered",
        all((directory / "training_logs").is_dir() for directory in {Path(v) for v in rows.get("result_dir", [])}),
        "train-only logs present",
    )
    return pd.DataFrame(checks)


def paired_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Create within-family paired rows where default and matched share a seed."""
    neural = rows[rows["layer"].isin(["B", "C"])].copy()
    if neural.empty or "arm" not in neural:
        return pd.DataFrame()
    default = neural[neural["requested_capacity"] == "default"].set_index(
        ["result_dir", "arm", "seed"]
    )
    matched = neural[neural["requested_capacity"] == "matched"].set_index(
        ["result_dir", "arm", "seed"]
    )
    records = []
    for key in default.index.intersection(matched.index):
        left = default.loc[key]
        right = matched.loc[key]
        records.append(
            {
                "result_dir": key[0],
                "arm": key[1],
                "seed": key[2],
                "matched_minus_default_rmse_ugm3": right["rmse_ugm3"] - left["rmse_ugm3"],
                "matched_minus_default_mae_ugm3": right["mae_ugm3"] - left["mae_ugm3"],
            }
        )
    return pd.DataFrame(records)


def write_tables_and_figures(
    rows: pd.DataFrame,
    comparisons: pd.DataFrame,
    compliance: pd.DataFrame,
    tables_dir: Path,
    figures_dir: Path,
) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(tables_dir / "B1_baseline_main.csv", index=False)
    neural = rows[rows["layer"].isin(["B", "C"])].copy()
    neural.to_csv(tables_dir / "B2_capacity_comparison.csv", index=False)
    paired_table(rows).to_csv(tables_dir / "B3_paired_differences.csv", index=False)
    rows[rows["layer"] == "A"].to_csv(
        tables_dir / "B4_traditional_details.csv", index=False
    )
    comparisons.to_csv(tables_dir / "independent_recalculation.csv", index=False)
    compliance.to_csv(tables_dir / "compliance_self_check.csv", index=False)

    plot_rows = rows.copy()
    plot_rows["label"] = plot_rows["variant"].astype(str)
    means = plot_rows.groupby("label", sort=False)["rmse_ugm3"].mean().sort_values()
    fig, axis = plt.subplots(figsize=(10, max(4, 0.3 * len(means))))
    axis.barh(means.index, means.values)
    axis.set_xlabel("Test RMSE (μg/m³)")
    axis.set_title("BL-F1 Baseline comparison")
    fig.tight_layout()
    fig.savefig(figures_dir / "BL-F1_main_comparison.png", dpi=180)
    plt.close(fig)

    if {"history", "horizon"} <= set(rows.columns):
        heat = rows.pivot_table(
            index="history", columns="horizon", values="rmse_ugm3", aggfunc="mean"
        )
        fig, axis = plt.subplots(figsize=(7, 4))
        image = axis.imshow(heat.to_numpy(), aspect="auto")
        axis.set_xticks(range(len(heat.columns)), heat.columns)
        axis.set_yticks(range(len(heat.index)), heat.index)
        axis.set_xlabel("Horizon")
        axis.set_ylabel("History")
        axis.set_title("BL-F2 Mean RMSE by configuration")
        fig.colorbar(image, ax=axis, label="RMSE (μg/m³)")
        fig.tight_layout()
        fig.savefig(figures_dir / "BL-F2_configuration_heatmap.png", dpi=180)
        plt.close(fig)

    if len(neural):
        fig, axis = plt.subplots(figsize=(7, 5))
        axis.scatter(neural["parameter_count"], neural["rmse_ugm3"], alpha=0.75)
        axis.set_xscale("log")
        axis.set_xlabel("Parameters (log scale)")
        axis.set_ylabel("Test RMSE (μg/m³)")
        axis.set_title("BL-F3 Capacity-performance")
        fig.tight_layout()
        fig.savefig(figures_dir / "BL-F3_capacity_performance.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default="experiments/results/baselines")
    parser.add_argument("--tables-dir", default="tables/baselines")
    parser.add_argument("--figures-dir", default="figures/baselines")
    args = parser.parse_args()
    result_root = Path(args.result_root)
    rows, comparisons = recompute_directory(result_root)
    if rows.empty:
        raise RuntimeError(f"No baseline raw_metrics.csv found below {result_root}")
    compliance = compliance_checks(result_root, rows, comparisons)
    write_tables_and_figures(
        rows,
        comparisons,
        compliance,
        Path(args.tables_dir),
        Path(args.figures_dir),
    )
    worst = comparisons["relative_difference"].max()
    recomputed_count = int(rows.get("independent_recomputed", pd.Series(dtype=bool)).fillna(False).sum())
    infeasible_count = int((rows.get("status", pd.Series(index=rows.index, dtype=object)) == "infeasible_oom").sum())
    print(
        f"recomputed_runs={recomputed_count} infeasible_oom_rows={infeasible_count} "
        f"max_relative_difference={worst:.3e}"
    )
    print(compliance.to_string(index=False))
    if not comparisons["pass"].all():
        raise SystemExit("Independent metric recalculation failed")
    print("BASELINE_SUMMARY_DONE")


if __name__ == "__main__":
    main()
