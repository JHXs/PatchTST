"""Independently recompute baseline metrics and emit protocol tables/figures."""

from __future__ import annotations

import argparse
import json
import re
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
SAME_INFORMATION_ARMS = frozenset(
    {
        "plain_mix_patchtst_all",
        "plain_mix_patchtst_top5",
        "concat_patchtst_all",
        "multi_gru",
        "multi_tst",
        "trad_spatial_linear",
    }
)
ST_VARIANT = "st_sparse_station_bias_delta_forecast"


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
        relative_parts = raw_path.parent.relative_to(result_root).parts
        city = relative_parts[0] if relative_parts else "unknown"
        station_id = next(
            (
                int(part.removeprefix("station_"))
                for part in relative_parts
                if part.startswith("station_")
            ),
            None,
        )
        config_path = raw_path.parent / "experiment_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
        # 目录名是 L/H 的权威来源：运行中或崩溃的配置不会有 experiment_config.json，
        # 之前回落到 config.get(...) 会得到 None → int(None) 崩溃。
        directory_match = re.match(r"(\d+)h_(\d+)h$", raw_path.parent.name)
        if directory_match is None:
            continue
        config_history = int(directory_match.group(1))
        config_horizon = int(directory_match.group(2))
        for _, recorded in raw.iterrows():
            status = str(recorded.get("status", "completed"))
            if status.startswith(("infeasible", "nonfinite")):
                row = recorded.to_dict()
                row["result_dir"] = str(raw_path.parent)
                row["history"] = config_history
                row["horizon"] = config_horizon
                row["smoke"] = bool(config.get("smoke", False))
                row["city"] = city
                row["station_id"] = station_id
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
            row["history"] = config_history
            row["horizon"] = config_horizon
            row["smoke"] = bool(config.get("smoke", False))
            row["city"] = city
            row["station_id"] = station_id
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
    # 不可复算的行有两类：OOM 与训练发散（nonfinite）——两者都没有预测文件，
    # 因此都必须从"应可独立复算"的集合里排除，否则自检会误报 False。
    not_recomputable = statuses.astype(str).str.startswith(("infeasible", "nonfinite"))
    feasible = rows[~not_recomputable]
    infeasible_count = int(statuses.astype(str).str.startswith("infeasible").sum())
    nonfinite_count = int(statuses.astype(str).str.startswith("nonfinite").sum())
    recomputed_ok = bool(feasible["independent_recomputed"].all()) if len(feasible) else False
    add(
        "predictions independently recomputed",
        recomputed_ok,
        f"feasible_rows={len(feasible)}, infeasible_oom_rows={infeasible_count}, "
        f"nonfinite_rows={nonfinite_count}",
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


def _rmse_from_npz(path: Path) -> float:
    with np.load(path) as payload:
        error = payload["prediction_ugm3"] - payload["target_ugm3"]
    return float(np.sqrt(np.mean(error ** 2)))


def load_st_seed_metrics(
    city: str,
    history: int,
    horizon: int,
    seeds: list[int],
    beijing_st_root: Path,
    guangzhou_st_root: Path,
) -> pd.DataFrame:
    """Load frozen ST results; absent artifacts are explicit missing rows."""
    records = []
    task_name = f"{int(history)}h_{int(horizon)}h"
    if city == "beijing":
        raw_path = beijing_st_root / task_name / "raw_metrics.csv"
        available = {}
        if raw_path.is_file():
            raw = pd.read_csv(raw_path)
            selected = raw[raw["variant"] == ST_VARIANT]
            available = {
                int(row["seed"]): float(row["rmse_ugm3"])
                for _, row in selected.iterrows()
            }
        for seed in seeds:
            value = available.get(int(seed))
            records.append(
                {
                    "seed": int(seed),
                    "st_rmse_ugm3": value if value is not None else np.nan,
                    "st_status": "available" if value is not None else "missing",
                    "st_source_count": 1 if value is not None else 0,
                }
            )
        return pd.DataFrame(records)

    task_dir = guangzhou_st_root / task_name
    station_dirs = sorted(path for path in task_dir.glob("station_*") if path.is_dir())
    by_seed = {int(seed): [] for seed in seeds}
    for station_dir in station_dirs:
        raw_path = station_dir / "raw_metrics.csv"
        if raw_path.is_file():
            raw = pd.read_csv(raw_path)
            selected = raw[raw["variant"] == ST_VARIANT]
            for _, row in selected.iterrows():
                seed = int(row["seed"])
                if seed in by_seed:
                    by_seed[seed].append(float(row["rmse_ugm3"]))
            continue
        manifest_path = station_dir / "run_manifest.csv"
        if not manifest_path.is_file():
            continue
        manifest = pd.read_csv(manifest_path)
        selected = manifest[manifest["arm"] == ST_VARIANT]
        for _, row in selected.iterrows():
            seed = int(row["seed"])
            artifact = station_dir / str(row["artifact"])
            if seed in by_seed and artifact.is_file():
                by_seed[seed].append(_rmse_from_npz(artifact))

    expected_sources = len(station_dirs)
    for seed in seeds:
        values = by_seed[int(seed)]
        complete = expected_sources > 0 and len(values) == expected_sources
        records.append(
            {
                "seed": int(seed),
                "st_rmse_ugm3": float(np.mean(values)) if complete else np.nan,
                "st_status": "available" if complete else "missing",
                "st_source_count": len(values),
            }
        )
    return pd.DataFrame(records)


def build_best_of_baselines_tables(
    rows: pd.DataFrame,
    beijing_st_root: str | Path,
    guangzhou_st_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build R8 best-of-baselines and R6 paired comparison tables."""
    beijing_st_root = Path(beijing_st_root)
    guangzhou_st_root = Path(guangzhou_st_root)
    working = rows.copy()
    if "arm" not in working:
        working["arm"] = working["variant"]
    working["arm"] = working["arm"].fillna(working["variant"])
    statuses = working.get("status", pd.Series("completed", index=working.index))
    working = working[
        working["arm"].isin(SAME_INFORMATION_ARMS)
        & (statuses.fillna("completed") == "completed")
        & np.isfinite(pd.to_numeric(working["rmse_ugm3"], errors="coerce"))
    ].copy()
    working["numeric_seed"] = pd.to_numeric(working["seed"], errors="coerce")

    detail_records = []
    paired_records = []
    summary_records = []
    config_columns = ["city", "history", "horizon"]
    for config_key, group in working.groupby(config_columns, dropna=False, sort=True):
        city, history, horizon = config_key
        seeds = sorted(group["numeric_seed"].dropna().astype(int).unique().tolist())
        if not seeds:
            continue
        stochastic = group[group["numeric_seed"].notna()].copy()
        for (candidate_arm, seed), candidate in stochastic.groupby(
            ["variant", "numeric_seed"], sort=True
        ):
            detail_records.append(
                {
                    "city": city,
                    "history": int(history),
                    "horizon": int(horizon),
                    "candidate_arm": candidate_arm,
                    "base_arm": candidate["arm"].iloc[0],
                    "seed": int(seed),
                    "rmse_ugm3": float(candidate["rmse_ugm3"].mean()),
                    "source_count": len(candidate),
                }
            )
        deterministic = group[group["numeric_seed"].isna()]
        for candidate_arm, candidate in deterministic.groupby("variant", sort=True):
            value = float(candidate["rmse_ugm3"].mean())
            for seed in seeds:
                detail_records.append(
                    {
                        "city": city,
                        "history": int(history),
                        "horizon": int(horizon),
                        "candidate_arm": candidate_arm,
                        "base_arm": candidate["arm"].iloc[0],
                        "seed": int(seed),
                        "rmse_ugm3": value,
                        "source_count": len(candidate),
                    }
                )

        config_detail = pd.DataFrame(
            [
                record
                for record in detail_records
                if record["city"] == city
                and record["history"] == int(history)
                and record["horizon"] == int(horizon)
            ]
        )
        arm_means = config_detail.groupby("candidate_arm")["rmse_ugm3"].mean()
        winner_arm = str(arm_means.idxmin())
        winner_mean = float(arm_means.min())
        st_rows = load_st_seed_metrics(
            str(city), int(history), int(horizon), seeds, beijing_st_root, guangzhou_st_root
        ).set_index("seed")
        config_pairs = []
        for seed in seeds:
            seed_rows = config_detail[config_detail["seed"] == seed]
            best_index = seed_rows["rmse_ugm3"].idxmin()
            best = seed_rows.loc[best_index]
            st = st_rows.loc[seed]
            st_available = st["st_status"] == "available"
            st_rmse = float(st["st_rmse_ugm3"]) if st_available else np.nan
            signed_difference = st_rmse - float(best["rmse_ugm3"]) if st_available else np.nan
            relative_change = (
                100 * signed_difference / float(best["rmse_ugm3"])
                if st_available
                else np.nan
            )
            record = {
                "city": city,
                "history": int(history),
                "horizon": int(horizon),
                "seed": int(seed),
                "winner_arm": winner_arm,
                "winner_arm_mean_rmse_ugm3": winner_mean,
                "best_seed_arm": best["candidate_arm"],
                "best_baseline_rmse_ugm3": float(best["rmse_ugm3"]),
                "st_rmse_ugm3": st_rmse,
                "st_status": st["st_status"],
                "st_source_count": int(st["st_source_count"]),
                "st_minus_best_baseline_rmse_ugm3": signed_difference,
                "absolute_paired_difference_ugm3": (
                    abs(signed_difference) if st_available else np.nan
                ),
                "st_relative_change_vs_best_percent": relative_change,
                "st_better_than_best": bool(st_rmse < best["rmse_ugm3"])
                if st_available
                else pd.NA,
            }
            paired_records.append(record)
            config_pairs.append(record)

        available_pairs = [row for row in config_pairs if row["st_status"] == "available"]
        better_count = sum(bool(row["st_better_than_best"]) for row in available_pairs)
        direction_count = f"{better_count}/{len(available_pairs)}"
        for record in config_pairs:
            record["st_better_direction_count"] = direction_count
        summary_records.append(
            {
                "city": city,
                "history": int(history),
                "horizon": int(horizon),
                "winner_arm": winner_arm,
                "winner_arm_mean_rmse_ugm3": winner_mean,
                "best_of_baselines_mean_rmse_ugm3": float(
                    np.mean([row["best_baseline_rmse_ugm3"] for row in config_pairs])
                ),
                "st_mean_rmse_ugm3": (
                    float(np.mean([row["st_rmse_ugm3"] for row in available_pairs]))
                    if available_pairs
                    else np.nan
                ),
                "mean_st_minus_best_rmse_ugm3": (
                    float(
                        np.mean(
                            [row["st_minus_best_baseline_rmse_ugm3"] for row in available_pairs]
                        )
                    )
                    if available_pairs
                    else np.nan
                ),
                "mean_st_relative_change_vs_best_percent": (
                    float(
                        np.mean(
                            [row["st_relative_change_vs_best_percent"] for row in available_pairs]
                        )
                    )
                    if available_pairs
                    else np.nan
                ),
                "st_better_direction_count": direction_count,
                "st_pairing_status": "available" if len(available_pairs) == len(seeds) else "missing",
            }
        )

    details = pd.DataFrame(detail_records)
    if len(details):
        details["arm_mean_rmse_ugm3"] = details.groupby(
            ["city", "history", "horizon", "candidate_arm"]
        )["rmse_ugm3"].transform("mean")
    return details, pd.DataFrame(paired_records), pd.DataFrame(summary_records)


def write_best_of_markdown(summary: pd.DataFrame, path: Path) -> None:
    columns = [
        "city",
        "history",
        "horizon",
        "winner_arm",
        "winner_arm_mean_rmse_ugm3",
        "best_of_baselines_mean_rmse_ugm3",
        "st_mean_rmse_ugm3",
        "mean_st_minus_best_rmse_ugm3",
        "mean_st_relative_change_vs_best_percent",
        "st_better_direction_count",
        "st_pairing_status",
    ]
    if summary.empty:
        path.write_text("# B3 Best-of-baselines 与 ST 配对\n\n无可用配置。\n", encoding="utf-8")
        return
    display = summary[columns].copy()
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    body = []
    for _, row in display.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                values.append("missing")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.6f}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    path.write_text(
        "# B3 Best-of-baselines 与 ST 配对\n\n"
        "负的 `ST−best` 与相对变化表示 ST 的 RMSE 更低。缺失 ST 产物显式标为 `missing`。\n\n"
        + "\n".join([header, separator, *body])
        + "\n",
        encoding="utf-8",
    )


def write_tables_and_figures(
    rows: pd.DataFrame,
    comparisons: pd.DataFrame,
    compliance: pd.DataFrame,
    tables_dir: Path,
    figures_dir: Path,
    beijing_st_root: Path,
    guangzhou_st_root: Path,
) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(tables_dir / "B1_baseline_main.csv", index=False)
    neural = rows[rows["layer"].isin(["B", "C"])].copy()
    neural.to_csv(tables_dir / "B2_capacity_comparison.csv", index=False)
    arm_details, paired, paired_summary = build_best_of_baselines_tables(
        rows, beijing_st_root, guangzhou_st_root
    )
    arm_details.to_csv(tables_dir / "B3_baseline_arm_details.csv", index=False)
    paired.to_csv(tables_dir / "B3_best_of_baselines_paired.csv", index=False)
    paired_summary.to_csv(tables_dir / "B3_best_of_baselines_summary.csv", index=False)
    write_best_of_markdown(paired_summary, tables_dir / "B3_best_of_baselines.md")
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
    parser.add_argument(
        "--beijing-st-root",
        default="experiments/results/beijing_leakfree_coverage",
    )
    parser.add_argument(
        "--guangzhou-st-root",
        default="experiments/results/guangzhou_horizon_coverage",
    )
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
        Path(args.beijing_st_root),
        Path(args.guangzhou_st_root),
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
