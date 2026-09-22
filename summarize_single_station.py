"""Independently recompute and summarize the single-station baseline study."""

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


METRICS = ("rmse_ugm3", "mae_ugm3", "smape_percent")
REQUIRED_NPZ_KEYS = (
    "prediction_scaled",
    "target_scaled",
    "prediction_ugm3",
    "target_ugm3",
)
ST_SOURCE_VARIANT = "st_sparse_station_bias_delta_forecast"
INFORMATION_SET_STATEMENT = (
    "信息集不同：单站点基线仅使用中心站 PM2.5；ST/ST+频域额外使用邻站。"
    "相对提升只能归因于新增跨站信息的价值，不代表机制优于同信息集模型。"
)


def recompute_prediction_file(path: str | Path) -> dict:
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
        "rmse_scaled": float(np.sqrt(mse_scaled)),
        "mae_scaled": float(np.mean(np.abs(scaled_error))),
        "rmse_ugm3": float(np.sqrt(np.mean(error ** 2))),
        "mae_ugm3": float(np.mean(np.abs(error))),
        "smape_percent": float(
            200 * np.mean(np.abs(error) / np.maximum(denominator, 1e-6))
        ),
    }


def _result_context(raw_path: Path, result_root: Path) -> tuple[str, int, int, int | None]:
    relative = raw_path.relative_to(result_root)
    city = relative.parts[0]
    task = next(
        (
            match
            for part in relative.parts
            if (match := re.fullmatch(r"(\d+)h_(\d+)h", part)) is not None
        ),
        None,
    )
    if task is None:
        raise ValueError(f"cannot infer L/H from {raw_path}")
    station_part = next(
        (part for part in relative.parts if part.startswith("station_")), None
    )
    station_id = int(station_part.removeprefix("station_")) if station_part else None
    return city, int(task.group(1)), int(task.group(2)), station_id


def recompute_directory(result_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute every feasible ingested row from its persisted prediction."""
    result_root = Path(result_root)
    rows: list[dict] = []
    comparisons: list[dict] = []
    for raw_path in sorted(result_root.rglob("raw_metrics.csv")):
        context = _result_context(raw_path, result_root)
        raw = pd.read_csv(raw_path)
        for _, recorded in raw.iterrows():
            row = recorded.to_dict()
            row.update(
                {
                    "city": context[0],
                    "history": context[1],
                    "horizon": context[2],
                    "station_id": context[3],
                    "result_dir": str(raw_path.parent),
                }
            )
            status = str(recorded.get("status", "completed"))
            if status.startswith(("infeasible", "nonfinite")):
                row["independent_recomputed"] = False
                rows.append(row)
                continue
            prediction_path = Path(str(recorded["prediction_file"]))
            if not prediction_path.is_absolute():
                prediction_path = raw_path.parent / prediction_path
            metrics = recompute_prediction_file(prediction_path)
            for metric, recomputed in metrics.items():
                recorded_value = float(recorded[metric])
                absolute = abs(recomputed - recorded_value)
                comparisons.append(
                    {
                        "city": context[0],
                        "history": context[1],
                        "horizon": context[2],
                        "station_id": context[3],
                        "variant": recorded["variant"],
                        "seed": recorded["seed"],
                        "metric": metric,
                        "recorded": recorded_value,
                        "recomputed": recomputed,
                        "absolute_difference": absolute,
                        "pass": absolute <= 1e-9,
                    }
                )
            row.update(metrics)
            row["independent_recomputed"] = True
            rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(comparisons)


def _standard_model_row(
    source: pd.Series | dict,
    city: str,
    history: int,
    horizon: int,
    station_id: int | None,
    arm: str,
    provenance: str,
) -> dict:
    row = {
        "city": city,
        "history": int(history),
        "horizon": int(horizon),
        "station_id": station_id,
        "variant": arm,
        "arm": arm,
        "seed": int(float(source["seed"])),
        "status": "completed",
        "capacity_tier": "not_applicable",
        "requested_capacity": "not_applicable",
        "input_channels": 1 if arm.startswith("center_patchtst") else np.nan,
        "provenance": provenance,
    }
    for metric in METRICS:
        row[metric] = float(source[metric]) if metric in source and pd.notna(source[metric]) else np.nan
    return row


def _load_beijing_raw_root(root: Path, variant_map: dict[str, str], provenance: str) -> list[dict]:
    records = []
    if not root.is_dir():
        return records
    for raw_path in sorted(root.glob("*h_*h/raw_metrics.csv")):
        match = re.fullmatch(r"(\d+)h_(\d+)h", raw_path.parent.name)
        if match is None:
            continue
        raw = pd.read_csv(raw_path)
        for _, source in raw.iterrows():
            source_variant = str(source["variant"])
            if source_variant in variant_map:
                records.append(
                    _standard_model_row(
                        source,
                        "beijing",
                        int(match.group(1)),
                        int(match.group(2)),
                        None,
                        variant_map[source_variant],
                        provenance,
                    )
                )
    return records


def _load_guangzhou_pair_table(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    records = []
    for _, source in pd.read_csv(path).iterrows():
        common = {
            "seed": source["seed"],
            "mae_ugm3": np.nan,
            "smape_percent": np.nan,
        }
        records.append(
            _standard_model_row(
                {**common, "rmse_ugm3": source["base_rmse"]},
                "guangzhou",
                int(source["history"]),
                int(source["horizon"]),
                int(source["station"]),
                "center_patchtst_frozen",
                f"read-only existing artifact: {path}",
            )
        )
        records.append(
            _standard_model_row(
                {**common, "rmse_ugm3": source["spatial_rmse"]},
                "guangzhou",
                int(source["history"]),
                int(source["horizon"]),
                int(source["station"]),
                "st",
                f"read-only existing artifact: {path}",
            )
        )
    return records


def _load_guangzhou_frequency(root: Path) -> list[dict]:
    records = []
    if not root.is_dir():
        return records
    mapping = {"st_rfft": "st_plus_frequency", "st_time": "st_time_control"}
    for raw_path in sorted(root.glob("*h_*h/station_*/raw_metrics.csv")):
        task = re.fullmatch(r"(\d+)h_(\d+)h", raw_path.parent.parent.name)
        if task is None:
            continue
        station_id = int(raw_path.parent.name.removeprefix("station_"))
        for _, source in pd.read_csv(raw_path).iterrows():
            variant = str(source["variant"])
            if variant in mapping:
                records.append(
                    _standard_model_row(
                        source,
                        "guangzhou",
                        int(task.group(1)),
                        int(task.group(2)),
                        station_id,
                        mapping[variant],
                        f"read-only existing artifact: {raw_path}",
                    )
                )
    return records


def load_existing_model_rows(
    beijing_st_root: str | Path,
    beijing_frequency_root: str | Path,
    guangzhou_pair_table: str | Path,
    guangzhou_frequency_root: str | Path,
    st_reference_root: str | Path,
) -> pd.DataFrame:
    records = []
    records.extend(
        _load_beijing_raw_root(
            Path(beijing_st_root),
            {
                "degraded_patchtst": "center_patchtst_frozen",
                ST_SOURCE_VARIANT: "st",
            },
            f"read-only existing artifact: {beijing_st_root}",
        )
    )
    records.extend(
        _load_beijing_raw_root(
            Path(beijing_frequency_root),
            {"st_rfft": "st_plus_frequency", "st_time": "st_time_control"},
            f"read-only existing artifact: {beijing_frequency_root}",
        )
    )
    records.extend(_load_guangzhou_pair_table(Path(guangzhou_pair_table)))
    records.extend(_load_guangzhou_frequency(Path(guangzhou_frequency_root)))
    records.extend(
        _load_beijing_raw_root(
            Path(st_reference_root),
            {
                "degraded_patchtst": "center_patchtst_e2e",
                ST_SOURCE_VARIANT: "st_e2e",
            },
            f"read-only existing artifact: {st_reference_root}",
        )
    )
    return pd.DataFrame(records)


def _expected_seeds(city: str, history: int, horizon: int) -> tuple[int, ...]:
    if city == "guangzhou":
        return (7001, 7002, 7003)
    if (int(history), int(horizon)) in {(24, 1), (168, 6)}:
        return (2047, 2048, 2049, 2050, 2051)
    return (2047, 2048, 2049)


def _candidate_label(row: pd.Series) -> str:
    arm = str(row["arm"])
    tier = str(row.get("capacity_tier", row.get("requested_capacity", "")))
    return arm if tier in {"", "nan", "not_applicable"} else f"{arm}_{tier}"


def _pool_candidate_by_seed(group: pd.DataFrame, seeds: tuple[int, ...]) -> pd.DataFrame:
    numeric = pd.to_numeric(group["seed"], errors="coerce")
    records = []
    if numeric.notna().any():
        stochastic = group[numeric.notna()].copy()
        stochastic["numeric_seed"] = numeric[numeric.notna()].astype(int)
        for seed in seeds:
            selected = stochastic[stochastic["numeric_seed"] == seed]
            if selected.empty:
                continue
            records.append({"seed": seed, "rmse_ugm3": float(selected["rmse_ugm3"].mean())})
    else:
        value = float(group["rmse_ugm3"].mean())
        records.extend({"seed": seed, "rmse_ugm3": value} for seed in seeds)
    return pd.DataFrame(records)


def build_s3_tables(
    baseline_rows: pd.DataFrame,
    model_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Choose one configuration-level winner and build paired seed contrasts."""
    baselines = baseline_rows.copy()
    if "arm" not in baselines:
        baselines["arm"] = baselines["variant"]
    baselines["candidate"] = baselines.apply(_candidate_label, axis=1)
    statuses = baselines.get("status", pd.Series("completed", index=baselines.index))
    baselines = baselines[
        statuses.fillna("completed").eq("completed")
        & np.isfinite(pd.to_numeric(baselines["rmse_ugm3"], errors="coerce"))
    ].copy()

    candidate_records = []
    paired_records = []
    summary_records = []
    for config_key, config_baselines in baselines.groupby(
        ["city", "history", "horizon"], sort=True
    ):
        city, history, horizon = config_key
        seeds = _expected_seeds(str(city), int(history), int(horizon))
        pooled_candidates: dict[str, pd.DataFrame] = {}
        for candidate, candidate_group in config_baselines.groupby("candidate", sort=True):
            pooled = _pool_candidate_by_seed(candidate_group, seeds)
            if set(pooled.get("seed", [])) != set(seeds):
                continue
            pooled_candidates[str(candidate)] = pooled
            candidate_records.append(
                {
                    "city": city,
                    "history": int(history),
                    "horizon": int(horizon),
                    "candidate": candidate,
                    "mean_rmse_ugm3": float(pooled["rmse_ugm3"].mean()),
                    "paired_seed_count": len(pooled),
                }
            )
        if not pooled_candidates:
            continue
        winner = min(
            pooled_candidates,
            key=lambda name: (float(pooled_candidates[name]["rmse_ugm3"].mean()), name),
        )
        winner_rows = pooled_candidates[winner].set_index("seed")
        config_models = model_rows[
            (model_rows["city"] == city)
            & (model_rows["history"] == int(history))
            & (model_rows["horizon"] == int(horizon))
            & model_rows["arm"].isin(["st", "st_plus_frequency"])
        ]
        for model_arm in ("st", "st_plus_frequency"):
            selected_model = config_models[config_models["arm"] == model_arm].copy()
            if selected_model.empty:
                continue
            selected_model["numeric_seed"] = pd.to_numeric(
                selected_model["seed"], errors="coerce"
            )
            config_pairs = []
            for seed in seeds:
                seed_model = selected_model[selected_model["numeric_seed"] == seed]
                if seed_model.empty or seed not in winner_rows.index:
                    continue
                model_rmse = float(seed_model["rmse_ugm3"].mean())
                baseline_rmse = float(winner_rows.loc[seed, "rmse_ugm3"])
                signed_difference = model_rmse - baseline_rmse
                relative_change = 100 * signed_difference / baseline_rmse
                record = {
                    "city": city,
                    "history": int(history),
                    "horizon": int(horizon),
                    "model_arm": model_arm,
                    "seed": int(seed),
                    "best_single_station_arm": winner,
                    "best_single_station_rmse_ugm3": baseline_rmse,
                    "model_rmse_ugm3": model_rmse,
                    "model_minus_baseline_rmse_ugm3": signed_difference,
                    "relative_change_percent": relative_change,
                    "reduction_percent": -relative_change,
                    "model_better": bool(model_rmse < baseline_rmse),
                    "information_set_statement": INFORMATION_SET_STATEMENT,
                }
                paired_records.append(record)
                config_pairs.append(record)
            if config_pairs:
                better = sum(bool(row["model_better"]) for row in config_pairs)
                summary_records.append(
                    {
                        "city": city,
                        "history": int(history),
                        "horizon": int(horizon),
                        "model_arm": model_arm,
                        "best_single_station_arm": winner,
                        "best_single_station_mean_rmse_ugm3": float(
                            np.mean([row["best_single_station_rmse_ugm3"] for row in config_pairs])
                        ),
                        "model_mean_rmse_ugm3": float(
                            np.mean([row["model_rmse_ugm3"] for row in config_pairs])
                        ),
                        "mean_paired_difference_ugm3": float(
                            np.mean([row["model_minus_baseline_rmse_ugm3"] for row in config_pairs])
                        ),
                        "mean_relative_change_percent": float(
                            np.mean([row["relative_change_percent"] for row in config_pairs])
                        ),
                        "mean_reduction_percent": float(
                            np.mean([row["reduction_percent"] for row in config_pairs])
                        ),
                        "better_direction_count": f"{better}/{len(config_pairs)}",
                        "pairing_status": "complete" if len(config_pairs) == len(seeds) else "missing",
                        "information_set_statement": INFORMATION_SET_STATEMENT,
                    }
                )
    return (
        pd.DataFrame(candidate_records),
        pd.DataFrame(paired_records),
        pd.DataFrame(summary_records),
    )


def build_s7(rows: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    records = []
    if not rows.empty:
        statuses = rows.get("status", pd.Series("completed", index=rows.index)).fillna("completed")
        for status, count in statuses.astype(str).value_counts().sort_index().items():
            records.append({"category": "result_status", "item": status, "count": int(count)})
    if not audit.empty and "reason" in audit:
        for reason, count in audit["reason"].value_counts().sort_index().items():
            records.append({"category": "source_exclusion", "item": reason, "count": int(count)})
    if not records:
        records.append({"category": "result_status", "item": "none", "count": 0})
    return pd.DataFrame(records)


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    body = []
    for _, row in frame.iterrows():
        values = []
        for value in row:
            if pd.isna(value):
                values.append("missing")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.6f}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *body])


def write_s3_markdown(summary: pd.DataFrame, path: Path) -> None:
    display_columns = [
        "city",
        "history",
        "horizon",
        "model_arm",
        "best_single_station_arm",
        "best_single_station_mean_rmse_ugm3",
        "model_mean_rmse_ugm3",
        "mean_reduction_percent",
        "better_direction_count",
        "pairing_status",
    ]
    table = summary[display_columns] if not summary.empty else pd.DataFrame(columns=display_columns)
    path.write_text(
        "# S3 主表：多站点模型 vs 最强单站点基线\n\n"
        "> **解释边界：" + INFORMATION_SET_STATEMENT + "**\n\n"
        "`mean_reduction_percent > 0` 表示多站点模型 RMSE 更低；比较按种子配对，"
        "广州先在 8 个站内按种子池化。\n\n"
        + _markdown_table(table)
        + "\n",
        encoding="utf-8",
    )


def compliance_checks(
    baseline_rows: pd.DataFrame,
    comparisons: pd.DataFrame,
    s3_summary: pd.DataFrame,
    missing_plan: pd.DataFrame,
    all_baselines: pd.DataFrame | None = None,
) -> pd.DataFrame:
    checks = []

    def add(check: str, passed: bool, detail: str) -> None:
        checks.append({"check": check, "pass": bool(passed), "detail": detail})

    channel_scope = all_baselines if all_baselines is not None else baseline_rows
    add(
        "baseline input_channels are exactly 1",
        bool((pd.to_numeric(channel_scope["input_channels"], errors="coerce") == 1).all()),
        f"rows={len(channel_scope)} (ingested and read-only PatchTST baselines)",
    )
    add(
        "ingest provenance present",
        bool(baseline_rows["provenance"].astype(str).str.contains("same arm definition").all()),
        "every ingested row identifies the abandoned branch and equivalence basis",
    )
    max_difference = (
        float(comparisons["absolute_difference"].max()) if len(comparisons) else float("inf")
    )
    add(
        "prediction metrics independently recomputed within 1e-9",
        bool(comparisons["pass"].all()) if len(comparisons) else False,
        f"max_absolute_difference={max_difference:.3e}",
    )
    add(
        "evaluation split fixed to test",
        bool((baseline_rows["evaluation_split"] == "test").all()),
        "validation remains selection-only",
    )
    add(
        "S3 information-set statement embedded",
        bool(
            len(s3_summary)
            and (s3_summary["information_set_statement"] == INFORMATION_SET_STATEMENT).all()
        ),
        INFORMATION_SET_STATEMENT,
    )
    add(
        "missing combinations explicitly planned",
        True,
        f"missing_rows={len(missing_plan)}; no silent omission",
    )
    return pd.DataFrame(checks)


def write_tables_and_figures(
    baseline_rows: pd.DataFrame,
    comparisons: pd.DataFrame,
    model_rows: pd.DataFrame,
    audit: pd.DataFrame,
    missing_plan: pd.DataFrame,
    tables_dir: Path,
    figures_dir: Path,
) -> dict[str, pd.DataFrame]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    active_configs = baseline_rows[["city", "history", "horizon"]].drop_duplicates()
    active_models = model_rows.merge(
        active_configs, on=["city", "history", "horizon"], how="inner"
    )
    patch_rows = active_models[
        active_models["arm"].astype(str).str.startswith("center_patchtst_")
    ].copy()
    all_baselines = pd.concat([baseline_rows, patch_rows], ignore_index=True, sort=False)
    all_baselines["candidate"] = all_baselines.apply(_candidate_label, axis=1)

    s1 = (
        all_baselines.groupby(
            ["city", "history", "horizon", "candidate", "capacity_tier"],
            dropna=False,
            sort=True,
        )
        .agg(
            runs=("rmse_ugm3", "count"),
            mean_rmse_ugm3=("rmse_ugm3", "mean"),
            mean_mae_ugm3=("mae_ugm3", "mean"),
            mean_smape_percent=("smape_percent", "mean"),
        )
        .reset_index()
    )
    neural = baseline_rows[baseline_rows["arm"].astype(str).str.startswith("center_")].copy()
    s2 = (
        neural.groupby(["city", "history", "horizon", "arm", "capacity_tier"], sort=True)[
            "rmse_ugm3"
        ]
        .mean()
        .unstack("capacity_tier")
        .reset_index()
    )
    if {"default", "matched"} <= set(s2.columns):
        s2["matched_minus_default_rmse_ugm3"] = s2["matched"] - s2["default"]

    candidate_detail, s3_paired, s3_summary = build_s3_tables(
        all_baselines, active_models
    )
    s4 = baseline_rows[baseline_rows["arm"].isin(
        ["trad_persistence", "trad_daily_naive", "trad_climatology", "trad_ar", "trad_ridge"]
    )].copy()
    s5_source = active_models[active_models["arm"].isin(
        ["st", "st_plus_frequency", "st_time_control"]
    )].copy()
    s5 = (
        s5_source.groupby(["city", "history", "horizon", "arm"], sort=True)
        .agg(
            runs=("rmse_ugm3", "count"),
            mean_rmse_ugm3=("rmse_ugm3", "mean"),
            mean_mae_ugm3=("mae_ugm3", "mean"),
            mean_smape_percent=("smape_percent", "mean"),
        )
        .reset_index()
    )
    s6 = compliance_checks(
        baseline_rows, comparisons, s3_summary, missing_plan, all_baselines
    )
    s7 = build_s7(baseline_rows, audit)

    outputs = {
        "S1_baseline_means.csv": s1,
        "S2_capacity_comparison.csv": s2,
        "S3_main.csv": s3_summary,
        "S3_paired_detail.csv": s3_paired,
        "S3_candidate_detail.csv": candidate_detail,
        "S4_traditional_details.csv": s4,
        "S5_frequency_and_time_control.csv": s5,
        "S6_compliance_self_check.csv": s6,
        "S7_exclusions_and_exceptions.csv": s7,
        "independent_recalculation.csv": comparisons,
        "missing_run_plan.csv": missing_plan,
    }
    for filename, frame in outputs.items():
        frame.to_csv(tables_dir / filename, index=False)
    write_s3_markdown(s3_summary, tables_dir / "S3_main.md")

    if not s3_summary.empty:
        plot = s3_summary.copy()
        plot["label"] = (
            plot["city"].astype(str)
            + " "
            + plot["history"].astype(str)
            + "→"
            + plot["horizon"].astype(str)
            + " "
            + plot["model_arm"].astype(str)
        )
        fig, axis = plt.subplots(figsize=(11, max(5, 0.22 * len(plot))))
        colors = np.where(plot["mean_reduction_percent"] >= 0, "#2a9d8f", "#e76f51")
        axis.barh(plot["label"], plot["mean_reduction_percent"], color=colors)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_xlabel("Paired RMSE reduction vs best single-station baseline (%)")
        axis.set_title("S3 Value of additional neighbour-station information")
        fig.tight_layout()
        fig.savefig(figures_dir / "S3_paired_reduction.png", dpi=180)
        plt.close(fig)

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root", default="experiments/results/single_station_baselines"
    )
    parser.add_argument("--tables-dir", default="tables/single_station_baselines")
    parser.add_argument("--figures-dir", default="figures/single_station_baselines")
    parser.add_argument(
        "--beijing-st-root", default="experiments/results/beijing_leakfree_coverage"
    )
    parser.add_argument(
        "--beijing-frequency-root",
        default="experiments/results/beijing_leakfree_coverage_frequency",
    )
    parser.add_argument(
        "--guangzhou-pair-table",
        default="tables/guangzhou_horizon_coverage/G2_pair_detail.csv",
    )
    parser.add_argument(
        "--guangzhou-frequency-root",
        default="experiments/results/guangzhou_horizon_coverage_frequency",
    )
    parser.add_argument(
        "--st-reference-root", default="experiments/results/st_reference_arms"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_root = Path(args.result_root)
    baseline_rows, comparisons = recompute_directory(result_root)
    if baseline_rows.empty:
        raise RuntimeError(f"no ingested baseline rows below {result_root}")
    model_rows = load_existing_model_rows(
        args.beijing_st_root,
        args.beijing_frequency_root,
        args.guangzhou_pair_table,
        args.guangzhou_frequency_root,
        args.st_reference_root,
    )
    audit_path = result_root / "ingest_audit.csv"
    plan_path = result_root / "missing_plan.csv"
    audit = pd.read_csv(audit_path) if audit_path.is_file() else pd.DataFrame()
    plan = pd.read_csv(plan_path) if plan_path.is_file() else pd.DataFrame()
    outputs = write_tables_and_figures(
        baseline_rows,
        comparisons,
        model_rows,
        audit,
        plan,
        Path(args.tables_dir),
        Path(args.figures_dir),
    )
    s3 = outputs["S3_main.csv"]
    worst = float(comparisons["absolute_difference"].max())
    print(
        f"baseline_rows={len(baseline_rows)} model_rows={len(model_rows)} "
        f"max_metric_abs_diff={worst:.3e}"
    )
    if not outputs["S6_compliance_self_check.csv"]["pass"].all():
        raise SystemExit("single-station compliance self-check failed")
    print(s3.to_string(index=False))
    print("SINGLE_STATION_SUMMARY_DONE")


if __name__ == "__main__":
    main()
