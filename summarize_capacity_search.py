"""Summarize the preregistered P1 spatial-branch capacity search."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_capacity_search import (
    BACKBONE_SELECTION,
    CAPACITY_CANDIDATES,
    CENTER_STATION_ID,
    GRID_SEEDS,
    GRID_TASKS,
    HEADLINE_SEEDS,
    HEADLINE_TASKS,
    OUTPUT_ROOT,
    select_capacity_by_validation,
)


S3_MAIN = Path(
    "/home/hansel/.herdr/worktrees/PatchTST/experiment-baseline-single-station/"
    "tables/single_station_baselines/S3_main.csv"
)
TABLE_ROOT = Path("tables/capacity_search")
GROUP_COLUMNS = ["city", "history", "horizon", "station_id", "seed"]


def nonfinite_status_checks(raw: pd.DataFrame) -> list[dict[str, Any]]:
    """Expose exceptional terminal counts without admitting them to pairings."""
    return [
        {
            "check": f"{status}_count",
            "pass": True,
            "detail": f"count={int((raw['status'] == status).sum())}",
        }
        for status in ("backbone_nonfinite", "no_usable_backbone")
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--table-root", default=str(TABLE_ROOT))
    parser.add_argument("--backbone-selection", default=str(BACKBONE_SELECTION))
    parser.add_argument("--s3-main", default=str(S3_MAIN))
    return parser.parse_args()


def _paired_row(
    comparison: str,
    candidate_rmse: float,
    reference_rmse: float,
    **keys: Any,
) -> dict[str, Any]:
    return {
        **keys,
        "comparison": comparison,
        "candidate_rmse_ugm3": candidate_rmse,
        "reference_rmse_ugm3": reference_rmse,
        "difference_ugm3": candidate_rmse - reference_rmse,
        "reduction_percent": 100 * (reference_rmse - candidate_rmse) / reference_rmse,
        "candidate_better": candidate_rmse < reference_rmse,
    }


def build_c0(raw: pd.DataFrame) -> pd.DataFrame:
    complete = raw[raw["status"] == "completed"].copy()
    baseline = complete[complete["capacity_candidate"] == "cap32_b4_a10"][
        GROUP_COLUMNS + ["rmse_ugm3"]
    ].rename(columns={"rmse_ugm3": "capacity_baseline_rmse_ugm3"})
    merged = complete.merge(baseline, on=GROUP_COLUMNS, how="left", validate="many_to_one")
    rows = []
    for _, row in merged.dropna(subset=["capacity_baseline_rmse_ugm3"]).iterrows():
        rows.append(
            _paired_row(
                "C0_candidate_vs_cap32_b4_a10",
                float(row["rmse_ugm3"]),
                float(row["capacity_baseline_rmse_ugm3"]),
                **{key: row[key] for key in GROUP_COLUMNS},
                capacity_candidate=row["capacity_candidate"],
            )
        )
    return pd.DataFrame(rows)


def build_c1(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in raw[raw["status"] == "completed"].iterrows():
        rows.append(
            _paired_row(
                "C1_candidate_vs_own_backbone",
                float(row["rmse_ugm3"]),
                float(row["backbone_rmse_ugm3"]),
                **{key: row[key] for key in GROUP_COLUMNS},
                capacity_candidate=row["capacity_candidate"],
                selected_backbone=row["selected_variant"],
            )
        )
    return pd.DataFrame(rows)


def load_s3_references(s3_main_path: str | Path) -> tuple[pd.DataFrame, Path]:
    main_path = Path(s3_main_path)
    detail_path = main_path.with_name("S3_paired_detail.csv")
    if not main_path.is_file() or not detail_path.is_file():
        raise FileNotFoundError(f"缺少 S3 主表或逐种子表: {main_path}, {detail_path}")
    main = pd.read_csv(main_path)
    choices = main[(main["city"] == "beijing") & (main["model_arm"] == "st")][
        ["city", "history", "horizon", "best_single_station_arm"]
    ]
    detail = pd.read_csv(detail_path)
    detail = detail[(detail["city"] == "beijing") & (detail["model_arm"] == "st")]
    merged = detail.merge(
        choices,
        on=["city", "history", "horizon"],
        how="inner",
        suffixes=("_detail", "_main"),
        validate="many_to_one",
    )
    if not (
        merged["best_single_station_arm_detail"]
        == merged["best_single_station_arm_main"]
    ).all():
        raise RuntimeError("S3_main 与 S3_paired_detail 的冻结单站点口径不一致")
    return merged, detail_path


def build_c2(raw: pd.DataFrame, s3: pd.DataFrame) -> pd.DataFrame:
    complete = raw[raw["status"] == "completed"].copy()
    references = s3[
        [
            "city", "history", "horizon", "seed",
            "best_single_station_arm_main", "best_single_station_rmse_ugm3",
        ]
    ]
    merged = complete.merge(
        references,
        on=["city", "history", "horizon", "seed"],
        how="left",
        validate="many_to_one",
    )
    rows = []
    for _, row in merged.dropna(subset=["best_single_station_rmse_ugm3"]).iterrows():
        rows.append(
            _paired_row(
                "C2_candidate_vs_strongest_single_station",
                float(row["rmse_ugm3"]),
                float(row["best_single_station_rmse_ugm3"]),
                **{key: row[key] for key in GROUP_COLUMNS},
                capacity_candidate=row["capacity_candidate"],
                reference_arm=row["best_single_station_arm_main"],
            )
        )
    return pd.DataFrame(rows)


def build_capacity_selection(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in raw.groupby(GROUP_COLUMNS, sort=True, dropna=False):
        rows.extend(select_capacity_by_validation(group.to_dict("records")))
    columns = GROUP_COLUMNS + [
        "capacity_candidate", "status", "best_valid_loss", "selected",
        "rmse_ugm3", "backbone_rmse_ugm3",
    ]
    selection = pd.DataFrame(rows)
    return selection[[column for column in columns if column in selection.columns]]


def expansion_gate(selection: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    completed = selection[selection["status"] == "completed"].copy()
    baseline = completed[completed["capacity_candidate"] == "cap32_b4_a10"][
        GROUP_COLUMNS + ["best_valid_loss"]
    ].rename(columns={"best_valid_loss": "baseline_best_valid_loss"})
    selected = completed[completed["selected"].astype(bool)][
        GROUP_COLUMNS + ["capacity_candidate", "best_valid_loss"]
    ].rename(
        columns={
            "capacity_candidate": "selected_capacity_candidate",
            "best_valid_loss": "selected_best_valid_loss",
        }
    )
    detail = baseline.merge(selected, on=GROUP_COLUMNS, how="inner", validate="one_to_one")
    detail["validation_reduction_percent"] = 100 * (
        detail["baseline_best_valid_loss"] - detail["selected_best_valid_loss"]
    ) / detail["baseline_best_valid_loss"]
    same_direction = int((detail["validation_reduction_percent"] > 0).sum())
    mean_reduction = (
        float(detail["validation_reduction_percent"].mean()) if len(detail) else math.nan
    )
    eligible = len(detail) > 0 and same_direction >= 1 and mean_reduction > 0
    gate = {
        "status": "worth_requesting_full_grid" if eligible else "stop_no_expansion",
        "eligible_group_count": len(detail),
        "strictly_better_group_count": same_direction,
        "mean_validation_reduction_percent": mean_reduction,
        "criterion_met": bool(eligible),
    }
    return gate, detail


def _comparison_summary(frame: pd.DataFrame, candidate: str) -> dict[str, Any]:
    selected = frame[frame["capacity_candidate"] == candidate]
    if selected.empty:
        return {"pairs": 0}
    return {
        "pairs": len(selected),
        "mean_reduction_percent": float(selected["reduction_percent"].mean()),
        "better_count": int(selected["candidate_better"].sum()),
    }


def candidate_summary(
    raw: pd.DataFrame, c0: pd.DataFrame, c1: pd.DataFrame, c2: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for spec in CAPACITY_CANDIDATES:
        runs = raw[
            (raw["capacity_candidate"] == spec.name) & (raw["status"] == "completed")
        ]
        summaries = {
            name: _comparison_summary(frame, spec.name)
            for name, frame in (("c0", c0), ("c1", c1), ("c2", c2))
        }
        trainable = sorted(set(runs["trainable_parameter_count"].dropna().astype(int)))
        rows.append(
            {
                "capacity_candidate": spec.name,
                "completed_runs": len(runs),
                "c0_mean_reduction_percent": summaries["c0"].get("mean_reduction_percent"),
                "c0_better_count": summaries["c0"].get("better_count", 0),
                "c1_mean_reduction_percent": summaries["c1"].get("mean_reduction_percent"),
                "c1_better_count": summaries["c1"].get("better_count", 0),
                "c2_mean_reduction_percent": summaries["c2"].get("mean_reduction_percent"),
                "c2_better_count": summaries["c2"].get("better_count", 0),
                "alpha_mean": float(runs["alpha"].mean()) if len(runs) else math.nan,
                "spatial_residual_rms_ratio_mean": (
                    float(runs["spatial_residual_rms_ratio"].mean())
                    if len(runs) else math.nan
                ),
                "best_epoch_mean": float(runs["best_epoch"].mean()) if len(runs) else math.nan,
                "trainable_parameter_counts": ";".join(map(str, trainable)),
            }
        )
    return pd.DataFrame(rows)


def recompute_prediction_metrics(path: Path) -> dict[str, float]:
    with np.load(path) as payload:
        target = payload["target_ugm3"]
        prediction = payload["prediction_ugm3"]
        backbone = payload["backbone_prediction_ugm3"]
    return {
        "rmse_ugm3": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "backbone_rmse_ugm3": float(np.sqrt(np.mean((backbone - target) ** 2))),
    }


def compliance_checks(
    raw: pd.DataFrame,
    selection: pd.DataFrame,
    output_root: Path,
    backbone_selection_path: Path,
    c2: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    checks = []
    expected = {
        ("beijing", history, horizon, CENTER_STATION_ID, seed, spec.name)
        for history, horizon in GRID_TASKS
        for seed in GRID_SEEDS
        for spec in CAPACITY_CANDIDATES
    }
    expected |= {
        ("beijing", history, horizon, CENTER_STATION_ID, seed, spec.name)
        for history, horizon in HEADLINE_TASKS
        for seed in set(HEADLINE_SEEDS) - set(GRID_SEEDS)
        for spec in CAPACITY_CANDIDATES
    }
    actual = {
        (
            row.city, int(row.history), int(row.horizon), int(row.station_id),
            int(row.seed), row.capacity_candidate,
        )
        for row in raw.itertuples()
    }
    checks.append(
        {
            "check": "exact_expected_run_identities",
            "pass": actual == expected,
            "detail": f"actual={len(actual)}, expected={len(expected)}",
        }
    )
    terminal = {
        "completed", "infeasible_oom", "nonfinite", "backbone_nonfinite",
        "no_usable_backbone",
    }
    checks.append(
        {
            "check": "all_expected_runs_terminal",
            "pass": actual == expected and set(raw["status"]).issubset(terminal),
            "detail": str(raw["status"].value_counts().to_dict()),
        }
    )
    checks.extend(nonfinite_status_checks(raw))

    specs = {spec.name: spec for spec in CAPACITY_CANDIDATES}
    hyperparameters_ok = True
    for row in raw.itertuples():
        spec = specs.get(row.capacity_candidate)
        hyperparameters_ok &= spec is not None and (
            int(row.neighbor_hidden_dim) == spec.neighbor_hidden_dim
            and int(row.spatial_pool_bins) == spec.spatial_pool_bins
            and math.isclose(float(row.forecast_alpha_init), spec.forecast_alpha_init)
            and math.isclose(float(row.forecast_alpha_max), spec.forecast_alpha_max)
        )
    checks.append(
        {"check": "registered_hyperparameters_exact", "pass": hyperparameters_ok, "detail": "3 fixed candidates"}
    )

    backbone = pd.read_csv(backbone_selection_path)
    selected_backbone = backbone[
        backbone["selected"].astype(str).str.lower().isin({"true", "1"})
    ][GROUP_COLUMNS + ["variant", "checkpoint_path"]]
    auditable_raw = raw[raw["status"] != "no_usable_backbone"]
    audit = auditable_raw.merge(
        selected_backbone,
        on=GROUP_COLUMNS,
        how="left",
        suffixes=("_run", "_registered"),
        validate="many_to_one",
    )
    backbone_ok = (
        len(audit) == len(auditable_raw)
        and audit["variant"].notna().all()
        and (audit["selected_variant"] == audit["variant"]).all()
        and (audit["source_checkpoint"] == audit["checkpoint_path"]).all()
    )
    checks.append(
        {"check": "backbone_selection_unchanged", "pass": backbone_ok, "detail": f"rows={len(audit)}"}
    )

    complete = raw[raw["status"] == "completed"]
    frozen_ok = len(complete) > 0 and complete["backbone_frozen"].astype(bool).all()
    zero_ok = len(complete) > 0 and complete["zero_init_ok"].astype(bool).all()
    checks.append({"check": "backbone_frozen", "pass": bool(frozen_ok), "detail": f"completed={len(complete)}"})
    checks.append(
        {
            "check": "zero_initialization_equivalent",
            "pass": bool(zero_ok),
            "detail": (
                "no completed runs" if complete.empty
                else f"max_abs={complete['zero_init_max_abs'].max():.3e}"
            ),
        }
    )

    group_counts = selection.groupby(GROUP_COLUMNS)["selected"].sum()
    unusable_groups = set(
        map(
            tuple,
            raw[raw["status"] == "no_usable_backbone"][GROUP_COLUMNS]
            .drop_duplicates()
            .to_numpy(),
        )
    )
    expected_group_count = len(GRID_TASKS) * len(GRID_SEEDS) + len(HEADLINE_TASKS) * (
        len(HEADLINE_SEEDS) - len(GRID_SEEDS)
    )
    group_selection_ok = all(
        int(count) == (0 if tuple(keys) in unusable_groups else 1)
        for keys, count in group_counts.items()
    )
    recomputed = build_capacity_selection(raw.drop(columns=[
        column for column in ("rmse_ugm3", "mae_ugm3", "smape_percent", "alpha")
        if column in raw.columns
    ]))
    selected_keys = set(
        map(tuple, selection[selection["selected"].astype(bool)][GROUP_COLUMNS + ["capacity_candidate"]].to_numpy())
    )
    recomputed_keys = set(
        map(tuple, recomputed[recomputed["selected"].astype(bool)][GROUP_COLUMNS + ["capacity_candidate"]].to_numpy())
    )
    validation_ok = (
        len(group_counts) == expected_group_count
        and group_selection_ok
        and selected_keys == recomputed_keys
    )
    checks.append(
        {
            "check": "validation_only_selection_recomputed",
            "pass": bool(validation_ok),
            "detail": f"groups={len(group_counts)}",
        }
    )

    recalculation_rows = []
    for _, row in complete.iterrows():
        metrics = recompute_prediction_metrics(output_root / row["prediction_file"])
        for metric, recomputed_value in metrics.items():
            recorded = float(row[metric])
            recalculation_rows.append(
                {
                    "run_id": row["run_id"],
                    "metric": metric,
                    "recorded": recorded,
                    "recomputed": recomputed_value,
                    "absolute_difference": abs(recorded - recomputed_value),
                    "pass": abs(recorded - recomputed_value) <= 1e-9,
                }
            )
    recalculation = pd.DataFrame(recalculation_rows)
    recalc_ok = not recalculation.empty and recalculation["pass"].all()
    checks.append(
        {
            "check": "c1_prediction_recalculation_le_1e-9",
            "pass": bool(recalc_ok),
            "detail": (
                "no completed runs" if recalculation.empty
                else f"max={recalculation['absolute_difference'].max():.3e}"
            ),
        }
    )
    c2_ok = len(c2) == len(complete) and c2["reference_rmse_ugm3"].notna().all()
    checks.append(
        {"check": "c2_fixed_s3_pairing_complete", "pass": bool(c2_ok), "detail": f"pairs={len(c2)}"}
    )
    return pd.DataFrame(checks), recalculation


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    table_root = Path(args.table_root)
    table_root.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(output_root / "raw_metrics.csv")
    selection = build_capacity_selection(raw)
    s3, s3_detail_path = load_s3_references(args.s3_main)
    c0 = build_c0(raw)
    c1 = build_c1(raw)
    c2 = build_c2(raw, s3)
    candidates = candidate_summary(raw, c0, c1, c2)
    gate, gate_detail = expansion_gate(selection)
    compliance, recalculation = compliance_checks(
        raw,
        selection,
        output_root,
        Path(args.backbone_selection),
        c2,
    )

    derived = {
        "capacity_selection.csv": selection,
        "paired_c0.csv": c0,
        "paired_c1.csv": c1,
        "paired_c2.csv": c2,
        "candidate_summary.csv": candidates,
        "expansion_gate_detail.csv": gate_detail,
        "compliance_self_check.csv": compliance,
        "independent_recalculation.csv": recalculation,
    }
    for name, frame in derived.items():
        frame.to_csv(output_root / name, index=False)
        frame.to_csv(table_root / name, index=False)

    distribution = (
        selection[selection["selected"].astype(bool)]["capacity_candidate"]
        .value_counts()
        .to_dict()
    )
    selected_rows = selection[selection["selected"].astype(bool)][
        GROUP_COLUMNS + ["capacity_candidate"]
    ]
    selected_c1 = c1.merge(
        selected_rows,
        on=GROUP_COLUMNS + ["capacity_candidate"],
        how="inner",
        validate="one_to_one",
    )
    selected_c2 = c2.merge(
        selected_rows,
        on=GROUP_COLUMNS + ["capacity_candidate"],
        how="inner",
        validate="one_to_one",
    )
    summary = {
        "run_status_counts": raw["status"].value_counts().to_dict(),
        "selection_distribution": distribution,
        "candidate_summary": candidates.to_dict("records"),
        "selected_C1": {
            "pairs": len(selected_c1),
            "mean_reduction_percent": float(selected_c1["reduction_percent"].mean()),
            "better_count": int(selected_c1["candidate_better"].sum()),
        },
        "selected_C2": {
            "pairs": len(selected_c2),
            "mean_reduction_percent": float(selected_c2["reduction_percent"].mean()),
            "better_count": int(selected_c2["candidate_better"].sum()),
        },
        "expansion_gate": gate,
        "compliance_all_pass": bool(compliance["pass"].all()),
        "backbone_selection_source": str(args.backbone_selection),
        "s3_main_source": str(args.s3_main),
        "s3_detail_source": str(s3_detail_path),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
