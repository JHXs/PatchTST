"""Independently recompute and verify L0 selection-interval result artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cross_variable_lag_l0_data import AUXILIARY_COLUMNS, MODEL_FEATURES
from run_cross_variable_lag_l0 import (
    DEFAULT_OUTPUT_DIR,
    SCOPE_LABEL,
    compute_gate,
    regression_metrics,
    relative_improvement_percent,
    select_alpha,
)


REQUIRED_CSVS = (
    "processing_ledger.csv",
    "fit_ledger.csv",
    "data_quality.csv",
    "window_manifest.csv",
    "boundary_checks.csv",
    "target_time_sets.csv",
    "embargoed_origins.csv",
    "inner_alpha_scores.csv",
    "inner_alpha_horizon_metrics.csv",
    "outer_predictions.csv",
    "fold_horizon_metrics.csv",
    "fold_metrics_all.csv",
    "paired_differences.csv",
    "task_summary.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recompute_metric_frames(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizon_rows = []
    all_rows = []
    group_keys = ["task", "fold", "model"]
    for (task, fold, model), group in predictions.groupby(group_keys, sort=True):
        alpha_values = group["selected_alpha"].unique()
        if len(alpha_values) != 1:
            raise AssertionError("Outer predictions contain multiple selected alphas")
        alpha = float(alpha_values[0])
        origins = group["origin_time"].nunique()
        for horizon_step, horizon_group in group.groupby("horizon_step", sort=True):
            metrics = regression_metrics(
                horizon_group["actual"].to_numpy(),
                horizon_group["prediction"].to_numpy(),
            )
            horizon_rows.append(
                {
                    "task": task,
                    "fold": int(fold),
                    "model": model,
                    "horizon_step": int(horizon_step),
                    "selected_alpha": alpha,
                    "validation_samples": origins,
                    **metrics,
                }
            )
        metrics = regression_metrics(
            group["actual"].to_numpy(), group["prediction"].to_numpy()
        )
        all_rows.append(
            {
                "task": task,
                "fold": int(fold),
                "model": model,
                "selected_alpha": alpha,
                "validation_samples": origins,
                "target_values": len(group),
                "rmse_all": metrics["rmse"],
                "mae_all": metrics["mae"],
                "smape_all_percent": metrics["smape_percent"],
            }
        )
    return pd.DataFrame(horizon_rows), pd.DataFrame(all_rows)


def recompute_paired_differences(fold_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task in sorted(fold_all["task"].unique()):
        for fold in range(1, 6):
            values = fold_all[
                (fold_all["task"] == task) & (fold_all["fold"] == fold)
            ].set_index("model")
            lag_rmse = float(values.loc["R_lag", "rmse_all"])
            for baseline in ("R_self", "R_recent"):
                baseline_rmse = float(values.loc[baseline, "rmse_all"])
                absolute = baseline_rmse - lag_rmse
                rows.append(
                    {
                        "task": task,
                        "fold": fold,
                        "baseline": baseline,
                        "candidate": "R_lag",
                        "baseline_rmse_all": baseline_rmse,
                        "candidate_rmse_all": lag_rmse,
                        "absolute_rmse_improvement": absolute,
                        "relative_rmse_improvement_percent": (
                            relative_improvement_percent(baseline_rmse, lag_rmse)
                        ),
                        "candidate_better": bool(absolute > 0),
                    }
                )
    return pd.DataFrame(rows)


def _assert_numeric_match(
    stored: pd.DataFrame,
    recomputed: pd.DataFrame,
    keys: list[str],
    numeric_columns: list[str],
    tolerance: float = 1e-10,
) -> float:
    left = stored[keys + numeric_columns].copy()
    right = recomputed[keys + numeric_columns].copy()
    merged = left.merge(right, on=keys, suffixes=("_stored", "_recomputed"))
    if len(merged) != len(left) or len(merged) != len(right):
        raise AssertionError(f"Key mismatch while recomputing {keys}")
    maximum = 0.0
    for column in numeric_columns:
        differences = np.abs(
            merged[f"{column}_stored"].to_numpy(dtype=float)
            - merged[f"{column}_recomputed"].to_numpy(dtype=float)
        )
        maximum = max(maximum, float(differences.max(initial=0.0)))
    if maximum > tolerance:
        raise AssertionError(
            f"Metric recomputation mismatch {maximum:.3e} > {tolerance:.3e}"
        )
    return maximum


def _verify_target_sets(target_sets: pd.DataFrame) -> int:
    checked = 0
    for keys, group in target_sets.groupby(
        ["task", "boundary_type", "boundary_id"], sort=True
    ):
        left = set(group.loc[group["side"] == "left", "target_time"])
        right = set(group.loc[group["side"] == "right", "target_time"])
        if left & right:
            raise AssertionError(f"Target time overlap in {keys}")
        checked += 1
    return checked


def _verify_alpha_selection(alpha_scores: pd.DataFrame) -> int:
    checked = 0
    for keys, group in alpha_scores.groupby(["task", "fold", "model"], sort=True):
        selected = group[group["selected"].astype(str).str.lower() == "true"]
        if len(selected) != 1:
            raise AssertionError(f"Expected exactly one selected alpha for {keys}")
        scores = {
            float(row.alpha): float(row.rmse_all)
            for row in group.itertuples(index=False)
        }
        expected = select_alpha(scores)
        actual = float(selected.iloc[0]["alpha"])
        if actual != expected:
            raise AssertionError(f"Alpha tie/selection mismatch for {keys}")
        checked += 1
    return checked


def verify_outputs(output_dir: Path, write_result: bool = True) -> dict[str, Any]:
    missing = [name for name in REQUIRED_CSVS if not (output_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing L0 artifacts: {missing}")
    for name in ("config.json", "task_metadata.json", "gate_status.json"):
        if not (output_dir / name).is_file():
            raise FileNotFoundError(f"Missing L0 artifact: {name}")

    frames = {name[:-4]: pd.read_csv(output_dir / name) for name in REQUIRED_CSVS}
    config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    metadata = json.loads(
        (output_dir / "task_metadata.json").read_text(encoding="utf-8")
    )
    stored_gate = json.loads(
        (output_dir / "gate_status.json").read_text(encoding="utf-8")
    )

    predictions = frames["outer_predictions"]
    if set(predictions["scope"]) != {SCOPE_LABEL}:
        raise AssertionError("Prediction scope label is missing or inconsistent")
    if predictions["formal_test_sample"].astype(str).str.lower().ne("false").any():
        raise AssertionError("A formal-test prediction was found")
    if config["formal_test_used"] or config["neural_model_used"]:
        raise AssertionError("Run config violates L0 selection-only scope")
    for task, values in metadata.items():
        access = values["access_metadata"]
        if access["test_feature_rows_loaded"] != 0:
            raise AssertionError(f"Test feature rows were loaded for {task}")
        for field in (
            "test_samples_constructed",
            "test_predictions_constructed",
            "test_metrics_constructed",
        ):
            if values[field] != 0:
                raise AssertionError(f"Nonzero {field} for {task}")

    recomputed_horizon, recomputed_all = recompute_metric_frames(predictions)
    horizon_max_error = _assert_numeric_match(
        frames["fold_horizon_metrics"],
        recomputed_horizon,
        keys=["task", "fold", "model", "horizon_step"],
        numeric_columns=["selected_alpha", "validation_samples", "rmse", "mae", "smape_percent"],
    )
    all_max_error = _assert_numeric_match(
        frames["fold_metrics_all"],
        recomputed_all,
        keys=["task", "fold", "model"],
        numeric_columns=[
            "selected_alpha",
            "validation_samples",
            "target_values",
            "rmse_all",
            "mae_all",
            "smape_all_percent",
        ],
    )

    recomputed_pairs = recompute_paired_differences(recomputed_all)
    pair_max_error = _assert_numeric_match(
        frames["paired_differences"],
        recomputed_pairs,
        keys=["task", "fold", "baseline", "candidate"],
        numeric_columns=[
            "baseline_rmse_all",
            "candidate_rmse_all",
            "absolute_rmse_improvement",
            "relative_rmse_improvement_percent",
        ],
    )

    primary = recomputed_all[recomputed_all["task"] == "168h_6h"]
    values = {
        model: primary[primary["model"] == model]
        .sort_values("fold")["rmse_all"]
        .to_numpy()
        for model in MODEL_FEATURES
    }
    recomputed_gate = compute_gate(
        values["R_self"], values["R_recent"], values["R_lag"]
    )
    if recomputed_gate["status"] != stored_gate["status"]:
        raise AssertionError("Stored gate status differs from prediction recomputation")
    for name, observed in recomputed_gate["observed"].items():
        stored = stored_gate["observed"][name]
        if isinstance(observed, float):
            if not math_isclose(float(stored), observed):
                raise AssertionError(f"Gate value mismatch: {name}")
        elif stored != observed:
            raise AssertionError(f"Gate value mismatch: {name}")
    if recomputed_gate["checks"] != stored_gate["checks"]:
        raise AssertionError("Stored gate checks differ from recomputation")

    boundary_checks = frames["boundary_checks"]
    if not boundary_checks["assertion_passed"].astype(str).str.lower().eq("true").all():
        raise AssertionError("A boundary assertion is not marked as passed")
    if not (boundary_checks["target_time_intersection_count"] == 0).all():
        raise AssertionError("A stored boundary intersection is nonzero")
    for task, expected in (("168h_6h", 5), ("24h_1h", 0)):
        values_for_task = boundary_checks.loc[
            boundary_checks["task"] == task, "embargo_origin_count"
        ]
        if len(values_for_task) != 17 or not (values_for_task == expected).all():
            raise AssertionError(f"Embargo contract failed for {task}")
    target_boundaries_checked = _verify_target_sets(frames["target_time_sets"])
    alpha_groups_checked = _verify_alpha_selection(frames["inner_alpha_scores"])

    for ledger_name in ("processing_ledger", "fit_ledger"):
        ledger = frames[ledger_name]
        required = ("fit_start", "fit_end", "row_count", "config_hash")
        if ledger.loc[:, required].isna().any().any():
            raise AssertionError(f"Incomplete range/hash fields in {ledger_name}")
        if (ledger["row_count"] <= 0).any():
            raise AssertionError(f"Nonpositive row count in {ledger_name}")
        if not ledger["config_hash"].str.fullmatch(r"[0-9a-f]{64}").all():
            raise AssertionError(f"Invalid config hash in {ledger_name}")

    fit_ledger = frames["fit_ledger"].copy()
    fit_ledger["fit_end"] = pd.to_datetime(fit_ledger["fit_end"])
    processing_ledger = frames["processing_ledger"].copy()
    processing_ledger["fit_end"] = pd.to_datetime(processing_ledger["fit_end"])
    for task, task_metadata in metadata.items():
        test_start = pd.Timestamp(task_metadata["official_boundaries"]["test_start"])
        if (fit_ledger.loc[fit_ledger["task"] == task, "fit_end"] >= test_start).any():
            raise AssertionError(f"A fitted object reached formal test time for {task}")
        if (
            processing_ledger.loc[processing_ledger["task"] == task, "fit_end"]
            >= test_start
        ).any():
            raise AssertionError(f"A processing object reached formal test time for {task}")
        for fold in range(1, 6):
            for boundary_type, prefix in (
                ("inner", ("inner_scaler:", "inner_ridge:")),
                ("outer", ("outer_scaler:", "outer_ridge:")),
            ):
                right_times = frames["target_time_sets"].loc[
                    (frames["target_time_sets"]["task"] == task)
                    & (frames["target_time_sets"]["boundary_type"] == boundary_type)
                    & (frames["target_time_sets"]["boundary_id"] == f"fold_{fold}")
                    & (frames["target_time_sets"]["side"] == "right"),
                    "target_time",
                ]
                right_start = pd.to_datetime(right_times).min()
                objects = fit_ledger.loc[
                    (fit_ledger["task"] == task)
                    & (fit_ledger["fold"] == fold)
                    & fit_ledger["object_id"].str.startswith(prefix)
                ]
                if objects.empty or (objects["fit_end"] >= right_start).any():
                    raise AssertionError(
                        f"Training-only fit-range contract failed for "
                        f"{task}/{boundary_type}/fold{fold}"
                    )

    data_quality = frames["data_quality"]
    aux_quality = data_quality[data_quality["variable"].isin(AUXILIARY_COLUMNS)]
    for task in ("168h_6h", "24h_1h"):
        task_train = aux_quality[
            (aux_quality["task"] == task) & (aux_quality["split"] == "train")
        ]
        if list(task_train["variable"]) != list(AUXILIARY_COLUMNS):
            raise AssertionError(f"P=7 order mismatch for {task}")
        if (task_train["residual_missing_rate"] > 0.10).any():
            raise AssertionError(f"Training missing-rate gate failed for {task}")

    expected_shapes = {"R_self": 12, "R_recent": 26, "R_lag": 96}
    for task, task_metadata in metadata.items():
        for model, feature_count in expected_shapes.items():
            if task_metadata["feature_shapes"][model][1] != feature_count:
                raise AssertionError(f"Feature shape mismatch for {task}/{model}")

    checks = {
        "all_required_files_present": True,
        "selection_scope_only": True,
        "test_access_counts_all_zero": True,
        "outer_predictions_recomputed": True,
        "paired_differences_recomputed": True,
        "gate_recomputed": True,
        "all_target_time_sets_disjoint": True,
        "embargo_168_to_6_equals_5": True,
        "embargo_24_to_1_equals_0": True,
        "alpha_selection_recomputed": True,
        "ledger_complete": True,
        "all_fit_ranges_training_only": True,
        "p7_order_and_missing_rate_gate_verified": True,
        "feature_shapes_verified": True,
    }
    result = {
        "scope": SCOPE_LABEL,
        "passed": bool(all(checks.values())),
        "checks": checks,
        "stored_gate_status": stored_gate["status"],
        "recomputed_gate_status": recomputed_gate["status"],
        "maximum_absolute_metric_errors": {
            "fold_horizon": horizon_max_error,
            "fold_all": all_max_error,
            "paired": pair_max_error,
        },
        "target_boundaries_checked": target_boundaries_checked,
        "alpha_groups_checked": alpha_groups_checked,
        "artifact_sha256": {
            name: _sha256(output_dir / name)
            for name in (*REQUIRED_CSVS, "config.json", "task_metadata.json", "gate_status.json")
        },
    }
    if write_result:
        (output_dir / "integrity_verification.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def math_isclose(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return bool(np.isclose(left, right, rtol=0.0, atol=tolerance))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify_outputs(args.output_dir, write_result=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
