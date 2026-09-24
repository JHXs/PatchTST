"""Independently summarize C1/C2/C3 for the backbone-upgrade experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backbone_candidates import expected_candidates


S3_DIRECTORIES = (
    Path("tables/single_station_baselines"),
    Path("/home/hansel/Documents/ITProject/Python/PatchTST/tables/single_station_baselines"),
    Path("/home/hansel/.herdr/worktrees/PatchTST/experiment-baseline-single-station/tables/single_station_baselines"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="experiments/results/backbone_upgrade")
    parser.add_argument("--run-kind", choices=("formal", "smoke"), default="formal")
    return parser.parse_args()


def locate_s3() -> tuple[Path | None, Path | None]:
    """Locate S3_main (the frozen arm choice) and its per-seed detail together."""
    for directory in S3_DIRECTORIES:
        main = directory / "S3_main.csv"
        detail = directory / "S3_paired_detail.csv"
        if main.is_file() and detail.is_file():
            return main, detail
    return None, None


def recompute_prediction_metrics(path: str | Path) -> dict[str, float]:
    """Recompute physical-unit RMSE for T6 and the compliance audit."""
    with np.load(path) as payload:
        # Preserve the artifact dtype: the recorded metric was computed from the
        # same float32 tensors.  Promoting only the audit path to float64 changes
        # reduction order enough to violate the preregistered 1e-9 equality gate.
        target = payload["target_ugm3"]
        prediction = payload["prediction_ugm3"]
        backbone = payload["backbone_prediction_ugm3"]
    return {
        "rmse_ugm3": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "backbone_rmse_ugm3": float(np.sqrt(np.mean((backbone - target) ** 2))),
    }


def _paired_row(prefix: str, upgraded: float, reference: float, **keys) -> dict:
    return {
        **keys,
        "comparison": prefix,
        "upgraded_rmse_ugm3": upgraded,
        "reference_rmse_ugm3": reference,
        "difference_ugm3": upgraded - reference,
        "reduction_percent": 100 * (reference - upgraded) / reference,
        "upgraded_better": upgraded < reference,
    }


def build_c1(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in raw.query("status == 'completed'").iterrows():
        rows.append(
            _paired_row(
                "C1_upgraded_vs_own_backbone",
                float(row["rmse_ugm3"]),
                float(row["backbone_rmse_ugm3"]),
                city=row["city"],
                history=int(row["history"]),
                horizon=int(row["horizon"]),
                station_id=int(row["station_id"]),
                seed=int(row["seed"]),
                selected_variant=row["selected_variant"],
            )
        )
    return pd.DataFrame(rows)


def _upgrade_seed_means(raw: pd.DataFrame) -> pd.DataFrame:
    complete = raw.query("status == 'completed'")
    return (
        complete.groupby(["city", "history", "horizon", "seed"], as_index=False)
        .agg(upgraded_rmse_ugm3=("rmse_ugm3", "mean"), station_count=("station_id", "nunique"))
    )


def build_c2_c3(
    raw: pd.DataFrame, s3_main_path: Path | None, s3_detail_path: Path | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if s3_main_path is None or s3_detail_path is None:
        return pd.DataFrame(), pd.DataFrame()
    s3_main = pd.read_csv(s3_main_path)
    frozen_choices = s3_main[s3_main["model_arm"] == "st"][
        ["city", "history", "horizon", "best_single_station_arm"]
    ]
    s3_detail = pd.read_csv(s3_detail_path)
    current = s3_detail[s3_detail["model_arm"] == "st"].copy()
    current = current.merge(
        frozen_choices,
        on=["city", "history", "horizon"],
        suffixes=("_detail", "_main"),
        how="inner",
        validate="many_to_one",
    )
    if not (
        current["best_single_station_arm_detail"]
        == current["best_single_station_arm_main"]
    ).all():
        raise RuntimeError("S3_main 与 S3_paired_detail 的最强单站点口径不一致")
    upgrades = _upgrade_seed_means(raw)
    merged = upgrades.merge(
        current,
        on=["city", "history", "horizon", "seed"],
        how="left",
        validate="one_to_one",
    )
    c2, c3 = [], []
    for _, row in merged.dropna(subset=["best_single_station_rmse_ugm3"]).iterrows():
        keys = {
            "city": row["city"],
            "history": int(row["history"]),
            "horizon": int(row["horizon"]),
            "seed": int(row["seed"]),
            "station_count": int(row["station_count"]),
        }
        c2.append(
            {
                **_paired_row(
                    "C2_upgraded_vs_strongest_single_station",
                    float(row["upgraded_rmse_ugm3"]),
                    float(row["best_single_station_rmse_ugm3"]),
                    **keys,
                ),
                "reference_arm": row["best_single_station_arm_main"],
            }
        )
        c3.append(
            _paired_row(
                "C3_upgraded_vs_current_frozen_st",
                float(row["upgraded_rmse_ugm3"]),
                float(row["model_rmse_ugm3"]),
                **keys,
            )
        )
    return pd.DataFrame(c2), pd.DataFrame(c3)


def compliance_checks(
    raw: pd.DataFrame,
    selection: pd.DataFrame,
    output_dir: Path,
    s3_main_path: Path | None,
    s3_detail_path: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    checks = []
    completed = raw.query("status == 'completed'")
    recalculations = []
    for _, row in completed.iterrows():
        metrics = recompute_prediction_metrics(output_dir / row["prediction_file"])
        for metric, recomputed in metrics.items():
            recorded = float(row[metric])
            recalculations.append(
                {
                    "run_id": row["run_id"],
                    "metric": metric,
                    "recorded": recorded,
                    "recomputed": recomputed,
                    "absolute_difference": abs(recorded - recomputed),
                    "pass": abs(recorded - recomputed) <= 1e-9,
                }
            )
    recalculation = pd.DataFrame(recalculations)
    checks.append(
        {
            "check": "prediction_metric_recalculation_le_1e-9",
            "pass": not recalculation.empty and bool(recalculation["pass"].all()),
            "detail": "无已完成运行" if recalculation.empty else f"max={recalculation['absolute_difference'].max():.3e}",
        }
    )
    checks.append(
        {
            "check": "backbone_frozen",
            "pass": not completed.empty and bool(completed["backbone_frozen"].astype(bool).all()),
            "detail": f"completed={len(completed)}",
        }
    )
    checks.append(
        {
            "check": "zero_initialization_le_1e-12",
            "pass": not completed.empty and bool((completed["zero_init_max_abs"] <= 1e-12).all()),
            "detail": "无已完成运行" if completed.empty else f"max={completed['zero_init_max_abs'].max():.3e}",
        }
    )
    group_cols = ["city", "history", "horizon", "station_id", "seed"]
    selected_counts = selection.groupby(group_cols)["selected"].sum() if not selection.empty else pd.Series(dtype=int)
    unusable_groups = set(
        map(
            tuple,
            raw[raw["status"] == "no_usable_backbone"][group_cols]
            .drop_duplicates()
            .to_numpy(),
        )
    )
    selection_counts_ok = all(
        int(count) == (0 if tuple(keys) in unusable_groups else 1)
        for keys, count in selected_counts.items()
    )
    checks.append(
        {
            "check": "validation_selection_or_explicitly_unusable",
            "pass": not selected_counts.empty and selection_counts_ok,
            "detail": (
                f"groups={len(selected_counts)}, no_usable={len(unusable_groups)}"
            ),
        }
    )
    expected_complete = True
    missing_details = []
    for keys, group in selection.groupby(group_cols):
        city, history, horizon, _, _ = keys
        expected = {spec.variant for spec in expected_candidates(city, history, horizon)}
        absent = expected - set(group["variant"])
        if absent:
            expected_complete = False
            missing_details.append(f"{keys}:{sorted(absent)}")
    checks.append(
        {
            "check": "expected_candidate_universe_recorded",
            "pass": not selection.empty and expected_complete,
            "detail": "; ".join(missing_details) if missing_details else "全部期望候选均有行",
        }
    )
    checks.append(
        {
            "check": "unavailable_candidates_explicit",
            "pass": "candidate_status" in selection,
            "detail": (
                selection["candidate_status"].value_counts().to_dict()
                if "candidate_status" in selection else "缺少 candidate_status"
            ),
        }
    )
    for status in ("backbone_nonfinite", "no_usable_backbone"):
        count = int((raw["status"] == status).sum())
        checks.append(
            {
                "check": f"{status}_count",
                "pass": True,
                "detail": f"count={count}",
            }
        )
    checks.append(
        {
            "check": "s3_pairing_source_found",
            "pass": s3_main_path is not None and s3_detail_path is not None,
            "detail": (
                "" if s3_main_path is None
                else f"main={s3_main_path}; detail={s3_detail_path}"
            ),
        }
    )
    return pd.DataFrame(checks), recalculation


def _comparison_summary(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"pairs": 0}
    return {
        "pairs": len(frame),
        "mean_reduction_percent": float(frame["reduction_percent"].mean()),
        "better_count": int(frame["upgraded_better"].sum()),
        "better_fraction": float(frame["upgraded_better"].mean()),
    }


def configuration_summary(*frames: pd.DataFrame) -> pd.DataFrame:
    """Collapse paired seeds/stations to the protocol's city-task configuration."""
    rows = []
    for frame in frames:
        if frame.empty:
            continue
        for keys, group in frame.groupby(["comparison", "city", "history", "horizon"]):
            comparison, city, history, horizon = keys
            rows.append(
                {
                    "comparison": comparison,
                    "city": city,
                    "history": int(history),
                    "horizon": int(horizon),
                    "pair_count": len(group),
                    "mean_difference_ugm3": float(group["difference_ugm3"].mean()),
                    "mean_reduction_percent": float(group["reduction_percent"].mean()),
                    "configuration_better": bool(group["difference_ugm3"].mean() < 0),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "comparison", "city", "history", "horizon", "pair_count",
            "mean_difference_ugm3", "mean_reduction_percent", "configuration_better",
        ],
    )


def protocol_gate(configurations: pd.DataFrame, smoke: bool) -> dict:
    if smoke:
        return {"status": "smoke_not_eligible", "reason": "两轮冒烟不得用于正式判定"}
    c1 = configurations[
        configurations["comparison"] == "C1_upgraded_vs_own_backbone"
    ]
    c2 = configurations[
        configurations["comparison"] == "C2_upgraded_vs_strongest_single_station"
    ]
    if c1.empty or c2.empty:
        return {"status": "incomplete", "reason": "缺少 C1 或 C2 配置级对照"}
    c1_mean = float(c1["mean_reduction_percent"].mean())
    c1_fraction = float(c1["configuration_better"].mean())
    c2_fraction = float(c2["configuration_better"].mean())
    if c1_mean <= 0:
        status = "failure_stop"
    elif c1_fraction >= 0.6 and c2_fraction >= 0.6:
        status = "success"
    elif c1_fraction >= 0.6:
        status = "partial_success"
    else:
        status = "failure_stop"
    return {
        "status": status,
        "c1_mean_reduction_percent": c1_mean,
        "c1_better_configuration_fraction": c1_fraction,
        "c2_better_configuration_fraction": c2_fraction,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_root) / args.run_kind
    raw = pd.read_csv(output_dir / "raw_metrics.csv")
    selection = pd.read_csv(output_dir / "selection.csv")
    s3_main_path, s3_detail_path = locate_s3()
    c1 = build_c1(raw)
    c2, c3 = build_c2_c3(raw, s3_main_path, s3_detail_path)
    configurations = configuration_summary(c1, c2, c3)
    diagnostics_columns = [
        "run_id", "city", "history", "horizon", "station_id", "seed",
        "selected_variant", "alpha", "spatial_residual_rms_ratio", "best_epoch",
        "best_valid_loss", "zero_init_max_abs", "backbone_frozen",
    ]
    diagnostics = raw.loc[raw["status"] == "completed", diagnostics_columns]
    compliance, recalculation = compliance_checks(
        raw, selection, output_dir, s3_main_path, s3_detail_path
    )
    for name, frame in (
        ("paired_c1.csv", c1),
        ("paired_c2.csv", c2),
        ("paired_c3.csv", c3),
        ("diagnostics.csv", diagnostics),
        ("compliance_self_check.csv", compliance),
        ("independent_recalculation.csv", recalculation),
        ("configuration_summary.csv", configurations),
    ):
        frame.to_csv(output_dir / name, index=False)
    distribution = (
        selection[selection["selected"].astype(bool)]["variant"].value_counts().to_dict()
    )
    summary = {
        "selection_distribution": distribution,
        "selection_status_counts": selection["candidate_status"].value_counts().to_dict(),
        "C1": _comparison_summary(c1),
        "C2": _comparison_summary(c2),
        "C3": _comparison_summary(c3),
        "protocol_gate": protocol_gate(configurations, args.run_kind == "smoke"),
        "diagnostics": {
            "alpha_mean": float(diagnostics["alpha"].mean()) if len(diagnostics) else None,
            "residual_ratio_mean": (
                float(diagnostics["spatial_residual_rms_ratio"].mean()) if len(diagnostics) else None
            ),
            "best_epoch_mean": float(diagnostics["best_epoch"].mean()) if len(diagnostics) else None,
        },
        "compliance_all_pass": bool(compliance["pass"].all()),
        "s3_main_source": None if s3_main_path is None else str(s3_main_path),
        "s3_detail_source": None if s3_detail_path is None else str(s3_detail_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
