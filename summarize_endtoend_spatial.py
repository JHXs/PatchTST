"""Summarize the preregistered end-to-end spatial comparison."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_endtoend_spatial import BACKBONES, expected_identities, run_id


BASELINE_RAW = Path("experiments/results/trainable_matched/raw_metrics.csv")
OUTPUT_ROOT = Path("experiments/results/endtoend_spatial")
TABLE_ROOT = Path("tables/endtoend_spatial")
FIGURE_ROOT = Path("figures/endtoend_spatial")
FROZEN_C1_REDUCTION_PERCENT = 1.43
TERMINAL_STATUSES = frozenset({"completed", "infeasible_oom", "nonfinite"})


def recompute_prediction_metrics(path: str | Path) -> dict[str, float]:
    with np.load(path) as payload:
        target = payload["target_ugm3"]
        prediction = payload["prediction_ugm3"]
    error = prediction - target
    return {
        "rmse_ugm3": float(np.sqrt(np.mean(error ** 2))),
        "mae_ugm3": float(np.mean(np.abs(error))),
    }


def registered_baselines(frame: pd.DataFrame) -> pd.DataFrame:
    registered = pd.DataFrame(BACKBONES, columns=["family", "hidden_size"])
    result = frame.merge(registered, on=["family", "hidden_size"], how="inner")
    return result.copy()


def _assert_unique(frame: pd.DataFrame, keys: list[str], label: str) -> None:
    duplicated = frame.duplicated(keys, keep=False)
    if duplicated.any():
        raise RuntimeError(
            f"{label} 键不唯一: {frame.loc[duplicated, keys].head().to_dict('records')}"
        )


def build_pairs(baseline: pd.DataFrame, ours: pd.DataFrame) -> pd.DataFrame:
    keys = ["history", "horizon", "seed", "family", "hidden_size"]
    _assert_unique(baseline, keys, "B")
    _assert_unique(ours, keys, "O")
    baseline_columns = keys + [
        "run_id", "status", "rmse_ugm3", "mae_ugm3", "best_valid_loss",
        "trainable_parameter_count", "prediction_file",
    ]
    ours_columns = keys + [
        "run_id", "status", "rmse_ugm3", "mae_ugm3", "best_valid_loss",
        "trainable_parameter_count", "backbone_parameter_count",
        "spatial_head_parameter_count", "prediction_file",
    ]
    paired = baseline[baseline_columns].merge(
        ours[ours_columns],
        on=keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_b", "_o"),
        indicator=True,
    )
    paired["backbone"] = paired.apply(
        lambda row: f"{row['family']}_h{int(row['hidden_size'])}", axis=1
    )
    paired["pair_completed"] = (
        paired["_merge"].eq("both")
        & paired["status_b"].eq("completed")
        & paired["status_o"].eq("completed")
    )
    paired["rmse_difference_ugm3"] = paired["rmse_ugm3_o"] - paired["rmse_ugm3_b"]
    paired["relative_change_percent"] = 100.0 * (
        paired["rmse_ugm3_o"] / paired["rmse_ugm3_b"] - 1.0
    )
    paired["o_better"] = paired["rmse_ugm3_o"] < paired["rmse_ugm3_b"]
    return paired.sort_values(keys).reset_index(drop=True)


def summarize_group(group: pd.DataFrame) -> dict[str, Any]:
    complete = group[group["pair_completed"]].copy()
    count = len(complete)
    better = int(complete["o_better"].sum())
    relative = float(complete["relative_change_percent"].mean()) if count else math.nan
    better_fraction = better / count if count else math.nan
    return {
        "registered_pairs": int(len(group)),
        "valid_pairs": count,
        "mean_b_rmse_ugm3": float(complete["rmse_ugm3_b"].mean()) if count else math.nan,
        "mean_o_rmse_ugm3": float(complete["rmse_ugm3_o"].mean()) if count else math.nan,
        "mean_rmse_difference_ugm3": (
            float(complete["rmse_difference_ugm3"].mean()) if count else math.nan
        ),
        "mean_relative_change_percent": relative,
        "o_better_pairs": better,
        "o_better_fraction": better_fraction,
        "gate_pass": bool(count and relative < 0.0 and better_fraction >= 0.60),
    }


def grouped_summary(paired: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame([{"stratum": "overall", **summarize_group(paired)}])
    rows = []
    grouping: str | list[str] = columns[0] if len(columns) == 1 else columns
    for key, group in paired.groupby(grouping, sort=True, dropna=False):
        values = (key,) if len(columns) == 1 else tuple(key)
        rows.append({**dict(zip(columns, values)), **summarize_group(group)})
    return pd.DataFrame(rows)


def independent_recalculation(
    ours: pd.DataFrame, output_root: Path
) -> pd.DataFrame:
    rows = []
    completed = ours[ours["status"] == "completed"]
    for row in completed.itertuples(index=False):
        path = output_root / str(row.prediction_file)
        metrics = recompute_prediction_metrics(path)
        rows.append(
            {
                "run_id": row.run_id,
                "prediction_file": str(path),
                "recorded_rmse_ugm3": float(row.rmse_ugm3),
                "recomputed_rmse_ugm3": metrics["rmse_ugm3"],
                "rmse_abs_difference": abs(metrics["rmse_ugm3"] - float(row.rmse_ugm3)),
                "recorded_mae_ugm3": float(row.mae_ugm3),
                "recomputed_mae_ugm3": metrics["mae_ugm3"],
                "mae_abs_difference": abs(metrics["mae_ugm3"] - float(row.mae_ugm3)),
            }
        )
    return pd.DataFrame(rows)


def per_lead_pairs(
    paired: pd.DataFrame,
    baseline_root: Path,
    output_root: Path,
) -> pd.DataFrame:
    rows = []
    for row in paired[paired["pair_completed"]].itertuples(index=False):
        with np.load(baseline_root / str(row.prediction_file_b)) as b_payload:
            b_prediction = b_payload["prediction_ugm3"]
            b_target = b_payload["target_ugm3"]
        with np.load(output_root / str(row.prediction_file_o)) as o_payload:
            o_prediction = o_payload["prediction_ugm3"]
            o_target = o_payload["target_ugm3"]
        if not np.array_equal(b_target, o_target):
            raise RuntimeError(f"配对目标不一致: {row.run_id_o}")
        for lead in range(int(row.horizon)):
            target = o_target[..., lead]
            b_rmse = float(np.sqrt(np.mean((b_prediction[..., lead] - target) ** 2)))
            o_rmse = float(np.sqrt(np.mean((o_prediction[..., lead] - target) ** 2)))
            rows.append(
                {
                    "history": int(row.history),
                    "horizon": int(row.horizon),
                    "lead": lead + 1,
                    "seed": int(row.seed),
                    "backbone": row.backbone,
                    "b_rmse_ugm3": b_rmse,
                    "o_rmse_ugm3": o_rmse,
                    "relative_change_percent": 100.0 * (o_rmse / b_rmse - 1.0),
                    "o_better": o_rmse < b_rmse,
                }
            )
    return pd.DataFrame(rows)


def summarize_per_lead(per_lead: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (horizon, lead), group in per_lead.groupby(["horizon", "lead"], sort=True):
        better = int(group["o_better"].sum())
        count = len(group)
        rows.append(
            {
                "horizon": int(horizon),
                "lead": int(lead),
                "valid_pairs": count,
                "mean_relative_change_percent": float(
                    group["relative_change_percent"].mean()
                ),
                "o_better_pairs": better,
                "o_better_fraction": better / count,
            }
        )
    return pd.DataFrame(rows)


def parameter_tables(
    baseline: pd.DataFrame, ours: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["family", "hidden_size", "horizon"]
    b = baseline.groupby(keys, as_index=False).agg(
        b_trainable_parameters=("trainable_parameter_count", "first"),
        b_parameter_variants=("trainable_parameter_count", "nunique"),
    )
    o = ours[ours["status"] == "completed"].groupby(keys, as_index=False).agg(
        o_trainable_parameters=("trainable_parameter_count", "first"),
        o_parameter_variants=("trainable_parameter_count", "nunique"),
        o_backbone_parameters=("backbone_parameter_count", "first"),
        spatial_head_parameters=("spatial_head_parameter_count", "first"),
    )
    detail = b.merge(o, on=keys, how="outer", validate="one_to_one")
    detail["backbone"] = detail.apply(
        lambda row: f"{row['family']}_h{int(row['hidden_size'])}", axis=1
    )
    detail["parameter_identity_ok"] = (
        detail["b_trainable_parameters"].eq(detail["o_backbone_parameters"])
        & detail["o_trainable_parameters"].eq(
            detail["b_trainable_parameters"] + detail["spatial_head_parameters"]
        )
        & detail["b_parameter_variants"].eq(1)
        & detail["o_parameter_variants"].eq(1)
    )
    summary = detail.groupby("backbone", as_index=False).agg(
        b_trainable_min=("b_trainable_parameters", "min"),
        b_trainable_max=("b_trainable_parameters", "max"),
        o_trainable_min=("o_trainable_parameters", "min"),
        o_trainable_max=("o_trainable_parameters", "max"),
        spatial_head_min=("spatial_head_parameters", "min"),
        spatial_head_max=("spatial_head_parameters", "max"),
        all_parameter_identities_ok=("parameter_identity_ok", "all"),
    )
    return detail, summary


def fairness_gates(
    baseline: pd.DataFrame,
    ours: pd.DataFrame,
    pairs: pd.DataFrame,
    parameter_detail: pd.DataFrame,
    recalculation: pd.DataFrame,
    preflight_path: Path,
) -> pd.DataFrame:
    preflight: dict[str, Any] = {}
    if preflight_path.is_file():
        payload = json.loads(preflight_path.read_text(encoding="utf-8"))
        preflight = {row["gate"]: row for row in payload.get("gates", [])}
    zero = preflight.get("zero_spatial_equivalence", {})
    baseline_inputs_ok = bool((baseline["input_channels"] == 1).all())
    ours_inputs_ok = bool((ours["input_channels"] == 18).all())
    no_peeking = bool(
        ours["selection_split"].eq("valid").all()
        and ours["evaluation_split"].eq("test").all()
        and ours.loc[ours["status"].eq("completed"), "test_evaluation_count"].eq(1).all()
    )
    recompute_max = (
        float(recalculation["rmse_abs_difference"].max())
        if not recalculation.empty else math.inf
    )
    rows = [
        {
            "gate": "zero_spatial_equivalence",
            "passed": bool(zero.get("passed", False)),
            "detail": zero.get("detail", "preflight result missing"),
        },
        {
            "gate": "parameter_registration",
            "passed": bool(
                not parameter_detail.empty
                and parameter_detail["parameter_identity_ok"].all()
            ),
            "detail": f"checked_rows={len(parameter_detail)}",
        },
        {
            "gate": "input_channel_boundary",
            "passed": baseline_inputs_ok and ours_inputs_ok,
            "detail": "B=1 and O=18" if baseline_inputs_ok and ours_inputs_ok else "channel mismatch",
        },
        {
            "gate": "no_test_peeking",
            "passed": no_peeking,
            "detail": "selection=valid; evaluation=test once" if no_peeking else "manifest mismatch",
        },
        {
            "gate": "independent_recalculation",
            "passed": bool(len(recalculation) == int(ours["status"].eq("completed").sum()) and recompute_max < 1e-9),
            "detail": f"rows={len(recalculation)}, max_rmse_abs_diff={recompute_max:.3e}",
        },
    ]
    return pd.DataFrame(rows)


def compliance_checks(
    baseline: pd.DataFrame,
    ours: pd.DataFrame,
    pairs: pd.DataFrame,
    gates: pd.DataFrame,
) -> pd.DataFrame:
    expected = expected_identities()
    actual = set(ours["run_id"].astype(str))
    baseline_expected = {
        run_id(int(row.history), int(row.horizon), int(row.seed), row.family, int(row.hidden_size))
        for row in baseline.itertuples(index=False)
    }
    rows = [
        ("o_identity_set_exact", actual == expected, f"actual={len(actual)}, expected={len(expected)}"),
        ("b_registered_rows_384", len(baseline) == 384 and len(baseline_expected) == 384, f"rows={len(baseline)}"),
        ("o_terminal_statuses", set(ours["status"]).issubset(TERMINAL_STATUSES), str(ours["status"].value_counts().to_dict())),
        ("pair_keys_complete", len(pairs) == 384 and pairs["_merge"].eq("both").all(), f"rows={len(pairs)}"),
        ("five_fairness_gates", len(gates) == 5 and gates["passed"].all(), str(dict(zip(gates["gate"], gates["passed"])))),
    ]
    return pd.DataFrame(rows, columns=["check", "passed", "detail"])


def preregistered_decision(
    overall: pd.DataFrame,
    by_history: pd.DataFrame,
    by_horizon: pd.DataFrame,
) -> tuple[str, str]:
    if bool(overall.iloc[0]["gate_pass"]):
        return "成功", "总体平均相对变化<0，且至少60%的配对中O优于B。"
    passing_history = by_history.loc[by_history["gate_pass"], "history"].astype(str).tolist()
    passing_horizon = by_horizon.loc[by_horizon["gate_pass"], "horizon"].astype(str).tolist()
    if passing_history or passing_horizon:
        return (
            "部分成功",
            f"总体门未通过；通过的history={passing_history}，horizon={passing_horizon}。",
        )
    return "失败", "总体门未通过，且无预注册history/horizon分层通过。"


def make_figure(
    overall: pd.DataFrame,
    by_history: pd.DataFrame,
    by_horizon: pd.DataFrame,
    table_root: Path,
    figure_root: Path,
) -> None:
    figure_rows = [
        {
            "family": "overall",
            "stratum": "overall",
            "mean_relative_change_percent": overall.iloc[0]["mean_relative_change_percent"],
            "o_better_fraction": overall.iloc[0]["o_better_fraction"],
        }
    ]
    figure_rows.extend(
        {
            "family": "history",
            "stratum": str(int(row.history)),
            "mean_relative_change_percent": row.mean_relative_change_percent,
            "o_better_fraction": row.o_better_fraction,
        }
        for row in by_history.itertuples(index=False)
    )
    figure_rows.extend(
        {
            "family": "horizon",
            "stratum": str(int(row.horizon)),
            "mean_relative_change_percent": row.mean_relative_change_percent,
            "o_better_fraction": row.o_better_fraction,
        }
        for row in by_horizon.itertuples(index=False)
    )
    data = pd.DataFrame(figure_rows)
    data.to_csv(table_root / "figure_stratified.csv", index=False)

    labels = [
        "Overall",
        *[f"L={value}" for value in by_history["history"]],
        *[f"H={value}" for value in by_horizon["horizon"]],
    ]
    values = data["mean_relative_change_percent"].to_numpy()
    colors = ["#009E73" if value < 0 else "#D55E00" for value in values]
    fig, axis = plt.subplots(figsize=(10.5, 4.8))
    axis.bar(np.arange(len(values)), values, color=colors, width=0.72)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(np.arange(len(values)), labels, rotation=40, ha="right")
    axis.set_ylabel("Mean relative RMSE change, 100×(O−B)/B (%)")
    axis.set_title("End-to-end spatial head effect by preregistered stratum")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    figure_root.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_root / "stratified_relative_change.png", dpi=450)
    fig.savefig(figure_root / "stratified_relative_change.svg")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-raw", default=str(BASELINE_RAW))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--table-root", default=str(TABLE_ROOT))
    parser.add_argument("--figure-root", default=str(FIGURE_ROOT))
    args = parser.parse_args()

    baseline_path = Path(args.baseline_raw)
    output_root = Path(args.output_root)
    table_root = Path(args.table_root)
    figure_root = Path(args.figure_root)
    table_root.mkdir(parents=True, exist_ok=True)

    baseline = registered_baselines(pd.read_csv(baseline_path))
    ours = pd.read_csv(output_root / "raw_metrics.csv")
    pairs = build_pairs(baseline, ours)
    overall = grouped_summary(pairs, [])
    by_horizon = grouped_summary(pairs, ["horizon"])
    by_history = grouped_summary(pairs, ["history"])
    by_backbone = grouped_summary(pairs, ["backbone"])
    recalculation = independent_recalculation(ours, output_root)
    per_lead = per_lead_pairs(pairs, baseline_path.parent, output_root)
    per_lead_summary = summarize_per_lead(per_lead)
    parameter_detail, parameter_summary = parameter_tables(baseline, ours)
    gates = fairness_gates(
        baseline,
        ours,
        pairs,
        parameter_detail,
        recalculation,
        table_root / "preflight_fairness_gates.json",
    )
    compliance = compliance_checks(baseline, ours, pairs, gates)
    decision, decision_reason = preregistered_decision(overall, by_history, by_horizon)
    all_checks_pass = bool(compliance["passed"].all())

    outputs = {
        "paired_results.csv": pairs,
        "summary_overall.csv": overall,
        "summary_by_horizon.csv": by_horizon,
        "summary_by_history.csv": by_history,
        "summary_by_backbone.csv": by_backbone,
        "per_lead_pairs.csv": per_lead,
        "per_lead_summary.csv": per_lead_summary,
        "parameter_counts.csv": parameter_detail,
        "parameter_counts_summary.csv": parameter_summary,
        "independent_recalculation.csv": recalculation,
        "fairness_gates.csv": gates,
        "compliance_checks.csv": compliance,
    }
    for name, frame in outputs.items():
        frame.to_csv(table_root / name, index=False)
    make_figure(overall, by_history, by_horizon, table_root, figure_root)

    endtoend_reduction = -float(overall.iloc[0]["mean_relative_change_percent"])
    summary = {
        "registered_pairs": 384,
        "status_counts": {str(k): int(v) for k, v in ours["status"].value_counts().items()},
        "overall": overall.iloc[0].to_dict(),
        "preregistered_decision": decision,
        "decision_reason": decision_reason,
        "all_compliance_checks_pass": all_checks_pass,
        "frozen_c1_reduction_percent": FROZEN_C1_REDUCTION_PERCENT,
        "endtoend_reduction_percent": endtoend_reduction,
        "endtoend_minus_frozen_percentage_points": (
            endtoend_reduction - FROZEN_C1_REDUCTION_PERCENT
        ),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nC1_endtoend 总体")
    print(overall.to_string(index=False))
    print("\n按预测步长 H")
    print(by_horizon.to_string(index=False))
    print("\n按历史长度 L")
    print(by_history.to_string(index=False))
    print("\n参数量范围")
    print(parameter_summary.to_string(index=False))
    print("\n公平性门")
    print(gates.to_string(index=False))
    print(f"\n预注册结论: {decision}；{decision_reason}")
    print(
        f"冻结C1改善={FROZEN_C1_REDUCTION_PERCENT:.2f}%，"
        f"端到端改善={endtoend_reduction:.4f}%，"
        f"差={endtoend_reduction - FROZEN_C1_REDUCTION_PERCENT:+.4f}个百分点"
    )
    if not all_checks_pass:
        raise RuntimeError("完整性或公平性门未全部通过，结果不可放行")
    print("ENDTOEND_MATCH_DONE")


if __name__ == "__main__":
    main()
