"""Summarize the preregistered trainable-parameter-matched baselines."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_trainable_matched_baselines import (
    FAMILIES,
    HIDDEN_SIZES,
    OUTPUT_ROOT,
    TASK_GRID,
    TERMINAL_STATUSES,
    build_single_station_model,
    expected_identities,
    parameter_counts,
)


CAPACITY_RAW = Path("experiments/results/capacity_search/raw_metrics.csv")
S3_DETAIL = Path(
    "/home/hansel/.herdr/worktrees/PatchTST/experiment-baseline-single-station/"
    "tables/single_station_baselines/S3_paired_detail.csv"
)
TABLE_ROOT = Path("tables/trainable_matched")
FIGURE_ROOT = Path("figures/trainable_matched")
RESULT_DOC = Path("docs/主干升级/06_同训练预算结果.md")
CAPACITY_CANDIDATES = ("cap32_b4_a10", "cap128_b8_a20")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--capacity-raw", default=str(CAPACITY_RAW))
    parser.add_argument("--s3-detail", default=str(S3_DETAIL))
    parser.add_argument("--table-root", default=str(TABLE_ROOT))
    parser.add_argument("--figure-root", default=str(FIGURE_ROOT))
    parser.add_argument("--result-doc", default=str(RESULT_DOC))
    return parser.parse_args()


def recompute_prediction_metrics(path: str | Path) -> dict[str, float]:
    with np.load(path) as payload:
        prediction = payload["prediction_ugm3"]
        target = payload["target_ugm3"]
    error = prediction - target
    return {
        "rmse_ugm3": float(np.sqrt(np.mean(error ** 2))),
        "mae_ugm3": float(np.mean(np.abs(error))),
    }


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1"})


def _complete(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["status"] == "completed"].copy()


def build_capacity_curve(raw: pd.DataFrame) -> pd.DataFrame:
    complete = _complete(raw)
    rows = []
    keys = ["history", "horizon", "family", "hidden_size"]
    for key, group in complete.groupby(keys, sort=True):
        counts = sorted(group["trainable_parameter_count"].astype(int).unique())
        totals = sorted(group["total_parameter_count"].astype(int).unique())
        rows.append(
            {
                **dict(zip(keys, key)),
                "variant": str(group["variant"].iloc[0]),
                "trainable_parameter_count": counts[0] if len(counts) == 1 else math.nan,
                "total_parameter_count": totals[0] if len(totals) == 1 else math.nan,
                "completed_runs": len(group),
                "rmse_ugm3_mean": float(group["rmse_ugm3"].mean()),
                "rmse_ugm3_std": (
                    float(group["rmse_ugm3"].std(ddof=1)) if len(group) > 1 else 0.0
                ),
                "best_valid_loss_mean": float(group["best_valid_loss"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_own_points(capacity: pd.DataFrame) -> pd.DataFrame:
    complete = capacity[
        (capacity["status"] == "completed")
        & capacity["capacity_candidate"].isin(CAPACITY_CANDIDATES)
    ].copy()
    rows = []
    for key, group in complete.groupby(
        ["history", "horizon", "capacity_candidate"], sort=True
    ):
        rows.append(
            {
                "history": int(key[0]),
                "horizon": int(key[1]),
                "capacity_candidate": key[2],
                "trainable_parameter_count_mean": float(
                    group["trainable_parameter_count"].mean()
                ),
                "trainable_parameter_count_min": int(
                    group["trainable_parameter_count"].min()
                ),
                "trainable_parameter_count_max": int(
                    group["trainable_parameter_count"].max()
                ),
                "completed_runs": len(group),
                "rmse_ugm3_mean": float(group["rmse_ugm3"].mean()),
                "rmse_ugm3_std": (
                    float(group["rmse_ugm3"].std(ddof=1)) if len(group) > 1 else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def build_matched_pairs(raw: pd.DataFrame, capacity: pd.DataFrame) -> pd.DataFrame:
    completed_baselines = _complete(raw)
    baselines = completed_baselines[
        completed_baselines["hidden_size"].astype(int) == 40
    ][
        [
            "history", "horizon", "seed", "family", "variant",
            "trainable_parameter_count", "rmse_ugm3",
        ]
    ].rename(
        columns={
            "variant": "baseline_variant",
            "trainable_parameter_count": "baseline_trainable_parameter_count",
            "rmse_ugm3": "baseline_rmse_ugm3",
        }
    )
    ours = capacity[
        (capacity["status"] == "completed")
        & (capacity["capacity_candidate"] == "cap32_b4_a10")
    ][
        ["history", "horizon", "seed", "trainable_parameter_count", "rmse_ugm3"]
    ].rename(
        columns={
            "trainable_parameter_count": "our_trainable_parameter_count",
            "rmse_ugm3": "our_rmse_ugm3",
        }
    )
    paired = baselines.merge(
        ours, on=["history", "horizon", "seed"], how="inner", validate="many_to_one"
    )
    paired["difference_ugm3"] = paired["our_rmse_ugm3"] - paired["baseline_rmse_ugm3"]
    paired["relative_change_percent"] = 100 * (
        paired["our_rmse_ugm3"] / paired["baseline_rmse_ugm3"] - 1
    )
    paired["our_model_better"] = paired["difference_ugm3"] < 0
    paired["parameter_gap"] = (
        paired["our_trainable_parameter_count"]
        - paired["baseline_trainable_parameter_count"]
    )
    return paired.sort_values(["history", "horizon", "seed", "family"])


def _summary_row(
    group: pd.DataFrame,
    scope_type: str,
    scope_value: str,
    family: str,
) -> dict[str, Any]:
    pairs = len(group)
    better = int(group["our_model_better"].sum()) if pairs else 0
    mean_difference = float(group["difference_ugm3"].mean()) if pairs else math.nan
    mean_relative = (
        float(group["relative_change_percent"].mean()) if pairs else math.nan
    )
    return {
        "scope_type": scope_type,
        "scope_value": scope_value,
        "family": family,
        "pairs": pairs,
        "mean_difference_ugm3": mean_difference,
        "mean_relative_change_percent": mean_relative,
        "our_model_better_count": better,
        "our_model_better_fraction": better / pairs if pairs else math.nan,
        "criterion_met": bool(
            pairs and mean_difference < 0 and better / pairs >= 0.60
        ),
    }


def summarize_matched_pairs(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family in (*FAMILIES, "pooled"):
        group = paired if family == "pooled" else paired[paired["family"] == family]
        rows.append(_summary_row(group, "overall", "all", family))
    for column in ("history", "horizon"):
        for value, group in paired.groupby(column, sort=True):
            rows.append(_summary_row(group, column, str(int(value)), "pooled"))
            for family, family_group in group.groupby("family", sort=True):
                rows.append(
                    _summary_row(family_group, column, str(int(value)), str(family))
                )
    return pd.DataFrame(rows)


def preregistered_verdict(summary: pd.DataFrame) -> tuple[str, str]:
    overall = summary[summary["scope_type"] == "overall"].set_index("family")
    strict = (
        {"gru", "lstm", "pooled"}.issubset(overall.index)
        and bool(overall.loc["gru", "criterion_met"])
        and bool(overall.loc["lstm", "criterion_met"])
        and float(overall.loc["pooled", "our_model_better_fraction"]) >= 0.60
    )
    if strict:
        return "成功", "GRU/LSTM h=40 均满足平均差<0，合并配对更优比例≥60%"
    strata = summary[
        (summary["scope_type"].isin(["history", "horizon"]))
        & (summary["criterion_met"].map(bool))
    ]
    family_partial = any(
        bool(overall.loc[family, "criterion_met"])
        for family in ("gru", "lstm")
        if family in overall.index
    )
    if family_partial or not strata.empty:
        return "部分成功", "总体严格门未同时通过，但至少一个模型族或预注册分层通过"
    return "失败", "同预算总体与预注册分层均未达到放行条件"


def build_large_budget_reference(
    raw: pd.DataFrame, capacity: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    completed_baselines = _complete(raw)
    baselines = completed_baselines[
        completed_baselines["hidden_size"].astype(int) == 64
    ][
        [
            "history", "horizon", "seed", "family", "variant",
            "trainable_parameter_count", "rmse_ugm3",
        ]
    ].rename(
        columns={
            "variant": "reference_variant",
            "trainable_parameter_count": "reference_trainable_parameter_count",
            "rmse_ugm3": "reference_rmse_ugm3",
        }
    )
    ours = capacity[
        (capacity["status"] == "completed")
        & (capacity["capacity_candidate"] == "cap128_b8_a20")
    ][
        ["history", "horizon", "seed", "trainable_parameter_count", "rmse_ugm3"]
    ].rename(
        columns={
            "trainable_parameter_count": "our_trainable_parameter_count",
            "rmse_ugm3": "our_rmse_ugm3",
        }
    )
    pairs = baselines.merge(
        ours, on=["history", "horizon", "seed"], how="inner", validate="many_to_one"
    )
    pairs["comparison_label"] = "largest_registered_baseline_below_budget_not_matched"
    pairs["parameter_gap"] = (
        pairs["our_trainable_parameter_count"]
        - pairs["reference_trainable_parameter_count"]
    )
    pairs["difference_ugm3"] = pairs["our_rmse_ugm3"] - pairs["reference_rmse_ugm3"]
    pairs["relative_change_percent"] = 100 * (
        pairs["our_rmse_ugm3"] / pairs["reference_rmse_ugm3"] - 1
    )
    pairs["our_model_better"] = pairs["difference_ugm3"] < 0
    rows = []
    for family, group in pairs.groupby("family", sort=True):
        rows.append(
            {
                "family": family,
                "pairs": len(group),
                "our_parameter_mean": float(group["our_trainable_parameter_count"].mean()),
                "reference_parameter_mean": float(
                    group["reference_trainable_parameter_count"].mean()
                ),
                "mean_parameter_gap": float(group["parameter_gap"].mean()),
                "mean_relative_change_percent": float(
                    group["relative_change_percent"].mean()
                ),
                "our_model_better_count": int(group["our_model_better"].sum()),
                "budget_interpretation": "not_matched_reference_is_below_budget",
            }
        )
    return pairs, pd.DataFrame(rows)


def build_reference_comparisons(
    capacity: pd.DataFrame, s3_detail: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    complete = capacity[
        (capacity["status"] == "completed")
        & capacity["capacity_candidate"].isin(CAPACITY_CANDIDATES)
    ].copy()
    rows: list[dict[str, Any]] = []
    for record in complete.to_dict("records"):
        for comparison, reference in (
            ("C1_vs_own_backbone", float(record["backbone_rmse_ugm3"])),
        ):
            candidate = float(record["rmse_ugm3"])
            rows.append(
                {
                    "history": int(record["history"]),
                    "horizon": int(record["horizon"]),
                    "seed": int(record["seed"]),
                    "capacity_candidate": record["capacity_candidate"],
                    "comparison": comparison,
                    "candidate_rmse_ugm3": candidate,
                    "reference_rmse_ugm3": reference,
                    "relative_change_percent": 100 * (candidate / reference - 1),
                    "candidate_better": candidate < reference,
                }
            )
    s3 = s3_detail[
        (s3_detail["city"] == "beijing") & (s3_detail["model_arm"] == "st")
    ][
        ["history", "horizon", "seed", "best_single_station_arm", "best_single_station_rmse_ugm3"]
    ]
    merged = complete.merge(
        s3, on=["history", "horizon", "seed"], how="left", validate="many_to_one"
    )
    for record in merged.dropna(subset=["best_single_station_rmse_ugm3"]).to_dict("records"):
        candidate = float(record["rmse_ugm3"])
        reference = float(record["best_single_station_rmse_ugm3"])
        rows.append(
            {
                "history": int(record["history"]),
                "horizon": int(record["horizon"]),
                "seed": int(record["seed"]),
                "capacity_candidate": record["capacity_candidate"],
                "comparison": "C2_vs_validation_selected_single_station",
                "reference_arm": record["best_single_station_arm"],
                "candidate_rmse_ugm3": candidate,
                "reference_rmse_ugm3": reference,
                "relative_change_percent": 100 * (candidate / reference - 1),
                "candidate_better": candidate < reference,
            }
        )
    detail = pd.DataFrame(rows)
    summary_rows = []
    for key, group in detail.groupby(["capacity_candidate", "comparison"], sort=True):
        summary_rows.append(
            {
                "capacity_candidate": key[0],
                "comparison": key[1],
                "pairs": len(group),
                "mean_relative_change_percent": float(
                    group["relative_change_percent"].mean()
                ),
                "candidate_better_count": int(group["candidate_better"].sum()),
            }
        )
    return detail, pd.DataFrame(summary_rows)


def independent_recalculation(
    raw: pd.DataFrame, output_root: Path
) -> pd.DataFrame:
    rows = []
    for record in _complete(raw).to_dict("records"):
        path = Path(str(record["prediction_file"]))
        if not path.is_absolute():
            path = output_root / path
        metrics = recompute_prediction_metrics(path)
        rows.append(
            {
                "run_id": record["run_id"],
                "prediction_file": str(path),
                "recorded_rmse_ugm3": float(record["rmse_ugm3"]),
                "recomputed_rmse_ugm3": metrics["rmse_ugm3"],
                "rmse_abs_difference": abs(
                    metrics["rmse_ugm3"] - float(record["rmse_ugm3"])
                ),
                "recorded_mae_ugm3": float(record["mae_ugm3"]),
                "recomputed_mae_ugm3": metrics["mae_ugm3"],
                "mae_abs_difference": abs(
                    metrics["mae_ugm3"] - float(record["mae_ugm3"])
                ),
            }
        )
    return pd.DataFrame(rows)


def compliance_checks(
    raw: pd.DataFrame,
    output_root: Path,
    recalculation: pd.DataFrame,
) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []
    actual = set(raw["run_id"].astype(str))
    expected = expected_identities()
    checks.append(
        {
            "check": "exact_expected_run_identities",
            "pass": actual == expected and len(raw) == len(expected),
            "detail": f"actual={len(actual)}, expected={len(expected)}, rows={len(raw)}",
        }
    )
    statuses = set(raw["status"].astype(str))
    checks.append(
        {
            "check": "all_runs_terminal",
            "pass": statuses.issubset(TERMINAL_STATUSES),
            "detail": str(raw["status"].value_counts().to_dict()),
        }
    )
    checks.append(
        {
            "check": "single_station_input_channels",
            "pass": bool((pd.to_numeric(raw["input_channels"]) == 1).all()),
            "detail": str(sorted(pd.to_numeric(raw["input_channels"]).unique())),
        }
    )
    completed = _complete(raw)
    finite_columns = ["best_valid_loss", "rmse_ugm3", "trainable_parameter_count"]
    finite = bool(
        len(completed)
        and set(finite_columns).issubset(completed.columns)
        and np.isfinite(completed[finite_columns].to_numpy(dtype=float)).all()
    )
    checks.append(
        {
            "check": "completed_metrics_finite",
            "pass": bool(len(completed) and finite),
            "detail": f"completed={len(completed)}",
        }
    )

    count_ok = True
    count_details = []
    for row in completed.itertuples():
        model = build_single_station_model(
            str(row.family), int(row.hidden_size), int(row.horizon), 0
        )
        total, trainable = parameter_counts(model)
        ok = (
            int(row.total_parameter_count) == total
            and int(row.trainable_parameter_count) == trainable
            and total == trainable
        )
        count_ok &= ok
        if not ok and len(count_details) < 5:
            count_details.append(str(row.run_id))
    checks.append(
        {
            "check": "parameter_counts_independently_reconstructed",
            "pass": bool(count_ok and len(completed)),
            "detail": "ok" if count_ok else f"mismatch={count_details}",
        }
    )

    max_recalc = (
        float(recalculation[["rmse_abs_difference", "mae_abs_difference"]].max().max())
        if len(recalculation) else math.inf
    )
    checks.append(
        {
            "check": "prediction_metrics_independently_recomputed",
            "pass": bool(len(recalculation) == len(completed) and max_recalc <= 1e-6),
            "detail": f"rows={len(recalculation)}, max_abs={max_recalc:.3e}",
        }
    )

    log_ok = True
    max_log_difference = 0.0
    for row in completed.itertuples():
        path = output_root / str(row.training_log_file)
        if not path.is_file():
            log_ok = False
            continue
        log = pd.read_csv(path)
        difference = abs(float(log["valid_loss"].min()) - float(row.best_valid_loss))
        max_log_difference = max(max_log_difference, difference)
        log_ok &= difference <= 1e-10
    checks.append(
        {
            "check": "best_valid_loss_matches_training_log",
            "pass": bool(log_ok and len(completed)),
            "detail": f"max_abs={max_log_difference:.3e}",
        }
    )

    metadata_ok = True
    for history, horizon in TASK_GRID:
        path = output_root / "metadata" / f"config_{history}h_{horizon}h.json"
        if not path.is_file():
            metadata_ok = False
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        semantics = payload.get("training_semantics", {})
        metadata_ok &= (
            semantics.get("optimizer") == "AdamW"
            and float(semantics.get("learning_rate", math.nan)) == 1e-3
            and float(semantics.get("weight_decay", math.nan)) == 1e-4
            and semantics.get("selection_split") == "valid"
            and semantics.get("evaluation_split") == "test"
            and float(semantics.get("gradient_clip_norm", math.nan)) == 1.0
            and semantics.get("loss") == "MSELoss"
        )
    checks.append(
        {
            "check": "training_semantics_manifest_exact",
            "pass": bool(metadata_ok),
            "detail": "20 formal config manifests",
        }
    )
    return pd.DataFrame(checks)


def build_figure_data(curve: pd.DataFrame, own: pd.DataFrame) -> pd.DataFrame:
    baseline = curve.rename(
        columns={
            "variant": "series",
            "trainable_parameter_count": "trainable_parameters",
        }
    )[
        [
            "history", "horizon", "series", "trainable_parameters",
            "rmse_ugm3_mean", "rmse_ugm3_std", "completed_runs",
        ]
    ].copy()
    baseline["model_type"] = "single_station"
    ours = own.rename(
        columns={
            "capacity_candidate": "series",
            "trainable_parameter_count_mean": "trainable_parameters",
        }
    )[
        [
            "history", "horizon", "series", "trainable_parameters",
            "rmse_ugm3_mean", "rmse_ugm3_std", "completed_runs",
        ]
    ].copy()
    ours["model_type"] = "spatial_frozen_backbone"
    return pd.concat([baseline, ours], ignore_index=True)


def plot_capacity_curve(data: pd.DataFrame, figure_root: Path) -> None:
    figure_root.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
            "savefig.dpi": 450,
            "savefig.bbox": "tight",
        }
    )
    colors = {"gru": "#0077BB", "lstm": "#EE7733"}
    point_colors = {"cap32_b4_a10": "#009988", "cap128_b8_a20": "#CC3311"}
    fig, axes = plt.subplots(4, 5, figsize=(13.5, 9.0), sharex=False, sharey=False)
    for axis, (history, horizon) in zip(axes.flat, TASK_GRID):
        task = data[(data["history"] == history) & (data["horizon"] == horizon)]
        for family in FAMILIES:
            family_rows = task[
                (task["model_type"] == "single_station")
                & task["series"].str.contains(f"_{family}_")
            ].sort_values("trainable_parameters")
            axis.plot(
                family_rows["trainable_parameters"],
                family_rows["rmse_ugm3_mean"],
                marker="o", markersize=2.8, linewidth=1.1,
                color=colors[family], label=family.upper(),
            )
        for candidate, marker in (("cap32_b4_a10", "D"), ("cap128_b8_a20", "X")):
            point = task[task["series"] == candidate]
            if not point.empty:
                axis.scatter(
                    point["trainable_parameters"], point["rmse_ugm3_mean"],
                    marker=marker, s=28, color=point_colors[candidate],
                    edgecolors="white", linewidths=0.4, zorder=4, label=candidate,
                )
        axis.set_xscale("log")
        axis.set_title(f"L={history}, H={horizon}")
        if horizon == 1:
            axis.set_ylabel("Test RMSE (ug/m3)")
        if history == 168:
            axis.set_xlabel("Trainable parameters (log)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Single-station capacity curves and frozen-backbone spatial points", y=1.035)
    fig.tight_layout()
    fig.savefig(figure_root / "capacity_curve.png", dpi=450)
    fig.savefig(figure_root / "capacity_curve.svg")
    plt.close(fig)


def _format_overall(summary: pd.DataFrame) -> list[str]:
    lines = []
    overall = summary[summary["scope_type"] == "overall"]
    for row in overall.itertuples():
        lines.append(
            f"| {row.family} | {row.pairs} | {row.mean_difference_ugm3:+.4f} | "
            f"{row.mean_relative_change_percent:+.4f}% | "
            f"{row.our_model_better_count}/{row.pairs} | "
            f"{'PASS' if row.criterion_met else 'FAIL'} |"
        )
    return lines


def write_result_doc(
    path: Path,
    raw: pd.DataFrame,
    curve: pd.DataFrame,
    matched_summary: pd.DataFrame,
    large_summary: pd.DataFrame,
    reference_summary: pd.DataFrame,
    checks: pd.DataFrame,
    verdict: str,
    verdict_reason: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    status_counts = raw["status"].value_counts().to_dict()
    failed_checks = checks[~checks["pass"].map(bool)]
    check_status = "PASS" if failed_checks.empty else "FAIL"
    curve_parameter_ranges = (
        curve.groupby(["family", "hidden_size"])["trainable_parameter_count"]
        .agg(["min", "max"])
        .reset_index()
    )
    lines = [
        "# 单站点同可训练参数预算对照结果",
        "",
        "> 预注册协议：`docs/主干升级/05_同训练预算对照协议.md`  ",
        "> 数据：北京 1013、无泄漏管线、20 任务配置  ",
        f"> 合规自检：{check_status}",
        "",
        "## 1. 执行完整性",
        "",
        f"正式矩阵共登记 {len(raw)} 个身份，终态分布为 `{status_counts}`。"
        f"合规检查 {int(checks['pass'].sum())}/{len(checks)} 通过。",
        "",
        "## 2. 单站点容量曲线",
        "",
        "下表给出参数量在不同 horizon 下的最小—最大值；逐20 配置 RMSE 均值见 "
        "`tables/trainable_matched/capacity_curve.csv`，图见 `figures/trainable_matched/capacity_curve.png`。",
        "",
        "| 模型族 | hidden | 可训练参数范围 |",
        "|---|---:|---:|",
    ]
    for row in curve_parameter_ranges.itertuples():
        lines.append(f"| {row.family.upper()} | {int(row.hidden_size)} | {int(row.min)}–{int(row.max)} |")
    lines.extend(
        [
            "",
            "## 3. 同预算主对照",
            "",
            "差值与相对变化均按“我们的 cap32 减单站点 h=40”计算，负值表示 cap32 更优。",
            "",
            "| 参考 | 配对数 | 平均差 (ug/m3) | 平均相对变化 | cap32 更优 | 分项门 |",
            "|---|---:|---:|---:|---:|---:|",
            *_format_overall(matched_summary),
            "",
            "## 4. cap128 的预算边界",
            "",
            "`cap128_b8_a20` 的可训练参数约为 55k–61k，而预注册单站点网格最大只到 h=64。"
            "因此下表是“注册网格内最大但低于预算”的参考，不是同预算证据。",
            "",
            "| 参考 | 配对数 | cap128 参数均值 | 基线参数均值 | 平均相对变化 | cap128 更优 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in large_summary.itertuples():
        lines.append(
            f"| {row.family.upper()} h=64 | {row.pairs} | {row.our_parameter_mean:.0f} | "
            f"{row.reference_parameter_mean:.0f} | {row.mean_relative_change_percent:+.4f}% | "
            f"{row.our_model_better_count}/{row.pairs} |"
        )
    lines.extend(
        [
            "",
            "## 5. C1/C2 参考",
            "",
            "| 容量点 | 对照 | 配对数 | 平均相对变化 | 候选更优 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in reference_summary.itertuples():
        lines.append(
            f"| {row.capacity_candidate} | {row.comparison} | {row.pairs} | "
            f"{row.mean_relative_change_percent:+.4f}% | "
            f"{row.candidate_better_count}/{row.pairs} |"
        )
    lines.extend(
        [
            "",
            "## 6. 预注册结论",
            "",
            f"判定：{verdict}。{verdict_reason}。",
            "",
            "该结论只适用于北京 1013 PM2.5、当前无泄漏划分、已注册任务和种子。"
            "本轮使用已消费种子，属于方法学补充对照，不是新独立确认。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _print_capacity_table(curve: pd.DataFrame) -> None:
    printable = curve.copy()
    printable["task"] = printable.apply(
        lambda row: f"{int(row.history)}x{int(row.horizon)}", axis=1
    )
    pivot = printable.pivot_table(
        index=["family", "hidden_size", "trainable_parameter_count"],
        columns="task", values="rmse_ugm3_mean", aggfunc="first"
    ).reset_index()
    print("\n容量曲线表（各配置 test RMSE 均值）")
    print(pivot.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    table_root = Path(args.table_root)
    figure_root = Path(args.figure_root)
    table_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)

    raw_path = output_root / "raw_metrics.csv"
    for path in (raw_path, Path(args.capacity_raw), Path(args.s3_detail)):
        if not path.is_file():
            raise FileNotFoundError(path)
    raw = pd.read_csv(raw_path)
    capacity = pd.read_csv(args.capacity_raw)
    s3_detail = pd.read_csv(args.s3_detail)

    curve = build_capacity_curve(raw)
    own_points = build_own_points(capacity)
    matched_pairs = build_matched_pairs(raw, capacity)
    matched_summary = summarize_matched_pairs(matched_pairs)
    verdict, verdict_reason = preregistered_verdict(matched_summary)
    large_detail, large_summary = build_large_budget_reference(raw, capacity)
    reference_detail, reference_summary = build_reference_comparisons(
        capacity, s3_detail
    )
    recalculation = independent_recalculation(raw, output_root)
    checks = compliance_checks(raw, output_root, recalculation)
    figure_data = build_figure_data(curve, own_points)

    artifacts = {
        "capacity_curve.csv": curve,
        "our_capacity_points.csv": own_points,
        "matched_paired.csv": matched_pairs,
        "matched_summary.csv": matched_summary,
        "large_budget_reference.csv": large_detail,
        "large_budget_reference_summary.csv": large_summary,
        "reference_comparisons.csv": reference_detail,
        "reference_comparison_summary.csv": reference_summary,
        "independent_recalculation.csv": recalculation,
        "compliance_self_check.csv": checks,
        "figure_capacity_curve.csv": figure_data,
    }
    for filename, frame in artifacts.items():
        frame.to_csv(table_root / filename, index=False)
    plot_capacity_curve(figure_data, figure_root)

    all_checks_pass = bool(len(checks) and checks["pass"].map(bool).all())
    summary_payload = {
        "registered_verdict": verdict,
        "verdict_reason": verdict_reason,
        "formal_run_count": len(raw),
        "status_counts": raw["status"].value_counts().to_dict(),
        "compliance_pass": all_checks_pass,
        "compliance_checks_passed": int(checks["pass"].sum()),
        "compliance_checks_total": len(checks),
        "cap128_budget_statement": (
            "No same-budget GRU/LSTM arm exists in the preregistered grid; "
            "h64 is reported only as the largest lower-budget reference."
        ),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_result_doc(
        Path(args.result_doc), raw, curve, matched_summary, large_summary,
        reference_summary, checks, verdict, verdict_reason,
    )

    _print_capacity_table(curve)
    print("\n同预算点对照（cap32 vs h40）")
    print(
        matched_summary.to_string(
            index=False, float_format=lambda value: f"{value:.6f}"
        )
    )
    print("\ncap128 vs 注册网格最大 h64（低于预算，不是同预算）")
    print(
        large_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )
    print("\nC1/C2 参考")
    print(
        reference_summary.to_string(
            index=False, float_format=lambda value: f"{value:.6f}"
        )
    )
    print(f"\n预注册判定：{verdict}（{verdict_reason}）")
    print(
        f"合规自检：{int(checks['pass'].sum())}/{len(checks)} "
        f"{'PASS' if all_checks_pass else 'FAIL'}"
    )
    print("TRAINABLE_MATCH_DONE")


if __name__ == "__main__":
    main()
