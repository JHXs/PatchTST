"""Summarize and independently audit the Baseline-v2 formal matrix."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import run_baseline_v2 as runner


DEFAULT_ROOT = Path("experiments/results/baseline_v2")
DEFAULT_TABLES = Path("tables/baseline_v2")
DEFAULT_FIGURES = Path("figures/baseline_v2")
REUSED = DEFAULT_ROOT / "reused"
O_LOCKED_ROOT = (
    REUSED
    / "beijing_leakfree/experiments/results/beijing_leakfree_coverage"
)
GRU_LSTM_ROOT = (
    REUSED
    / "backbone_upgrade/experiments/results/trainable_matched"
)
TRADITIONAL_TABLE = (
    REUSED
    / "single_station/tables/single_station_baselines/S4_traditional_details.csv"
)
O_VARIANT = "st_sparse_station_bias_delta_forecast"
KEYS = ["history", "horizon", "seed"]
COLORS = {
    "informer": "#0077BB",
    "tst": "#EE7733",
    "gru": "#009988",
    "lstm": "#CC3311",
    "o_locked": "#AA3377",
}
METRIC_COLUMNS = [
    "best_valid_loss",
    "mse_scaled",
    "rmse_scaled",
    "mae_scaled",
    "rmse_ugm3",
    "mae_ugm3",
    "smape_percent",
]


def _relative_difference(actual: float, expected: float) -> float:
    return abs(float(actual) - float(expected)) / max(abs(float(expected)), 1e-30)


def _write_table(df: pd.DataFrame, path: Path, *, markdown: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    if markdown:
        try:
            text = df.to_markdown(index=False)
        except ImportError:
            text = df.to_csv(index=False)
        path.with_suffix(".md").write_text(text + "\n", encoding="utf-8")


def load_new_results(root: Path) -> pd.DataFrame:
    path = root / "raw_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(f"缺少正式矩阵: {path}")
    data = pd.read_csv(path)
    for column in ("history", "horizon", "seed"):
        data[column] = pd.to_numeric(data[column], errors="raise").astype(int)
    return data


def load_o_locked(root: Path = O_LOCKED_ROOT) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    pattern = re.compile(r"(?P<history>\d+)h_(?P<horizon>\d+)h")
    for raw_path in sorted(root.glob("*h_*h/raw_metrics.csv")):
        match = pattern.fullmatch(raw_path.parent.name)
        if match is None:
            continue
        data = pd.read_csv(raw_path)
        data = data[data["variant"].eq(O_VARIANT)].copy()
        data["history"] = int(match.group("history"))
        data["horizon"] = int(match.group("horizon"))
        data["seed"] = data["seed"].astype(int)
        data["family"] = "o_locked"
        data["arm"] = "o_locked"
        data["status"] = "completed"
        data["input_channels"] = 18
        data["prediction_path"] = data["seed"].map(
            lambda seed: raw_path.parent
            / "predictions"
            / f"{O_VARIANT}_seed{int(seed)}.npz"
        )
        rows.append(data)
    if not rows:
        raise FileNotFoundError(f"未找到 O-locked 复用结果: {root}")
    result = pd.concat(rows, ignore_index=True)
    return result.sort_values(KEYS).reset_index(drop=True)


def load_gru_lstm(root: Path = GRU_LSTM_ROOT) -> pd.DataFrame:
    data = pd.read_csv(root / "raw_metrics.csv")
    data = data[data["status"].eq("completed")].copy()
    data["arm"] = data["variant"].astype(str)
    data["prediction_path"] = data["prediction_file"].map(root.__truediv__)
    return data


def load_traditional(path: Path = TRADITIONAL_TABLE) -> pd.DataFrame:
    data = pd.read_csv(path)
    data = data[data["city"].eq("beijing") & data["status"].eq("completed")].copy()
    data["arm"] = data["variant"].astype(str)
    data["family"] = "traditional"
    return data


def paired_with_o(
    baseline: pd.DataFrame,
    o_locked: pd.DataFrame,
    *,
    arm: str | None = None,
) -> pd.DataFrame:
    left = baseline[baseline["status"].eq("completed")].copy()
    if arm is not None:
        left = left[left["variant"].eq(arm)]
    right = o_locked[o_locked["status"].eq("completed")].copy()
    merged = left.merge(
        right[KEYS + ["rmse_ugm3", "mae_ugm3"]],
        on=KEYS,
        how="inner",
        suffixes=("_baseline", "_o_locked"),
        validate="many_to_one",
    )
    merged["baseline_relative_to_o_percent"] = 100.0 * (
        merged["rmse_ugm3_baseline"] - merged["rmse_ugm3_o_locked"]
    ) / merged["rmse_ugm3_o_locked"]
    merged["baseline_better"] = (
        merged["rmse_ugm3_baseline"] < merged["rmse_ugm3_o_locked"]
    )
    return merged


def summarize_pair_rows(
    pairs: pd.DataFrame,
    *,
    category: str,
    family: str,
    arm: str,
    alignment: str,
) -> dict[str, Any]:
    relative = pairs["baseline_relative_to_o_percent"]
    return {
        "category": category,
        "family": family,
        "arm": arm,
        "capacity_alignment": alignment,
        "pairing_unit": "config_seed",
        "completed_runs": len(pairs),
        "pool_mean_baseline_rmse_ugm3": pairs["rmse_ugm3_baseline"].mean(),
        "pool_mean_o_locked_rmse_ugm3": pairs["rmse_ugm3_o_locked"].mean(),
        "mean_baseline_relative_to_o_percent": relative.mean(),
        "median_baseline_relative_to_o_percent": relative.median(),
        "baseline_better_count": int(pairs["baseline_better"].sum()),
        "paired_total": len(pairs),
        "information_set": "baseline=single_station; o_locked=18_stations",
        "reported_our_reduction_percent": -relative.mean(),
    }


def build_main_table(
    new: pd.DataFrame,
    o_locked: pd.DataFrame,
    recurrent: pd.DataFrame,
    traditional: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    details: list[pd.DataFrame] = []
    for arm in runner.ARMS:
        pairs = paired_with_o(new, o_locked, arm=arm)
        pairs["family"] = runner.ARM_SPECS[arm]["family"]
        pairs["arm"] = arm
        details.append(pairs)
        rows.append(
            summarize_pair_rows(
                pairs,
                category="new",
                family=runner.ARM_SPECS[arm]["family"],
                arm=arm,
                alignment=(
                    "matched"
                    if runner.ARM_SPECS[arm]["family"] == "informer"
                    else "non_aligned_as_is"
                ),
            )
        )
    for arm, group in recurrent.groupby("arm", sort=True):
        pairs = paired_with_o(group, o_locked)
        pairs["family"] = str(group["family"].iloc[0])
        pairs["arm"] = arm
        details.append(pairs)
        rows.append(
            summarize_pair_rows(
                pairs,
                category="reused_capacity_curve",
                family=str(group["family"].iloc[0]),
                arm=str(arm),
                alignment="capacity_curve",
            )
        )
    o_config = (
        o_locked.groupby(["history", "horizon"], as_index=False)["rmse_ugm3"]
        .mean()
        .rename(columns={"rmse_ugm3": "rmse_ugm3_o_locked"})
    )
    for arm, group in traditional.groupby("arm", sort=True):
        config_rows = group.merge(o_config, on=["history", "horizon"], how="inner")
        config_rows["baseline_relative_to_o_percent"] = 100.0 * (
            config_rows["rmse_ugm3"] - config_rows["rmse_ugm3_o_locked"]
        ) / config_rows["rmse_ugm3_o_locked"]
        rows.append(
            {
                "category": "reused_traditional",
                "family": "traditional",
                "arm": arm,
                "capacity_alignment": "not_applicable",
                "pairing_unit": "config_deterministic",
                "completed_runs": len(config_rows),
                "pool_mean_baseline_rmse_ugm3": config_rows["rmse_ugm3"].mean(),
                "pool_mean_o_locked_rmse_ugm3": config_rows["rmse_ugm3_o_locked"].mean(),
                "mean_baseline_relative_to_o_percent": config_rows[
                    "baseline_relative_to_o_percent"
                ].mean(),
                "median_baseline_relative_to_o_percent": config_rows[
                    "baseline_relative_to_o_percent"
                ].median(),
                "baseline_better_count": int(
                    (config_rows["rmse_ugm3"] < config_rows["rmse_ugm3_o_locked"]).sum()
                ),
                "paired_total": len(config_rows),
                "information_set": "baseline=single_station; o_locked=18_stations",
                "reported_our_reduction_percent": -config_rows[
                    "baseline_relative_to_o_percent"
                ].mean(),
            }
        )
    prior = [
        ("direction20_multistation_gru", -2.82, "same_information_multistation_gru"),
        ("direction20_single_station_gru_valid", -1.38, "validation_selection"),
        ("direction20_single_station_gru_test", -3.44, "test_selection_optimistic"),
        ("direction20_end_to_end_same_backbone", -23.47, "same_backbone_end_to_end"),
    ]
    for arm, effect, note in prior:
        rows.append(
            {
                "category": "mandatory_prior_negative_result",
                "family": "direction20",
                "arm": arm,
                "capacity_alignment": note,
                "pairing_unit": "archived_report",
                "completed_runs": np.nan,
                "pool_mean_baseline_rmse_ugm3": np.nan,
                "pool_mean_o_locked_rmse_ugm3": np.nan,
                "mean_baseline_relative_to_o_percent": np.nan,
                "median_baseline_relative_to_o_percent": np.nan,
                "baseline_better_count": np.nan,
                "paired_total": np.nan,
                "information_set": "see direction20 archived protocol/report",
                "reported_our_reduction_percent": effect,
            }
        )
    return pd.DataFrame(rows), pd.concat(details, ignore_index=True)


def build_stratified(pair_details: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for (family, arm, level), group in pair_details.groupby(["family", "arm", column]):
        rows.append(
            {
                "family": family,
                "arm": arm,
                column: level,
                "pairs": len(group),
                "mean_baseline_rmse_ugm3": group["rmse_ugm3_baseline"].mean(),
                "mean_o_locked_rmse_ugm3": group["rmse_ugm3_o_locked"].mean(),
                "mean_baseline_relative_to_o_percent": group[
                    "baseline_relative_to_o_percent"
                ].mean(),
                "baseline_better_count": int(group["baseline_better"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _family_selection_rows(
    data: pd.DataFrame,
    o_locked: pd.DataFrame,
    family_name: str,
    arms: Iterable[str],
) -> pd.DataFrame:
    subset = data[data["variant"].isin(tuple(arms)) & data["status"].eq("completed")].copy()
    output = []
    for criterion, metric in (("validation", "best_valid_loss"), ("test", "rmse_ugm3")):
        means = (
            subset.groupby(["history", "horizon", "variant"], as_index=False)[metric]
            .mean()
            .sort_values(["history", "horizon", metric, "variant"])
        )
        selected = means.groupby(["history", "horizon"], as_index=False).first()
        selected = selected.rename(columns={"variant": "selected_arm", metric: "selection_value"})
        selected_rows = subset.merge(
            selected[["history", "horizon", "selected_arm", "selection_value"]],
            left_on=["history", "horizon", "variant"],
            right_on=["history", "horizon", "selected_arm"],
            how="inner",
        )
        paired = selected_rows.merge(
            o_locked[KEYS + ["rmse_ugm3"]],
            on=KEYS,
            how="inner",
            suffixes=("_baseline", "_o_locked"),
        )
        paired["family"] = family_name
        paired["criterion"] = criterion
        paired["baseline_relative_to_o_percent"] = 100.0 * (
            paired["rmse_ugm3_baseline"] - paired["rmse_ugm3_o_locked"]
        ) / paired["rmse_ugm3_o_locked"]
        output.append(paired)
    return pd.concat(output, ignore_index=True)


def build_selection_and_dual(
    new: pd.DataFrame, recurrent: pd.DataFrame, o_locked: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selection = pd.concat(
        [
            _family_selection_rows(
                new,
                o_locked,
                "Informer",
                [arm for arm in runner.ARMS if arm.startswith("informer_")],
            ),
            _family_selection_rows(
                new,
                o_locked,
                "TST",
                [arm for arm in runner.ARMS if arm.startswith("tst_")],
            ),
            _family_selection_rows(
                recurrent,
                o_locked,
                "GRU-LSTM",
                recurrent["variant"].unique(),
            ),
        ],
        ignore_index=True,
    )
    dual_rows = []
    for (family, criterion), group in selection.groupby(["family", "criterion"]):
        relative = group["baseline_relative_to_o_percent"]
        dual_rows.append(
            {
                "family": family,
                "criterion": criterion,
                "pairs": len(group),
                "mean_baseline_rmse_ugm3": group["rmse_ugm3_baseline"].mean(),
                "mean_o_locked_rmse_ugm3": group["rmse_ugm3_o_locked"].mean(),
                "mean_baseline_relative_to_o_percent": relative.mean(),
                "median_baseline_relative_to_o_percent": relative.median(),
                "baseline_better_count": int((relative < 0).sum()),
            }
        )
    return selection, pd.DataFrame(dual_rows)


def build_rank_correlations(new: pd.DataFrame, recurrent: pd.DataFrame) -> pd.DataFrame:
    families = {
        "Informer": new[new["family"].eq("informer")],
        "TST": new[new["family"].eq("tst")],
        "GRU-LSTM": recurrent,
    }
    rows = []
    for family, data in families.items():
        complete = data[data["status"].eq("completed")].copy()
        pooled = spearmanr(complete["best_valid_loss"], complete["rmse_ugm3"])
        within = []
        for _, group in complete.groupby(["history", "horizon"]):
            arm_means = group.groupby("variant")[["best_valid_loss", "rmse_ugm3"]].mean()
            if len(arm_means) >= 3:
                result = spearmanr(arm_means["best_valid_loss"], arm_means["rmse_ugm3"])
                if math.isfinite(float(result.statistic)):
                    within.append(float(result.statistic))
        rows.append(
            {
                "family": family,
                "completed_runs": len(complete),
                "pooled_spearman": float(pooled.statistic),
                "pooled_pvalue": float(pooled.pvalue),
                "within_config_count": len(within),
                "mean_within_config_spearman": float(np.mean(within)),
                "median_within_config_spearman": float(np.median(within)),
                "selection_warning": (
                    "validation ranking may diverge from test ranking"
                    if float(np.mean(within)) <= 0.2
                    else "positive rank agreement"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_independent_recalculation(new: pd.DataFrame, root: Path) -> pd.DataFrame:
    rows = []
    for row in new[new["status"].eq("completed")].itertuples(index=False):
        path = root / str(row.prediction_file)
        with np.load(path) as payload:
            prediction_scaled = payload["prediction_scaled"]
            target_scaled = payload["target_scaled"]
            prediction_ugm3 = payload["prediction_ugm3"]
            target_ugm3 = payload["target_ugm3"]
        mse_scaled = float(np.mean((prediction_scaled - target_scaled) ** 2))
        rmse_ugm3 = float(np.sqrt(np.mean((prediction_ugm3 - target_ugm3) ** 2)))
        rows.append(
            {
                "run_id": row.run_id,
                "prediction_file": row.prediction_file,
                "stored_mse_scaled": row.mse_scaled,
                "recomputed_mse_scaled": mse_scaled,
                "mse_relative_difference": _relative_difference(mse_scaled, row.mse_scaled),
                "stored_rmse_ugm3": row.rmse_ugm3,
                "recomputed_rmse_ugm3": rmse_ugm3,
                "rmse_relative_difference": _relative_difference(rmse_ugm3, row.rmse_ugm3),
                "prediction_finite": bool(np.isfinite(prediction_ugm3).all()),
            }
        )
    return pd.DataFrame(rows)


def build_parameter_audit(new: pd.DataFrame) -> pd.DataFrame:
    rows = []
    complete = new[new["status"].eq("completed")]
    for (arm, history, horizon), group in complete.groupby(["variant", "history", "horizon"]):
        total, trainable = runner.independent_parameter_count(
            str(arm), int(history), int(horizon), 0
        )
        recorded_total = sorted(group["total_parameter_count"].astype(int).unique())
        recorded_trainable = sorted(group["trainable_parameter_count"].astype(int).unique())
        registered = np.nan
        if arm in runner.INFORMER_REGISTERED_COUNTS:
            registered = runner.INFORMER_REGISTERED_COUNTS[str(arm)]
        elif (int(history), int(horizon)) in runner.TST_REGISTERED_COUNTS:
            registered = runner.TST_REGISTERED_COUNTS[(int(history), int(horizon))].get(
                str(arm), np.nan
            )
        rows.append(
            {
                "arm": arm,
                "history": history,
                "horizon": horizon,
                "independent_total": total,
                "independent_trainable": trainable,
                "recorded_total_values": json.dumps([int(v) for v in recorded_total]),
                "recorded_trainable_values": json.dumps([int(v) for v in recorded_trainable]),
                "protocol_registered_value_if_explicit": registered,
                "recorded_match": recorded_total == [total] and recorded_trainable == [trainable],
                "protocol_match_if_explicit": bool(
                    pd.isna(registered) or int(registered) == trainable
                ),
            }
        )
    return pd.DataFrame(rows)


def build_training_log_audit(new: pd.DataFrame, root: Path) -> pd.DataFrame:
    rows = []
    for row in new[new["status"].eq("completed")].itertuples(index=False):
        log = pd.read_csv(root / str(row.training_log_file))
        selected = log[log["epoch"].eq(int(row.best_epoch))]
        logged = float(selected.iloc[0]["valid_loss"]) if len(selected) == 1 else np.nan
        rows.append(
            {
                "run_id": row.run_id,
                "best_epoch": row.best_epoch,
                "stored_best_valid_loss": row.best_valid_loss,
                "logged_best_valid_loss": logged,
                "relative_difference": _relative_difference(logged, row.best_valid_loss),
                "matched": bool(
                    len(selected) == 1
                    and _relative_difference(logged, row.best_valid_loss) < 1e-12
                ),
            }
        )
    return pd.DataFrame(rows)


def _prediction_rmse_by_lead(path: Path) -> np.ndarray:
    with np.load(path) as payload:
        prediction = payload["prediction_ugm3"]
        target = payload["target_ugm3"]
    return np.sqrt(np.mean((prediction - target) ** 2, axis=(0, 1)))


def build_per_lead(
    new: pd.DataFrame,
    o_locked: pd.DataFrame,
    recurrent: pd.DataFrame,
    root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    horizons = {6, 12, 24}
    o_lookup: dict[tuple[int, int, int], tuple[np.ndarray, Path]] = {}
    for row in o_locked[o_locked["horizon"].isin(horizons)].itertuples(index=False):
        path = Path(row.prediction_path)
        o_lookup[(int(row.history), int(row.horizon), int(row.seed))] = (
            _prediction_rmse_by_lead(path),
            path,
        )
    rows = []
    sources = [
        (new, root, "variant"),
        (recurrent, GRU_LSTM_ROOT, "variant"),
    ]
    for data, source_root, arm_column in sources:
        for row in data[
            data["status"].eq("completed") & data["horizon"].isin(horizons)
        ].itertuples(index=False):
            key = (int(row.history), int(row.horizon), int(row.seed))
            if key not in o_lookup:
                continue
            prediction_file = Path(getattr(row, "prediction_path", ""))
            if not prediction_file.is_file():
                prediction_file = source_root / str(row.prediction_file)
            baseline_rmse = _prediction_rmse_by_lead(prediction_file)
            o_rmse = o_lookup[key][0]
            arm = str(getattr(row, arm_column))
            family = str(row.family)
            for lead, (baseline_value, o_value) in enumerate(
                zip(baseline_rmse, o_rmse), start=1
            ):
                rows.append(
                    {
                        "row_type": "run",
                        "history": key[0],
                        "horizon": key[1],
                        "seed": key[2],
                        "family": family,
                        "arm": arm,
                        "lead": lead,
                        "baseline_rmse_ugm3": float(baseline_value),
                        "o_locked_rmse_ugm3": float(o_value),
                        "baseline_relative_to_o_percent": float(
                            100.0 * (baseline_value - o_value) / o_value
                        ),
                        "o_locked_wins": bool(o_value < baseline_value),
                    }
                )
    detail = pd.DataFrame(rows)
    summaries = []
    for (family, arm, horizon, lead), group in detail.groupby(
        ["family", "arm", "horizon", "lead"]
    ):
        summaries.append(
            {
                "row_type": "summary",
                "history": "pooled",
                "horizon": horizon,
                "seed": "pooled",
                "family": family,
                "arm": arm,
                "lead": lead,
                "baseline_rmse_ugm3": group["baseline_rmse_ugm3"].mean(),
                "o_locked_rmse_ugm3": group["o_locked_rmse_ugm3"].mean(),
                "baseline_relative_to_o_percent": group[
                    "baseline_relative_to_o_percent"
                ].mean(),
                "o_locked_wins": bool(
                    group["o_locked_rmse_ugm3"].mean()
                    < group["baseline_rmse_ugm3"].mean()
                ),
            }
        )
    summary = pd.DataFrame(summaries)
    win_rows = []
    for (family, arm, horizon), group in summary.groupby(["family", "arm", "horizon"]):
        wins = int(group["o_locked_wins"].sum())
        win_rows.append(
            {
                "family": family,
                "arm": arm,
                "horizon": horizon,
                "o_locked_winning_leads": wins,
                "o_locked_losing_leads": int(horizon) - wins,
                "total_leads": int(horizon),
            }
        )
    combined = pd.concat([detail, summary], ignore_index=True)
    return combined, summary, pd.DataFrame(win_rows)


def build_capacity_curve(
    new: pd.DataFrame, recurrent: pd.DataFrame, o_locked: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for source, category in ((new, "new"), (recurrent, "reused")):
        complete = source[source["status"].eq("completed")]
        for (family, arm, history, horizon, parameters), group in complete.groupby(
            ["family", "variant", "history", "horizon", "trainable_parameter_count"]
        ):
            rows.append(
                {
                    "category": category,
                    "family": family,
                    "arm": arm,
                    "history": history,
                    "horizon": horizon,
                    "trainable_parameter_count": int(parameters),
                    "mean_rmse_ugm3": group["rmse_ugm3"].mean(),
                    "std_rmse_ugm3": group["rmse_ugm3"].std(ddof=1),
                    "runs": len(group),
                    "capacity_alignment": (
                        "non_aligned_as_is" if family == "tst" else "capacity_curve"
                    ),
                }
            )
    for (history, horizon, parameters), group in o_locked.groupby(
        ["history", "horizon", "trainable_parameter_count"]
    ):
        rows.append(
            {
                "category": "o_locked",
                "family": "o_locked",
                "arm": "o_locked",
                "history": history,
                "horizon": horizon,
                "trainable_parameter_count": int(parameters),
                "mean_rmse_ugm3": group["rmse_ugm3"].mean(),
                "std_rmse_ugm3": group["rmse_ugm3"].std(ddof=1),
                "runs": len(group),
                "capacity_alignment": "our_budget_point_18_stations",
            }
        )
    return pd.DataFrame(rows)


def build_budget_neighborhood(capacity: pd.DataFrame) -> pd.DataFrame:
    in_budget = capacity[capacity["trainable_parameter_count"].between(4588, 5418)].copy()
    informer_exact = capacity[capacity["arm"].eq("informer_d12_e2")]
    result = pd.concat([in_budget, informer_exact], ignore_index=True).drop_duplicates()
    return result.sort_values(["history", "horizon", "family", "trainable_parameter_count"])


def _all_o_metadata_have_18() -> tuple[bool, str]:
    counts = []
    for path in sorted(O_LOCKED_ROOT.glob("*h_*h/dataset_metadata.json")):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        counts.append((path.parent.name, len(metadata["station_ids"])))
    return bool(counts and all(count == 18 for _, count in counts)), json.dumps(counts)


def build_compliance(
    new: pd.DataFrame,
    recalculation: pd.DataFrame,
    parameter_audit: pd.DataFrame,
    log_audit: pd.DataFrame,
    root: Path,
) -> pd.DataFrame:
    expected = runner.expected_identities()
    actual = set(new["run_id"].astype(str))
    duplicate_count = int(new["run_id"].duplicated().sum())
    terminal = new["status"].isin(runner.TERMINAL_STATUSES)
    completed = new[new["status"].eq("completed")]

    center_ok = True
    center_details = []
    for path in sorted((root / "metadata").glob("dataset_*h_*h.json")):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        station_ids = [int(value) for value in metadata["station_ids"]]
        resolved = station_ids.index(runner.CENTER_STATION_ID)
        center_details.append((path.name, resolved, metadata["center_station_idx"]))
        center_ok &= resolved == int(metadata["center_station_idx"]) == 9
    center_ok &= bool(
        len(completed)
        and completed["input_channels"].eq(1).all()
        and completed["selected_channel_index"].eq(9).all()
    )
    o_18_ok, o_18_details = _all_o_metadata_have_18()
    finite_ok = bool(
        len(completed)
        and np.isfinite(completed[METRIC_COLUMNS].to_numpy(dtype=float)).all()
    )
    parameter_ok = bool(
        len(parameter_audit)
        and parameter_audit["recorded_match"].all()
        and parameter_audit["protocol_match_if_explicit"].all()
    )
    recalculation_ok = bool(
        len(recalculation) == len(completed)
        and recalculation["prediction_finite"].all()
        and recalculation["rmse_relative_difference"].max() < 1e-9
        and recalculation["mse_relative_difference"].max() < 1e-9
    )
    logs_ok = bool(len(log_audit) == len(completed) and log_audit["matched"].all())
    split_ok = bool(
        len(completed)
        and completed["selection_split"].eq("valid").all()
        and completed["evaluation_split"].eq("test").all()
        and completed["test_evaluation_count"].eq(1).all()
    )
    reproduction_path = root / "verification/reproduction_gate.json"
    wrapper_path = root / "verification/wrapper_gate.json"
    reproduction = (
        json.loads(reproduction_path.read_text(encoding="utf-8"))
        if reproduction_path.is_file()
        else {}
    )
    wrapper = (
        json.loads(wrapper_path.read_text(encoding="utf-8"))
        if wrapper_path.is_file()
        else {}
    )
    semantic_ok = bool(reproduction.get("passed") and wrapper.get("passed"))
    checks = [
        (1, "运行身份数", actual == expected and len(new) == 448 and duplicate_count == 0,
         f"actual={len(new)}, expected=448, missing={len(expected-actual)}, extra={len(actual-expected)}, duplicates={duplicate_count}"),
        (2, "全部身份为终态", bool(len(new) == 448 and terminal.all()),
         json.dumps(new["status"].value_counts().to_dict())),
        (3, "单站点输入与中心索引", center_ok, json.dumps(center_details)),
        (4, "O-locked 输入通道为18", o_18_ok, o_18_details),
        (5, "completed 指标有限", finite_ok, f"completed={len(completed)}"),
        (6, "参数量独立重构一致", parameter_ok, f"rows={len(parameter_audit)}"),
        (7, "预测独立复算一致", recalculation_ok,
         f"max_rmse_rel={recalculation['rmse_relative_difference'].max() if len(recalculation) else np.nan}"),
        (8, "best_valid_loss 与日志一致", logs_ok,
         f"max_rel={log_audit['relative_difference'].max() if len(log_audit) else np.nan}"),
        (9, "选择/评估划分与次数", split_ok, "selection=valid; evaluation=test; count=1"),
        (10, "语义等价门", semantic_ok,
         json.dumps({"reproduction": reproduction, "wrapper": wrapper}, ensure_ascii=False)),
    ]
    return pd.DataFrame(
        [
            {"item": item, "check": name, "passed": bool(passed), "details": details}
            for item, name, passed, details in checks
        ]
    )


def make_figures(
    capacity: pd.DataFrame,
    lead_summary: pd.DataFrame,
    pair_details: pd.DataFrame,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
            "savefig.dpi": 450,
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(4, 5, figsize=(14, 10), sharex=False, sharey=False)
    for ax, (history, horizon) in zip(axes.flat, runner.TASK_GRID):
        subset = capacity[
            capacity["history"].eq(history) & capacity["horizon"].eq(horizon)
        ]
        for family in ("informer", "tst", "gru", "lstm"):
            family_rows = subset[subset["family"].eq(family)].sort_values(
                "trainable_parameter_count"
            )
            if len(family_rows):
                label = "TST (non-aligned)" if family == "tst" else family.upper()
                ax.plot(
                    family_rows["trainable_parameter_count"],
                    family_rows["mean_rmse_ugm3"],
                    marker="o",
                    markersize=2.5,
                    linewidth=0.8,
                    color=COLORS[family],
                    label=label,
                )
        ours = subset[subset["family"].eq("o_locked")]
        ax.scatter(
            ours["trainable_parameter_count"],
            ours["mean_rmse_ugm3"],
            marker="*",
            s=28,
            color=COLORS["o_locked"],
            label="O-locked (18 stations)",
            zorder=4,
        )
        ax.axvspan(4588, 5418, color="#BBBBBB", alpha=0.12)
        ax.set_xscale("log")
        ax.set_title(f"L={history}, H={horizon}")
        if ax in axes[:, 0]:
            ax.set_ylabel("RMSE (µg/m³)")
        if ax in axes[-1, :]:
            ax.set_xlabel("Trainable parameters (log)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(figures_dir / "capacity_curves.png", dpi=450)
    fig.savefig(figures_dir / "capacity_curves.svg")
    plt.close(fig)

    new_leads = lead_summary[
        lead_summary["arm"].isin(runner.ARMS)
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    palette = ["#0077BB", "#33BBEE", "#009988", "#EE7733", "#CC3311", "#EE3377", "#BBBB44"]
    for ax, horizon in zip(axes, (6, 12, 24)):
        subset = new_leads[new_leads["horizon"].eq(horizon)]
        o_curve = subset.groupby("lead")["o_locked_rmse_ugm3"].mean()
        ax.plot(o_curve.index, o_curve.values, color=COLORS["o_locked"], linewidth=2.2, label="O-locked")
        for color, (arm, group) in zip(palette, subset.groupby("arm", sort=True)):
            ax.plot(group["lead"], group["baseline_rmse_ugm3"], color=color, linewidth=1, label=arm)
        ax.set_title(f"H={horizon}")
        ax.set_xlabel("Lead")
        ax.set_ylabel("RMSE (µg/m³)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=6)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(figures_dir / "per_lead_curves.png", dpi=450)
    fig.savefig(figures_dir / "per_lead_curves.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, horizon in zip(axes, (6, 12, 24)):
        subset = new_leads[new_leads["horizon"].eq(horizon)]
        for color, (arm, group) in zip(palette, subset.groupby("arm", sort=True)):
            ax.plot(
                group["lead"],
                -group["baseline_relative_to_o_percent"],
                color=color,
                linewidth=1,
                label=arm,
            )
        ax.axhline(0, color="#444444", linewidth=0.8)
        ax.set_title(f"H={horizon}")
        ax.set_xlabel("Lead")
        ax.set_ylabel("O-locked reduction (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=6)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(figures_dir / "per_lead_relative.png", dpi=450)
    fig.savefig(figures_dir / "per_lead_relative.svg")
    plt.close(fig)

    heat = (
        pair_details[pair_details["arm"].isin(runner.ARMS)]
        .groupby(["arm", "history", "horizon"])["baseline_relative_to_o_percent"]
        .mean()
        .unstack(["history", "horizon"])
        .reindex(index=runner.ARMS)
    )
    fig, ax = plt.subplots(figsize=(12, 3.8))
    image = ax.imshow(-heat.to_numpy(), cmap="RdBu_r", aspect="auto", vmin=-10, vmax=10)
    ax.set_yticks(range(len(heat.index)), heat.index)
    ax.set_xticks(
        range(len(heat.columns)),
        [f"{history}→{horizon}" for history, horizon in heat.columns],
        rotation=60,
        ha="right",
    )
    ax.set_title("O-locked paired RMSE reduction (%)")
    fig.colorbar(image, ax=ax, label="Reduction (%)")
    fig.tight_layout()
    fig.savefig(figures_dir / "paired_reduction_heatmap.png", dpi=450)
    fig.savefig(figures_dir / "paired_reduction_heatmap.svg")
    plt.close(fig)


def _write_manifests(figures_dir: Path, tables_dir: Path) -> None:
    (figures_dir / "data-manifest.md").write_text(
        """# Baseline v2 figure data manifest

| Figure | Data | Real/mock | Source script | Outputs |
|---|---|---|---|---|
| Capacity curves | `tables/baseline_v2/capacity_curve.csv` | real | `summarize_baseline_v2.py` | PNG/SVG |
| Per-lead curves | `tables/baseline_v2/per_lead.csv` | real | `summarize_baseline_v2.py` | PNG/SVG |
| Per-lead relative curves | `tables/baseline_v2/per_lead.csv` | real | `summarize_baseline_v2.py` | PNG/SVG |
| Paired reduction heatmap | `tables/baseline_v2/paired_detail.csv` | real | `summarize_baseline_v2.py` | PNG/SVG |

All figures use completed formal runs and archived, checksum-verified reuse artifacts; no mock data are used.
""",
        encoding="utf-8",
    )
    (tables_dir / "table-schema.md").write_text(
        """# Baseline v2 table schema

| Table | Purpose | Unit / aggregation | Data source |
|---|---|---|---|
| main_table | All new arms, reused capacity curves, traditional baselines, mandatory prior negatives | common `(L,H,seed)`; deterministic arms by config | formal + archives |
| capacity_curve | Parameter/RMSE points; TST remains per `(L,H)` and non-aligned | arm × config mean | formal + archives |
| per_lead | Lead-wise RMSE and paired difference | run and pooled lead | saved predictions |
| stratified_H/L | Horizon/history stratification | common paired subset | paired detail |
| rank_correlation | Validation/test rank agreement | pooled and within-config Spearman | completed runs |
| dual_criterion | Validation-selected and test-selected upper-bound views | common paired subset | completed runs |
| compliance_self_check | Protocol §8 ten gates | boolean gate | independent audits |
""",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    (args.root / "verification").mkdir(parents=True, exist_ok=True)

    new = load_new_results(args.root)
    o_locked = load_o_locked()
    recurrent = load_gru_lstm()
    traditional = load_traditional()

    main_table, pair_details = build_main_table(new, o_locked, recurrent, traditional)
    stratified_h = build_stratified(pair_details, "horizon")
    stratified_l = build_stratified(pair_details, "history")
    selection, dual = build_selection_and_dual(new, recurrent, o_locked)
    rank = build_rank_correlations(new, recurrent)
    recalculation = build_independent_recalculation(new, args.root)
    parameter_audit = build_parameter_audit(new)
    log_audit = build_training_log_audit(new, args.root)
    per_lead, lead_summary, lead_wins = build_per_lead(new, o_locked, recurrent, args.root)
    capacity = build_capacity_curve(new, recurrent, o_locked)
    budget = build_budget_neighborhood(capacity)
    compliance = build_compliance(
        new, recalculation, parameter_audit, log_audit, args.root
    )

    _write_table(main_table, args.tables_dir / "main_table.csv", markdown=True)
    _write_table(pair_details, args.tables_dir / "paired_detail.csv")
    _write_table(capacity, args.tables_dir / "capacity_curve.csv")
    _write_table(budget, args.tables_dir / "budget_neighborhood.csv", markdown=True)
    _write_table(per_lead, args.tables_dir / "per_lead.csv", markdown=True)
    _write_table(lead_wins, args.tables_dir / "per_lead_win_counts.csv", markdown=True)
    _write_table(stratified_h, args.tables_dir / "stratified_H.csv")
    _write_table(stratified_l, args.tables_dir / "stratified_L.csv")
    _write_table(rank, args.tables_dir / "rank_correlation.csv", markdown=True)
    _write_table(dual, args.tables_dir / "dual_criterion.csv", markdown=True)
    _write_table(compliance, args.tables_dir / "compliance_self_check.csv", markdown=True)
    _write_table(
        recalculation,
        args.tables_dir / "independent_recalculation.csv",
    )
    _write_table(parameter_audit, args.root / "verification/parameter_audit.csv")
    _write_table(log_audit, args.root / "verification/training_log_audit.csv")
    selection.to_csv(args.root / "selection.csv", index=False)

    make_figures(capacity, lead_summary, pair_details, args.figures_dir)
    _write_manifests(args.figures_dir, args.tables_dir)

    statuses = new["status"].value_counts().to_dict()
    summary = {
        "formal_identity_count": int(len(new)),
        "expected_identity_count": 448,
        "status_counts": {str(key): int(value) for key, value in statuses.items()},
        "completed_count": int(new["status"].eq("completed").sum()),
        "failed_or_excluded_count": int((~new["status"].eq("completed")).sum()),
        "training_seconds_sum": float(
            pd.to_numeric(new.get("training_seconds"), errors="coerce").sum()
        ),
        "all_compliance_passed": bool(compliance["passed"].all()),
        "new_arm_results": main_table[main_table["category"].eq("new")].to_dict("records"),
        "rank_correlations": rank.to_dict("records"),
        "dual_criterion": dual.to_dict("records"),
        "per_lead_win_counts": lead_wins[lead_wins["arm"].isin(runner.ARMS)].to_dict("records"),
        "max_independent_rmse_relative_difference": float(
            recalculation["rmse_relative_difference"].max()
        ),
    }
    (args.root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        f"BASELINE_V2_SUMMARY_DONE identities={len(new)} "
        f"compliance={compliance['passed'].sum()}/{len(compliance)}"
    )


if __name__ == "__main__":
    main()
