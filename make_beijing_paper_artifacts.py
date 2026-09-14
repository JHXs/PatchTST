"""Generate paper-ready tables and figures for the Beijing 1013 first-innovation results.

Everything is derived from committed evidence under
`experiments/results/st_patchtst_ablation/`; nothing is re-trained or re-selected.
The final confirmation attempt is `frozen_topk5_2047_2051` (variant
`st_sparse_station_bias_delta_forecast`, k=5), and the four earlier attempts are shown
as well: the project requires that failed rounds are not hidden.

Usage:
    python make_beijing_paper_artifacts.py \
        --results experiments/results/st_patchtst_ablation \
        --out-tables tables/beijing --out-figures figures/beijing
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLOR_BASE = "#4c72b0"
COLOR_SPATIAL = "#dd8452"
COLOR_FAIL = "#c44e52"
COLOR_PASS = "#55a868"
TASK_LABELS = {"24h_to_1h": "24$\\rightarrow$1", "168h_to_6h": "168$\\rightarrow$6"}
FINAL_ATTEMPT = "frozen_topk5_2047_2051"
FINAL_ROUND_DIRS = {"24h_to_1h": "stability_confirmation_topk5_24h_1h", "168h_to_6h": "stability_confirmation_topk5_168h_6h"}
SPATIAL = "st_sparse_station_bias_delta_forecast"


def to_markdown_table(frame: pd.DataFrame, floatfmt: str = ".4f") -> str:
    def render(value) -> str:
        if isinstance(value, (float, np.floating)):
            return "nan" if np.isnan(value) else format(float(value), floatfmt)
        if isinstance(value, (bool, np.bool_)):
            return "true" if value else "false"
        return str(value)

    columns = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(render(row[c]) for c in frame.columns) + " |")
    return "\n".join(lines) + "\n"


def write_table(frame: pd.DataFrame, out_dir: Path, stem: str, latex: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / f"{stem}.csv", index=False, float_format="%.4f")
    (out_dir / f"{stem}.md").write_text(to_markdown_table(frame), encoding="utf-8")
    if latex:
        (out_dir / f"{stem}.tex").write_text(
            frame.to_latex(index=False, float_format="%.4f", escape=True, caption=stem, label=f"tab:{stem}"),
            encoding="utf-8",
        )


def build_tables(results: Path, out_dir: Path, release: dict) -> dict[str, pd.DataFrame]:
    paired = pd.read_csv(results / "stability_confirmation_paired.csv")
    summary = pd.read_csv(results / "stability_confirmation_summary.csv")

    # B1: final gates
    final = summary[summary.attempt == FINAL_ATTEMPT].copy()
    b1 = final[["task", "runs", "baseline_rmse_ugm3_mean", "candidate_rmse_ugm3_mean", "rmse_reduction_percent_mean",
                "rmse_reduction_percent_std", "rmse_reduction_bootstrap_ci95_low", "rmse_reduction_bootstrap_ci95_high",
                "rmse_improved_seed_count", "one_sided_exact_sign_p", "practical_threshold_percent",
                "strict_task_gate_pass", "baseline_mae_ugm3_mean", "candidate_mae_ugm3_mean", "mae_improved_seed_count"]].copy()
    b1["task"] = b1["task"].map(TASK_LABELS)
    write_table(b1, out_dir, "B1_final_gates", latex=True)

    # B2: per-seed pairs of the final attempt
    b2 = paired[paired.attempt == FINAL_ATTEMPT].copy()
    b2["task"] = b2["task"].map(TASK_LABELS)
    b2 = b2[["task", "seed", "candidate_variant", "baseline_rmse_ugm3", "candidate_rmse_ugm3", "rmse_reduction_percent",
             "baseline_mae_ugm3", "candidate_mae_ugm3", "mae_reduction_percent"]]
    write_table(b2, out_dir, "B2_final_per_seed", latex=True)

    # B3: every attempt (failed rounds included)
    b3 = summary[["attempt", "task", "runs", "rmse_reduction_percent_mean", "rmse_reduction_percent_std",
                  "rmse_improved_seed_count", "one_sided_exact_sign_p", "practical_threshold_percent",
                  "direction_gate_pass", "practical_gate_pass", "strict_task_gate_pass"]].copy()
    b3["task"] = b3["task"].map(TASK_LABELS)
    b3["is_final"] = b3["attempt"].eq(FINAL_ATTEMPT)
    write_table(b3, out_dir, "B3_all_attempts", latex=True)

    # B4: neighbor interventions on the final round
    rows = []
    for task, dirname in FINAL_ROUND_DIRS.items():
        raw = pd.read_csv(results / dirname / "raw_metrics.csv")
        for variant, group in raw.groupby("variant"):
            rows.append({
                "task": TASK_LABELS[task],
                "variant": variant,
                "seeds": len(group),
                "rmse": group.rmse_ugm3.mean(),
                "rmse_std": group.rmse_ugm3.std(),
                "disable_neighbor_rmse": group.disable_neighbor_rmse_ugm3.mean(),
                "zero_neighbor_rmse": group.zero_neighbor_rmse_ugm3.mean(),
                "shuffle_neighbor_rmse": group.shuffle_neighbor_rmse_ugm3.mean(),
                "center_copy_rmse": group.center_copy_neighbor_rmse_ugm3.mean(),
            })
    b4 = pd.DataFrame(rows)
    for column in ("disable_neighbor_rmse", "zero_neighbor_rmse", "shuffle_neighbor_rmse", "center_copy_rmse"):
        b4[f"{column}_delta_percent"] = 100 * (b4[column] - b4.rmse) / b4.rmse
    write_table(b4, out_dir, "B4_neighbor_interventions", latex=True)

    # B5: gate diagnostics of the final round
    rows = []
    for task, dirname in FINAL_ROUND_DIRS.items():
        s = pd.read_csv(results / dirname / "summary_metrics.csv")
        for _, row in s.iterrows():
            rows.append({
                "task": TASK_LABELS[task], "variant": row["variant"], "runs": int(row["runs"]),
                "alpha_mean": row["alpha_mean"], "spatial_residual_rms_ratio_mean": row["spatial_residual_rms_ratio_mean"],
                "gate_normalized_entropy_mean": row["gate_normalized_entropy_mean"],
                "null_neighbor_weight_mean": row["null_neighbor_weight_mean_mean"],
                "gate_sample_std_mean": row["gate_sample_std_mean_mean"],
                "spatial_forecast_weight_norm_mean": row["spatial_forecast_weight_norm_mean"],
            })
    write_table(pd.DataFrame(rows), out_dir, "B5_gate_diagnostics", latex=True)

    # B6: top-k validation selection
    b6 = pd.read_csv(results / "topk_validation_selection.csv")
    b6["task"] = b6["task"].map(TASK_LABELS)
    scores = pd.read_csv(results / "topk_validation_scores.csv")
    write_table(b6, out_dir, "B6_topk_validation")
    write_table(scores, out_dir, "B6b_topk_scores")

    # B7: lag diagnostics (negative result)
    b7 = pd.read_csv(results / "lag_diagnostics" / "lag_cv_summary.csv")
    write_table(b7, out_dir, "B7_lag_diagnostics")

    return {"paired": paired, "summary": summary, "topk": b6, "lag": b7, "b4": b4}


def figure_final_per_seed(paired: pd.DataFrame, out: Path) -> None:
    final = paired[paired.attempt == FINAL_ATTEMPT]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, task in zip(axes, ["24h_to_1h", "168h_to_6h"]):
        sel = final[final.task == task].sort_values("seed")
        threshold = 1.0 if task == "24h_to_1h" else 0.5
        x = np.arange(len(sel))
        ax.bar(x, sel.rmse_reduction_percent, color=COLOR_SPATIAL, edgecolor="black", linewidth=0.4)
        ax.axhline(threshold, color="black", linestyle="--", linewidth=1.0, label=f"pre-registered threshold ({threshold}%)")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x, [str(int(s)) for s in sel.seed])
        ax.set_xlabel("confirmation seed")
        ax.set_ylabel("paired RMSE reduction (%)")
        ax.set_title(TASK_LABELS[task], fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.annotate(f"mean {sel.rmse_reduction_percent.mean():.3f}%\n{int((sel.rmse_reduction_percent > 0).sum())}/{len(sel)} seeds improved",
                    xy=(0.03, 0.94), xycoords="axes fraction", fontsize=8, va="top")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Beijing 1013 final confirmation: per-seed paired RMSE reduction (seeds 2047–2051)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "BF1_final_per_seed.pdf")
    fig.savefig(out / "BF1_final_per_seed.png", dpi=300)
    plt.close(fig)


def figure_attempts(summary: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), sharey=True)
    order = ["joint_2027_2031", "frozen_2032_2036", "anchored_2037_2041", "frozen_sparse_station_2042_2046", FINAL_ATTEMPT]
    for ax, task in zip(axes, ["24h_to_1h", "168h_to_6h"]):
        sel = summary[summary.task == task].set_index("attempt").loc[order]
        threshold = 1.0 if task == "24h_to_1h" else 0.5
        x = np.arange(len(sel))
        # Colour by the pre-registered strict task gate (direction + practical effect), not by the mean alone.
        colors = [COLOR_PASS if bool(passed) else COLOR_FAIL for passed in sel.strict_task_gate_pass]
        ax.bar(x, sel.rmse_reduction_percent_mean, color=colors, edgecolor="black", linewidth=0.4)
        ax.errorbar(x, sel.rmse_reduction_percent_mean, yerr=sel.rmse_reduction_percent_std, fmt="none", ecolor="black", capsize=3, linewidth=1)
        ax.axhline(threshold, color="black", linestyle="--", linewidth=1.0, label=f"practical threshold ({threshold}%)")
        ax.axhline(0, color="black", linewidth=0.8)
        for xi, (_, row) in zip(x, sel.iterrows()):
            ax.annotate(f"{int(row.rmse_improved_seed_count)}/{int(row.runs)}", (xi, row.rmse_reduction_percent_mean),
                        ha="center", va="bottom", fontsize=7, xytext=(0, 6), textcoords="offset points")
        ax.set_xticks(x, [name.replace("_", "\n") for name in sel.index], fontsize=7)
        ax.set_ylabel("mean paired RMSE reduction (%)")
        ax.set_title(TASK_LABELS[task], fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR_PASS), plt.Rectangle((0, 0), 1, 1, color=COLOR_FAIL)]
    axes[0].legend(handles + [plt.Line2D([], [], color="black", linestyle="--")],
                   ["strict gate passed", "strict gate failed", "practical threshold"], fontsize=7, loc="upper left")
    fig.suptitle("All five confirmation attempts (labels: improved seeds / runs); earlier failures are shown deliberately", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "BF2_attempts.pdf")
    fig.savefig(out / "BF2_attempts.png", dpi=300)
    plt.close(fig)


def figure_interventions(results: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for ax, (task, dirname) in zip(axes, FINAL_ROUND_DIRS.items()):
        raw = pd.read_csv(results / dirname / "raw_metrics.csv")
        spatial = raw[raw.variant == SPATIAL]
        degraded = raw[raw.variant == "degraded_patchtst"]
        labels = ["normal", "no neighbour", "zeroed", "shuffled", "centre copy"]
        values = [spatial.rmse_ugm3.mean(), spatial.disable_neighbor_rmse_ugm3.mean(), spatial.zero_neighbor_rmse_ugm3.mean(),
                  spatial.shuffle_neighbor_rmse_ugm3.mean(), spatial.center_copy_neighbor_rmse_ugm3.mean()]
        errors = [spatial.rmse_ugm3.std(), spatial.disable_neighbor_rmse_ugm3.std(), spatial.zero_neighbor_rmse_ugm3.std(),
                  spatial.shuffle_neighbor_rmse_ugm3.std(), spatial.center_copy_neighbor_rmse_ugm3.std()]
        x = np.arange(len(labels))
        ax.bar(x, values, yerr=errors, capsize=3, color=[COLOR_SPATIAL] + [COLOR_BASE] * 4, edgecolor="black", linewidth=0.4)
        ax.axhline(degraded.rmse_ugm3.mean(), color="black", linestyle="--", linewidth=1.0, label="degraded baseline")
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_ylabel("RMSE ($\\mu g/m^3$)")
        ax.set_title(TASK_LABELS[task], fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("Neighbour reliance of the frozen spatial branch (final round; mean $\\pm$ sd over 5 seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "BF3_interventions.pdf")
    fig.savefig(out / "BF3_interventions.png", dpi=300)
    plt.close(fig)


def figure_topk(topk: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for ax, task in zip(axes, ["24$\\rightarrow$1", "168$\\rightarrow$6"]):
        sel = topk[topk.task == task].sort_values("top_k")
        ax.plot(sel.top_k, sel.normalized_margin, marker="o", color=COLOR_SPATIAL, label="normalised margin")
        ax.plot(sel.top_k, sel.mean_rmse_reduction_percent, marker="s", color=COLOR_BASE, label="mean RMSE reduction (%)")
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="margin = 1")
        chosen = sel.loc[sel.normalized_margin.idxmax(), "top_k"]
        ax.axvline(chosen, color="gray", linestyle=":", linewidth=1.2, label=f"selected k={int(chosen)}")
        ax.set_xlabel("k (sparse neighbours)")
        ax.set_title(task, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("validation score")
    axes[0].legend(fontsize=7)
    fig.suptitle("Top-k selection on the 2024–2026 validation segment (structure selection, not confirmation)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "BF4_topk_selection.pdf")
    fig.savefig(out / "BF4_topk_selection.png", dpi=300)
    plt.close(fig)


def figure_lag(lag: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(lag.lag_hours, lag.rmse_improvement_percent_mean, marker="o", color=COLOR_BASE)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle=":", linewidth=1.2)
    for _, row in lag.iterrows():
        ax.annotate(f"{int(row.positive_station_count)}", (row.lag_hours, row.rmse_improvement_percent_mean),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("assumed neighbour lag (hours)")
    ax.set_ylabel("RMSE improvement (%)")
    ax.set_title("Neighbour-lag diagnostic (labels: stations improved); trained-window only", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "BF5_lag_diagnostics.pdf")
    fig.savefig(out / "BF5_lag_diagnostics.png", dpi=300)
    plt.close(fig)


def figure_training_curves(results: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), sharey=False)
    for ax, (task, dirname) in zip(axes, FINAL_ROUND_DIRS.items()):
        log_dir = results / dirname / "training_logs"
        for path in sorted(log_dir.glob(f"{SPATIAL}_seed*.csv")):
            frame = pd.read_csv(path)
            ax.plot(frame.epoch, frame.train_loss, color=COLOR_SPATIAL, alpha=0.35, linewidth=1.0)
            ax.plot(frame.epoch, frame.valid_loss, color=COLOR_BASE, alpha=0.35, linewidth=1.0)
        ax.plot([], [], color=COLOR_SPATIAL, label="train loss (5 seeds)")
        ax.plot([], [], color=COLOR_BASE, label="valid loss (5 seeds)")
        ax.set_xlabel("epoch")
        ax.set_ylabel("scaled MSE")
        ax.set_title(TASK_LABELS[task], fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("Training and validation loss of the frozen spatial variant (final round)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "BF6_training_curves.pdf")
    fig.savefig(out / "BF6_training_curves.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="experiments/results/st_patchtst_ablation")
    parser.add_argument("--out-tables", default="tables/beijing")
    parser.add_argument("--out-figures", default="figures/beijing")
    args = parser.parse_args()

    results = Path(args.results)
    out_tables = Path(args.out_tables)
    out_figures = Path(args.out_figures)
    out_figures.mkdir(parents=True, exist_ok=True)

    release = json.loads((results / "stability_release_status.json").read_text(encoding="utf-8"))
    frames = build_tables(results, out_tables, release)
    ablation = build_structure_ablation(results, out_tables)
    figure_final_per_seed(frames["paired"], out_figures)
    figure_attempts(frames["summary"], out_figures)
    figure_interventions(results, out_figures)
    figure_topk(frames["topk"], out_figures)
    figure_lag(frames["lag"], out_figures)
    figure_training_curves(results, out_figures)
    figure_structure_ablation(ablation, out_figures)

    print(json.dumps({
        "tables": sorted(p.name for p in out_tables.glob("*.csv")),
        "figures": sorted(p.name for p in out_figures.glob("*.pdf")),
        "final_attempt": release["final_attempt"],
        "strict_release_gate_pass": release["strict_release_gate_pass"],
    }, ensure_ascii=False, indent=2))




# ---------------------------------------------------------------------------
# Structure ablation (development / selection runs; seeds 2024-2026)
#
# The project trained the spatial variants under two different protocols:
#   * block "end_to_end_dev": rounds 1-2, no frozen backbone, no degraded init;
#   * block "frozen_production": rounds 3-6 and the confirmation, frozen backbone
#     with degraded initialisation.
# A candidate is only comparable to the degraded baseline trained in the same
# directory (same protocol, same seeds); comparisons across blocks are confounded
# by the training protocol and are reported separately on purpose.
# ---------------------------------------------------------------------------

# Every row below was evaluated on the TEST split. Development rounds 3-6 ran on
# the VALIDATION split and are therefore listed separately (B9); mixing them into
# this table would compare different evaluation sets.
ABLATION_SOURCES: list[dict[str, object]] = [
    {"block": "end_to_end_dev", "task": "24h_to_1h", "dir": "24h_1h", "variants": ["st_center_only", "st_pairwise"], "k": ""},
    {"block": "end_to_end_dev", "task": "24h_to_1h", "dir": "24h_1h_delta", "variants": ["st_pairwise_delta"], "k": ""},
    {"block": "end_to_end_dev", "task": "24h_to_1h", "dir": "round2_forecast_24h_1h", "variants": ["st_pairwise_delta_forecast", "st_sparse_delta_forecast"], "k": 7},
    {"block": "end_to_end_dev", "task": "24h_to_1h", "dir": "round2_station_bias_24h_1h", "variants": ["st_station_bias_delta_forecast"], "k": ""},
    {"block": "end_to_end_dev", "task": "168h_to_6h", "dir": "168h_6h_delta", "variants": ["st_pairwise_delta"], "k": ""},
    {"block": "end_to_end_dev", "task": "168h_to_6h", "dir": "round2_forecast_168h_6h", "variants": ["st_pairwise_delta_forecast", "st_sparse_delta_forecast"], "k": 7},
    {"block": "end_to_end_dev", "task": "168h_to_6h", "dir": "round2_station_bias_168h_6h", "variants": ["st_station_bias_delta_forecast"], "k": ""},
    {"block": "frozen_production", "task": "24h_to_1h", "dir": "stability_confirmation_topk5_24h_1h", "variants": ["st_sparse_station_bias_delta_forecast"], "k": 5},
    {"block": "frozen_production", "task": "168h_to_6h", "dir": "stability_confirmation_topk5_168h_6h", "variants": ["st_sparse_station_bias_delta_forecast"], "k": 5},
]

# Development selection rounds evaluated on the VALIDATION split (seeds 2024-2026).
DEV_SELECTION_SOURCES: list[dict[str, object]] = [
    {"task": "24h_to_1h", "dir": "round3_selection_frozen_station_bias_24h_1h", "variants": ["st_station_bias_delta_forecast"], "note": "frozen backbone"},
    {"task": "24h_to_1h", "dir": "round4_selection_anchored_station_bias_24h_1h", "variants": ["st_station_bias_delta_forecast"], "note": "anchored first lead"},
    {"task": "24h_to_1h", "dir": "round5_selection_frozen_sparse_station_24h_1h", "variants": ["st_sparse_station_bias_delta_forecast"], "note": "sparse+station, k=7"},
    {"task": "24h_to_1h", "dir": "round6_selection_topk3_24h_1h", "variants": ["st_sparse_station_bias_delta_forecast"], "note": "k=3"},
    {"task": "24h_to_1h", "dir": "round6_selection_topk5_24h_1h", "variants": ["st_sparse_station_bias_delta_forecast"], "note": "k=5"},
    {"task": "24h_to_1h", "dir": "round6_selection_topk9_24h_1h", "variants": ["st_sparse_station_bias_delta_forecast"], "note": "k=9"},
    {"task": "24h_to_1h", "dir": "round6_selection_topk12_24h_1h", "variants": ["st_sparse_station_bias_delta_forecast"], "note": "k=12"},
    {"task": "168h_to_6h", "dir": "round3_selection_frozen_station_bias_168h_6h", "variants": ["st_station_bias_delta_forecast"], "note": "frozen backbone"},
    {"task": "168h_to_6h", "dir": "round4_selection_anchored_station_bias_168h_6h", "variants": ["st_station_bias_delta_forecast"], "note": "anchored first lead"},
    {"task": "168h_to_6h", "dir": "round5_selection_frozen_sparse_station_168h_6h", "variants": ["st_sparse_station_bias_delta_forecast"], "note": "sparse+station, k=7"},
    {"task": "168h_to_6h", "dir": "round6_selection_topk3_168h_6h", "variants": ["st_sparse_station_bias_delta_forecast"], "note": "k=3"},
    {"task": "168h_to_6h", "dir": "round6_selection_topk5_168h_6h", "variants": ["st_sparse_station_bias_delta_forecast"], "note": "k=5"},
    {"task": "168h_to_6h", "dir": "round6_selection_topk9_168h_6h", "variants": ["st_sparse_station_bias_delta_forecast"], "note": "k=9"},
    {"task": "168h_to_6h", "dir": "round6_selection_topk12_168h_6h", "variants": ["st_sparse_station_bias_delta_forecast"], "note": "k=12"},
]

VARIANT_LABELS = {
    "degraded_patchtst": "degraded (no spatial)",
    "st_center_only": "centre-only gate",
    "st_pairwise": "pairwise gate",
    "st_pairwise_delta": "pairwise + delta (input)",
    "st_pairwise_delta_forecast": "pairwise + delta (forecast)",
    "st_sparse_delta_forecast": "sparse + delta (forecast)",
    "st_station_bias_delta_forecast": "station-bias + delta (forecast)",
    "st_sparse_station_bias_delta_forecast": "sparse + station-bias + delta",
}


def build_structure_ablation(results: Path, out_dir: Path) -> pd.DataFrame:
    rows = []
    for source in ABLATION_SOURCES:
        raw = pd.read_csv(results / str(source["dir"]) / "raw_metrics.csv")
        baseline = raw[raw.variant == "degraded_patchtst"].set_index("seed")
        for variant in source["variants"]:  # type: ignore[union-attr]
            candidate = raw[raw.variant == variant].set_index("seed")
            common = sorted(set(baseline.index) & set(candidate.index))
            reductions, mae_reductions, smape_deltas = [], [], []
            disable_increases, shuffle_increases = [], []
            for seed in common:
                b, c = baseline.loc[seed], candidate.loc[seed]
                reductions.append(100 * (b.rmse_ugm3 - c.rmse_ugm3) / b.rmse_ugm3)
                mae_reductions.append(100 * (b.mae_ugm3 - c.mae_ugm3) / b.mae_ugm3)
                smape_deltas.append(c.smape_percent - b.smape_percent)
                if pd.notna(c.get("disable_neighbor_rmse_ugm3")):
                    disable_increases.append(100 * (c.disable_neighbor_rmse_ugm3 - c.rmse_ugm3) / c.rmse_ugm3)
                    shuffle_increases.append(100 * (c.shuffle_neighbor_rmse_ugm3 - c.rmse_ugm3) / c.rmse_ugm3)
            rows.append({
                "block": source["block"],
                "task": TASK_LABELS[str(source["task"])],
                "variant": VARIANT_LABELS.get(str(variant), str(variant)),
                "k": source["k"],
                "seeds": len(common),
                "rmse_mean": candidate.rmse_ugm3.mean(),
                "rmse_std": candidate.rmse_ugm3.std(),
                "rmse_reduction_mean": float(np.mean(reductions)),
                "rmse_reduction_std": float(np.std(reductions, ddof=1)) if len(reductions) > 1 else 0.0,
                "improved_seeds": int(sum(1 for value in reductions if value > 0)),
                "mae_reduction_mean": float(np.mean(mae_reductions)),
                "smape_delta_mean": float(np.mean(smape_deltas)),
                "disable_rmse_increase_mean": float(np.mean(disable_increases)) if disable_increases else np.nan,
                "shuffle_rmse_increase_mean": float(np.mean(shuffle_increases)) if shuffle_increases else np.nan,
            })
    frame = pd.DataFrame(rows)
    frame.insert(0, "split", "test")
    write_table(frame, out_dir, "B8_architecture_ablation", latex=True)

    dev_rows = []
    for source in DEV_SELECTION_SOURCES:
        raw = pd.read_csv(results / str(source["dir"]) / "raw_metrics.csv")
        baseline = raw[raw.variant == "degraded_patchtst"].set_index("seed")
        for variant in source["variants"]:  # type: ignore[union-attr]
            candidate = raw[raw.variant == variant].set_index("seed")
            common = sorted(set(baseline.index) & set(candidate.index))
            reductions = [100 * (baseline.loc[s].rmse_ugm3 - candidate.loc[s].rmse_ugm3) / baseline.loc[s].rmse_ugm3 for s in common]
            dev_rows.append({
                "split": "valid", "task": TASK_LABELS[str(source["task"])],
                "variant": VARIANT_LABELS.get(str(variant), str(variant)), "note": source["note"],
                "seeds": len(common), "rmse_mean": candidate.rmse_ugm3.mean(), "rmse_std": candidate.rmse_ugm3.std(),
                "rmse_reduction_mean": float(np.mean(reductions)),
                "improved_seeds": int(sum(1 for value in reductions if value > 0)),
            })
    write_table(pd.DataFrame(dev_rows), out_dir, "B9_dev_selection_validation")
    return frame


def figure_structure_ablation(ablation: pd.DataFrame, out: Path) -> None:
    block_colors = {"end_to_end_dev": "#8c8c8c", "frozen_production": COLOR_SPATIAL}
    ablation = ablation[ablation.split == "test"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.2), sharex="col")
    for column, task in enumerate([TASK_LABELS["24h_to_1h"], TASK_LABELS["168h_to_6h"]]):
        sel = ablation[ablation.task == task].reset_index(drop=True)
        ax = axes[0][column]
        x = np.arange(len(sel))
        colors = [block_colors[b] for b in sel.block]
        ax.bar(x, sel.rmse_reduction_mean, yerr=sel.rmse_reduction_std, capsize=3, color=colors,
               edgecolor="black", linewidth=0.4)
        ax.axhline(0, color="black", linewidth=0.8)
        for xi, (_, row) in zip(x, sel.iterrows()):
            ax.annotate(f"{int(row.improved_seeds)}/{int(row.seeds)}", (xi, row.rmse_reduction_mean), ha="center",
                        va="bottom", fontsize=7, xytext=(0, 4), textcoords="offset points")
        ax.set_ylabel("paired RMSE reduction (%)")
        ax.set_title(f"{task} — structure ablation (dev seeds 2024–2026)", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_xticks(x)

        ax2 = axes[1][column]
        reliance = sel.disable_rmse_increase_mean.fillna(0.0)
        shuffle = sel.shuffle_rmse_increase_mean.fillna(0.0)
        width = 0.38
        ax2.bar(x - width / 2, reliance, width, color="#4c72b0", edgecolor="black", linewidth=0.4, label="neighbours disabled")
        ax2.bar(x + width / 2, shuffle, width, color="#8172b2", edgecolor="black", linewidth=0.4, label="neighbours shuffled")
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_ylabel("RMSE increase (%)")
        ax2.set_xlabel("")
        ax2.grid(axis="y", alpha=0.3)
        ax2.set_xticks(x)
        short_block = {"end_to_end_dev": "dev, e2e, test", "frozen_production": "final, frozen, test"}
        ax2.set_xticklabels([f"{v}\n({short_block.get(b, b)}, k={int(k)})" if k != "" and k == k else f"{v}\n({short_block.get(b, b)})"
                             for v, b, k in zip(sel.variant, sel.block, sel.k)], rotation=35, ha="right", fontsize=6.5)
    axes[0][0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=block_colors["end_to_end_dev"]),
                              plt.Rectangle((0, 0), 1, 1, color=block_colors["frozen_production"])],
                     labels=["end-to-end dev, test split (rounds 1–2)", "frozen + degraded init, test split (final confirmation)"], fontsize=7, loc="upper left")
    axes[1][0].legend(fontsize=7)
    fig.suptitle("Architecture ablation and neighbour reliance (all rows on the TEST split); validation-split development sweeps are reported separately in B9",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "BF7_structure_ablation.pdf")
    fig.savefig(out / "BF7_structure_ablation.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
