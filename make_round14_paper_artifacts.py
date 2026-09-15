"""Generate paper-ready tables and figures for the round14 cross-city generalization study.

Inputs are the committed summary CSVs (and optionally the external archive holding the
full run artifacts). Nothing here re-trains or re-tunes: every number is derived from the
frozen evidence produced by
`experiment/cross-city-generalization-ablation@5211d7b`.

Usage:
    python make_round14_paper_artifacts.py \
        --summary-dir experiments/results/round14/cross_city_generalization_summary \
        --archive /home/hansel/.herdr/artifacts/PatchTST/round14-cross-city-generalization-71bb1d9.tar.zst \
        --out-tables tables/round14 --out-figures figures/round14
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import os

# Keep PDF/PNG output byte-stable across regenerations (matplotlib embeds dates otherwise).
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TASKS = [(24, 1), (168, 6)]
TASK_LABELS = {(24, 1): "24$\\rightarrow$1", (168, 6): "168$\\rightarrow$6"}
ARM_BASE = "degraded_patchtst"
ARM_SPATIAL = "st_sparse_station_bias_delta_forecast"
COLOR_BASE = "#4c72b0"
COLOR_SPATIAL = "#dd8452"


def load_frames(summary_dir: Path) -> dict[str, pd.DataFrame]:
    names = ["run_metrics", "paired_effects", "station_metrics", "block_metrics", "lead_metrics", "peak_metrics", "quantile_smape_decomposition"]
    return {name: pd.read_csv(summary_dir / f"{name}.csv") for name in names}


def to_markdown_table(frame: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Minimal Markdown renderer (avoids the optional `tabulate` dependency)."""
    def render(value) -> str:
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return "nan"
            return format(float(value), floatfmt)
        if isinstance(value, (bool, np.bool_)):
            return "true" if value else "false"
        return str(value)

    columns = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(render(row[c]) for c in frame.columns) + " |")
    return "\n".join(lines) + "\n"


def write_table(frame: pd.DataFrame, out_dir: Path, stem: str, latex: bool = False, float_format: str = "%.4f") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / f"{stem}.csv", index=False, float_format=float_format)
    (out_dir / f"{stem}.md").write_text(to_markdown_table(frame), encoding="utf-8")
    if latex:
        (out_dir / f"{stem}.tex").write_text(
            frame.to_latex(index=False, float_format=float_format, escape=True, caption=stem, label=f"tab:{stem}"),
            encoding="utf-8",
        )


def summarize_tables(frames: dict[str, pd.DataFrame], gate: dict, out_dir: Path) -> None:
    paired = frames["paired_effects"]
    station = frames["station_metrics"]
    block = frames["block_metrics"]
    lead = frames["lead_metrics"]
    peak = frames["peak_metrics"]
    run = frames["run_metrics"]

    # T1: headline gates
    rows = []
    for history, horizon in TASKS:
        entry = next(t for t in gate["tasks"] if t["task"] == f"{history}h_{horizon}h")
        rows.append({
            "task": TASK_LABELS[(history, horizon)],
            "pool_effect_percent": entry["pool_effect_percent"],
            "G1_threshold_percent": entry["G1_threshold_percent"],
            "G1": entry["gates"]["G1_mean_effect"],
            "pairs_improved": entry["successes"],
            "pairs_total": entry["successes"] + entry["failures"] + entry["ties"],
            "G2": entry["gates"]["G2_pair_consistency"],
            "stations_improved": entry["positive_station_count"],
            "G3": entry["gates"]["G3_station_consistency"],
            "blocks_improved": entry["positive_block_pool_count"],
            "G4": entry["gates"]["G4_block_robustness"],
            "task_pass": entry["task_pass"],
            "sign_tail_nominal": entry["nominal_one_sided_sign_tail_probability"],
        })
    t1 = pd.DataFrame(rows)
    t1["sign_tail_nominal"] = [f"{v:.2e}" for v in t1["sign_tail_nominal"]]
    t1.attrs["caption"] = "Pre-registered gates for the cross-city generalization confirmation."
    write_table(t1, out_dir, "T1_main_gates", latex=True)

    # T2: per-station
    key = ["history", "horizon", "station"]
    t2 = station.copy()
    t2["task"] = [TASK_LABELS[(h, hz)] for h, hz in zip(t2.history, t2.horizon)]
    t2 = t2[["task", "station", "base_rmse", "spatial_rmse", "mean_rmse_reduction_percent", "base_mae", "spatial_mae", "base_smape_percent", "spatial_smape_percent"]]
    t2["smape_change_percent_points"] = t2.spatial_smape_percent - t2.base_smape_percent
    write_table(t2, out_dir, "T2_per_station", latex=True)

    # T3: per lead
    t3 = lead.copy()
    t3["task"] = [TASK_LABELS[(h, hz)] for h, hz in zip(t3.history, t3.horizon)]
    t3 = t3.groupby(["task", "lead"], as_index=False)[["base_rmse", "spatial_rmse", "rmse_reduction_percent", "base_mae", "spatial_mae", "base_smape_percent", "spatial_smape_percent"]].mean()
    write_table(t3, out_dir, "T3_per_lead", latex=True)

    # T4: per block
    t4 = block.copy()
    t4["task"] = [TASK_LABELS[(h, hz)] for h, hz in zip(t4.history, t4.horizon)]
    t4 = t4.groupby(["task", "block"], as_index=False)[["base_rmse", "spatial_rmse", "rmse_reduction_percent"]].mean()
    write_table(t4, out_dir, "T4_per_block", latex=True)

    # T5: peak subset (q90 per station, training segment)
    t5 = peak.copy()
    t5["task"] = [TASK_LABELS[(h, hz)] for h, hz in zip(t5.history, t5.horizon)]
    pivot = t5.pivot_table(index=["task", "station"], columns="arm", values=["rmse", "mae", "coverage_count"], aggfunc="mean").reset_index()
    pivot.columns = ["_".join([str(part) for part in col if part]) if isinstance(col, tuple) else col for col in pivot.columns]
    rename = {
        "rmse_degraded_patchtst": "base_rmse", "rmse_st_sparse_station_bias_delta_forecast": "spatial_rmse",
        "mae_degraded_patchtst": "base_mae", "mae_st_sparse_station_bias_delta_forecast": "spatial_mae",
        "coverage_count_degraded_patchtst": "peak_samples",
    }
    pivot = pivot.rename(columns=rename)
    pivot["rmse_reduction_percent"] = 100 * (pivot.base_rmse - pivot.spatial_rmse) / pivot.base_rmse
    keep = ["task", "station", "peak_samples", "base_rmse", "spatial_rmse", "rmse_reduction_percent", "base_mae", "spatial_mae"]
    write_table(pivot[keep], out_dir, "T5_peak_subset", latex=True)

    # T6: headline metrics per arm
    rows = []
    for history, horizon in TASKS:
        sel = run[(run.history == history) & (run.horizon == horizon)]
        for arm, label in ((ARM_BASE, "degraded"), (ARM_SPATIAL, "spatial")):
            arm_sel = sel[sel.arm == arm]
            sse = float(arm_sel.sse.sum())
            n = float(arm_sel.element_count.sum())
            rmse = float(np.sqrt(sse / n))
            mae = float((arm_sel.mae * arm_sel.element_count).sum() / n)
            smape = float((arm_sel.smape_percent * arm_sel.element_count).sum() / n)
            rows.append({"task": TASK_LABELS[(history, horizon)], "arm": label, "rmse": rmse, "mae": mae, "smape_percent": smape, "elements": int(n)})
    write_table(pd.DataFrame(rows), out_dir, "T6_headline_metrics", latex=True)

    # T7: SMAPE decomposition by true-value quantile bin
    q = frames["quantile_smape_decomposition"].copy()
    q["task"] = [TASK_LABELS[(h, hz)] for h, hz in zip(q.history, q.horizon)]
    q = q.groupby(["task", "quantile_bin"], as_index=False)[["count", "base_smape_percent", "spatial_smape_percent", "smape_change_percent_points", "base_rmse", "spatial_rmse"]].sum()
    write_table(q, out_dir, "T7_smape_quantile_decomposition")


def load_metadata(predictions_root: Path) -> pd.DataFrame:
    """One row per center station (the training-segment Top-5 screening is task-independent)."""
    rows = {}
    for path in sorted(predictions_root.glob("*/*/run_metadata.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        station = int(meta["station_ids"][0])
        selected = [int(s) for s in meta["selected_top5"]]
        correlations = {int(k): float(v) for k, v in meta["candidate_correlations"].items()}
        rows[station] = {
            "station": station,
            "selected_top5": ", ".join(str(s) for s in selected),
            "top5_correlations": ", ".join(f"{correlations[s]:.3f}" for s in selected),
        }
    return pd.DataFrame(sorted(rows.values(), key=lambda r: r["station"]))


def figure_station_effects(paired: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (history, horizon) in zip(axes, TASKS):
        sel = paired[(paired.history == history) & (paired.horizon == horizon)]
        per_station = sel.groupby("station")["rmse_reduction_percent"].agg(["mean", "min", "max"]).sort_values("mean", ascending=False)
        x = np.arange(len(per_station))
        ax.bar(x, per_station["mean"], color=COLOR_SPATIAL, edgecolor="black", linewidth=0.4, label="mean paired reduction")
        ax.vlines(x, per_station["min"], per_station["max"], color="black", linewidth=1.0, label="min–max over 5 seeds")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x, [str(s) for s in per_station.index], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("RMSE reduction (%)")
        ax.set_title(TASK_LABELS[(history, horizon)], fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Cross-city generalization: per-station paired RMSE reduction (Guangzhou, 8 stations)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "F1_per_station_reduction.pdf")
    fig.savefig(out / "F1_per_station_reduction.png", dpi=300)
    plt.close(fig)


def figure_lead_effects(lead: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    sel = lead[(lead.history == 168) & (lead.horizon == 6)]
    agg = sel.groupby("lead")[["base_rmse", "spatial_rmse"]].mean()
    x = agg.index.to_numpy()
    width = 0.36
    axes[0].bar(x - width / 2, agg.base_rmse, width, label="degraded PatchTST", color=COLOR_BASE, edgecolor="black", linewidth=0.4)
    axes[0].bar(x + width / 2, agg.spatial_rmse, width, label="+ spatial residual", color=COLOR_SPATIAL, edgecolor="black", linewidth=0.4)
    axes[0].set_xticks(x)
    axes[0].set_xlabel("forecast lead (h)")
    axes[0].set_ylabel("RMSE ($\\mu g/m^3$)")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)
    red = sel.groupby("lead")["rmse_reduction_percent"].mean()
    axes[1].plot(red.index, red.values, marker="o", color=COLOR_SPATIAL)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xlabel("forecast lead (h)")
    axes[1].set_ylabel("RMSE reduction (%)")
    axes[1].grid(alpha=0.3)
    fig.suptitle("168$\\rightarrow$6: error by forecast lead (mean over 8 stations $\\times$ 5 seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "F2_per_lead.pdf")
    fig.savefig(out / "F2_per_lead.png", dpi=300)
    plt.close(fig)


def figure_block_effects(block: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (history, horizon) in zip(axes, TASKS):
        sel = block[(block.history == history) & (block.horizon == horizon)]
        agg = sel.groupby("block")["rmse_reduction_percent"].agg(["mean", "min", "max"])
        x = agg.index.to_numpy()
        ax.bar(x, agg["mean"], color=COLOR_SPATIAL, edgecolor="black", linewidth=0.4)
        ax.vlines(x, agg["min"], agg["max"], color="black", linewidth=1.0)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xlabel("confirmation block")
        ax.set_ylabel("RMSE reduction (%)")
        ax.set_title(TASK_LABELS[(history, horizon)], fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Robustness across the five confirmation blocks (bars: mean over 40 station-seed pairs)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "F3_per_block.pdf")
    fig.savefig(out / "F3_per_block.png", dpi=300)
    plt.close(fig)


def figure_pair_distribution(paired: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for (history, horizon), offset, color in zip(TASKS, (-0.12, 0.12), (COLOR_BASE, COLOR_SPATIAL)):
        sel = paired[(paired.history == history) & (paired.horizon == horizon)]
        jitter = np.linspace(-0.05, 0.05, len(sel))
        ax.scatter(np.full(len(sel), offset) + jitter, sel.rmse_reduction_percent, s=18, color=color, alpha=0.8,
                   label=f"{TASK_LABELS[(history, horizon)]} (n={len(sel)})")
        ax.hlines(sel.rmse_reduction_percent.mean(), offset - 0.09, offset + 0.09, color="black", linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks([-0.12, 0.12], [TASK_LABELS[t] for t in TASKS])
    ax.set_ylabel("paired RMSE reduction (%)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("All 40 station-seed pairs per task (black bars: pool effect)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "F4_pair_distribution.pdf")
    fig.savefig(out / "F4_pair_distribution.png", dpi=300)
    plt.close(fig)


def figure_rmse_scatter(station: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, (history, horizon) in zip(axes, TASKS):
        sel = station[(station.history == history) & (station.horizon == horizon)]
        ax.scatter(sel.base_rmse, sel.spatial_rmse, s=40, color=COLOR_SPATIAL, edgecolor="black", linewidth=0.5)
        lo = float(min(sel.base_rmse.min(), sel.spatial_rmse.min())) * 0.98
        hi = float(max(sel.base_rmse.max(), sel.spatial_rmse.max())) * 1.02
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray", linewidth=1)
        for _, row in sel.iterrows():
            ax.annotate(str(int(row.station)), (row.base_rmse, row.spatial_rmse), fontsize=7, xytext=(3, 2), textcoords="offset points")
        ax.set_xlabel("degraded PatchTST RMSE ($\\mu g/m^3$)")
        ax.set_ylabel("+ spatial residual RMSE ($\\mu g/m^3$)")
        ax.set_title(TASK_LABELS[(history, horizon)], fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle("Per-station RMSE (each point: one center station, mean over 5 seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "F5_rmse_scatter.pdf")
    fig.savefig(out / "F5_rmse_scatter.png", dpi=300)
    plt.close(fig)


def figure_smape_quantiles(q: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (history, horizon) in zip(axes, TASKS):
        sel = q[(q.history == history) & (q.horizon == horizon)]
        agg = sel.groupby("quantile_bin")[["base_smape_percent", "spatial_smape_percent", "count"]].sum()
        x = agg.index.to_numpy()
        width = 0.36
        ax.bar(x - width / 2, agg.base_smape_percent, width, color=COLOR_BASE, edgecolor="black", linewidth=0.4, label="degraded PatchTST")
        ax.bar(x + width / 2, agg.spatial_smape_percent, width, color=COLOR_SPATIAL, edgecolor="black", linewidth=0.4, label="+ spatial residual")
        ax.set_xticks(x)
        ax.set_xlabel("true-value quantile bin (training segment)")
        ax.set_ylabel("SMAPE (%)")
        ax.set_title(TASK_LABELS[(history, horizon)], fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("SMAPE by true-value quantile bin (pooled over stations and seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "F6_smape_quantiles.pdf")
    fig.savefig(out / "F6_smape_quantiles.png", dpi=300)
    plt.close(fig)


def figure_peak(peak: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (history, horizon) in zip(axes, TASKS):
        sel = peak[(peak.history == history) & (peak.horizon == horizon)]
        agg = sel.groupby(["station", "arm"])["rmse"].mean().unstack()
        x = np.arange(len(agg))
        width = 0.36
        ax.bar(x - width / 2, agg[ARM_BASE], width, color=COLOR_BASE, edgecolor="black", linewidth=0.4, label="degraded PatchTST")
        ax.bar(x + width / 2, agg[ARM_SPATIAL], width, color=COLOR_SPATIAL, edgecolor="black", linewidth=0.4, label="+ spatial residual")
        ax.set_xticks(x, [str(s) for s in agg.index], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("RMSE on q90 subset ($\\mu g/m^3$)")
        ax.set_title(TASK_LABELS[(history, horizon)], fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("High-concentration subset (per-station training-segment 90th percentile)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "F7_peak_subset.pdf")
    fig.savefig(out / "F7_peak_subset.png", dpi=300)
    plt.close(fig)


def figure_example_series(predictions_root: Path, out: Path, station: int = 9033, seed: int = 7001,
                          history: int = 24, horizon: int = 1, zoom_hours: int = 120) -> bool:
    base_path = predictions_root / f"{history}h_{horizon}h" / f"station_{station}" / "predictions" / f"{ARM_BASE}_seed{seed}.npz"
    spatial_path = predictions_root / f"{history}h_{horizon}h" / f"station_{station}" / "predictions" / f"{ARM_SPATIAL}_seed{seed}.npz"
    if not (base_path.exists() and spatial_path.exists()):
        return False
    base = np.load(base_path)
    spatial = np.load(spatial_path)
    ts = base["target_timestamp_ns"]
    order = np.argsort(ts)
    ts = ts[order]
    target = base["target_ugm3"][order]
    pred_base = base["prediction_ugm3"][order]
    pred_spatial = spatial["prediction_ugm3"][order]
    block = base["block_label"][order]
    mask = block == 1
    t = pd.to_datetime(ts[mask])
    obs = target[mask]
    pb = pred_base[mask]
    ps = pred_spatial[mask]
    zoom = slice(0, min(zoom_hours, len(t)))

    fig, axes = plt.subplots(2, 1, figsize=(11, 5.4), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1.0]}, constrained_layout=True)
    ax = axes[0]
    ax.plot(t[zoom], obs[zoom], label="observed PM$_{2.5}$", color="black", linewidth=1.0)
    ax.plot(t[zoom], pb[zoom], label="degraded PatchTST", color=COLOR_BASE, linewidth=1.1, alpha=0.9)
    ax.plot(t[zoom], ps[zoom], label="+ spatial residual", color=COLOR_SPATIAL, linewidth=1.1, alpha=0.9)
    ax.set_ylabel("PM$_{2.5}$ ($\\mu g/m^3$)")
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.grid(alpha=0.3)
    rmse_b = float(np.sqrt(np.mean((pb - obs) ** 2)))
    rmse_s = float(np.sqrt(np.mean((ps - obs) ** 2)))
    ax.set_title(f"Guangzhou station {station}, seed {seed}, {history}$\\rightarrow${horizon}; first confirmation block "
                 f"(RMSE {rmse_b:.3f} vs {rmse_s:.3f}, reduction {100 * (rmse_b - rmse_s) / rmse_b:.2f}%)", fontsize=10)

    ax2 = axes[1]
    ax2.plot(t[zoom], ps[zoom] - pb[zoom], color=COLOR_SPATIAL, linewidth=1.0)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("correction ($\\mu g/m^3$)")
    ax2.set_xlabel(f"target timestamp (first {zoom_hours} h of the confirmation block)")
    ax2.grid(alpha=0.3)
    ax2.set_title("spatial residual contribution", fontsize=9)

    fig.savefig(out / f"F8_example_series_{station}_{history}x{horizon}.pdf")
    fig.savefig(out / f"F8_example_series_{station}_{history}x{horizon}.png", dpi=300)
    plt.close(fig)
    return True


def coverage_table(run: pd.DataFrame, metadata_root: Path, out_dir: Path) -> None:
    """Data-coverage table: common-axis bounds and per-task sample counts."""
    rows = []
    for path in sorted(metadata_root.glob("*/*/run_metadata.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        station = int(meta["station_ids"][0])
        task_dir = path.parent.parent.name
        bounds = meta["bounds"]
        rows.append({
            "task_dir": task_dir,
            "station": station,
            "common_rows": int(bounds["confirm"][1]),
            "fit_rows": int(bounds["fit"][1] - bounds["fit"][0]),
            "val_rows": int(bounds["val"][1] - bounds["val"][0]),
            "confirm_rows": int(bounds["confirm"][1] - bounds["confirm"][0]),
            "n_candidates": len(meta["candidate_correlations"]),
        })
    frame = pd.DataFrame(rows).drop_duplicates(subset=["station", "fit_rows", "val_rows", "confirm_rows"])
    frame = frame[frame.task_dir == frame.task_dir.iloc[0]].drop(columns=["task_dir"]).sort_values("station").reset_index(drop=True)
    elements = run.groupby(["history", "horizon", "arm"])["element_count"].agg("sum").unstack()
    frame.attrs["confirm_scalars_per_arm"] = {f"{h}h_{hz}h": int(v) for (h, hz), v in elements[ARM_BASE].items()}
    write_table(frame, out_dir, "T9_data_coverage")
    (out_dir / "T9_data_coverage_note.md").write_text(
        "Confirm scalar targets per arm: " + ", ".join(f"{k} = {v}" for k, v in frame.attrs["confirm_scalars_per_arm"].items())
        + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-dir", required=True)
    parser.add_argument("--archive", default=None, help="Optional .tar.zst with the full run artifacts (for the example-series figure).")
    parser.add_argument("--out-tables", required=True)
    parser.add_argument("--out-figures", required=True)
    args = parser.parse_args()

    summary_dir = Path(args.summary_dir)
    out_tables = Path(args.out_tables)
    out_figures = Path(args.out_figures)
    out_figures.mkdir(parents=True, exist_ok=True)

    frames = load_frames(summary_dir)
    gate = json.loads((summary_dir / "gate_summary.json").read_text(encoding="utf-8"))
    summarize_tables(frames, gate, out_tables)

    paired, station, block, lead, peak = (frames["paired_effects"], frames["station_metrics"], frames["block_metrics"], frames["lead_metrics"], frames["peak_metrics"])
    figure_station_effects(paired, out_figures)
    figure_lead_effects(lead, out_figures)
    figure_block_effects(block, out_figures)
    figure_pair_distribution(paired, out_figures)
    figure_rmse_scatter(station, out_figures)
    figure_smape_quantiles(frames["quantile_smape_decomposition"], out_figures)
    figure_peak(peak, out_figures)

    example_written = False
    if args.archive:
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                ["tar", "--zstd", "-xf", str(args.archive), "-C", tmp,
                 "--wildcards", "*/predictions/*", "*/run_metadata.json"],
                check=True,
            )
            root = Path(tmp) / "experiments" / "results" / "round14" / "cross_city_generalization"
            if root.exists():
                meta = load_metadata(root)
                write_table(meta, out_tables, "T8_top5_neighbours")
                coverage_table(frames["run_metrics"], root, out_tables)
                example_written = figure_example_series(root, out_figures)

    print(json.dumps({
        "tables_dir": str(out_tables),
        "figures_dir": str(out_figures),
        "example_series_written": example_written,
        "tables": sorted(p.name for p in out_tables.glob("*.csv")),
        "figures": sorted(p.name for p in out_figures.glob("*.pdf")),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
