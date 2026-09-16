"""Paper tables and figures for the frequency-residual comparison (locked ST vs ST + rFFT).

Requested scope (user, 2026-09-15): show only the locked ST and the ST + frequency arms.
The capacity-matched control and the leakage discussion are deliberately out of scope here;
the full record (including the control) stays in
`experiments/results/frequency_legacy_rerun/` and `docs/频域旧协议重跑/00_协议与结果.md`.

Usage:
    python make_frequency_paper_artifacts.py \
        --run-24h /tmp/freq_legacy_24 --run-168h /tmp/freq_legacy_168 \
        --out-tables tables/frequency_legacy --out-figures figures/frequency_legacy
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # byte-stable PDF/PNG output

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ST = "st_sparse_station_bias_delta_forecast"
RFFT = "st_rfft"
DEGRADED = "degraded_patchtst"
TASKS = [("24h_1h", "24$\\rightarrow$1"), ("168h_6h", "168$\\rightarrow$6")]
COLOR_DEGRADED = "#8c8c8c"
COLOR_ST = "#4c72b0"
COLOR_RFFT = "#dd8452"


def to_markdown(frame: pd.DataFrame, floatfmt: str = ".4f") -> str:
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
    frame.to_csv(out_dir / f"{stem}.csv", index=False, float_format="%.6f")
    (out_dir / f"{stem}.md").write_text(to_markdown(frame), encoding="utf-8")
    if latex:
        (out_dir / f"{stem}.tex").write_text(
            frame.to_latex(index=False, float_format="%.6f", escape=True, caption=stem, label=f"tab:{stem}"),
            encoding="utf-8",
        )


def load_runs(run_dirs: dict[str, Path]) -> dict[str, pd.DataFrame]:
    frames = {}
    for task, path in run_dirs.items():
        frames[task] = pd.read_csv(path / "raw_metrics.csv")
    return frames


def build_tables(frames: dict[str, pd.DataFrame], run_dirs: dict[str, Path], out_dir: Path) -> pd.DataFrame:
    rows = []
    per_seed_rows = []
    for task, label in TASKS:
        frame = frames[task]
        pivot = frame.pivot_table(index="seed", columns="variant", values="rmse_ugm3")
        mae_pivot = frame.pivot_table(index="seed", columns="variant", values="mae_ugm3")
        reduction = 100 * (pivot[ST] - pivot[RFFT]) / pivot[ST]
        rows.append({
            "task": label,
            "seeds": len(pivot),
            "degraded_rmse": pivot[DEGRADED].mean(),
            "degraded_rmse_std": pivot[DEGRADED].std(),
            "locked_st_rmse": pivot[ST].mean(),
            "locked_st_rmse_std": pivot[ST].std(),
            "st_plus_rfft_rmse": pivot[RFFT].mean(),
            "st_plus_rfft_rmse_std": pivot[RFFT].std(),
            "improvement_over_st_percent": reduction.mean(),
            "improvement_over_st_std": reduction.std(),
            "improved_seeds": int((reduction > 0).sum()),
            "improvement_over_degraded_percent": 100 * (pivot[DEGRADED].mean() - pivot[RFFT].mean()) / pivot[DEGRADED].mean(),
        })
        for seed in pivot.index:
            per_seed_rows.append({
                "task": label, "seed": int(seed),
                "degraded_rmse": pivot.loc[seed, DEGRADED],
                "locked_st_rmse": pivot.loc[seed, ST],
                "st_plus_rfft_rmse": pivot.loc[seed, RFFT],
                "rmse_reduction_percent": reduction.loc[seed],
                "locked_st_mae": mae_pivot.loc[seed, ST],
                "st_plus_rfft_mae": mae_pivot.loc[seed, RFFT],
                "mae_reduction_percent": 100 * (mae_pivot.loc[seed, ST] - mae_pivot.loc[seed, RFFT]) / mae_pivot.loc[seed, ST],
            })
    main = pd.DataFrame(rows)
    write_table(main, out_dir, "F1_main_results", latex=True)
    write_table(pd.DataFrame(per_seed_rows), out_dir, "F2_per_seed", latex=True)

    configs = []
    for task, label in TASKS:
        cfg = json.loads((run_dirs[task] / "experiment_config.json").read_text(encoding="utf-8"))
        adapter_row = frames[task][frames[task].variant == RFFT].iloc[0]
        configs.append({
            "task": label, "history": cfg["history"], "horizon": cfg["horizon"],
            "epochs": cfg["epochs"], "patience": cfg["patience"], "batch_size": cfg["batch_size"],
            "sparse_neighbor_top_k": cfg["sparse_neighbor_top_k"], "evaluation_split": cfg["evaluation_split"],
            "freeze_backbone": cfg["freeze_backbone"], "initialize_from_degraded": cfg["initialize_from_degraded"],
            "adapter_trainable_parameters": int(adapter_row["trainable_parameter_count"]),
            "mean_beta": float(frames[task][frames[task].variant == RFFT].beta.mean()),
        })
    write_table(pd.DataFrame(configs), out_dir, "F3_configuration")
    return main


def build_vs_degraded(frames: dict[str, pd.DataFrame], out_dir: Path) -> pd.DataFrame:
    """Paired comparison of the frozen ST and the ST + frequency arms against the PatchTST baseline."""
    rows = []
    per_seed = []
    for task, label in TASKS:
        pivot = frames[task].pivot_table(index="seed", columns="variant", values="rmse_ugm3")
        degraded, st, rfft = pivot[DEGRADED], pivot[ST], pivot[RFFT]
        red_st = 100 * (degraded - st) / degraded
        red_rfft = 100 * (degraded - rfft) / degraded
        rows.append({
            "task": label,
            "seeds": len(pivot),
            "degraded_rmse": degraded.mean(),
            "locked_st_rmse": st.mean(),
            "st_plus_rfft_rmse": rfft.mean(),
            "st_reduction_vs_degraded_percent": red_st.mean(),
            "st_reduction_vs_degraded_std": red_st.std(),
            "st_improved_seeds": int((red_st > 0).sum()),
            "st_plus_rfft_reduction_vs_degraded_percent": red_rfft.mean(),
            "st_plus_rfft_reduction_vs_degraded_std": red_rfft.std(),
            "st_plus_rfft_improved_seeds": int((red_rfft > 0).sum()),
            "extra_from_frequency_percent": (red_rfft - red_st).mean(),
        })
        for seed in pivot.index:
            per_seed.append({
                "task": label, "seed": int(seed),
                "degraded_rmse": degraded.loc[seed],
                "locked_st_rmse": st.loc[seed],
                "st_plus_rfft_rmse": rfft.loc[seed],
                "st_reduction_vs_degraded_percent": red_st.loc[seed],
                "st_plus_rfft_reduction_vs_degraded_percent": red_rfft.loc[seed],
                "extra_from_frequency_percent": red_rfft.loc[seed] - red_st.loc[seed],
            })
    frame = pd.DataFrame(rows)
    write_table(frame, out_dir, "F4_vs_degraded_patchtst", latex=True)
    write_table(pd.DataFrame(per_seed), out_dir, "F5_vs_degraded_per_seed", latex=True)
    return frame


def figure_reduction_vs_degraded(frames: dict[str, pd.DataFrame], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    for ax, (task, label) in zip(axes, TASKS):
        pivot = frames[task].pivot_table(index="seed", columns="variant", values="rmse_ugm3")
        degraded, st, rfft = pivot[DEGRADED], pivot[ST], pivot[RFFT]
        red_st = (100 * (degraded - st) / degraded).sort_index()
        red_rfft = (100 * (degraded - rfft) / degraded).sort_index()
        x = np.arange(len(red_st))
        width = 0.38
        ax.bar(x - width / 2, red_st.values, width, color=COLOR_ST, edgecolor="black", linewidth=0.4, label="frozen ST")
        ax.bar(x + width / 2, red_rfft.values, width, color=COLOR_RFFT, edgecolor="black", linewidth=0.4, label="frozen ST + frequency")
        for xi, (a, b) in enumerate(zip(red_st.values, red_rfft.values)):
            ax.annotate(f"{a:.2f}", (xi - width / 2, a), ha="center", va="bottom", fontsize=6.5, xytext=(0, 2), textcoords="offset points")
            ax.annotate(f"{b:.2f}", (xi + width / 2, b), ha="center", va="bottom", fontsize=6.5, xytext=(0, 2), textcoords="offset points")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x, [str(int(s)) for s in red_st.index])
        ax.set_xlabel("seed")
        ax.set_ylabel("RMSE reduction vs degraded PatchTST (%)")
        ax.set_title(f"{label} — mean: ST {red_st.mean():.3f}% , ST+frequency {red_rfft.mean():.3f}%", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Paired RMSE reduction against the degraded PatchTST baseline (test split, seeds 2047–2051)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "FF5_reduction_vs_degraded.pdf")
    fig.savefig(out / "FF5_reduction_vs_degraded.png", dpi=300)
    plt.close(fig)


def figure_per_seed_reduction(frames: dict[str, pd.DataFrame], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    for ax, (task, label) in zip(axes, TASKS):
        frame = frames[task]
        pivot = frame.pivot_table(index="seed", columns="variant", values="rmse_ugm3")
        reduction = (100 * (pivot[ST] - pivot[RFFT]) / pivot[ST]).sort_index()
        x = np.arange(len(reduction))
        ax.bar(x, reduction.values, color=COLOR_RFFT, edgecolor="black", linewidth=0.4)
        ax.hlines(reduction.mean(), -0.5, len(reduction) - 0.5, color="black", linewidth=1.2,
                  linestyle="--", label=f"mean {reduction.mean():.3f}%")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x, [str(int(s)) for s in reduction.index])
        ax.set_xlabel("seed")
        ax.set_ylabel("RMSE reduction vs frozen ST (%)")
        ax.set_title(label, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Frequency residual on top of the frozen ST: per-seed paired RMSE reduction (test split)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "FF1_per_seed_reduction.pdf")
    fig.savefig(out / "FF1_per_seed_reduction.png", dpi=300)
    plt.close(fig)


def figure_rmse_levels(frames: dict[str, pd.DataFrame], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    for ax, (task, label) in zip(axes, TASKS):
        frame = frames[task]
        arms = [DEGRADED, ST, RFFT]
        names = ["degraded\nPatchTST", "frozen ST", "frozen ST\n+ frequency"]
        colors = [COLOR_DEGRADED, COLOR_ST, COLOR_RFFT]
        means = [frame[frame.variant == a].rmse_ugm3.mean() for a in arms]
        stds = [frame[frame.variant == a].rmse_ugm3.std() for a in arms]
        x = np.arange(len(arms))
        ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.4)
        for xi, (m, s) in enumerate(zip(means, stds)):
            ax.annotate(f"{m:.2f}", (xi, m), ha="center", va="bottom", fontsize=8, xytext=(0, 4), textcoords="offset points")
        ax.set_xticks(x, names, fontsize=8)
        ax.set_ylabel("RMSE ($\\mu g/m^3$)")
        ax.set_title(label, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Test-split RMSE: degraded PatchTST, frozen ST, and frozen ST + frequency residual (mean $\\pm$ sd over seeds 2047–2051)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "FF2_rmse_levels.pdf")
    fig.savefig(out / "FF2_rmse_levels.png", dpi=300)
    plt.close(fig)


def figure_cumulative(main: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    tasks = main["task"].tolist()
    x = np.arange(len(tasks))
    width = 0.36
    st_gain = [100 * (d - s) / d for d, s in zip(main.degraded_rmse, main.locked_st_rmse)]
    rfft_gain = main.improvement_over_degraded_percent.tolist()
    ax.bar(x - width / 2, st_gain, width, color=COLOR_ST, edgecolor="black", linewidth=0.4, label="frozen ST")
    ax.bar(x + width / 2, rfft_gain, width, color=COLOR_RFFT, edgecolor="black", linewidth=0.4, label="frozen ST + frequency")
    for xi, (a, b) in enumerate(zip(st_gain, rfft_gain)):
        ax.annotate(f"{a:.2f}%", (xi - width / 2, a), ha="center", va="bottom", fontsize=8)
        ax.annotate(f"{b:.2f}%", (xi + width / 2, b), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, tasks)
    ax.set_ylabel("RMSE improvement over degraded PatchTST (%)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Cumulative improvement over the degraded PatchTST baseline", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "FF3_cumulative_improvement.pdf")
    fig.savefig(out / "FF3_cumulative_improvement.png", dpi=300)
    plt.close(fig)


def _timestamps(meta: dict, n_test: int, history: int, horizon: int, lead: int = 0) -> pd.DatetimeIndex:
    start = pd.Timestamp(meta["start_time"])
    offset = history + meta["split_sizes"]["train"] + meta["split_sizes"]["valid"] + lead
    return pd.date_range(start=start + pd.Timedelta(hours=offset), periods=n_test, freq="h")


def _seed_with_largest_improvement(frame: pd.DataFrame) -> int:
    """Deterministic rule: the seed with the largest paired RMSE reduction (stated in the caption)."""
    pivot = frame.pivot_table(index="seed", columns="variant", values="rmse_ugm3")
    reduction = 100 * (pivot[ST] - pivot[RFFT]) / pivot[ST]
    return int(reduction.idxmax())


def figure_example_series(run_dirs: dict[str, Path], frames: dict[str, pd.DataFrame],
                          out: Path, zoom_hours: int = 120) -> list[str]:
    written = []
    for task, label in TASKS:
        run = run_dirs[task]
        seed = _seed_with_largest_improvement(frames[task])
        base = np.load(run / "predictions" / f"{ST}_seed{seed}.npz")
        cand = np.load(run / "predictions" / f"{RFFT}_seed{seed}.npz")
        meta = json.loads((run / "dataset_metadata.json").read_text(encoding="utf-8"))
        horizon = 6 if task == "168h_6h" else 1
        history = 168 if task == "168h_6h" else 24
        # Prediction files may hold either scaled tensors or physical concentrations depending on
        # which revision of the runner produced them; detect and normalise both cases.
        center_mean, center_std = float(meta["center_mean"]), float(meta["center_std"])

        def to_physical(array: np.ndarray) -> np.ndarray:
            """Normalise per file: runs patched mid-flight may store scaled or physical tensors."""
            return array * center_std + center_mean if float(np.nanmean(array)) < 20 else array

        target_all = to_physical(base["target_ugm3"][:, 0, :])
        st_all = to_physical(base["prediction_ugm3"][:, 0, :])
        rfft_all = to_physical(cand["prediction_ugm3"][:, 0, :])
        for name, array in (("target", target_all), ("frozen ST", st_all), ("ST+frequency", rfft_all)):
            if not (0 < float(np.nanmean(array)) < 500):
                raise AssertionError(f"example-series {name} values are not physically plausible: mean={np.nanmean(array):.3f}")
        targets, st_pred, rfft_pred = target_all[:, 0], st_all[:, 0], rfft_all[:, 0]
        times = _timestamps(meta, len(targets), history, horizon, lead=0)
        start = int(np.nanargmax(pd.Series(targets).rolling(min(zoom_hours, len(targets))).std().to_numpy()))
        sl = slice(start, start + min(zoom_hours, len(targets)))

        fig, axes = plt.subplots(2, 1, figsize=(11, 5.4), sharex=True,
                                 gridspec_kw={"height_ratios": [2.2, 1.0]}, constrained_layout=True)
        ax = axes[0]
        ax.plot(times[sl], targets[sl], color="black", linewidth=1.0, label="observed PM$_{2.5}$")
        ax.plot(times[sl], st_pred[sl], color=COLOR_ST, linewidth=1.1, alpha=0.9, label="frozen ST")
        ax.plot(times[sl], rfft_pred[sl], color=COLOR_RFFT, linewidth=1.1, alpha=0.9, label="frozen ST + frequency")
        ax.set_ylabel("PM$_{2.5}$ ($\\mu g/m^3$)")
        ax.legend(fontsize=8, ncol=3, loc="upper right")
        ax.grid(alpha=0.3)
        rmse_st = float(np.sqrt(np.mean((st_pred - targets) ** 2)))
        rmse_rfft = float(np.sqrt(np.mean((rfft_pred - targets) ** 2)))
        ax.set_title(f"Beijing 1013, {label}, test split; seed {seed} (largest paired improvement); "
                     f"full-test RMSE {rmse_st:.3f} vs {rmse_rfft:.3f} "
                     f"({100 * (rmse_st - rmse_rfft) / rmse_st:+.2f}%)", fontsize=10)
        ax2 = axes[1]
        ax2.plot(times[sl], rfft_pred[sl] - st_pred[sl], color=COLOR_RFFT, linewidth=1.0)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_ylabel("frequency\ncorrection")
        ax2.set_xlabel(f"target timestamp (most variable {zoom_hours} h window)")
        ax2.grid(alpha=0.3)
        name = f"FF4_example_series_{task}"
        fig.savefig(out / f"{name}.pdf")
        fig.savefig(out / f"{name}.png", dpi=300)
        plt.close(fig)
        written.append(name)
    return written


def _load_arm_arrays(run: Path, variant: str, seed: int, meta: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (prediction, target) in physical units for one arm and seed."""
    center_mean, center_std = float(meta["center_mean"]), float(meta["center_std"])
    data = np.load(run / "predictions" / f"{variant}_seed{seed}.npz")
    prediction = data["prediction_ugm3"]
    target = data["target_ugm3"]
    if float(np.nanmean(target)) < 20:  # scaled storage
        prediction = prediction * center_std + center_mean
        target = target * center_std + center_mean
    return prediction[:, 0, :], target[:, 0, :]


def figure_per_lead_168h(run_dirs: dict[str, Path], out: Path) -> None:
    """Dual panel, per seed: RMSE by forecast lead and paired reduction over the frozen ST."""
    task = "168h_6h"
    run = run_dirs[task]
    meta = json.loads((run / "dataset_metadata.json").read_text(encoding="utf-8"))
    seeds = sorted({int(p.stem.split("seed")[1]) for p in (run / "predictions").glob(f"{ST}_seed*.npz")})
    arms = [(DEGRADED, "degraded PatchTST", COLOR_DEGRADED), (ST, "frozen ST", COLOR_ST), (RFFT, "frozen ST + frequency", COLOR_RFFT)]

    per_lead = {variant: [] for variant, _, _ in arms}
    targets_ref = None
    for seed in seeds:
        for variant, _, _ in arms:
            prediction, target = _load_arm_arrays(run, variant, seed, meta)
            if targets_ref is None:
                targets_ref = target
            per_lead[variant].append(np.sqrt(np.mean((prediction - target) ** 2, axis=0)))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    leads = np.arange(1, per_lead[ST][0].size + 1)
    ax = axes[0]
    reference = np.stack(per_lead[ST])
    for variant, label, color in ((DEGRADED, "degraded PatchTST", COLOR_DEGRADED),
                                  (RFFT, "frozen ST + frequency", COLOR_RFFT)):
        curves = np.stack(per_lead[variant]) - reference  # difference against the frozen ST, per seed
        for curve in curves:
            ax.plot(leads, curve, color=color, alpha=0.28, linewidth=0.9)
        ax.plot(leads, curves.mean(axis=0), color=color, linewidth=2.0, marker="o", markersize=4,
                label=f"{label} (mean)")
    ax.axhline(0, color="black", linewidth=1.0, label="frozen ST")
    ax.set_xlabel("forecast lead (h)")
    ax.set_ylabel("RMSE difference vs frozen ST ($\\mu g/m^3$)")
    ax.set_xticks(leads)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title("Per-seed error difference vs frozen ST by lead (thin lines: seeds)", fontsize=9)

    ax2 = axes[1]
    st_curves = np.stack(per_lead[ST])
    rfft_curves = np.stack(per_lead[RFFT])
    reductions = 100 * (st_curves - rfft_curves) / st_curves
    for curve in reductions:
        ax2.plot(leads, curve, color=COLOR_RFFT, alpha=0.35, linewidth=0.9)
    ax2.plot(leads, reductions.mean(axis=0), color=COLOR_RFFT, linewidth=2.0, marker="o", markersize=4,
             label="mean reduction")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("forecast lead (h)")
    ax2.set_ylabel("RMSE reduction vs frozen ST (%)")
    ax2.set_xticks(leads)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7)
    ax2.set_title("Per-seed paired reduction of ST+frequency over frozen ST", fontsize=9)
    fig.suptitle("168$\\rightarrow$6, test split: per-lead error and paired reduction, seed by seed", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "FF6_per_lead_168h6h.pdf")
    fig.savefig(out / "FF6_per_lead_168h6h.png", dpi=300)
    plt.close(fig)


def figure_per_seed_levels(frames: dict[str, pd.DataFrame], out: Path) -> None:
    """Dual panel, per seed: absolute RMSE of the three arms across the five seeds."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for ax, (task, label) in zip(axes, TASKS):
        pivot = frames[task].pivot_table(index="seed", columns="variant", values="rmse_ugm3").sort_index()
        x = np.arange(len(pivot))
        for variant, name, color in ((DEGRADED, "degraded PatchTST", COLOR_DEGRADED),
                                     (ST, "frozen ST", COLOR_ST),
                                     (RFFT, "frozen ST + frequency", COLOR_RFFT)):
            ax.plot(x, pivot[variant].values, marker="o", markersize=4, linewidth=1.6, color=color, label=name)
        ax.set_xticks(x, [str(int(s)) for s in pivot.index])
        ax.set_xlabel("seed")
        ax.set_ylabel("test-split RMSE ($\\mu g/m^3$)")
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("Per-seed RMSE of the three arms (paired by seed)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "FF7_per_seed_levels.pdf")
    fig.savefig(out / "FF7_per_seed_levels.png", dpi=300)
    plt.close(fig)


def build_component_contribution(frames: dict[str, pd.DataFrame], run_dirs: dict[str, Path],
                                 clean_ref: Path, out_dir: Path) -> pd.DataFrame:
    """Decompose the model into its components: PatchTST -> + spatial structure -> + frequency adapter."""
    rows = []
    for task, label in TASKS:
        pivot = frames[task].pivot_table(index="seed", columns="variant", values="rmse_ugm3")
        degraded = float(pivot[DEGRADED].mean())
        spatial = float(pivot[ST].mean())
        final = float(pivot[RFFT].mean())
        rows.append({
            "pipeline": "legacy (test, 5 seeds)",
            "task": label,
            "patchtst_rmse": degraded,
            "spatial_structure_gain_ugm3": degraded - spatial,
            "spatial_structure_gain_percent": 100 * (degraded - spatial) / degraded,
            "after_spatial_rmse": spatial,
            "frequency_adapter_gain_ugm3": spatial - final,
            "frequency_adapter_gain_percent": 100 * (spatial - final) / spatial,
            "final_rmse": final,
            "total_gain_percent": 100 * (degraded - final) / degraded,
            "spatial_share_of_total_percent": 100 * (degraded - spatial) / (degraded - final),
            "frequency_share_of_total_percent": 100 * (spatial - final) / (degraded - final),
        })
    if clean_ref.is_dir():
        base = pd.read_csv(clean_ref / "p0_bridge_168h_6h.csv").set_index("variant").rmse_ugm3_mean
        fixed = pd.read_csv(clean_ref / "p1_fixed_168h_6h.csv").set_index("variant").rmse_ugm3_mean
        degraded = float(base["degraded_patchtst_clean"])
        spatial = float(base["locked_st_clean"])
        final = float(fixed["st_fixed_frequency_residual"])
        rows.append({
            "pipeline": "leak-free (validation, 3 seeds)",
            "task": dict(TASKS)["168h_6h"],
            "patchtst_rmse": degraded,
            "spatial_structure_gain_ugm3": degraded - spatial,
            "spatial_structure_gain_percent": 100 * (degraded - spatial) / degraded,
            "after_spatial_rmse": spatial,
            "frequency_adapter_gain_ugm3": spatial - final,
            "frequency_adapter_gain_percent": 100 * (spatial - final) / spatial,
            "final_rmse": final,
            "total_gain_percent": 100 * (degraded - final) / degraded,
            "spatial_share_of_total_percent": 100 * (degraded - spatial) / (degraded - final),
            "frequency_share_of_total_percent": 100 * (spatial - final) / (degraded - final),
        })
    frame = pd.DataFrame(rows)
    write_table(frame, out_dir, "F6_component_contribution", latex=True)
    return frame


def figure_component_contribution(table: pd.DataFrame, out: Path) -> None:
    panels = [(row.pipeline, row.task, row) for row in table.itertuples()]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 4.2), sharey=False)
    if len(panels) == 1:
        axes = [axes]
    for ax, (pipeline, task, row) in zip(axes, panels):
        steps = ["PatchTST", "+ spatial" + chr(10) + "structure", "+ frequency" + chr(10) + "adapter"]
        values = [row.patchtst_rmse, row.after_spatial_rmse, row.final_rmse]
        x = np.arange(len(steps))
        ax.bar(x, values, color=[COLOR_DEGRADED, COLOR_ST, COLOR_RFFT], edgecolor="black", linewidth=0.4)
        for xi, value in enumerate(values):
            ax.annotate(f"{value:.3f}", (xi, value), ha="center", va="bottom", fontsize=8, xytext=(0, 4),
                        textcoords="offset points")
        for xi in range(len(steps) - 1):
            gain = values[xi] - values[xi + 1]
            pct = 100 * gain / values[xi]
            ax.annotate(f"−{gain:.3f}\n(−{pct:.2f}%)",
                        ((xi + xi + 1) / 2, max(values[xi], values[xi + 1])),
                        ha="center", va="bottom", fontsize=7.5, color="#333333")
        ax.set_xticks(x, steps, fontsize=8)
        ax.set_ylabel("test-split RMSE ($\\mu g/m^3$)" if "legacy" in pipeline else "validation RMSE ($\\mu g/m^3$)")
        ax.set_title(f"{task} — {pipeline}", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        lo = min(values) * 0.985
        ax.set_ylim(lo, max(values) * 1.01)
    fig.suptitle("Component contributions: PatchTST baseline, spatial structure, and frequency adapter", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "FF8_component_contribution.pdf")
    fig.savefig(out / "FF8_component_contribution.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-24h", required=True)
    parser.add_argument("--run-168h", required=True)
    parser.add_argument("--out-tables", default="tables/frequency_legacy")
    parser.add_argument("--out-figures", default="figures/frequency_legacy")
    parser.add_argument("--clean-reference", default="experiments/results/frequency_legacy_rerun/clean_pipeline_reference")
    args = parser.parse_args()

    run_dirs = {"24h_1h": Path(args.run_24h), "168h_6h": Path(args.run_168h)}
    out_tables, out_figures = Path(args.out_tables), Path(args.out_figures)
    out_figures.mkdir(parents=True, exist_ok=True)
    frames = load_runs(run_dirs)

    main_table = build_tables(frames, run_dirs, out_tables)
    build_vs_degraded(frames, out_tables)
    contribution = build_component_contribution(frames, run_dirs, Path(args.clean_reference), out_tables)
    figure_component_contribution(contribution, out_figures)
    figure_reduction_vs_degraded(frames, out_figures)
    figure_per_seed_reduction(frames, out_figures)
    figure_rmse_levels(frames, out_figures)
    figure_cumulative(main_table, out_figures)
    figure_per_lead_168h(run_dirs, out_figures)
    figure_per_seed_levels(frames, out_figures)
    examples = figure_example_series(run_dirs, frames, out_figures)

    print(json.dumps({
        "tables": sorted(p.name for p in out_tables.glob("*.csv")),
        "figures": sorted(p.name for p in out_figures.glob("*.pdf")),
        "examples": examples,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
