"""汇总并独立复算"预测步覆盖扩展"的结果，输出网格表与趋势图。

数据来源：`experiments/results/horizon_coverage/*/raw_metrics.csv` 与同目录 `predictions/*.npz`。
复算方式：从每个种子的逐样本预测重新计算 RMSE，与运行器写入的 raw_metrics 比对（不一致即报错），
再用同种子配对计算 ST 相对 degraded PatchTST 的降幅。

用法：
    python summarize_horizon_coverage.py --results experiments/results/horizon_coverage \
        --out-tables tables/horizon_coverage --out-figures figures/horizon_coverage
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = "degraded_patchtst"
SPATIAL = "st_sparse_station_bias_delta_forecast"
COLORS = {24: "#4c72b0", 48: "#55a868", 72: "#dd8452", 168: "#c44e52"}
CONFIRMED = {(24, 1): 3.1188, (168, 6): 0.8220}  # 主确认的配对降幅（%），用于补齐网格


def to_markdown(frame: pd.DataFrame, floatfmt: str = ".3f") -> str:
    def render(value) -> str:
        if isinstance(value, (float, np.floating)):
            return "nan" if np.isnan(value) else format(float(value), floatfmt)
        return str(value)

    columns = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(render(row[c]) for c in frame.columns) + " |")
    return "\n".join(lines) + "\n"


def write_table(frame: pd.DataFrame, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / f"{stem}.csv", index=False, float_format="%.6f")
    (out_dir / f"{stem}.md").write_text(to_markdown(frame), encoding="utf-8")
    (out_dir / f"{stem}.tex").write_text(
        frame.to_latex(index=False, float_format="%.6f", escape=True, caption=stem, label=f"tab:{stem}"),
        encoding="utf-8",
    )


def recompute(config_dir: Path) -> dict:
    """Independent rebuild: RMSE from saved predictions, checked against the runner's metric."""
    raw = pd.read_csv(config_dir / "raw_metrics.csv")
    rows = []
    for variant in (BASE, SPATIAL):
        for _, record in raw[raw.variant == variant].iterrows():
            seed = int(record.seed)
            data = np.load(config_dir / "predictions" / f"{variant}_seed{seed}.npz")
            prediction = data["prediction_ugm3"]
            target = data["target_ugm3"]
            rmse = float(np.sqrt(np.mean((prediction - target) ** 2)))
            if abs(rmse - float(record.rmse_ugm3)) > 1e-4:
                raise AssertionError(f"{config_dir.name} {variant} seed{seed}: 复算 {rmse} != 记录 {record.rmse_ugm3}")
            rows.append({"variant": variant, "seed": seed, "rmse": rmse,
                         "alpha": float(record.get("alpha", np.nan)),
                         "residual_ratio": float(record.get("spatial_residual_rms_ratio", np.nan))})
    frame = pd.DataFrame(rows)
    pivot = frame.pivot_table(index="seed", columns="variant", values="rmse")
    reduction = 100 * (pivot[BASE] - pivot[SPATIAL]) / pivot[BASE]
    return {
        "seeds": len(pivot),
        "base_rmse": float(pivot[BASE].mean()),
        "spatial_rmse": float(pivot[SPATIAL].mean()),
        "reduction_mean": float(reduction.mean()),
        "reduction_std": float(reduction.std()),
        "improved_seeds": int((reduction > 0).sum()),
        "per_seed_reduction": {int(s): float(v) for s, v in reduction.items()},
        "alpha_mean": float(frame[frame.variant == SPATIAL].alpha.mean()),
        "residual_ratio_mean": float(frame[frame.variant == SPATIAL].residual_ratio.mean()),
    }


def recompute_frequency(freq_dir: Path, stage_a_dir: Path) -> dict:
    """独立复算频域臂：从预测重算 RMSE，并与阶段 A 的 ST 臂配对。"""
    raw = pd.read_csv(freq_dir / "raw_metrics.csv")
    stage_a = pd.read_csv(stage_a_dir / "raw_metrics.csv")
    meta = json.loads((freq_dir / "dataset_metadata.json").read_text(encoding="utf-8"))
    center_mean, center_std = float(meta["center_mean"]), float(meta["center_std"])

    def to_physical(array: np.ndarray) -> np.ndarray:
        return array * center_std + center_mean if float(np.nanmean(array)) < 20 else array
    st_map = {int(r.seed): float(r.rmse_ugm3) for r in stage_a[stage_a.variant == SPATIAL].itertuples()}
    base_map = {int(r.seed): float(r.rmse_ugm3) for r in stage_a[stage_a.variant == BASE].itertuples()}
    out = {}
    for variant in ("st_rfft", "st_time"):
        rows = raw[raw.variant == variant]
        rmse = {}
        for record in rows.itertuples():
            data = np.load(freq_dir / "predictions" / f"{variant}_seed{int(record.seed)}.npz")
            prediction = to_physical(data["prediction_ugm3"])
            target = to_physical(data["target_ugm3"])
            value = float(np.sqrt(np.mean((prediction - target) ** 2)))
            if abs(value - float(record.rmse_ugm3)) > 1e-4:
                raise AssertionError(f"{freq_dir.name} {variant} seed{record.seed}: 复算 {value} != 记录 {record.rmse_ugm3}")
            rmse[int(record.seed)] = value
        vs_st = 100 * np.array([(st_map[s] - rmse[s]) / st_map[s] for s in sorted(rmse)])
        vs_base = 100 * np.array([(base_map[s] - rmse[s]) / base_map[s] for s in sorted(rmse)])
        out[variant] = {"rmse_mean": float(np.mean(list(rmse.values()))),
                        "reduction_vs_st": float(vs_st.mean()), "reduction_vs_st_std": float(vs_st.std()),
                        "improved_vs_st": int((vs_st > 0).sum()), "seeds": len(rmse),
                        "reduction_vs_patchtst": float(vs_base.mean())}
    out["frequency_minus_control"] = out["st_rfft"]["reduction_vs_st"] - out["st_time"]["reduction_vs_st"]
    return out

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="experiments/results/horizon_coverage")
    parser.add_argument("--out-tables", default="tables/horizon_coverage")
    parser.add_argument("--out-figures", default="figures/horizon_coverage")
    args = parser.parse_args()

    results = Path(args.results)
    out_tables, out_figures = Path(args.out_tables), Path(args.out_figures)
    out_figures.mkdir(parents=True, exist_ok=True)

    rows = []
    for config_dir in sorted(results.iterdir()):
        if not config_dir.is_dir() or not (config_dir / "raw_metrics.csv").exists():
            continue
        history, horizon = config_dir.name.split("h_")
        metrics = recompute(config_dir)
        rows.append({"history": int(history), "horizon": int(horizon.rstrip("h")),
                     **{k: v for k, v in metrics.items() if k != "per_seed_reduction"},
                     "per_seed_reduction": json.dumps(metrics["per_seed_reduction"], ensure_ascii=False)})
    grid = pd.DataFrame(rows).sort_values(["history", "horizon"]).reset_index(drop=True)

    # 主确认的两个格子不在本轮结果目录里，用确认值补齐并标注
    supplements = []
    for (history, horizon), value in CONFIRMED.items():
        if not ((grid.history == history) & (grid.horizon == horizon)).any():
            supplements.append({"history": history, "horizon": horizon, "seeds": 5,
                                "base_rmse": np.nan, "spatial_rmse": np.nan, "reduction_mean": value,
                                "reduction_std": np.nan, "improved_seeds": 5, "alpha_mean": np.nan,
                                "residual_ratio_mean": np.nan, "per_seed_reduction": "{}", "source": "confirmed"})
    grid["source"] = "horizon_coverage"
    if supplements:
        grid = pd.concat([grid, pd.DataFrame(supplements)], ignore_index=True).sort_values(["history", "horizon"]).reset_index(drop=True)
    grid["reduction_mean"] = grid.reduction_mean.astype(float)
    write_table(grid.drop(columns=["per_seed_reduction"]), out_tables, "H1_grid_detail")

    reduction_grid = grid.pivot(index="history", columns="horizon", values="reduction_mean")
    improved_grid = grid.pivot(index="history", columns="horizon", values="improved_seeds")
    write_table(reduction_grid.reset_index(), out_tables, "H2_reduction_grid")
    write_table(improved_grid.reset_index(), out_tables, "H3_improved_seeds_grid")
    write_table(grid.pivot(index="history", columns="horizon", values="base_rmse").reset_index(),
                out_tables, "H4_patchtst_rmse_grid")
    write_table(grid.pivot(index="history", columns="horizon", values="spatial_rmse").reset_index(),
                out_tables, "H5_st_rmse_grid")

    # 趋势图：降幅随预测步变化（每个历史窗口一条线）
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    ax = axes[0]
    for history, group in grid.groupby("history"):
        group = group.sort_values("horizon")
        ax.plot(group.horizon, group.reduction_mean, marker="o", markersize=5, linewidth=1.8,
                color=COLORS.get(int(history), "#555555"), label=f"L={int(history)}")
        for _, row in group.iterrows():
            ax.annotate(f"{row.reduction_mean:.2f}", (row.horizon, row.reduction_mean), fontsize=6.5,
                        xytext=(3, 4), textcoords="offset points")
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xlabel("forecast horizon H (h)")
    ax.set_ylabel("RMSE reduction vs PatchTST (%)")
    ax.set_xticks(sorted(grid.horizon.unique()))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, title="history L")
    ax.set_title("Spatial structure gain vs forecast horizon", fontsize=10)

    ax2 = axes[1]
    for history, group in grid.groupby("history"):
        group = group.sort_values("horizon")
        ax2.plot(group.horizon, group.base_rmse, marker="o", markersize=4, linewidth=1.5,
                 color=COLORS.get(int(history), "#555555"), label=f"PatchTST L={int(history)}")
        ax2.plot(group.horizon, group.spatial_rmse, marker="s", markersize=4, linewidth=1.5, linestyle="--",
                 color=COLORS.get(int(history), "#555555"), alpha=0.7, label=f"ST L={int(history)}")
    ax2.set_xlabel("forecast horizon H (h)")
    ax2.set_ylabel("test-split RMSE ($\\mu g/m^3$)")
    ax2.set_xticks(sorted(grid.horizon.unique()))
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=6.5, ncol=2)
    ax2.set_title("RMSE levels (solid: PatchTST, dashed: ST)", fontsize=10)
    fig.suptitle("Horizon coverage: frozen spatial structure vs PatchTST across history lengths", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_figures / "HF1_horizon_trend.pdf")
    fig.savefig(out_figures / "HF1_horizon_trend.png", dpi=300)
    plt.close(fig)

    # 热图：降幅网格
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    data = reduction_grid.values.astype(float)
    im = ax.imshow(data, cmap="RdYlGn", vmin=-0.5, vmax=4.0, aspect="auto")
    ax.set_xticks(range(data.shape[1]), [str(int(h)) for h in reduction_grid.columns])
    ax.set_yticks(range(data.shape[0]), [f"L={int(l)}" for l in reduction_grid.index])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isnan(data[i, j]):
                continue
            ax.text(j, i, f"{data[i, j]:+.2f}%", ha="center", va="center", fontsize=8)
    ax.set_xlabel("forecast horizon H (h)")
    ax.set_ylabel("history length L (h)")
    fig.colorbar(im, ax=ax, label="RMSE reduction vs PatchTST (%)")
    ax.set_title("Gain grid (5 seeds, paired)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_figures / "HF2_gain_heatmap.pdf")
    fig.savefig(out_figures / "HF2_gain_heatmap.png", dpi=300)
    plt.close(fig)

    # ---- 频域阶段（阶段 B）：ST+频域 vs ST，以及与时域对照的内部比较 ----
    freq_root = results.parent / "horizon_coverage_frequency"
    freq_rows = []
    if freq_root.is_dir():
        for freq_dir in sorted(freq_root.iterdir()):
            stage_a_dir = results / freq_dir.name
            if not (freq_dir / "raw_metrics.csv").exists() or not (stage_a_dir / "raw_metrics.csv").exists():
                continue
            history, horizon = freq_dir.name.split("h_")
            try:
                stats = recompute_frequency(freq_dir, stage_a_dir)
            except FileNotFoundError as exc:
                print(f"[跳过频域] {freq_dir.name}: 缺少预测文件（{exc.filename}）")
                continue
            freq_rows.append({"history": int(history), "horizon": int(horizon.rstrip("h")),
                              "st_rfft_rmse": stats["st_rfft"]["rmse_mean"],
                              "st_time_rmse": stats["st_time"]["rmse_mean"],
                              "rfft_reduction_vs_st": stats["st_rfft"]["reduction_vs_st"],
                              "rfft_improved_seeds": stats["st_rfft"]["improved_vs_st"],
                              "time_reduction_vs_st": stats["st_time"]["reduction_vs_st"],
                              "rfft_reduction_vs_patchtst": stats["st_rfft"]["reduction_vs_patchtst"],
                              "frequency_minus_control": stats["frequency_minus_control"]})
    if freq_rows:
        freq = pd.DataFrame(freq_rows).sort_values(["history", "horizon"]).reset_index(drop=True)
        write_table(freq, out_tables, "H6_frequency_detail")
        for column, stem in (("rfft_reduction_vs_st", "H7_frequency_vs_st_grid"),
                             ("rfft_reduction_vs_patchtst", "H8_frequency_vs_patchtst_grid"),
                             ("frequency_minus_control", "H9_frequency_minus_control_grid")):
            write_table(freq.pivot(index="history", columns="horizon", values=column).reset_index(), out_tables, stem)

        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
        ax = axes[0]
        for history, group in freq.groupby("history"):
            group = group.sort_values("horizon")
            ax.plot(group.horizon, group.rfft_reduction_vs_st, marker="o", markersize=4, linewidth=1.8,
                    color=COLORS.get(int(history), "#555555"), label=f"L={int(history)}")
        ax.axhline(0, color="black", linewidth=0.9)
        ax.set_xlabel("forecast horizon H (h)")
        ax.set_ylabel("RMSE reduction of ST+frequency over ST (%)")
        ax.set_xticks(sorted(freq.horizon.unique()))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, title="history L")
        ax.set_title("Frequency adapter gain over the frozen ST", fontsize=10)

        ax2 = axes[1]
        for history, group in freq.groupby("history"):
            group = group.sort_values("horizon")
            ax2.plot(group.horizon, group.frequency_minus_control, marker="s", markersize=4, linewidth=1.6,
                     color=COLORS.get(int(history), "#555555"), label=f"L={int(history)}")
        ax2.axhline(0, color="black", linewidth=0.9)
        ax2.set_xlabel("forecast horizon H (h)")
        ax2.set_ylabel("frequency minus time-domain control (pp)")
        ax2.set_xticks(sorted(freq.horizon.unique()))
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=7, ncol=2)
        ax2.set_title("Internal check: frequency vs equal-capacity control", fontsize=10)
        fig.suptitle("Frequency adapter across history lengths and horizons (5 seeds, test split)", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_figures / "HF3_frequency_trend.pdf")
        fig.savefig(out_figures / "HF3_frequency_trend.png", dpi=300)
        plt.close(fig)

        print(json.dumps({"frequency_configs": len(freq),
                          "rfft_vs_st_grid": freq.pivot(index="history", columns="horizon",
                                                        values="rfft_reduction_vs_st").round(3).to_dict()},
                         ensure_ascii=False, indent=2))
    print(json.dumps({"configs": len(grid), "grid": reduction_grid.round(3).to_dict()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
