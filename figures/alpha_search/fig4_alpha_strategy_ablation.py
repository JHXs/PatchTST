"""Figure 4: three-seed alpha strategy ablation (real experiment data)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "figures/data"
OUTPUT_DIR = Path(__file__).resolve().parent
FONT_PATH = Path("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc")
FONT = fm.FontProperties(fname=FONT_PATH) if FONT_PATH.exists() else None
ORDER = ["alpha_zero", "uniform_alpha", "per_pollutant_alpha", "learnable_alpha"]
LABELS = ["alpha=0", "统一 alpha", "逐污染物 alpha", "可学习 alpha"]
COLORS = ["#999999", "#33BBEE", "#0077BB", "#EE7733"]


def main():
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "savefig.dpi": 450,
        "savefig.bbox": "tight",
    })
    raw = pd.read_csv(DATA_DIR / "alpha_ablation_seed_results.csv")
    summary = pd.read_csv(DATA_DIR / "alpha_ablation_summary.csv").set_index("condition")
    pairwise = pd.read_csv(DATA_DIR / "alpha_ablation_pairwise.csv")

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.1), constrained_layout=True)
    x = np.arange(len(ORDER))
    for index, condition in enumerate(ORDER):
        values = raw.loc[raw["condition"] == condition, "valid_mse"].to_numpy()
        jitter = np.array([-0.06, 0.0, 0.06])
        axes[0].scatter(np.full(3, index) + jitter, values, color=COLORS[index], s=24, zorder=3)
        mean = summary.loc[condition, "valid_mse_mean"]
        low = summary.loc[condition, "valid_mse_ci_low"]
        high = summary.loc[condition, "valid_mse_ci_high"]
        axes[0].errorbar(index, mean, yerr=[[mean-low], [high-mean]], fmt="_",
                         color="black", markersize=12, capsize=3, linewidth=1)
    axes[0].set_xticks(x, LABELS, rotation=25, ha="right", fontproperties=FONT)
    axes[0].set_ylabel("Validation MSE")
    axes[0].set_title("三随机种子结果与 95% CI", fontproperties=FONT)
    axes[0].grid(axis="y", alpha=0.25, linewidth=0.6)

    comparisons = ["uniform_alpha", "per_pollutant_alpha", "learnable_alpha"]
    diffs = [pairwise.loc[
        pairwise["comparison"] == f"alpha_zero - {name}", "mean_mse_difference"
    ].iloc[0] for name in comparisons]
    axes[1].bar(np.arange(3), diffs, color=[COLORS[1], COLORS[2], COLORS[3]], width=0.65)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(np.arange(3), ["统一", "逐污染物", "可学习"],
                      rotation=20, ha="right", fontproperties=FONT)
    axes[1].set_ylabel("MSE reduction")
    axes[1].set_title("相对 alpha=0 的配对改善", fontproperties=FONT)
    axes[1].grid(axis="y", alpha=0.25, linewidth=0.6)
    for i, value in enumerate(diffs):
        axes[1].text(i, value, f"{value:.4f}", ha="center",
                     va="bottom" if value >= 0 else "top", fontsize=7)

    times = [summary.loc[name, "elapsed_minutes_mean"] for name in ORDER]
    axes[2].bar(x, times, color=COLORS, width=0.68)
    axes[2].set_xticks(x, LABELS, rotation=25, ha="right", fontproperties=FONT)
    axes[2].set_ylabel("Training time (min)")
    axes[2].set_title("单次训练平均耗时", fontproperties=FONT)
    axes[2].grid(axis="y", alpha=0.25, linewidth=0.6)
    for i, value in enumerate(times):
        axes[2].text(i, value, f"{value:.1f}", ha="center", va="bottom", fontsize=7)

    fig.savefig(OUTPUT_DIR / "fig4_alpha_strategy_ablation.png", dpi=450)
    fig.savefig(OUTPUT_DIR / "fig4_alpha_strategy_ablation.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
