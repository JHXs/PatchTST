"""Figure 1: alpha search trajectory and validation MSE (real data)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "figures/data/alpha_search_trajectory.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
FONT_PATH = Path("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc")
FONT = fm.FontProperties(fname=FONT_PATH) if FONT_PATH.exists() else None
COLORS = ["#0077BB", "#33BBEE", "#009988", "#EE7733", "#CC3311"]


def apply_style():
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "savefig.dpi": 450,
        "savefig.bbox": "tight",
    })


def main():
    apply_style()
    df = pd.read_csv(DATA_PATH).sort_values("stage_order")
    pollutants = ["PM2.5", "PM10", "NO2", "CO", "O3", "SO2"]
    matrix = df[pollutants].to_numpy().T
    labels = df["stage"].tolist()

    fig, (ax_heat, ax_mse) = plt.subplots(
        2, 1, figsize=(7.2, 6.4), gridspec_kw={"height_ratios": [2.1, 1]},
        constrained_layout=True,
    )
    cmap = LinearSegmentedColormap.from_list(
        "alpha_scale", ["#F7FBFF", "#9ECAE1", "#2E86AB", "#A23B72"]
    )
    image = ax_heat.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=0.4)
    ax_heat.set_xticks(np.arange(len(labels)), labels, rotation=18, ha="right")
    ax_heat.set_yticks(np.arange(len(pollutants)), pollutants)
    ax_heat.set_title("逐污染物 alpha 搜索轨迹", fontproperties=FONT)
    ax_heat.set_xlabel("实验阶段", fontproperties=FONT)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            color = "white" if value >= 0.18 else "#222222"
            ax_heat.text(col, row, f"{value:.3f}", ha="center", va="center",
                         fontsize=7.5, color=color)
    colorbar = fig.colorbar(image, ax=ax_heat, fraction=0.025, pad=0.02)
    colorbar.set_label("alpha")

    x = np.arange(len(df))
    mse = df["valid_mse"].to_numpy()
    ax_mse.plot(x, mse, color=COLORS[0], marker="o", linewidth=1.8, markersize=5)
    ax_mse.fill_between(x, mse, mse.max() + 0.001, color=COLORS[1], alpha=0.12)
    ax_mse.set_xticks(x, labels, rotation=18, ha="right")
    ax_mse.set_ylabel("Validation MSE")
    ax_mse.set_title("验证集总体误差变化", fontproperties=FONT)
    ax_mse.grid(axis="y", alpha=0.25, linewidth=0.6)
    margin = (mse.max() - mse.min()) * 0.18
    ax_mse.set_ylim(mse.min() - margin, mse.max() + margin)
    for xpos, value in zip(x, mse):
        ax_mse.annotate(f"{value:.4f}", (xpos, value), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=7.5)

    fig.savefig(OUTPUT_DIR / "fig1_alpha_search_trajectory.png", dpi=450)
    fig.savefig(OUTPUT_DIR / "fig1_alpha_search_trajectory.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
