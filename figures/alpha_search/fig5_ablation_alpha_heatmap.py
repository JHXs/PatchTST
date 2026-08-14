"""Figure 5: selected alpha means and variability for the ablation."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "figures/data/alpha_ablation_selected_alpha.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
FONT_PATH = Path("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc")
FONT = fm.FontProperties(fname=FONT_PATH) if FONT_PATH.exists() else None
ORDER = ["alpha_zero", "uniform_alpha", "per_pollutant_alpha", "learnable_alpha"]
LABELS = ["alpha=0", "统一 alpha", "逐污染物 alpha", "可学习 alpha"]
POLLUTANTS = ["PM2.5", "PM10", "NO2", "CO", "O3", "SO2"]


def main():
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "savefig.dpi": 450,
        "savefig.bbox": "tight",
    })
    df = pd.read_csv(DATA_PATH)
    mean = np.zeros((len(ORDER), len(POLLUTANTS)))
    std = np.zeros_like(mean)
    for row_index, condition in enumerate(ORDER):
        group = df[df["condition"] == condition].set_index("pollutant")
        mean[row_index] = group.loc[POLLUTANTS, "alpha_mean"]
        std[row_index] = group.loc[POLLUTANTS, "alpha_std"]

    fig, ax = plt.subplots(figsize=(7.2, 3.3), constrained_layout=True)
    image = ax.imshow(mean, cmap="Blues", vmin=0, vmax=max(0.4, mean.max()))
    for row in range(mean.shape[0]):
        for col in range(mean.shape[1]):
            color = "white" if mean[row, col] > 0.22 else "#222222"
            ax.text(col, row, f"{mean[row,col]:.3f}\n±{std[row,col]:.3f}",
                    ha="center", va="center", fontsize=7, color=color)
    ax.set_xticks(np.arange(len(POLLUTANTS)), POLLUTANTS)
    ax.set_yticks(np.arange(len(ORDER)), LABELS, fontproperties=FONT)
    ax.set_xlabel("污染物", fontproperties=FONT)
    ax.set_title("不同融合策略的 alpha 均值与标准差（3 seeds）", fontproperties=FONT)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label("alpha mean")
    fig.savefig(OUTPUT_DIR / "fig5_ablation_alpha_heatmap.png", dpi=450)
    fig.savefig(OUTPUT_DIR / "fig5_ablation_alpha_heatmap.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
