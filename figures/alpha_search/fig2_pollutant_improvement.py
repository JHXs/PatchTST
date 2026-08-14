"""Figure 2: per-pollutant validation MSE improvement (real data)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "figures/data/pollutant_mse_improvement.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
FONT_PATH = Path("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc")
FONT = fm.FontProperties(fname=FONT_PATH) if FONT_PATH.exists() else None
COLORS = ["#B8D8E8", "#0077BB", "#EE7733"]


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
    df = pd.read_csv(DATA_PATH)
    x = np.arange(len(df))
    width = 0.36

    fig, (ax_mse, ax_gain) = plt.subplots(
        2, 1, figsize=(7.2, 6.0), gridspec_kw={"height_ratios": [1.65, 1]},
        constrained_layout=True,
    )
    ax_mse.bar(x - width / 2, df["mse_alpha_0"], width,
               color=COLORS[0], edgecolor="white", label="alpha=0")
    ax_mse.bar(x + width / 2, df["mse_calibrated"], width,
               color=COLORS[1], edgecolor="white", label="校准 alpha")
    ax_mse.set_xticks(x, df["pollutant"])
    ax_mse.set_ylabel("Validation MSE")
    ax_mse.set_title("最佳模型校准前后的逐污染物误差", fontproperties=FONT)
    ax_mse.legend(frameon=False, prop=FONT)
    ax_mse.grid(axis="y", alpha=0.25, linewidth=0.6)

    gain = df["relative_improvement_percent"].to_numpy()
    bars = ax_gain.bar(x, gain, color=COLORS[2], edgecolor="white", width=0.62)
    ax_gain.set_xticks(x, df["pollutant"])
    ax_gain.set_ylabel("MSE improvement (%)")
    ax_gain.set_title("邻站残差带来的相对改善", fontproperties=FONT)
    ax_gain.grid(axis="y", alpha=0.25, linewidth=0.6)
    for bar, value in zip(bars, gain):
        ax_gain.text(bar.get_x() + bar.get_width() / 2, value + 0.14,
                     f"{value:.2f}%", ha="center", va="bottom", fontsize=8)
    ax_gain.set_ylim(0, gain.max() * 1.18)

    fig.savefig(OUTPUT_DIR / "fig2_pollutant_improvement.png", dpi=450)
    fig.savefig(OUTPUT_DIR / "fig2_pollutant_improvement.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
