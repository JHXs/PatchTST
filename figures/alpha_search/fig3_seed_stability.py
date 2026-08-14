"""Figure 3: calibrated alpha stability across two random seeds (real data)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "figures/data/alpha_seed_stability.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
FONT_PATH = Path("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc")
FONT = fm.FontProperties(fname=FONT_PATH) if FONT_PATH.exists() else None
COLORS = ["#0077BB", "#EE7733"]


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
    pollutants = ["PM2.5", "PM10", "NO2", "CO", "O3", "SO2"]
    y = np.arange(len(pollutants))

    fig, (ax_alpha, ax_mse) = plt.subplots(
        1, 2, figsize=(7.2, 3.7), gridspec_kw={"width_ratios": [2.4, 1]},
        constrained_layout=True,
    )
    for idx, (_, row) in enumerate(df.iterrows()):
        values = row[pollutants].to_numpy(dtype=float)
        plot_values = np.maximum(values, 0.0018)
        offset = (idx - 0.5) * 0.13
        ax_alpha.scatter(plot_values, y + offset, s=38, color=COLORS[idx],
                         label=f"seed {int(row['seed'])}", zorder=3)
    for pollutant_idx in range(len(pollutants)):
        pair = np.maximum(df[pollutants[pollutant_idx]].to_numpy(), 0.0018)
        ax_alpha.plot(pair,
                      [y[pollutant_idx] - 0.065, y[pollutant_idx] + 0.065],
                      color="#9A9A9A", linewidth=0.8, zorder=1)
    ax_alpha.set_yticks(y, pollutants)
    ax_alpha.invert_yaxis()
    ax_alpha.set_xscale("log")
    ax_alpha.set_xlim(0.0013, 0.45)
    ax_alpha.set_xlabel("Calibrated alpha (log scale; O3=0 shown at left)")
    ax_alpha.set_title("双随机种子的 alpha 稳定性", fontproperties=FONT)
    ax_alpha.grid(axis="x", alpha=0.25, linewidth=0.6)
    ax_alpha.legend(frameon=False, fontsize=8)

    bars = ax_mse.bar(
        [str(int(value)) for value in df["seed"]], df["valid_mse"],
        color=COLORS, edgecolor="white", width=0.58,
    )
    ax_mse.set_ylabel("Validation MSE")
    ax_mse.set_xlabel("Random seed")
    ax_mse.set_title("复核结果", fontproperties=FONT)
    ax_mse.set_ylim(0.69, 0.708)
    ax_mse.grid(axis="y", alpha=0.25, linewidth=0.6)
    for bar, value in zip(bars, df["valid_mse"]):
        ax_mse.text(bar.get_x() + bar.get_width() / 2, value + 0.00035,
                    f"{value:.4f}", ha="center", va="bottom", fontsize=8)

    fig.savefig(OUTPUT_DIR / "fig3_seed_stability.png", dpi=450)
    fig.savefig(OUTPUT_DIR / "fig3_seed_stability.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
