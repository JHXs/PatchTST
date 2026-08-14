"""Figure 6: strict PatchTST baseline versus spatial fusion strategies."""

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
ORDER = [
    "strict_patchtst",
    "uniform_alpha",
    "per_pollutant_alpha",
    "learnable_alpha",
]
LABELS = ["严格 PatchTST", "统一 alpha", "逐污染物 alpha", "可学习 alpha"]
COLORS = ["#999999", "#33BBEE", "#0077BB", "#EE7733"]
POLLUTANTS = ["PM2.5", "PM10", "NO2", "CO", "O3", "SO2"]


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
    raw = pd.read_csv(DATA_DIR / "strict_patchtst_seed_results.csv")
    summary = pd.read_csv(DATA_DIR / "strict_patchtst_summary.csv").set_index("condition")
    pollutant = pd.read_csv(DATA_DIR / "strict_patchtst_pollutant_improvement.csv")

    fig, (ax_mse, ax_heat) = plt.subplots(
        1, 2, figsize=(7.2, 3.5), gridspec_kw={"width_ratios": [1.05, 1.55]},
        constrained_layout=True,
    )
    for index, condition in enumerate(ORDER):
        values = raw.loc[raw["condition"] == condition, "valid_mse"].to_numpy()
        ax_mse.scatter(
            np.full(3, index) + np.array([-0.06, 0, 0.06]), values,
            color=COLORS[index], s=26, zorder=3,
        )
        mean = summary.loc[condition, "valid_mse_mean"]
        low = summary.loc[condition, "valid_mse_ci_low"]
        high = summary.loc[condition, "valid_mse_ci_high"]
        ax_mse.errorbar(
            index, mean, yerr=[[mean - low], [high - mean]], fmt="_",
            color="black", markersize=13, capsize=3, linewidth=1,
        )
    ax_mse.set_xticks(np.arange(4), LABELS, rotation=25, ha="right", fontproperties=FONT)
    ax_mse.set_ylabel("Validation MSE")
    ax_mse.set_title("严格基线与空间融合策略", fontproperties=FONT)
    ax_mse.grid(axis="y", alpha=0.25, linewidth=0.6)

    spatial = ORDER[1:]
    matrix = np.zeros((3, len(POLLUTANTS)))
    for row, condition in enumerate(spatial):
        group = pollutant[pollutant["condition"] == condition].set_index("pollutant")
        matrix[row] = group.loc[POLLUTANTS, "improvement_percent"]
    limit = max(abs(matrix.min()), abs(matrix.max()))
    image = ax_heat.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax_heat.text(
                col, row, f"{matrix[row, col]:+.1f}%", ha="center", va="center",
                fontsize=7, color="white" if abs(matrix[row, col]) > 0.65 * limit else "#222222",
            )
    ax_heat.set_xticks(np.arange(len(POLLUTANTS)), POLLUTANTS)
    ax_heat.set_yticks(np.arange(3), LABELS[1:], fontproperties=FONT)
    ax_heat.set_title("相对严格 PatchTST 的逐污染物改善", fontproperties=FONT)
    colorbar = fig.colorbar(image, ax=ax_heat, fraction=0.045, pad=0.03)
    colorbar.set_label("MSE improvement (%)")

    fig.savefig(OUTPUT_DIR / "fig6_strict_patchtst_comparison.png", dpi=450)
    fig.savefig(OUTPUT_DIR / "fig6_strict_patchtst_comparison.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
