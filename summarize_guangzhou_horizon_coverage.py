"""汇总阶段 C（广州跨预测步覆盖）：从逐标量预测独立复算 RMSE，并给出各配置的配对降幅。

- 每个 (配置, 中心站, 种子) 的降幅 = (RMSE_degraded − RMSE_ST) / RMSE_degraded，其中 RMSE 由该臂全部
  标量的 SSE/元素数合并计算（与 round14 汇总口径一致：先合并 SSE，不平均块 RMSE）。
- 复算与产物内的 `sse_by_block` 交叉核对，不一致即报错。
- 输出：配置级明细表、降幅网格、改善站数网格、趋势图与热图。

用法：
    python summarize_guangzhou_horizon_coverage.py \
        --results experiments/results/guangzhou_horizon_coverage \
        --out-tables tables/guangzhou_horizon_coverage --out-figures figures/guangzhou_horizon_coverage
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


def to_markdown(frame: pd.DataFrame, floatfmt: str = ".3f") -> str:
    def render(value) -> str:
        if isinstance(value, (float, np.floating)):
            return "nan" if np.isnan(value) else format(float(value), floatfmt)
        return str(value)

    labels = list(frame.columns)
    lines = ["| " + " | ".join(str(c) for c in labels) + " |",
             "| " + " | ".join(["---"] * len(labels)) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(render(row[c]) for c in labels) + " |")
    return "\n".join(lines) + "\n"


def write_table(frame: pd.DataFrame, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / f"{stem}.csv", index=False, float_format="%.6f")
    (out_dir / f"{stem}.md").write_text(to_markdown(frame), encoding="utf-8")
    (out_dir / f"{stem}.tex").write_text(
        frame.to_latex(index=False, float_format="%.6f", escape=True, caption=stem, label=f"tab:{stem}"),
        encoding="utf-8",
    )


def load_arm_sse(path: Path) -> tuple[float, int]:
    """Rebuild pooled SSE and element count from raw scalars; verify against the stored per-block values."""
    data = np.load(path)
    prediction, target = data["prediction_ugm3"], data["target_ugm3"]
    errors = prediction - target
    sse = float(np.square(errors).sum())
    stored_sse = float(np.asarray(data["sse_by_block"], dtype=np.float64).sum())
    stored_count = int(np.asarray(data["element_count_by_block"], dtype=np.int64).sum())
    if abs(sse - stored_sse) > 1e-6 * max(1.0, abs(sse)):
        raise AssertionError(f"{path.name}: 复算 SSE {sse} != 落盘 {stored_sse}")
    if errors.size != stored_count:
        raise AssertionError(f"{path.name}: 标量数 {errors.size} != 落盘 {stored_count}")
    return sse, int(errors.size)


def station_is_complete(station_dir: Path, seeds: list[int]) -> bool:
    predictions = station_dir / "predictions"
    return all((predictions / f"{arm}_seed{seed}.npz").exists()
               for arm in (BASE, SPATIAL) for seed in seeds)


def config_stats(config_dir: Path, seeds: list[int]) -> dict:
    pairs, stations, skipped = [], {}, []
    for station_dir in sorted(config_dir.glob("station_*")):
        if "_partial_" in station_dir.name:  # 续跑时被移开的未完成目录，不参与统计
            continue
        center = int(station_dir.name.split("_")[1])
        if not station_is_complete(station_dir, seeds):
            skipped.append(center)
            continue
        station_reductions = []
        for seed in seeds:
            base_path = station_dir / "predictions" / f"{BASE}_seed{seed}.npz"
            spatial_path = station_dir / "predictions" / f"{SPATIAL}_seed{seed}.npz"
            base_sse, base_n = load_arm_sse(base_path)
            spatial_sse, spatial_n = load_arm_sse(spatial_path)
            base_rmse = float(np.sqrt(base_sse / base_n))
            spatial_rmse = float(np.sqrt(spatial_sse / spatial_n))
            reduction = 100 * (base_rmse - spatial_rmse) / base_rmse
            pairs.append({"station": center, "seed": seed, "base_rmse": base_rmse,
                          "spatial_rmse": spatial_rmse, "reduction": reduction})
            station_reductions.append(reduction)
        stations[center] = float(np.mean(station_reductions))
    frame = pd.DataFrame(pairs)
    if frame.empty:
        return {"pairs": frame, "skipped_stations": skipped, "pool_effect": float("nan"),
                "pool_std": float("nan"), "improved_pairs": 0, "n_pairs": 0,
                "improved_stations": 0, "n_stations": 0, "station_effects": {}}
    return {"pairs": frame, "skipped_stations": skipped,
            "pool_effect": float(frame.reduction.mean()),
            "pool_std": float(frame.reduction.std()),
            "improved_pairs": int((frame.reduction > 0).sum()),
            "n_pairs": int(len(frame)),
            "improved_stations": int(sum(1 for v in stations.values() if v > 0)),
            "n_stations": int(len(stations)),
            "station_effects": stations}


def arm_sse(path: Path) -> tuple[float, int]:
    data = np.load(path)
    errors = data["prediction_ugm3"] - data["target_ugm3"]
    sse = float(np.square(errors).sum())
    stored = float(np.asarray(data["sse_by_block"], dtype=np.float64).sum())
    if abs(sse - stored) > 1e-6 * max(1.0, abs(sse)):
        raise AssertionError(f"{path.name}: 复算 SSE 与落盘不一致")
    return sse, int(errors.size)


def frequency_stats(freq_config_dir: Path, stage_c_config_dir: Path, seeds: list[int]) -> dict:
    """频域臂：相对冻结 ST 与相对 PatchTST 的配对降幅，以及与时域对照的差。"""
    rows = []
    for freq_station in sorted(freq_config_dir.glob("station_*")):
        if "_partial_" in freq_station.name:
            continue
        center = int(freq_station.name.split("_")[1])
        base_station = stage_c_config_dir / freq_station.name
        if not base_station.is_dir():
            continue
        complete = all((freq_station / "predictions" / f"{arm}_seed{seed}.npz").exists()
                       for arm in ("st_rfft", "st_time") for seed in seeds)
        if not complete:
            continue
        for seed in seeds:
            sse_base, n_base = arm_sse(base_station / "predictions" / f"{BASE}_seed{seed}.npz")
            sse_st, n_st = arm_sse(base_station / "predictions" / f"{SPATIAL}_seed{seed}.npz")
            sse_freq, n_freq = arm_sse(freq_station / "predictions" / f"st_rfft_seed{seed}.npz")
            sse_time, n_time = arm_sse(freq_station / "predictions" / f"st_time_seed{seed}.npz")
            rmse = lambda sse, n: float(np.sqrt(sse / n))
            rows.append({"station": center, "seed": seed,
                         "rfft_vs_st": 100 * (rmse(sse_st, n_st) - rmse(sse_freq, n_freq)) / rmse(sse_st, n_st),
                         "rfft_vs_patchtst": 100 * (rmse(sse_base, n_base) - rmse(sse_freq, n_freq)) / rmse(sse_base, n_base),
                         "time_vs_st": 100 * (rmse(sse_st, n_st) - rmse(sse_time, n_time)) / rmse(sse_st, n_st)})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {"n_pairs": 0}
    return {"n_pairs": int(len(frame)),
            "rfft_vs_st": float(frame.rfft_vs_st.mean()),
            "rfft_vs_patchtst": float(frame.rfft_vs_patchtst.mean()),
            "time_vs_st": float(frame.time_vs_st.mean()),
            "frequency_minus_control": float((frame.rfft_vs_st - frame.time_vs_st).mean()),
            "improved_vs_st": int((frame.rfft_vs_st > 0).sum())}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="experiments/results/guangzhou_horizon_coverage")
    parser.add_argument("--seeds", default="7001,7002,7003,7004,7005")
    parser.add_argument("--out-tables", default="tables/guangzhou_horizon_coverage")
    parser.add_argument("--out-figures", default="figures/guangzhou_horizon_coverage")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s]
    results = Path(args.results)
    out_tables, out_figures = Path(args.out_tables), Path(args.out_figures)
    out_figures.mkdir(parents=True, exist_ok=True)

    rows, detail = [], []
    for config_dir in sorted(results.iterdir()):
        if not config_dir.is_dir() or not list(config_dir.glob("station_*")):
            continue
        history, horizon = config_dir.name.split("h_")
        stats = config_stats(config_dir, seeds)
        if stats["n_pairs"] == 0:
            print(f"[跳过] {config_dir.name}: 尚无完成的中心站")
            continue
        rows.append({"history": int(history), "horizon": int(horizon.rstrip("h")),
                     "stations": stats["n_stations"], "pairs": stats["n_pairs"],
                     "skipped_stations": ",".join(str(s) for s in stats["skipped_stations"]),
                     "complete": bool(stats["n_stations"] == 8 and not stats["skipped_stations"]),
                     "pool_effect": stats["pool_effect"], "pool_std": stats["pool_std"],
                     "improved_pairs": stats["improved_pairs"],
                     "improved_stations": stats["improved_stations"]})
        for _, pair in stats["pairs"].iterrows():
            detail.append({"history": int(history), "horizon": int(horizon.rstrip("h")),
                           "station": int(pair.station), "seed": int(pair.seed),
                           "base_rmse": pair.base_rmse, "spatial_rmse": pair.spatial_rmse,
                           "reduction": pair.reduction})
    grid = pd.DataFrame(rows).sort_values(["history", "horizon"]).reset_index(drop=True)
    if grid.empty:
        raise SystemExit("尚未有可汇总的配置")
    write_table(grid, out_tables, "G1_config_summary")
    write_table(pd.DataFrame(detail), out_tables, "G2_pair_detail")
    write_table(grid.pivot(index="history", columns="horizon", values="pool_effect").reset_index(),
                out_tables, "G3_pool_effect_grid")
    write_table(grid.pivot(index="history", columns="horizon", values="improved_stations").reset_index(),
                out_tables, "G4_improved_stations_grid")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    ax = axes[0]
    for history, group in grid.groupby("history"):
        group = group.sort_values("horizon")
        ax.plot(group.horizon, group.pool_effect, marker="o", markersize=5, linewidth=1.8,
                color=COLORS.get(int(history), "#555555"), label=f"L={int(history)}")
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xlabel("forecast horizon H (h)")
    ax.set_ylabel("pooled RMSE reduction vs PatchTST (%)")
    ax.set_xticks(sorted(grid.horizon.unique()))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, title="history L")
    ax.set_title("Guangzhou: spatial structure gain vs horizon", fontsize=10)

    ax2 = axes[1]
    data = grid.pivot(index="history", columns="horizon", values="pool_effect").values.astype(float)
    im = ax2.imshow(data, cmap="RdYlGn", vmin=-0.5, vmax=6.0, aspect="auto")
    ax2.set_xticks(range(data.shape[1]), [str(int(h)) for h in sorted(grid.horizon.unique())])
    ax2.set_yticks(range(data.shape[0]), [f"L={int(l)}" for l in sorted(grid.history.unique())])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                ax2.text(j, i, f"{data[i, j]:+.2f}", ha="center", va="center", fontsize=7.5)
    ax2.set_xlabel("forecast horizon H (h)")
    ax2.set_ylabel("history length L (h)")
    fig.colorbar(im, ax=ax2, label="pooled reduction (%)")
    ax2.set_title("Gain grid (8 stations $\\times$ 5 seeds, pooled)", fontsize=10)
    fig.suptitle("Guangzhou cross-horizon coverage: frozen spatial structure vs PatchTST", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_figures / "GF1_guangzhou_horizon.pdf")
    fig.savefig(out_figures / "GF1_guangzhou_horizon.png", dpi=300)
    plt.close(fig)

    # ---- 阶段 D：广州频域臂 ----
    freq_root = results.parent / "guangzhou_horizon_coverage_frequency"
    freq_rows = []
    if freq_root.is_dir():
        for freq_config_dir in sorted(freq_root.iterdir()):
            if not freq_config_dir.is_dir():
                continue
            stage_c_config_dir = results / freq_config_dir.name
            if not stage_c_config_dir.is_dir():
                continue
            stats = frequency_stats(freq_config_dir, stage_c_config_dir, seeds)
            if stats.get("n_pairs", 0) == 0:
                continue
            history, horizon = freq_config_dir.name.split("h_")
            freq_rows.append({"history": int(history), "horizon": int(horizon.rstrip("h")),
                              "pairs": stats["n_pairs"], "rfft_vs_st": stats["rfft_vs_st"],
                              "rfft_vs_patchtst": stats["rfft_vs_patchtst"],
                              "time_vs_st": stats["time_vs_st"],
                              "frequency_minus_control": stats["frequency_minus_control"],
                              "improved_pairs": stats["improved_vs_st"]})
    if freq_rows:
        freq = pd.DataFrame(freq_rows).sort_values(["history", "horizon"]).reset_index(drop=True)
        write_table(freq, out_tables, "G5_frequency_detail")
        for column, stem in (("rfft_vs_st", "G6_frequency_vs_st_grid"),
                             ("rfft_vs_patchtst", "G7_frequency_vs_patchtst_grid"),
                             ("frequency_minus_control", "G8_frequency_minus_control_grid")):
            write_table(freq.pivot(index="history", columns="horizon", values=column).reset_index(),
                        out_tables, stem)
        fig2, axes2 = plt.subplots(1, 2, figsize=(11.5, 4.0))
        ax3 = axes2[0]
        for history, group in freq.groupby("history"):
            group = group.sort_values("horizon")
            ax3.plot(group.horizon, group.rfft_vs_st, marker="o", markersize=4, linewidth=1.8,
                     color=COLORS.get(int(history), "#555555"), label=f"L={int(history)}")
        ax3.axhline(0, color="black", linewidth=0.9)
        ax3.set_xlabel("forecast horizon H (h)")
        ax3.set_ylabel("RMSE reduction of ST+frequency over ST (%)")
        ax3.set_xticks(sorted(freq.horizon.unique()))
        ax3.grid(alpha=0.3)
        ax3.legend(fontsize=8, title="history L")
        ax3.set_title("Guangzhou: frequency adapter gain over the frozen ST", fontsize=10)
        ax4 = axes2[1]
        for history, group in freq.groupby("history"):
            group = group.sort_values("horizon")
            ax4.plot(group.horizon, group.frequency_minus_control, marker="s", markersize=4, linewidth=1.6,
                     color=COLORS.get(int(history), "#555555"), label=f"L={int(history)}")
        ax4.axhline(0, color="black", linewidth=0.9)
        ax4.set_xlabel("forecast horizon H (h)")
        ax4.set_ylabel("frequency minus time-domain control (pp)")
        ax4.set_xticks(sorted(freq.horizon.unique()))
        ax4.grid(alpha=0.3)
        ax4.legend(fontsize=7, ncol=2)
        ax4.set_title("Internal check: frequency vs equal-capacity control", fontsize=10)
        fig2.suptitle("Guangzhou frequency adapter across history lengths and horizons (8 stations$\\times$5 seeds)",
                      fontsize=10)
        fig2.tight_layout()
        fig2.savefig(out_figures / "GF2_guangzhou_frequency.pdf")
        fig2.savefig(out_figures / "GF2_guangzhou_frequency.png", dpi=300)
        plt.close(fig2)
        print(json.dumps({"frequency_configs": len(freq),
                          "rfft_vs_st_grid": freq.pivot(index="history", columns="horizon",
                                                        values="rfft_vs_st").round(3).to_dict()},
                         ensure_ascii=False, indent=2))
    print(json.dumps({"configs": len(grid),
                      "grid": grid.pivot(index="history", columns="horizon", values="pool_effect").round(3).to_dict()},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
