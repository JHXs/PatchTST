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

    print(json.dumps({"configs": len(grid),
                      "grid": grid.pivot(index="history", columns="horizon", values="pool_effect").round(3).to_dict()},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
