"""端到端参考臂分析：在"不冻结主干、同等训练容量"条件下重做结构对照。

背景
----
主模型论文口径的 ST 臂是**冻结主干 + 约 1k 可训练参数**，而基线对比里的多变量基线
（`multi_gru` 等）是**端到端训练 21k 参数**。两者的差异既包含"空间结构"也包含"训练制度"，
因此需要本脚本对应的参考臂：同一 ST 结构、不冻结、不从退化基线初始化，逐字段复用
`run_beijing_leakfree_coverage.config_for`（epochs/patience/batch/lr/划分/Top-k 完全一致）。

产出（写入 --out-csv）
---------------------
每个配置一行：端到端 ST、端到端 degraded、以及同日同环境跑出的 `multi_gru` 基线，
并给出两个关键相对变化。配对严格限制在"同一 run 内两边都有的种子"。

卫生规则（预注册）
------------------
训练发散但损失仍有限时会产出巨大但有限的 RMSE（实测：168→6 seed 2047 达 5.07e6）。
规则：**RMSE > 5 × 该配置全部行的中位数**即判为发散，从配对中剔除，并在输出里计数。

用法：
    python analyze_reference_arms.py --out-csv tables/baselines/B5_reference_arms.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE_ARMS = ("degraded_patchtst", "st_sparse_station_bias_delta_forecast")
DIVERGENCE_MULTIPLIER = 5.0


def collect(reference_root: Path, baseline_root: Path) -> tuple[pd.DataFrame, list[dict]]:
    rows, diverged_notes = [], []
    pattern = str(reference_root / "*" / "raw_metrics.csv")
    for path in sorted(glob.glob(pattern)):
        config_name = os.path.basename(os.path.dirname(path))
        match = re.match(r"(\d+)h_(\d+)h$", config_name)
        if match is None:
            continue
        history, horizon = int(match.group(1)), int(match.group(2))
        frame = pd.read_csv(path)
        frame["rmse"] = pd.to_numeric(frame["rmse_ugm3"], errors="coerce")
        median = float(frame["rmse"].median())
        frame["diverged"] = frame["rmse"] > DIVERGENCE_MULTIPLIER * median
        for _, bad in frame[frame["diverged"]].iterrows():
            diverged_notes.append(
                {"config": config_name, "arm": bad["variant"], "seed": int(bad["seed"]),
                 "rmse_ugm3": float(bad["rmse"]), "median_rmse_ugm3": median}
            )
        clean = frame[~frame["diverged"]]
        spatial = clean[clean["variant"] == REFERENCE_ARMS[1]].set_index("seed")["rmse"]
        degraded = clean[clean["variant"] == REFERENCE_ARMS[0]].set_index("seed")["rmse"]
        paired_seeds = sorted(set(spatial.index) & set(degraded.index))
        if not paired_seeds:
            continue

        baseline_path = baseline_root / "beijing" / config_name / "raw_metrics.csv"
        multichannel = float("nan")
        multichannel_seeds = 0
        if baseline_path.is_file():
            raw = pd.read_csv(baseline_path)
            raw["rmse"] = pd.to_numeric(raw["rmse_ugm3"], errors="coerce")
            status = raw["status"].astype(str) if "status" in raw else pd.Series(
                ["completed"] * len(raw))
            arm = raw[(raw["arm"] == "multi_gru") & (status == "completed")
                      & np.isfinite(raw["rmse"])].copy()
            arm["seed_i"] = pd.to_numeric(arm["seed"], errors="coerce")
            selected = arm[arm["seed_i"].isin(paired_seeds)]
            if len(selected):
                multichannel = float(selected.groupby("seed_i")["rmse"].min().mean())
                multichannel_seeds = int(selected["seed_i"].nunique())

        spatial_mean = float(spatial.loc[paired_seeds].mean())
        degraded_mean = float(degraded.loc[paired_seeds].mean())
        rows.append({
            "history": history, "horizon": horizon, "paired_seeds": len(paired_seeds),
            "e2e_st_rmse_ugm3": spatial_mean,
            "e2e_degraded_rmse_ugm3": degraded_mean,
            "e2e_st_vs_degraded_percent": 100 * (spatial_mean - degraded_mean) / degraded_mean,
            "multi_gru_rmse_ugm3": multichannel,
            "multi_gru_seeds": multichannel_seeds,
            "e2e_st_vs_multi_gru_percent": (
                100 * (spatial_mean - multichannel) / multichannel
                if np.isfinite(multichannel) else np.nan
            ),
        })
    return pd.DataFrame(rows), diverged_notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", default="experiments/results/st_reference_arms")
    parser.add_argument("--baseline-root", default="experiments/results/baselines")
    parser.add_argument("--out-csv", default="tables/baselines/B5_reference_arms.csv")
    parser.add_argument("--out-md", default="tables/baselines/B5_reference_arms.md")
    args = parser.parse_args()

    table, diverged = collect(Path(args.reference_root), Path(args.baseline_root))
    if table.empty:
        raise SystemExit("未找到参考臂结果")
    table = table.sort_values(["history", "horizon"]).reset_index(drop=True)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False, float_format="%.6f")

    against_degraded = table["e2e_st_vs_degraded_percent"]
    against_gru = table["e2e_st_vs_multi_gru_percent"].dropna()
    lines = [
        "# B5 端到端参考臂（不冻结主干 / 同等训练容量）",
        "",
        "负值表示端到端 ST 更优。配对仅使用同一 run 内两边共有的种子。",
        f"卫生规则：RMSE > {DIVERGENCE_MULTIPLIER:g} × 该配置中位数判为发散并剔除"
        f"（本次剔除 {len(diverged)} 行）。",
        "",
        "| L | H | 配对种子 | 端到端 ST | 端到端 degraded | 相对自身基线 | multi_gru | 相对 multi_gru |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"| {int(row['history'])} | {int(row['horizon'])} | {int(row['paired_seeds'])} | "
            f"{row['e2e_st_rmse_ugm3']:.3f} | {row['e2e_degraded_rmse_ugm3']:.3f} | "
            f"{row['e2e_st_vs_degraded_percent']:+.3f}% | "
            f"{row['multi_gru_rmse_ugm3']:.3f} | "
            f"{row['e2e_st_vs_multi_gru_percent']:+.3f}% |"
        )
    lines += [
        "",
        "## 汇总",
        "",
        f"- 端到端 ST vs 端到端 degraded：平均 **{against_degraded.mean():+.3f}%**，"
        f"更优 **{int((against_degraded < 0).sum())}/{len(table)}** 配置",
        f"- 端到端 ST vs `multi_gru`（同容量、同信息集）：平均 **{against_gru.mean():+.3f}%**，"
        f"更优 **{int((against_gru < 0).sum())}/{len(against_gru)}** 配置",
        "",
        "## 判读",
        "",
        "① 端到端条件下，ST 结构相对同条件自身基线仍然稳定为负（更优），说明空间残差不是",
        "冻结训练制度的产物；② 但与同容量、同信息集的多变量 GRU 相比仍平均更差，",
        "因此论文不能声称该结构在全信息集上优于常规多变量建模。",
    ]
    if diverged:
        lines += ["", "## 被剔除的发散行", ""]
        lines += [
            f"- {item['config']} / {item['arm']} / seed {item['seed']}："
            f"RMSE {item['rmse_ugm3']:.1f}（该配置中位数 {item['median_rmse_ugm3']:.1f}）"
            for item in diverged
        ]
    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已写出 {out_csv} 与 {args.out_md}")
    print(f"端到端 ST vs degraded：{against_degraded.mean():+.3f}%（更优 "
          f"{int((against_degraded < 0).sum())}/{len(table)}）")
    print(f"端到端 ST vs multi_gru：{against_gru.mean():+.3f}%（更优 "
          f"{int((against_gru < 0).sum())}/{len(against_gru)}）")
    print(f"剔除发散行：{len(diverged)}")


if __name__ == "__main__":
    main()
