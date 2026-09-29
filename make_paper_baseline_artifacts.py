"""生成论文口径的基线对比表与图（方向 21：Transformer 族 vs 锁定结构）。

数据来源：``tables/paper/source/*.csv``
（从 ``experiment/baseline-comparison-v2-ablation:tables/baseline_v2/`` 复制，校验见同目录 SHA256 清单）。
本脚本**不做任何训练**，只做后处理；所有数字均可由源 CSV 重算。

用法::

    python make_paper_baseline_artifacts.py \
        --source tables/paper/source --out-tables tables/paper --out-figures figures/paper
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUR_BUDGET = (4588, 5418)          # O-locked 空间头可训练参数区间
MATCHED_ARM = "informer_d12_e2"    # 精确匹配点（5,125）
FAMILY_COLOR = {"informer": "#c0392b", "gru": "#2471a3", "lstm": "#1e8449", "tst": "#7d3c98"}


def load(source: Path) -> dict[str, pd.DataFrame]:
    names = (
        "main_table",
        "capacity_curve",
        "per_lead",
        "per_lead_win_counts",
        "dual_criterion",
        "rank_correlation",
        "compliance_self_check",
        "independent_recalculation",
    )
    return {name: pd.read_csv(source / f"{name}.csv") for name in names}


def table_informer(main_table: pd.DataFrame, capacity: pd.DataFrame,
                   dual: pd.DataFrame) -> pd.DataFrame:
    """PT1：论文主表 —— Informer 四个臂 vs O-locked（配对，共同子集）。"""
    rows = []
    params = (
        capacity[capacity["family"] == "informer"]
        .groupby("arm")["trainable_parameter_count"].median().astype(int)
    )
    for _, item in main_table[
        (main_table["category"] == "new") & (main_table["family"] == "informer")
    ].iterrows():
        arm = item["arm"]
        pair = dual[(dual["family"] == "Informer") & (dual["criterion"] == "validation")]
        rows.append(
            {
                "arm": arm,
                "trainable_parameters": int(params[arm]),
                "budget_ratio_to_ours": round(int(params[arm]) / np.mean(OUR_BUDGET), 2),
                "matched_budget_point": arm == MATCHED_ARM,
                "pooled_baseline_rmse": round(item["pool_mean_baseline_rmse_ugm3"], 4),
                "pooled_our_rmse": round(item["pool_mean_o_locked_rmse_ugm3"], 4),
                "our_advantage_percent": round(item["mean_baseline_relative_to_o_percent"], 3),
                "paired_total": int(item["paired_total"]),
                "our_better_runs": int(item["paired_total"] - item["baseline_better_count"]),
            }
        )
    table = pd.DataFrame(rows).sort_values("trainable_parameters").reset_index(drop=True)
    table["dual_criterion_validation_percent"] = (
        round(pair["mean_baseline_relative_to_o_percent"].iloc[0], 3) if len(pair) else np.nan
    )
    pair_test = dual[(dual["family"] == "Informer") & (dual["criterion"] == "test")]
    table["dual_criterion_test_percent"] = (
        round(pair_test["mean_baseline_relative_to_o_percent"].iloc[0], 3) if len(pair_test) else np.nan
    )
    return table


def table_capacity(main_table: pd.DataFrame, capacity: pd.DataFrame) -> pd.DataFrame:
    """PT2：容量曲线数据点（每臂聚合：可训练参数中位数、池化 RMSE）。"""
    keep = capacity[capacity["category"].isin(["new", "reused"])].copy()
    grouped = (
        keep.groupby(["family", "arm"])
        .agg(
            trainable_parameters=("trainable_parameter_count", "median"),
            parameter_min=("trainable_parameter_count", "min"),
            parameter_max=("trainable_parameter_count", "max"),
            runs=("runs", "sum"),
        )
        .reset_index()
    )
    pooled = main_table[["family", "arm", "pool_mean_baseline_rmse_ugm3"]].drop_duplicates()
    table = grouped.merge(pooled, on=["family", "arm"], how="left")
    table["pooled_rmse"] = table["pool_mean_baseline_rmse_ugm3"].round(4)
    table = table.rename(columns={"pool_mean_baseline_rmse_ugm3": "pooled_baseline_rmse"})
    return table[["family", "arm", "trainable_parameters", "parameter_min", "parameter_max",
                  "runs", "pooled_baseline_rmse", "pooled_rmse"]]


def table_capacity_our_reference(main_table: pd.DataFrame) -> float:
    """O-locked 的池化 RMSE（跨 20 配置、多臂共用同一列）。"""
    return float(main_table["pool_mean_o_locked_rmse_ugm3"].dropna().iloc[0])


def table_dual(dual: pd.DataFrame) -> pd.DataFrame:
    """PT3：双口径（验证集 / 测试集选择规则）。"""
    out = dual.copy()
    out["our_advantage_percent"] = out["mean_baseline_relative_to_o_percent"].round(3)
    return out[[
        "family", "criterion", "pairs", "mean_baseline_rmse_ugm3", "mean_o_locked_rmse_ugm3",
        "our_advantage_percent", "baseline_better_count",
    ]].rename(columns={
        "mean_baseline_rmse_ugm3": "pooled_baseline_rmse",
        "mean_o_locked_rmse_ugm3": "pooled_our_rmse",
        "baseline_better_count": "baseline_better_runs",
    })


def table_per_lead(per_lead: pd.DataFrame, wins: pd.DataFrame) -> pd.DataFrame:
    """PT4：逐 lead（池化聚合行）+ 胜负计数。"""
    pooled = per_lead[(per_lead["row_type"] == "summary") & (per_lead["history"] == "pooled")].copy()
    pooled["our_advantage_percent"] = pooled["baseline_relative_to_o_percent"].round(3)
    pooled = pooled.rename(columns={
        "baseline_rmse_ugm3": "baseline_rmse", "o_locked_rmse_ugm3": "our_rmse",
    })
    keep = ["family", "arm", "horizon", "lead", "baseline_rmse", "our_rmse",
            "our_advantage_percent", "o_locked_wins"]
    return pooled[keep].sort_values(["family", "arm", "horizon", "lead"]).reset_index(drop=True)


def figure_capacity(table_capacity: pd.DataFrame, main_table: pd.DataFrame,
                    capacity_raw: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ours = table_capacity_our_reference(main_table)
    ax.scatter([np.mean(OUR_BUDGET)], [ours], marker="*", s=280, color="#111111",
               zorder=6, label="Ours (frozen backbone, 18 stations)")
    for family in ("informer", "gru", "lstm"):
        block = table_capacity[table_capacity["family"] == family].sort_values("trainable_parameters")
        if block.empty:
            continue
        ax.plot(block["trainable_parameters"], block["pooled_baseline_rmse"], marker="o",
                color=FAMILY_COLOR[family], linewidth=1.8, markersize=6,
                label=f"{family.upper()} (single station)")
    # TST：参数量随 (L,H) 变化，逐点散点画出并标注为非容量对齐
    tst = capacity_raw[(capacity_raw["family"] == "tst") & (capacity_raw["category"] == "new")]
    if not tst.empty:
        ax.scatter(tst["trainable_parameter_count"], tst["mean_rmse_ugm3"], s=16, alpha=0.55,
                   color=FAMILY_COLOR["tst"], label="TST per (L,H), not capacity-aligned")
    matched = table_capacity[table_capacity["arm"] == MATCHED_ARM]
    if not matched.empty:
        px, py = matched["trainable_parameters"].iloc[0], matched["pooled_baseline_rmse"].iloc[0]
        ax.annotate("matched budget\n(5,125 params)", xy=(px, py), xytext=(px * 0.34, py + 4.5),
                    fontsize=9, arrowprops=dict(arrowstyle="->", color="#555555"))
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel(r"Pooled RMSE ($\mu g/m^3$)")
    ax.set_title("Baselines vs ours on the Beijing 20-task grid (paired, same seeds)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "PF1_capacity_curve.png", dpi=220)
    fig.savefig(out / "PF1_capacity_curve.svg")
    plt.close(fig)


def figure_per_lead(per_lead: pd.DataFrame, out: Path) -> None:
    arms = ["informer_d8_e1", MATCHED_ARM, "informer_d16_e1", "informer_d32_e1"]
    horizons = [6, 12, 24]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.6), sharey=True)
    for ax, horizon in zip(axes, horizons):
        block = per_lead[
            (per_lead["row_type"] == "summary")
            & (per_lead["history"] == "pooled")
            & (per_lead["horizon"] == horizon)
            & (per_lead["arm"].isin(arms))
        ]
        for arm in arms:
            curve = block[block["arm"] == arm].sort_values("lead")
            if curve.empty:
                continue
            style = dict(linewidth=2.4, marker="o", markersize=4) if arm == MATCHED_ARM else dict(
                linewidth=1.3, alpha=0.85)
            ax.plot(curve["lead"], curve["baseline_relative_to_o_percent"], label=arm, **style)
        ax.axhline(0, color="#333333", linewidth=1.0, linestyle="--")
        ax.set_title(f"Horizon H={horizon}")
        ax.set_xlabel("Lead step")
        ax.grid(alpha=0.3)
        ax.set_xticks(sorted(block["lead"].unique()))
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("Our advantage over Informer (%)")
    axes[-1].legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Per-lead advantage of ours over Informer arms (positive = ours better)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "PF2_per_lead.png", dpi=220)
    fig.savefig(out / "PF2_per_lead.svg")
    plt.close(fig)


def _markdown_table(table: pd.DataFrame, floatfmt: str = ".4g") -> str:
    """不依赖 tabulate 的 markdown 表格输出。"""
    def cell(value: object) -> str:
        if isinstance(value, float):
            if np.isnan(value):
                return ""
            return format(value, floatfmt)
        return str(value)

    header = "| " + " | ".join(str(col) for col in table.columns) + " |"
    divider = "|" + "|".join(["---"] * len(table.columns)) + "|"
    body = [
        "| " + " | ".join(cell(v) for v in row) + " |"
        for row in table.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *body])


def write_markdown(tables: dict[str, pd.DataFrame], out_tables: Path) -> None:
    for name, table in tables.items():
        table.to_csv(out_tables / f"{name}.csv", index=False)
        (out_tables / f"{name}.md").write_text(
            f"# {name}\n\n" + _markdown_table(table) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="tables/paper/source")
    parser.add_argument("--out-tables", default="tables/paper")
    parser.add_argument("--out-figures", default="figures/paper")
    args = parser.parse_args()

    source, out_tables, out_figures = (Path(args.source), Path(args.out_tables),
                                       Path(args.out_figures))
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)
    data = load(source)

    tables = {
        "PT1_informer_main": table_informer(
            data["main_table"], data["capacity_curve"], data["dual_criterion"]),
        "PT2_capacity_points": table_capacity(data["main_table"], data["capacity_curve"]),
        "PT3_dual_criterion": table_dual(data["dual_criterion"]),
        "PT4_per_lead": table_per_lead(data["per_lead"], data["per_lead_win_counts"]),
        "PT5_rank_correlation": data["rank_correlation"].round(4),
        "PT6_compliance_self_check": data["compliance_self_check"],
    }
    write_markdown(tables, out_tables)
    figure_capacity(tables["PT2_capacity_points"], data["main_table"],
                     data["capacity_curve"], out_figures)
    figure_per_lead(data["per_lead"], out_figures)
    print("PAPER_BASELINE_ARTIFACTS_DONE",
          f"tables={len(tables)}", f"figures=2", f"out={out_tables}/{out_figures}")


if __name__ == "__main__":
    main()
