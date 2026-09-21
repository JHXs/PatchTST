"""端到端（不冻结主干、不从退化基线初始化）参考臂：ST 结构 vs 退化基线。

用途：基线对比（`experiments/results/baselines/`）里的多变量基线是**端到端自由训练**的，
而论文主模型的 ST 臂是**冻结主干 + 约 1k 可训练参数**。两者训练制度不同，
不能直接回答"我们的空间结构是否有效"。本脚本在**完全相同的无泄漏数据、划分、种子、训练预算**下
训练两条**端到端**参考臂：

  - `degraded_patchtst`：中心站单通道 PatchTST（自身基线），端到端
  - `st_sparse_station_bias_delta_forecast`：Top-5 稀疏门控 + 站点身份偏置 + 预测端残差，端到端

除 `freeze_backbone=False, initialize_from_degraded=False` 外，其余配置与
`run_beijing_leakfree_coverage.config_for` 逐字段一致（epochs/patience/batch/lr/评估划分/Top-k）。

产物目录布局与 `beijing_leakfree_coverage` 一致，便于汇总器按 (L, H, seed) 配对。

用法：
    python run_st_reference_arms.py [--device cuda] [--seeds ...] [--configs 24x1,...]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch

import run_beijing_leakfree_coverage as leakfree
import run_st_patchtst_ablation as legacy

REFERENCE_ARMS = ("degraded_patchtst", "st_sparse_station_bias_delta_forecast")
HEADLINE_TASKS = ((24, 1), (168, 6))
GRID_SEEDS = (2047, 2048, 2049)
HEADLINE_SEEDS = (2047, 2048, 2049, 2050, 2051)


def config_for_end_to_end(history: int, horizon: int):
    """与 leakfree.config_for 一致，仅放开冻结与初始化。"""
    return replace(
        leakfree.config_for(history, horizon),
        freeze_backbone=False,
        initialize_from_degraded=False,
    )


def task_dir_for(root: Path, history: int, horizon: int) -> Path:
    return root / f"{history}h_{horizon}h"


def expected_runs(seeds) -> int:
    return len(REFERENCE_ARMS) * len(seeds)


def is_complete(task_dir: Path, seeds) -> bool:
    csv = task_dir / "raw_metrics.csv"
    if not csv.is_file():
        return False
    recorded = pd.read_csv(csv)
    if "status" in recorded.columns:
        bad = recorded["status"].astype(str).str.startswith(("infeasible", "nonfinite"))
        recorded = recorded[~bad]
    have = set(zip(recorded["variant"], recorded["seed"].astype(int)))
    want = {(arm, seed) for arm in REFERENCE_ARMS for seed in seeds}
    if want.issubset(have):
        return True
    # 已登记为不可行/发散的组合也算“已处理”，避免续跑反复重试同一失败组合
    if "status" in pd.read_csv(csv).columns:
        raw = pd.read_csv(csv)
        handled = raw[raw["status"].astype(str).str.startswith(("infeasible", "nonfinite"))]
        have |= set(zip(handled["variant"], handled["seed"].astype(int)))
    return want.issubset(have)


def train_reference_arm(config, datasets, metadata, arm, seed, task_dir, device, registration_note=""):
    """训练单个参考臂；发散时按统一规则处理（lr/10 重试一次 → 仍发散登记 nonfinite）。

    与基线运行器保持同一条规则（工作单 §8.2）：任何臂的非有限损失都不应中断整轮，
    也不应被静默丢弃。
    """
    import gc

    def attempt(run_config):
        row = legacy.train_one_run(
            run_config, datasets, metadata, arm, seed, task_dir, device,
            initialization_checkpoint=None,
        )
        row["arm"] = arm
        row["freeze_backbone"] = bool(run_config.freeze_backbone)
        row["initialized_from_degraded"] = False
        return row

    try:
        return attempt(config)
    except FloatingPointError as error:
        retry_config = replace(config, learning_rate=config.learning_rate / 10.0)
        print(f"[{arm} seed={seed}] 非有限损失，改用 lr/10 重试一次")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            row = attempt(retry_config)
            row["learning_rate_retry"] = True
            row["note"] = "nonfinite loss at protocol lr; retried once with lr/10"
            return row
        except FloatingPointError:
            print(f"[{arm} seed={seed}] status=nonfinite（lr 与 lr/10 均发散）")
            return {
                "variant": arm,
                "arm": arm,
                "seed": seed,
                "status": "nonfinite",
                "evaluation_split": config.evaluation_split,
                "best_epoch": 0,
                "best_valid_loss": float("nan"),
                "training_seconds": float("nan"),
                "test_inference_seconds": float("nan"),
                "parameter_count": float("nan"),
                "trainable_parameter_count": float("nan"),
                "rmse_ugm3": float("nan"),
                "mae_ugm3": float("nan"),
                "smape_percent": float("nan"),
                "mse_scaled": float("nan"),
                "rmse_scaled": float("nan"),
                "mae_scaled": float("nan"),
                "infeasible_reason": f"non-finite training loss at protocol lr and lr/10: {str(error)[:400]}",
                "freeze_backbone": bool(config.freeze_backbone),
                "initialized_from_degraded": False,
                "note": registration_note,
            }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="experiments/results/st_reference_arms")
    parser.add_argument("--configs", default=None, help="逗号分隔 LxH；缺省=全网格")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    grid = ([(int(v.split("x")[0]), int(v.split("x")[1])) for v in args.configs.split(",")]
            if args.configs else leakfree.TASK_GRID)
    forced_seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    output_root = Path(args.output_root)
    device = torch.device(args.device)
    started = time.perf_counter()

    for history, horizon in grid:
        seeds = forced_seeds or (
            HEADLINE_SEEDS if (history, horizon) in HEADLINE_TASKS else GRID_SEEDS
        )
        task_dir = task_dir_for(output_root, history, horizon)
        if is_complete(task_dir, seeds):
            print(f"[跳过] {history}→{horizon} 已完成")
            continue
        task_dir.mkdir(parents=True, exist_ok=True)
        config = config_for_end_to_end(history, horizon)
        datasets, metadata = leakfree.prepare_datasets_leakfree(config)
        rows = []
        for seed in seeds:
            for arm in REFERENCE_ARMS:
                row = train_reference_arm(config, datasets, metadata, arm, seed, task_dir, device,
                                          registration_note="end-to-end reference arm")
                rows.append(row)
                pd.DataFrame(rows).to_csv(task_dir / "raw_metrics.csv", index=False)
        (task_dir / "dataset_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        (task_dir / "experiment_config.json").write_text(
            json.dumps({**config.__dict__, "seeds": list(seeds), "arms": list(REFERENCE_ARMS),
                        "note": "end-to-end reference arms (no freezing, no degraded init)"},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        frame = pd.read_csv(task_dir / "raw_metrics.csv")
        for arm in REFERENCE_ARMS:
            part = frame[frame["variant"] == arm]["rmse_ugm3"]
            if len(part):
                print(f"[{history}→{horizon}] {arm}: RMSE={part.mean():.4f} "
                      f"(n={len(part)}, {time.perf_counter() - started:.0f}s)")
    print("ST_REFERENCE_ARMS_DONE")


if __name__ == "__main__":
    main()
