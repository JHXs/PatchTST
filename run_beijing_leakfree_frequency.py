"""无泄漏版北京的频域臂：接在 `beijing_leakfree_coverage` 的冻结 ST 检查点之上。

与 `run_horizon_stage_b.py` 的唯一差别是**数据集来自无泄漏管线**
（`run_beijing_leakfree_coverage.prepare_datasets_leakfree`：训练段筛站 + 因果填充），
以保证与无泄漏 ST 检查点的站点数一致（18 站）并保持协议统一。

用法：
    python run_beijing_leakfree_frequency.py \
        --stage-a-root experiments/results/beijing_leakfree_coverage \
        --output-root experiments/results/beijing_leakfree_coverage_frequency
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_beijing_leakfree_coverage as lf
import run_horizon_stage_b as stage_b

SPATIAL_ARM = "st_sparse_station_bias_delta_forecast"
ADAPTER_VARIANTS = ("st_rfft", "st_time")


def expected_artifacts(seeds) -> set[str]:
    return {f"{arm}_seed{seed}.npz" for arm in ADAPTER_VARIANTS for seed in seeds}


def is_complete(task_dir: Path, seeds) -> bool:
    return (task_dir / "raw_metrics.csv").is_file() and expected_artifacts(seeds).issubset(
        {p.name for p in (task_dir / "predictions").glob("*.npz")})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-a-root", default="experiments/results/beijing_leakfree_coverage")
    parser.add_argument("--output-root", default="experiments/results/beijing_leakfree_coverage_frequency")
    parser.add_argument("--seeds", default="2047,2048,2049,2050,2051")
    parser.add_argument("--configs", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s]
    grid = ([(int(v.split("x")[0]), int(v.split("x")[1])) for v in args.configs.split(",")]
            if args.configs else lf.TASK_GRID)
    stage_a_root, output_root = Path(args.stage_a_root), Path(args.output_root)
    device = torch.device(args.device)
    started = time.perf_counter()

    for history, horizon in grid:
        name = f"{history}h_{horizon}h"
        stage_a_dir = stage_a_root / name
        if not (stage_a_dir / "raw_metrics.csv").exists():
            print(f"[跳过] 缺阶段 A 结果: {name}")
            continue
        out = output_root / name
        if out.exists() and is_complete(out, seeds):
            print(f"[跳过] 已完成 {name}")
            continue
        out.mkdir(parents=True, exist_ok=True)
        config = lf.config_for(history, horizon)
        datasets, metadata = lf.prepare_datasets_leakfree(config)
        rows = []
        for seed in seeds:
            checkpoint = stage_a_dir / "checkpoints" / f"{SPATIAL_ARM}_seed{seed}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"缺少无泄漏 ST 检查点: {checkpoint}")
            for variant in ADAPTER_VARIANTS:
                record, prediction, base_prediction, target = stage_b.train_adapter(
                    config, datasets, metadata, variant, seed, checkpoint, device)
                rows.append(record)
                predictions_dir = out / "predictions"
                predictions_dir.mkdir(parents=True, exist_ok=True)
                center_mean, center_std = metadata["center_mean"], metadata["center_std"]
                np.savez_compressed(
                    predictions_dir / f"{variant}_seed{seed}.npz",
                    prediction_ugm3=prediction * center_std + center_mean,
                    base_prediction_ugm3=base_prediction * center_std + center_mean,
                    target_ugm3=target * center_std + center_mean,
                    note="physical units (ug/m3)")
                print(f"[{name}] {variant} seed={seed} RMSE={record['rmse_ugm3']:.4f} beta={record['beta']:.4f}")
        pd.DataFrame(rows).to_csv(out / "raw_metrics.csv", index=False)
        (out / "dataset_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "experiment_config.json").write_text(
            json.dumps({**config.__dict__, "seeds": seeds, "arms": list(ADAPTER_VARIANTS),
                        "pipeline": "leak-free", "note": lf.LEAKFREE_NOTE}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"[完成] {name} (累计 {time.perf_counter() - started:.0f}s)")
    print("BEIJING_LEAKFREE_FREQUENCY_DONE")


if __name__ == "__main__":
    main()
