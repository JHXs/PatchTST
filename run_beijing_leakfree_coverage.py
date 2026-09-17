"""方案 B：北京 1013 的 20 配置覆盖实验，改用**无泄漏管线**重跑。

与既有 `run_st_patchtst_ablation.py` 的唯一差别（两处，其余训练/评估完全复用）：

1. **筛站只用训练段**：先按窗口数确定 70/10/20 划分，再用"训练窗口覆盖的行"计算中心站与候选站的
   相关性并筛站；原管线用完整序列相关性（存在前视）。
2. **缺失填充只做因果前向**：`ffill()`，并裁掉公共不完整开头；原管线用 `ffill().bfill()`（回填使用未来值）。

其余保持与主确认一致：k=5、冻结主干、退化初始化、同一优化器/epoch/patience/batch、同一评估划分与指标。

用法：
    python run_beijing_leakfree_coverage.py --output-root experiments/results/beijing_leakfree_coverage
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_st_patchtst_ablation as legacy

TASK_GRID = [(24, 1), (24, 3), (24, 6), (24, 12), (24, 24),
             (48, 1), (48, 3), (48, 6), (48, 12), (48, 24),
             (72, 1), (72, 3), (72, 6), (72, 12), (72, 24),
             (168, 1), (168, 3), (168, 6), (168, 12), (168, 24)]
SEEDS = (2047, 2048, 2049, 2050, 2051)
ARMS = ("degraded_patchtst", "st_sparse_station_bias_delta_forecast")
LEAKFREE_NOTE = (
    "Leakage-free pipeline: station screening uses only the rows covered by training windows; "
    "missing values are filled causally (ffill only) and the common incomplete prefix is dropped."
)


def load_station_data_leakfree(config) -> tuple[pd.DataFrame, int]:
    """Causal-only filling; the returned frame keeps the common (finite) prefix."""
    frames = []
    station_ids = list(range(config.station_start, config.station_end + 1))
    for station_id in station_ids:
        path = Path(config.data_dir) / f"df_station_{station_id}.csv"
        df = pd.read_csv(path, usecols=["time", "PM25_Concentration"])
        df["time"] = pd.to_datetime(df["time"])
        series = df.drop_duplicates("time").set_index("time")["PM25_Concentration"]
        frames.append(series.rename(station_id))
    concatenated = pd.concat(frames, axis=1, join="inner").sort_index()
    aligned = concatenated.ffill()
    # 公共不完整开头：任何站点仍缺失的行一律裁掉，保证后续窗口与缩放都在完整数据上
    complete = aligned.notna().all(axis=1)
    dropped = int((~complete).sum())
    return aligned.loc[complete], dropped


def train_segment_correlations(aligned: pd.DataFrame, config, train_row_stop: int) -> pd.Series:
    """Correlations computed only on rows covered by training windows."""
    train_rows = aligned.iloc[:train_row_stop]
    if len(train_rows) < 3:
        raise ValueError("训练段行数不足，无法计算相关性")
    return train_rows.corr()[config.center_station_id]


def prepare_datasets_leakfree(config):
    aligned, dropped_prefix_rows = load_station_data_leakfree(config)
    num_samples = len(aligned) - config.history - config.horizon + 1
    if num_samples < 3:
        raise ValueError("时间序列太短，无法构造训练、验证和测试窗口")
    train_end = int(num_samples * config.train_fraction)
    valid_end = int(num_samples * (config.train_fraction + config.valid_fraction))
    train_row_stop = train_end + config.history + config.horizon - 1  # 与旧管线口径一致

    correlations_all = train_segment_correlations(aligned, config, train_row_stop)
    station_ids = [int(value) for value in aligned.columns]
    selected_ids = [sid for sid in station_ids
                    if correlations_all.loc[sid] > config.correlation_threshold]
    if config.center_station_id not in selected_ids:
        selected_ids.append(config.center_station_id)
        selected_ids.sort()
    frame = aligned[selected_ids]
    correlations = correlations_all.loc[selected_ids]

    raw_values = frame.to_numpy(dtype=np.float32)
    train_values = raw_values[:train_row_stop]
    means = train_values.mean(axis=0)
    stds = train_values.std(axis=0)
    stds = np.where(stds < 1e-6, 1.0, stds)
    normalized = ((raw_values - means) / stds).astype(np.float32)

    center_idx = selected_ids.index(config.center_station_id)
    split_indices = {
        "train": np.arange(0, train_end),
        "valid": np.arange(train_end, valid_end),
        "test": np.arange(valid_end, num_samples),
    }
    datasets = {
        split: legacy.ForecastWindowDataset(normalized, indices, config.history, config.horizon, center_idx)
        for split, indices in split_indices.items()
    }
    metadata = {
        "rows": len(frame),
        "dropped_incomplete_prefix_rows": dropped_prefix_rows,
        "start_time": str(frame.index.min()),
        "end_time": str(frame.index.max()),
        "num_samples": num_samples,
        "split_sizes": {key: len(value) for key, value in datasets.items()},
        "station_ids": [int(s) for s in selected_ids],
        "station_correlations": {str(k): float(v) for k, v in correlations.items()},
        "center_station_idx": int(center_idx),
        "center_mean": float(means[center_idx]),
        "center_std": float(stds[center_idx]),
        "station_means": means.tolist(),
        "station_stds": stds.tolist(),
        "station_selection_note": LEAKFREE_NOTE,
        "screened_candidates": len(selected_ids),
    }
    return datasets, metadata


def config_for(history: int, horizon: int):
    config = legacy.ExperimentConfig(history=history, horizon=horizon)
    if history <= 48:
        config = replace(config, epochs=40, patience=8, batch_size=256)
    else:
        config = replace(config, epochs=30, patience=6, batch_size=512)
    return replace(config, sparse_neighbor_top_k=5, evaluation_split="test",
                   initialize_from_degraded=True, freeze_backbone=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="experiments/results/beijing_leakfree_coverage")
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--configs", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s]
    grid = ([(int(v.split("x")[0]), int(v.split("x")[1])) for v in args.configs.split(",")]
            if args.configs else TASK_GRID)
    output_root = Path(args.output_root)
    started = time.perf_counter()
    for history, horizon in grid:
        config = config_for(history, horizon)
        out = output_root / f"{history}h_{horizon}h"
        if (out / "raw_metrics.csv").exists():
            print(f"[跳过] 已完成 {out.name}")
            continue
        out.mkdir(parents=True, exist_ok=True)
        datasets, metadata = prepare_datasets_leakfree(config)
        print(f"[{out.name}] 筛出 {metadata['screened_candidates']} 站（训练段相关性 > {config.correlation_threshold}）"
              f"，样本 {metadata['num_samples']}，划分 {metadata['split_sizes']}")
        rows = []
        for seed in seeds:
            initialization = None
            for variant in ARMS:
                if variant != "degraded_patchtst":
                    initialization = out / "checkpoints" / f"degraded_patchtst_seed{seed}.pt"
                    if not initialization.is_file():
                        raise RuntimeError("ST 臂必须在同种子退化基线之后训练")
                rows.append(legacy.train_one_run(config, datasets, metadata, variant, seed, out,
                                                 torch.device(args.device), initialization_checkpoint=initialization))
        pd.DataFrame(rows).to_csv(out / "raw_metrics.csv", index=False)
        (out / "dataset_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "experiment_config.json").write_text(
            json.dumps({**config.__dict__, "seeds": seeds, "station_selection_note": LEAKFREE_NOTE},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[完成] {out.name} (累计 {time.perf_counter() - started:.0f}s)")
    print("BEIJING_LEAKFREE_DONE")


if __name__ == "__main__":
    main()
