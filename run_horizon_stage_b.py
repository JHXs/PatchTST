"""阶段 B：在阶段 A 已训练好的冻结 ST 检查点之上，训练频域/时域适配器（跨预测步覆盖）。

与阶段 A 的差别：阶段 A 训练 `degraded_patchtst` 与 `st_sparse_station_bias_delta_forecast`；
本阶段**复用**阶段 A 的 ST 检查点，只训练适配器（ST 与 PatchTST 全程冻结），因此每个配置只需
2 次适配器训练/种子，而不是重训整个模型。

用法：
    python run_horizon_stage_b.py --stage-a-root experiments/results/horizon_coverage \
        --output-root experiments/results/horizon_coverage_frequency
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import run_st_patchtst_ablation as legacy
from frequency_residual_adapter import (
    FixedFrequencyResidualAdapter,
    FrozenSTForecastResidual,
    TimeResidualAdapter,
    trainable_parameter_count,
)

SPATIAL_VARIANT = "st_sparse_station_bias_delta_forecast"
ADAPTER_VARIANTS = ("st_rfft", "st_time")   # st_time 为等容量时域对照，仅内部记录
BETA_MAX, BETA_INIT = 0.5, 0.1
ADAPTER_EPOCHS, ADAPTER_PATIENCE = 30, 6
HIDDEN, POOL_BINS, STREAMS, DROPOUT = 32, 4, 3, 0.2
DEFAULT_CONFIGS = [(24, 1), (24, 3), (24, 6), (24, 12), (24, 24),
                   (48, 1), (48, 3), (48, 6), (48, 12), (48, 24),
                   (72, 1), (72, 3), (72, 6), (72, 12), (72, 24),
                   (168, 1), (168, 3), (168, 6), (168, 12), (168, 24)]


def build_adapter(variant: str, seq_len: int, pred_len: int) -> nn.Module:
    kwargs = dict(seq_len=seq_len, pred_len=pred_len, hidden_dim=HIDDEN, pool_bins=POOL_BINS,
                  dropout=DROPOUT, num_streams=STREAMS)
    return TimeResidualAdapter(**kwargs) if variant == "st_time" else FixedFrequencyResidualAdapter(**kwargs)


def config_for(history: int, horizon: int):
    config = legacy.ExperimentConfig(history=history, horizon=horizon)
    if history <= 48:
        config = replace(config, epochs=40, patience=8, batch_size=256)
    else:
        config = replace(config, epochs=30, patience=6, batch_size=512)
    return replace(config, sparse_neighbor_top_k=5, evaluation_split="test",
                   initialize_from_degraded=True, freeze_backbone=True)


def train_adapter(config, datasets, metadata, variant, seed, st_checkpoint, device):
    base = legacy.build_model(config, SPATIAL_VARIANT, num_stations=len(metadata["station_ids"]),
                              center_idx=metadata["center_station_idx"]).to(device)
    base.load_state_dict(torch.load(st_checkpoint, map_location=device, weights_only=True))
    base.eval()
    legacy.set_seed(seed)  # 两个适配器使用完全相同的初始化
    model = FrozenSTForecastResidual(base, build_adapter(variant, config.history, config.horizon),
                                     beta_max=BETA_MAX, beta_init=BETA_INIT).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if any(id(p) in {id(q) for q in model.base_model.parameters()} for p in trainable):
        raise AssertionError("冻结 ST 参数进入适配器优化器")

    train_loader = legacy.make_loader(datasets["train"], config, True, seed)
    valid_loader = legacy.make_loader(datasets["valid"], config, False, seed)
    test_loader = legacy.make_loader(datasets[config.evaluation_split], config, False, seed)
    model.eval()
    first_x, _ = next(iter(valid_loader))
    with torch.no_grad():
        init = model.forward_components(first_x.to(device))
    if float((init["prediction"] - init["base_prediction"]).abs().max().item()) != 0.0:
        raise AssertionError("零初始化未严格退化为锁定 ST")

    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    loss_fn = nn.MSELoss()
    best_loss, best_state, best_epoch, bad = math.inf, None, 0, 0
    for epoch in range(1, ADAPTER_EPOCHS + 1):
        model.train(); model.base_model.eval()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss_fn(model(x), y).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_loss = float(np.mean([loss_fn(model(x.to(device)), y.to(device)).item() for x, y in valid_loader]))
        scheduler.step(valid_loss)
        if valid_loss < best_loss - 1e-8:
            best_loss, best_epoch, bad = valid_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= ADAPTER_PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    base_prediction, _, _ = legacy.predict(model, test_loader, device, metadata["center_station_idx"])
    prediction, target, _ = legacy.predict(model, test_loader, device, metadata["center_station_idx"])
    metrics = legacy.regression_metrics(target, prediction, metadata["center_mean"], metadata["center_std"])
    beta = float((model.max_beta * torch.sigmoid(model.beta_logit)).item())
    return {"variant": variant, "seed": seed, "best_epoch": best_epoch, "beta": beta,
            "trainable_parameter_count": trainable_parameter_count(model), **metrics}, prediction, base_prediction, target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-a-root", default="experiments/results/horizon_coverage")
    parser.add_argument("--output-root", default="experiments/results/horizon_coverage_frequency")
    parser.add_argument("--seeds", default="2047,2048,2049,2050,2051")
    parser.add_argument("--configs", default=None, help="可选：如 24x3,168x24，默认全部 20 个配置")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s]
    device = torch.device(args.device)
    stage_a, output_root = Path(args.stage_a_root), Path(args.output_root)
    if args.configs:
        configs = [tuple(int(v) for v in item.split("x")) for item in args.configs.split(",")]
    else:
        configs = DEFAULT_CONFIGS

    for history, horizon in configs:
        name = f"{history}h_{horizon}h"
        stage_a_dir = stage_a / name
        if not (stage_a_dir / "raw_metrics.csv").exists():
            print(f"[跳过] 阶段 A 缺少 {name}")
            continue
        out = output_root / name
        expected_rows = len(seeds) * len(ADAPTER_VARIANTS)
        if (out / "raw_metrics.csv").exists():
            try:
                done = pd.read_csv(out / "raw_metrics.csv")
                complete = (len(done) == expected_rows
                            and set(done.seed.astype(int)) == set(seeds)
                            and set(done.variant) == set(ADAPTER_VARIANTS))
            except Exception:
                complete = False
            if complete:
                print(f"[跳过] 已完成 {name}")
                continue
            print(f"[重跑] {name} 结果不完整，重新执行")
            (out / "raw_metrics.csv").unlink(missing_ok=True)
        out.mkdir(parents=True, exist_ok=True)
        config = config_for(history, horizon)
        datasets, metadata = legacy.prepare_datasets(config)
        if name == "24h_1h" or name == "168h_6h":
            # 复用已有的频域结果时（主确认两个配置已有专门产物），仍按新网格重跑以保证一致性
            pass
        rows = []
        for seed in seeds:
            checkpoint = stage_a_dir / "checkpoints" / f"{SPATIAL_VARIANT}_seed{seed}.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(f"缺少阶段 A 检查点: {checkpoint}")
            for variant in ADAPTER_VARIANTS:
                record, prediction, base_prediction, target = train_adapter(
                    config, datasets, metadata, variant, seed, checkpoint, device)
                rows.append(record)
                predictions_dir = out / "predictions"
                predictions_dir.mkdir(parents=True, exist_ok=True)
                center_mean, center_std = metadata["center_mean"], metadata["center_std"]
                np.savez_compressed(predictions_dir / f"{variant}_seed{seed}.npz",
                                    prediction_ugm3=prediction * center_std + center_mean,
                                    base_prediction_ugm3=base_prediction * center_std + center_mean,
                                    target_ugm3=target * center_std + center_mean,
                                    note="physical units (ug/m3)")
                print(f"[{name}] {variant} seed={seed} RMSE={record['rmse_ugm3']:.4f} beta={record['beta']:.4f}")
        pd.DataFrame(rows).to_csv(out / "raw_metrics.csv", index=False)
        (out / "experiment_config.json").write_text(json.dumps({**config.__dict__, "seeds": seeds}, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "dataset_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[完成] {name}")


if __name__ == "__main__":
    main()
