"""阶段 D：广州跨预测步覆盖的频域适配器（接在阶段 C 的冻结 ST 检查点之上）。

复用 `run_cross_city_generalization.py` 的数据组件（白名单读取、因果填充、训练段筛站、
两臂共享样本）与 `frequency_residual_adapter.py` 的适配器实现。

- 每个 (配置, 中心站, 种子) 训练两个适配器：`st_rfft`（固定三段 rFFT）与 `st_time`（等容量时域对照）。
- ST 与 PatchTST 全程冻结并保持 eval，只训练适配器与其有界 β。
- 产物写入 `guangzhou_horizon_coverage_frequency/<L>h_<H>h/station_<id>/`，含逐标量预测（物理量）。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import run_cross_city_generalization as gz
from frequency_residual_adapter import (
    FixedFrequencyResidualAdapter,
    FrozenSTForecastResidual,
    TimeResidualAdapter,
    trainable_parameter_count,
)

TASK_GRID = [(24, 1), (24, 3), (24, 6), (24, 12), (24, 24),
             (48, 1), (48, 3), (48, 6), (48, 12), (48, 24),
             (72, 1), (72, 3), (72, 6), (72, 12), (72, 24),
             (168, 1), (168, 3), (168, 6), (168, 12), (168, 24)]
SEEDS = (7001, 7002, 7003, 7004, 7005)
SPATIAL_ARM = "st_sparse_station_bias_delta_forecast"
ADAPTER_VARIANTS = ("st_rfft", "st_time")
BETA_MAX, BETA_INIT = 0.5, 0.1
ADAPTER_EPOCHS, ADAPTER_PATIENCE = 30, 6
HIDDEN, POOL_BINS, STREAMS, DROPOUT = 32, 4, 3, 0.2


def stage_config(history: int, horizon: int) -> gz.RunConfig:
    if history <= 48:
        return gz.RunConfig(history, horizon, batch_size=256, epochs=40, patience=8)
    return gz.RunConfig(history, horizon, batch_size=512, epochs=30, patience=6)


def physical_metrics(prediction_ugm3: np.ndarray, target_ugm3: np.ndarray) -> dict:
    """逐标量物理尺度指标（与 round14 汇总口径一致：展平后统一计算）。"""
    error = prediction_ugm3 - target_ugm3
    denominator = np.abs(prediction_ugm3) + np.abs(target_ugm3)
    return {"rmse_ugm3": float(np.sqrt(np.mean(error ** 2))),
            "mae_ugm3": float(np.mean(np.abs(error))),
            "smape_percent": float(200 * np.mean(np.abs(error) / np.maximum(denominator, 1e-6))),
            "element_count": int(error.size)}


def build_adapter(variant: str, seq_len: int, pred_len: int) -> nn.Module:
    kwargs = dict(seq_len=seq_len, pred_len=pred_len, hidden_dim=HIDDEN, pool_bins=POOL_BINS,
                  dropout=DROPOUT, num_streams=STREAMS)
    return TimeResidualAdapter(**kwargs) if variant == "st_time" else FixedFrequencyResidualAdapter(**kwargs)


def verify_code_state(output_root: Path) -> str:
    head = gz.git_commit()
    if re.fullmatch(r"[0-9a-f]{40}", head) is None:
        raise gz.ProtocolViolation("HEAD 必须是 40 位小写十六进制 commit")
    for relative in ("run_guangzhou_horizon_coverage_d.py", "run_guangzhou_horizon_coverage.py",
                     "run_cross_city_generalization.py", "frequency_residual_adapter.py"):
        tracked = subprocess.run(["git", "cat-file", "-e", f"HEAD:{relative}"],
                                 check=False, capture_output=True, text=True)
        if tracked.returncode != 0:
            raise gz.ProtocolViolation(f"实现文件必须已被 HEAD 跟踪: {relative}")
    allowed = (f"?? {output_root.as_posix().rstrip('/')}",
               "?? experiments/results/guangzhou_horizon_coverage")
    dirty = [line for line in subprocess.run(["git", "status", "--porcelain"], check=False,
                                             capture_output=True, text=True).stdout.splitlines()
             if not line.startswith(allowed)]
    if dirty:
        raise gz.ProtocolViolation(f"存在未提交的源码/数据改动，拒绝运行: {dirty[:5]}")
    return head


def train_adapter(config, datasets, metadata, variant, seed, st_checkpoint, device):
    base = gz.build_model(config, SPATIAL_ARM).to(device)
    base.load_state_dict(torch.load(st_checkpoint, map_location=device, weights_only=True))
    base.eval()
    gz.set_seed(seed)  # 两个适配器共享同一初始化
    model = FrozenSTForecastResidual(base, build_adapter(variant, config.history, config.horizon),
                                     beta_max=BETA_MAX, beta_init=BETA_INIT).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if any(id(p) in {id(q) for q in model.base_model.parameters()} for p in trainable):
        raise AssertionError("冻结 ST 参数进入适配器优化器")

    fit_loader = gz.make_loader(datasets["fit"], config, True, seed)
    val_loader = gz.make_loader(datasets["val"], config, False, seed)
    confirm_loader = gz.make_loader(datasets["confirm"], config, False, seed)

    model.eval()
    first_x, _ = next(iter(val_loader))
    with torch.no_grad():
        init = model.forward_components(first_x.to(device))
    if float((init["prediction"] - init["base_prediction"]).abs().max().item()) != 0.0:
        raise AssertionError("零初始化未严格退化为锁定 ST")

    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    loss_fn = nn.MSELoss()
    best_loss, best_state, best_epoch, stale = math.inf, None, 0, 0
    history_rows = []
    for epoch in range(1, ADAPTER_EPOCHS + 1):
        model.train(); model.base_model.eval()
        total, count = 0.0, 0
        for x, y in fit_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total += float(loss) * len(x); count += len(x)
        prediction, target = gz.predict(model, val_loader, device)
        val_loss = float(np.mean((prediction - target) ** 2))
        scheduler.step(val_loss)
        history_rows.append({"epoch": epoch, "train_loss": total / count, "valid_loss": val_loss})
        if val_loss < best_loss - 1e-7:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= ADAPTER_PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    prediction, target = gz.predict(model, confirm_loader, device)
    scalar = gz.scalarize_predictions(prediction, target, metadata, config.horizon)
    beta = float((model.max_beta * torch.sigmoid(model.beta_logit)).item())
    metrics = physical_metrics(scalar[0], scalar[1])
    return ({"variant": variant, "seed": seed, "best_epoch": best_epoch, "beta": beta,
             "trained_epochs": len(history_rows),
             "trainable_parameter_count": trainable_parameter_count(model), **metrics},
            history_rows, scalar)


def expected_artifacts(seeds) -> set[str]:
    return {f"{arm}_seed{seed}.npz" for arm in ADAPTER_VARIANTS for seed in seeds}


def is_complete(task_dir: Path, seeds) -> bool:
    return (task_dir / "raw_metrics.csv").is_file() and expected_artifacts(seeds).issubset(
        {p.name for p in (task_dir / "predictions").glob("*.npz")})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-c-root", default="experiments/results/guangzhou_horizon_coverage")
    parser.add_argument("--output-root", default="experiments/results/guangzhou_horizon_coverage_frequency")
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--configs", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    seeds = tuple(int(s) for s in args.seeds.split(",") if s)
    stage_c_root, output_root = Path(args.stage_c_root), Path(args.output_root)
    device = torch.device(args.device)
    grid = ([(int(v.split("x")[0]), int(v.split("x")[1])) for v in args.configs.split(",")]
            if args.configs else TASK_GRID)

    access_plan = {str(center): list(gz.authorized_station_ids(center)) for center in gz.B1_STATIONS}
    for center, ids in access_plan.items():
        gz.validate_authorized_station_set(int(center), ids)
    print(json.dumps({"planned_station_access": access_plan, "forbidden": sorted(gz.FORBIDDEN_STATIONS),
                      "configs": [f"{l}h_{h}h" for l, h in grid], "seeds": list(seeds)}, ensure_ascii=False))
    if args.smoke:
        config = replace(stage_config(24, 1), batch_size=32, neighbor_hidden_dim=8, n_layers=1,
                         d_model=8, n_heads=2, d_ff=16, dropout=0.0)
        frame = gz.synthetic_frame()
        datasets, metadata = gz.prepare_shared_data(frame, 100, config)
        base = gz.build_model(config, SPATIAL_ARM).to(device)
        model = FrozenSTForecastResidual(base, build_adapter("st_rfft", config.history, config.horizon),
                                         beta_max=BETA_MAX, beta_init=BETA_INIT).to(device)
        prediction, target = gz.predict(model, gz.make_loader(datasets["confirm"], config, False, 1), device)
        scalar = gz.scalarize_predictions(prediction, target, metadata, config.horizon)
        metrics = physical_metrics(scalar[0], scalar[1])
        print(json.dumps({"smoke_ok": True, "rmse": metrics["rmse_ugm3"],
                          "degenerate": float(np.abs(prediction - target).mean()) < 1e6}, ensure_ascii=False))
        return

    verify_code_state(output_root)
    started = time.perf_counter()
    for center in gz.B1_STATIONS:
        frame, opened = gz.load_center_candidate_frame(center)
        for history, horizon in grid:
            stage_c_dir = stage_c_root / f"{history}h_{horizon}h" / f"station_{center}"
            checkpoint_dir = stage_c_dir / "checkpoints"
            if not checkpoint_dir.is_dir():
                print(f"[跳过] 阶段 C 缺少检查点: {history}h_{horizon}h station={center}")
                continue
            task_dir = output_root / f"{history}h_{horizon}h" / f"station_{center}"
            if task_dir.exists() and is_complete(task_dir, seeds):
                print(f"[跳过] 已完成 {task_dir.relative_to(output_root)}")
                continue
            if task_dir.exists():
                shutil.move(str(task_dir), str(task_dir.with_name(task_dir.name + f"_partial_{int(time.time())}")))
            config = stage_config(history, horizon)
            datasets, metadata = gz.prepare_shared_data(frame, center, config)
            (task_dir / "predictions").mkdir(parents=True, exist_ok=True)
            (task_dir / "training_logs").mkdir(exist_ok=True)
            rows = []
            for seed in seeds:
                checkpoint = checkpoint_dir / f"{SPATIAL_ARM}_seed{seed}.pt"
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"缺少冻结 ST 检查点: {checkpoint}")
                for variant in ADAPTER_VARIANTS:
                    record, history_rows, scalar = train_adapter(
                        config, datasets, metadata, variant, seed, checkpoint, device)
                    rows.append(record)
                    gz.save_prediction_artifact(task_dir / "predictions" / f"{variant}_seed{seed}.npz", *scalar)
                    pd.DataFrame(history_rows).to_csv(
                        task_dir / "training_logs" / f"{variant}_seed{seed}.csv", index=False)
            pd.DataFrame(rows).to_csv(task_dir / "raw_metrics.csv", index=False)
            signature = {**config.__dict__, "seeds": list(seeds), "arms": list(ADAPTER_VARIANTS),
                         "frozen_st_checkpoint_source": str(stage_c_dir)}
            (task_dir / "experiment_config.json").write_text(
                json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf-8")
            (task_dir / "dataset_metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[完成] {history}h_{horizon}h station={center} (累计 {time.perf_counter() - started:.0f}s)")
    print("STAGE_D_DONE")


if __name__ == "__main__":
    main()
