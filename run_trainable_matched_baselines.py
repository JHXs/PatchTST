"""Train preregistered single-station GRU/LSTM capacity curves.

Every model sees only the centre-station PM2.5 channel.  Data construction and
the optimizer/early-stopping semantics are shared with the leakage-free Beijing
coverage experiment.  Rows are flushed after every run so the matrix is safely
resumable.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
from tsai.models.RNN import GRU, LSTM

import run_beijing_leakfree_coverage as beijing
import run_st_patchtst_ablation as legacy


TASK_GRID = tuple(beijing.TASK_GRID)
HEADLINE_TASKS = ((24, 1), (168, 6))
GRID_SEEDS = (2047, 2048, 2049)
HEADLINE_SEEDS = (2047, 2048, 2049, 2050, 2051)
FAMILIES = ("gru", "lstm")
HIDDEN_SIZES = (8, 16, 32, 40, 48, 64)
CENTER_STATION_ID = 1013
OUTPUT_ROOT = Path("experiments/results/trainable_matched")
TERMINAL_STATUSES = frozenset({"completed", "infeasible_oom", "nonfinite"})
TRAINING_SEMANTICS = {
    "optimizer": "AdamW",
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "scheduler": "ReduceLROnPlateau",
    "scheduler_mode": "min",
    "scheduler_factor": 0.5,
    "scheduler_patience": 3,
    "loss": "MSELoss",
    "gradient_clip_norm": 1.0,
    "improvement_delta": 1e-7,
    "selection_split": "valid",
    "evaluation_split": "test",
    "nonfinite_retry_learning_rate_factor": 0.1,
    "nonfinite_retry_count": 1,
}


class SingleStationAdapter(nn.Module):
    """Expose a centre-only tsai model to shared ``[B,S,L]`` datasets."""

    def __init__(
        self,
        model: nn.Module,
        center_station_idx: int,
        horizon: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.center_station_idx = int(center_station_idx)
        self.horizon = int(horizon)
        self.input_channels = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"期望 [B,S,L]，实际 {tuple(x.shape)}")
        if not 0 <= self.center_station_idx < x.shape[1]:
            raise ValueError("center_station_idx 超出输入通道范围")
        # [B, S, L] -> [B, 1, L]; no neighbour value crosses this boundary.
        center_x = x[:, self.center_station_idx:self.center_station_idx + 1, :]
        output = self.model(center_x)
        if output.ndim == 2:
            output = output.unsqueeze(1)
        elif output.ndim == 3 and output.shape[1:] == (self.horizon, 1):
            output = output.transpose(1, 2)
        if output.ndim != 3 or output.shape[1:] != (1, self.horizon):
            raise RuntimeError(
                f"tsai 输出形状 {tuple(output.shape)}，期望 [B,1,{self.horizon}]"
            )
        return output


def build_single_station_model(
    family: str,
    hidden_size: int,
    horizon: int,
    center_station_idx: int,
) -> SingleStationAdapter:
    """Build one fully trainable registered arm."""
    if family not in FAMILIES:
        raise ValueError(f"未注册模型族: {family}")
    if int(hidden_size) not in HIDDEN_SIZES:
        raise ValueError(f"未注册 hidden_size: {hidden_size}")
    constructor = GRU if family == "gru" else LSTM
    model = constructor(1, int(horizon), hidden_size=int(hidden_size))
    adapted = SingleStationAdapter(model, center_station_idx, horizon)
    if adapted.input_channels != 1:
        raise AssertionError("单站点基线必须只登记一个输入通道")
    return adapted


def parameter_counts(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def config_for(history: int, horizon: int, *, smoke: bool = False):
    """Return the locked leakage-free recipe for one task."""
    config = beijing.config_for(int(history), int(horizon))
    if smoke:
        config = replace(config, epochs=2, patience=2)
    assert_training_semantics(config, smoke=smoke)
    return config


def training_semantics_manifest(config, *, smoke: bool = False) -> dict[str, Any]:
    return {
        **TRAINING_SEMANTICS,
        "learning_rate": float(config.learning_rate),
        "weight_decay": float(config.weight_decay),
        "epochs": int(config.epochs),
        "early_stopping_patience": int(config.patience),
        "batch_size": int(config.batch_size),
        "smoke": bool(smoke),
    }


def assert_training_semantics(config, *, smoke: bool = False) -> None:
    expected_epochs = 2 if smoke else (40 if config.history <= 48 else 30)
    expected_patience = 2 if smoke else (8 if config.history <= 48 else 6)
    expected_batch = 256 if config.history <= 48 else 512
    actual = training_semantics_manifest(config, smoke=smoke)
    expected = {
        **TRAINING_SEMANTICS,
        "epochs": expected_epochs,
        "early_stopping_patience": expected_patience,
        "batch_size": expected_batch,
        "smoke": bool(smoke),
    }
    if actual != expected:
        raise AssertionError(f"训练语义偏离: actual={actual}, expected={expected}")


def arm_name(family: str, hidden_size: int) -> str:
    return f"center_{family}_h{int(hidden_size)}"


def run_id(history: int, horizon: int, seed: int, family: str, hidden_size: int) -> str:
    return (
        f"beijing_{history}h_{horizon}h_station{CENTER_STATION_ID}_"
        f"seed{seed}_{arm_name(family, hidden_size)}"
    )


def expected_identities(
    tasks: tuple[tuple[int, int], ...] = TASK_GRID,
    families: tuple[str, ...] = FAMILIES,
    hidden_sizes: tuple[int, ...] = HIDDEN_SIZES,
) -> set[str]:
    identities = set()
    for history, horizon in tasks:
        seeds = HEADLINE_SEEDS if (history, horizon) in HEADLINE_TASKS else GRID_SEEDS
        for seed in seeds:
            for family in families:
                for hidden_size in hidden_sizes:
                    identities.add(run_id(history, horizon, seed, family, hidden_size))
    return identities


def train_one_baseline(
    config,
    datasets: dict,
    metadata: dict,
    family: str,
    hidden_size: int,
    seed: int,
    output_root: Path,
    identity: str,
    device: torch.device,
) -> dict[str, Any]:
    """Train one arm using the locked ``run_st_patchtst_ablation`` semantics."""
    if config.evaluation_split != "test":
        raise ValueError("正式评估划分必须为 test")
    legacy.set_seed(seed)
    model = build_single_station_model(
        family,
        hidden_size,
        config.horizon,
        metadata["center_station_idx"],
    ).to(device)
    total_parameters, trainable_parameters = parameter_counts(model)
    if total_parameters != trainable_parameters:
        raise AssertionError("单站点基线必须端到端训练全部参数")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    loss_fn = nn.MSELoss()
    train_loader = legacy.make_loader(datasets["train"], config, True, seed)
    valid_loader = legacy.make_loader(datasets["valid"], config, False, seed)
    test_loader = legacy.make_loader(datasets["test"], config, False, seed)

    checkpoint_path = output_root / "checkpoints" / f"{identity}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    history_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = loss_fn(prediction, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{identity} 出现非有限训练损失")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(x)
            train_count += len(x)

        valid_prediction, valid_target, _ = legacy.predict(
            model, valid_loader, device, metadata["center_station_idx"]
        )
        valid_loss = float(np.mean((valid_prediction - valid_target) ** 2))
        if not math.isfinite(valid_loss):
            raise FloatingPointError(f"{identity} 出现非有限验证损失")
        scheduler.step(valid_loss)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_sum / train_count,
                "valid_loss": valid_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if valid_loss < best_loss - 1e-7:
            best_loss = valid_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break

    training_seconds = time.perf_counter() - started
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    prediction, target, inference_seconds = legacy.predict(
        model, test_loader, device, metadata["center_station_idx"]
    )
    if not np.isfinite(prediction).all():
        raise FloatingPointError(f"{identity} 出现非有限测试预测")
    metrics = legacy.regression_metrics(
        target, prediction, metadata["center_mean"], metadata["center_std"]
    )

    log_path = output_root / "training_logs" / f"{identity}.csv"
    prediction_path = output_root / "predictions" / f"{identity}.npz"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history_rows).to_csv(log_path, index=False)
    np.savez_compressed(
        prediction_path,
        prediction_scaled=prediction,
        target_scaled=target,
        prediction_ugm3=prediction * metadata["center_std"] + metadata["center_mean"],
        target_ugm3=target * metadata["center_std"] + metadata["center_mean"],
    )
    return {
        "status": "completed",
        "best_epoch": best_epoch,
        "best_valid_loss": best_loss,
        "training_seconds": training_seconds,
        "test_inference_seconds": inference_seconds,
        "evaluation_split": "test",
        "input_channels": 1,
        "selected_channel_index": int(metadata["center_station_idx"]),
        "total_parameter_count": total_parameters,
        "trainable_parameter_count": trainable_parameters,
        "parameter_count": total_parameters,
        "prediction_file": str(prediction_path.relative_to(output_root)),
        "checkpoint_file": str(checkpoint_path.relative_to(output_root)),
        "training_log_file": str(log_path.relative_to(output_root)),
        **metrics,
    }


def _is_oom(error: BaseException) -> bool:
    message = str(error).lower()
    return isinstance(error, torch.OutOfMemoryError) or any(
        phrase in message
        for phrase in ("out of memory", "hip error out of memory", "cuda error: out of memory")
    )


def _failure_result(status: str, error: BaseException, retried: bool) -> dict[str, Any]:
    return {
        "status": status,
        "failure_reason": str(error),
        "learning_rate_retry": retried,
        "evaluation_split": "test",
        "input_channels": 1,
    }


def run_one_with_failure_policy(
    config,
    datasets: dict,
    metadata: dict,
    family: str,
    hidden_size: int,
    seed: int,
    output_root: Path,
    identity: str,
    device: torch.device,
    train_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Apply the registered one-retry nonfinite and terminal OOM policy."""
    train_callable = train_fn or train_one_baseline
    try:
        result = train_callable(
            config, datasets, metadata, family, hidden_size, seed,
            output_root, identity, device,
        )
        result["learning_rate_retry"] = False
        return result
    except FloatingPointError as first_error:
        retry_config = replace(config, learning_rate=config.learning_rate * 0.1)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        try:
            result = train_callable(
                retry_config, datasets, metadata, family, hidden_size, seed,
                output_root, identity, device,
            )
            result["learning_rate_retry"] = True
            result["retry_reason"] = str(first_error)
            return result
        except FloatingPointError as retry_error:
            return _failure_result("nonfinite", retry_error, True)
        except (RuntimeError, torch.OutOfMemoryError) as retry_error:
            if _is_oom(retry_error):
                return _failure_result("infeasible_oom", retry_error, True)
            raise
    except (RuntimeError, torch.OutOfMemoryError) as error:
        if _is_oom(error):
            return _failure_result("infeasible_oom", error, False)
        raise


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return pd.read_csv(path).to_dict("records") if path.is_file() else []


def _save_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _upsert(rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    rows[:] = [item for item in rows if item.get("run_id") != row.get("run_id")]
    rows.append(row)


def _parse_tasks(value: str | None) -> tuple[tuple[int, int], ...]:
    if not value:
        return TASK_GRID
    tasks = tuple(
        tuple(int(part) for part in item.lower().split("x"))
        for item in value.split(",")
    )
    unknown = set(tasks) - set(TASK_GRID)
    if unknown:
        raise ValueError(f"任务不在预注册网格: {sorted(unknown)}")
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default=None, help="逗号分隔 LxH")
    parser.add_argument("--seeds", default=None, help="显式覆盖种子，逗号分隔")
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--hidden-sizes", default=",".join(map(str, HIDDEN_SIZES)))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="24→1/seed2047/全12臂，每臂两轮，单独写入 smoke 目录",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    families = tuple(item.strip() for item in args.families.split(",") if item.strip())
    hidden_sizes = tuple(int(item) for item in args.hidden_sizes.split(",") if item)
    if not set(families).issubset(FAMILIES):
        raise ValueError(f"模型族越界: {families}")
    if not set(hidden_sizes).issubset(HIDDEN_SIZES):
        raise ValueError(f"hidden_size 越界: {hidden_sizes}")
    tasks = ((24, 1),) if args.smoke else _parse_tasks(args.configs)
    explicit_seeds = (
        tuple(int(item) for item in args.seeds.split(",") if item)
        if args.seeds else None
    )
    if args.smoke:
        explicit_seeds = (2047,)
    output_root = Path(args.output_root) if args.output_root else (
        OUTPUT_ROOT / "smoke" if args.smoke else OUTPUT_ROOT
    )
    output_root.mkdir(parents=True, exist_ok=True)
    raw_path = output_root / "raw_metrics.csv"
    rows = _read_rows(raw_path)
    device = torch.device(args.device)

    for history, horizon in tasks:
        config = config_for(history, horizon, smoke=args.smoke)
        seeds = explicit_seeds or (
            HEADLINE_SEEDS if (history, horizon) in HEADLINE_TASKS else GRID_SEEDS
        )
        datasets, metadata = beijing.prepare_datasets_leakfree(config)
        if int(metadata["screened_candidates"]) != len(metadata["station_ids"]):
            raise AssertionError("站点元数据不一致")
        metadata_dir = output_root / "metadata"
        metadata_dir.mkdir(exist_ok=True)
        (metadata_dir / f"dataset_{history}h_{horizon}h.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (metadata_dir / f"config_{history}h_{horizon}h.json").write_text(
            json.dumps(
                {
                    "config": asdict(config),
                    "training_semantics": training_semantics_manifest(
                        config, smoke=args.smoke
                    ),
                    "smoke": args.smoke,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"[{history}h_{horizon}h] stations={len(metadata['station_ids'])} "
            f"splits={metadata['split_sizes']} seeds={seeds}",
            flush=True,
        )
        for seed in seeds:
            for family in families:
                for hidden_size in hidden_sizes:
                    identity = run_id(history, horizon, seed, family, hidden_size)
                    prior = [item for item in rows if item.get("run_id") == identity]
                    if prior and str(prior[-1].get("status")) in TERMINAL_STATUSES:
                        print(f"[跳过] {identity}: {prior[-1]['status']}", flush=True)
                        continue
                    base = {
                        "run_id": identity,
                        "city": "beijing",
                        "history": history,
                        "horizon": horizon,
                        "station_id": CENTER_STATION_ID,
                        "seed": seed,
                        "family": family,
                        "hidden_size": hidden_size,
                        "variant": arm_name(family, hidden_size),
                    }
                    result = run_one_with_failure_policy(
                        config,
                        datasets,
                        metadata,
                        family,
                        hidden_size,
                        seed,
                        output_root,
                        identity,
                        device,
                    )
                    row = {**base, **result}
                    _upsert(rows, row)
                    _save_rows(rows, raw_path)
                    if result["status"] == "completed":
                        print(
                            f"[完成] {identity}: params={result['trainable_parameter_count']} "
                            f"valid={result['best_valid_loss']:.6f} "
                            f"RMSE={result['rmse_ugm3']:.4f}",
                            flush=True,
                        )
                    else:
                        print(f"[登记] {identity}: {result['status']}", flush=True)
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

    print("TRAINABLE_MATCH_SMOKE_DONE" if args.smoke else "TRAINABLE_MATCH_MATRIX_DONE")


if __name__ == "__main__":
    main()
