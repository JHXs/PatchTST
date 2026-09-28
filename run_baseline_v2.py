"""Run the preregistered Baseline-v2 Informer/TST single-station matrix.

The optimizer, scheduler, early-stopping and evaluation loop is copied from
``experiment/backbone-upgrade-ablation-failed:run_trainable_matched_baselines.py``.
Every registered arm sees only Beijing station 1013's PM2.5 channel.
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
from tsai.models.RNN import GRU
from tsai.models.TST import TST

from informer_tsai import build_informer_arm
import run_beijing_leakfree_coverage as beijing
import run_st_patchtst_ablation as legacy


TASK_GRID = tuple(beijing.TASK_GRID)
HEADLINE_TASKS = ((24, 1), (168, 6))
GRID_SEEDS = (2047, 2048, 2049)
HEADLINE_SEEDS = (2047, 2048, 2049, 2050, 2051)
CENTER_STATION_ID = 1013
OUTPUT_ROOT = Path("experiments/results/baseline_v2")
TERMINAL_STATUSES = frozenset(
    {"completed", "infeasible_oom", "nonfinite", "nonfinite_prediction"}
)

ARM_SPECS: dict[str, dict[str, Any]] = {
    "informer_d8_e1": {"family": "informer", "d_model": 8, "e_layers": 1},
    "informer_d12_e2": {"family": "informer", "d_model": 12, "e_layers": 2},
    "informer_d16_e1": {"family": "informer", "d_model": 16, "e_layers": 1},
    "informer_d32_e1": {"family": "informer", "d_model": 32, "e_layers": 1},
    "tst_d8_n1": {"family": "tst", "d_model": 8, "n_layers": 1},
    "tst_d12_n1": {"family": "tst", "d_model": 12, "n_layers": 1},
    "tst_d24_n1": {"family": "tst", "d_model": 24, "n_layers": 1},
}
ARMS = tuple(ARM_SPECS)
REPRODUCTION_ARM = "gru_h16"
INFORMER_REGISTERED_COUNTS = {
    "informer_d8_e1": 1609,
    "informer_d12_e2": 5125,
    "informer_d16_e1": 5777,
    "informer_d32_e1": 21793,
}
TST_REGISTERED_COUNTS = {
    (24, 1): {"tst_d8_n1": 969, "tst_d12_n1": 1837, "tst_d24_n1": 5977},
    (24, 6): {"tst_d8_n1": 1934, "tst_d12_n1": 3282, "tst_d24_n1": 8862},
    (168, 6): {"tst_d8_n1": 9998, "tst_d12_n1": 15378, "tst_d24_n1": 33054},
    (168, 24): {"tst_d8_n1": 34208, "tst_d12_n1": 51684, "tst_d24_n1": 105648},
}
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


class NonfinitePredictionError(FloatingPointError):
    """A terminal non-finite test prediction; unlike training it is not retried."""


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
    arm: str,
    history: int,
    horizon: int,
    center_station_idx: int,
) -> SingleStationAdapter:
    """Build one registered arm, plus the isolated GRU reproduction arm."""
    if arm == REPRODUCTION_ARM:
        backbone = GRU(1, int(horizon), hidden_size=16)
    elif arm in ARM_SPECS:
        spec = ARM_SPECS[arm]
        if spec["family"] == "informer":
            backbone = build_informer_arm(
                spec["d_model"],
                spec["e_layers"],
                int(history),
                int(horizon),
                int(center_station_idx),
            )
        else:
            backbone = TST(
                c_in=1,
                c_out=int(horizon),
                seq_len=int(history),
                n_heads=1,
                d_model=int(spec["d_model"]),
                d_ff=2 * int(spec["d_model"]),
                n_layers=int(spec["n_layers"]),
                dropout=0.1,
            )
    else:
        raise ValueError(f"未注册模型臂: {arm}")
    adapted = SingleStationAdapter(backbone, center_station_idx, horizon)
    if adapted.input_channels != 1:
        raise AssertionError("单站点基线必须只登记一个输入通道")
    return adapted


def parameter_counts(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def independent_parameter_count(
    arm: str, history: int, horizon: int, center_station_idx: int = 0
) -> tuple[int, int]:
    """Construct a fresh model through the audit path and count its parameters."""
    return parameter_counts(
        build_single_station_model(arm, history, horizon, center_station_idx)
    )


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


def arm_family(arm: str) -> str:
    if arm == REPRODUCTION_ARM:
        return "gru"
    return str(ARM_SPECS[arm]["family"])


def run_id(history: int, horizon: int, seed: int, arm: str) -> str:
    return (
        f"beijing_{history}h_{horizon}h_station{CENTER_STATION_ID}_"
        f"seed{seed}_{arm}"
    )


def expected_identities(
    tasks: tuple[tuple[int, int], ...] = TASK_GRID,
    arms: tuple[str, ...] = ARMS,
) -> set[str]:
    identities = set()
    for history, horizon in tasks:
        seeds = HEADLINE_SEEDS if (history, horizon) in HEADLINE_TASKS else GRID_SEEDS
        for seed in seeds:
            for arm in arms:
                identities.add(run_id(history, horizon, seed, arm))
    return identities


def train_one_baseline(
    config,
    datasets: dict,
    metadata: dict,
    arm: str,
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
        arm,
        config.history,
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
        raise NonfinitePredictionError(f"{identity} 出现非有限测试预测")
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
        "selection_split": "valid",
        "evaluation_split": "test",
        "test_evaluation_count": 1,
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
        "selection_split": "valid",
        "evaluation_split": "test",
        "test_evaluation_count": 1 if status == "nonfinite_prediction" else 0,
        "input_channels": 1,
    }


def run_one_with_failure_policy(
    config,
    datasets: dict,
    metadata: dict,
    arm: str,
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
            config, datasets, metadata, arm, seed, output_root, identity, device
        )
        result["learning_rate_retry"] = False
        return result
    except NonfinitePredictionError as error:
        return _failure_result("nonfinite_prediction", error, False)
    except FloatingPointError as first_error:
        retry_config = replace(config, learning_rate=config.learning_rate * 0.1)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        try:
            result = train_callable(
                retry_config,
                datasets,
                metadata,
                arm,
                seed,
                output_root,
                identity,
                device,
            )
            result["learning_rate_retry"] = True
            result["retry_reason"] = str(first_error)
            return result
        except NonfinitePredictionError as retry_error:
            return _failure_result("nonfinite_prediction", retry_error, True)
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


def should_skip_identity(rows: list[dict[str, Any]], identity: str) -> bool:
    """Return true only when the latest persisted row is terminal."""
    prior = [item for item in rows if item.get("run_id") == identity]
    return bool(prior and str(prior[-1].get("status")) in TERMINAL_STATUSES)


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
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="24→1、168→6、seed2047、全7臂，每臂两轮，写入 smoke 目录",
    )
    return parser.parse_args()


def _assert_center_resolution(metadata: dict[str, Any]) -> None:
    station_ids = [int(value) for value in metadata["station_ids"]]
    resolved = station_ids.index(CENTER_STATION_ID)
    if resolved != int(metadata["center_station_idx"]):
        raise AssertionError("中心站索引未由 station_ids 正确解析")
    if CENTER_STATION_ID == 1013 and resolved != 9:
        raise AssertionError(f"北京 1013 中心站索引应为 9，实际 {resolved}")


def main() -> None:
    args = parse_args()
    arms = tuple(item.strip() for item in args.arms.split(",") if item.strip())
    if not set(arms).issubset(ARMS):
        raise ValueError(f"模型臂越界: {arms}")
    tasks = HEADLINE_TASKS if args.smoke else _parse_tasks(args.configs)
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
    started_all = time.perf_counter()

    for history, horizon in tasks:
        config = config_for(history, horizon, smoke=args.smoke)
        seeds = explicit_seeds or (
            HEADLINE_SEEDS if (history, horizon) in HEADLINE_TASKS else GRID_SEEDS
        )
        datasets, metadata = beijing.prepare_datasets_leakfree(config)
        if int(metadata["screened_candidates"]) != len(metadata["station_ids"]):
            raise AssertionError("站点元数据不一致")
        _assert_center_resolution(metadata)
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
                    "registered_arms": list(arms),
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
            for arm in arms:
                identity = run_id(history, horizon, seed, arm)
                prior = [item for item in rows if item.get("run_id") == identity]
                if should_skip_identity(rows, identity):
                    print(f"[跳过] {identity}: {prior[-1]['status']}", flush=True)
                    continue
                spec = ARM_SPECS[arm]
                base = {
                    "run_id": identity,
                    "city": "beijing",
                    "history": history,
                    "horizon": horizon,
                    "station_id": CENTER_STATION_ID,
                    "seed": seed,
                    "family": arm_family(arm),
                    "backbone": arm,
                    "variant": arm,
                    "d_model": spec["d_model"],
                    "e_layers": spec.get("e_layers", np.nan),
                    "n_layers": spec.get("n_layers", np.nan),
                    "capacity_alignment": (
                        "matched" if spec["family"] == "informer" else "non_aligned_as_is"
                    ),
                }
                result = run_one_with_failure_policy(
                    config,
                    datasets,
                    metadata,
                    arm,
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

    run_manifest = {
        "smoke": bool(args.smoke),
        "tasks": [list(task) for task in tasks],
        "arms": list(arms),
        "formal_expected_runs": 448,
        "elapsed_seconds": time.perf_counter() - started_all,
        "device": str(device),
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("BASELINE_V2_SMOKE_DONE" if args.smoke else "BASELINE_V2_MATRIX_DONE")


if __name__ == "__main__":
    main()
