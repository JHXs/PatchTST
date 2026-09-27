"""Run the preregistered end-to-end spatial-vs-single-station comparison.

The formal matrix trains only O(b).  B(b) is read unchanged from the completed
``trainable_matched`` experiment during summarization.  The optimization loop
below is a literal reuse of ``train_one_baseline``'s locked recipe.
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

import run_beijing_leakfree_coverage as beijing
import run_st_patchtst_ablation as legacy
from backbone_candidates import BackboneAdapter, build_center_backbone
from run_trainable_matched_baselines import (
    HEADLINE_SEEDS,
    HEADLINE_TASKS,
    GRID_SEEDS,
    TASK_GRID,
    TERMINAL_STATUSES,
    assert_training_semantics,
    build_single_station_model,
    config_for,
    parameter_counts,
    training_semantics_manifest,
)


CENTER_STATION_ID = 1013
EXPECTED_STATIONS = 18
OUTPUT_ROOT = Path("experiments/results/endtoend_spatial")
SPATIAL_VARIANT = "st_sparse_station_bias_delta_forecast"
BACKBONES = (
    ("gru", 8),
    ("gru", 16),
    ("gru", 32),
    ("gru", 64),
    ("lstm", 16),
    ("lstm", 32),
)


def backbone_name(family: str, hidden_size: int) -> str:
    return f"{family}_h{int(hidden_size)}"


def run_id(history: int, horizon: int, seed: int, family: str, hidden_size: int) -> str:
    return (
        f"beijing_{history}h_{horizon}h_station{CENTER_STATION_ID}_seed{seed}_"
        f"endtoend_{backbone_name(family, hidden_size)}"
    )


def expected_identities(
    tasks: tuple[tuple[int, int], ...] = tuple(TASK_GRID),
    backbones: tuple[tuple[str, int], ...] = BACKBONES,
) -> set[str]:
    identities: set[str] = set()
    for history, horizon in tasks:
        seeds = HEADLINE_SEEDS if (history, horizon) in HEADLINE_TASKS else GRID_SEEDS
        for seed in seeds:
            for family, hidden_size in backbones:
                identities.add(run_id(history, horizon, seed, family, hidden_size))
    return identities


class EndToEndSpatialAdapter(nn.Module):
    """Expose ST_PatchTST and provide a true no-spatial bypass for audits."""

    def __init__(self, model: nn.Module, spatial_enabled: bool = True) -> None:
        super().__init__()
        self.model = model
        self.spatial_enabled = bool(spatial_enabled)
        self.input_channels = int(model.num_stations)

    @property
    def patch_tst(self) -> nn.Module:
        return self.model.patch_tst

    def _backbone_only(self, x: torch.Tensor) -> torch.Tensor:
        # [B, S, L] -> [B, S, 1, L] -> [B, 1, L]
        reshaped = self.model._reshape_input(x)
        center_x, _ = self.model._split_center_and_neighbors(reshaped)
        return self.model.patch_tst(center_x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.spatial_enabled:
            return self._backbone_only(x)
        return self.model(x)

    def forward_components(
        self, x: torch.Tensor, disable_spatial: bool = False
    ) -> dict[str, torch.Tensor | bool]:
        if disable_spatial or not self.spatial_enabled:
            prediction = self._backbone_only(x)
            return {"prediction": prediction, "spatial_disabled": True}
        return self.model.forward_components(x)


def apply_backbone_candidate(
    model: nn.Module,
    family: str,
    hidden_size: int,
    horizon: int,
    seed: int,
) -> nn.Module:
    """Mount the same freshly initialized B(b) backbone into ST_PatchTST."""
    if (family, int(hidden_size)) not in BACKBONES:
        raise ValueError(f"未注册主干: {family}_h{hidden_size}")
    # ST head construction consumes RNG.  Reset immediately before constructing
    # b so its tensors and the post-construction RNG state exactly match B(b).
    legacy.set_seed(int(seed))
    backbone = build_center_backbone(
        f"center_{family}",
        history=int(model.seq_len),
        horizon=int(horizon),
        hyperparameters={"hidden_size": int(hidden_size)},
    )
    model.patch_tst = BackboneAdapter(backbone, int(horizon))
    return model


def build_endtoend_model(
    config,
    metadata: dict[str, Any],
    family: str,
    hidden_size: int,
    seed: int,
    *,
    spatial_enabled: bool = True,
) -> EndToEndSpatialAdapter:
    if len(metadata["station_ids"]) != EXPECTED_STATIONS:
        raise AssertionError(
            f"O臂必须为{EXPECTED_STATIONS}站，实际{len(metadata['station_ids'])}站"
        )
    model = legacy.build_model(
        config,
        SPATIAL_VARIANT,
        num_stations=len(metadata["station_ids"]),
        center_idx=int(metadata["center_station_idx"]),
    )
    apply_backbone_candidate(model, family, hidden_size, config.horizon, seed)
    adapted = EndToEndSpatialAdapter(model, spatial_enabled=spatial_enabled)
    if adapted.input_channels != EXPECTED_STATIONS:
        raise AssertionError("O臂多站点输入登记失败")
    if not all(parameter.requires_grad for parameter in adapted.parameters()):
        raise AssertionError("O臂必须端到端训练全部参数")
    return adapted


def spatial_parameter_count(model: EndToEndSpatialAdapter) -> int:
    backbone_ids = {id(parameter) for parameter in model.patch_tst.parameters()}
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if id(parameter) not in backbone_ids
    )


@torch.no_grad()
def initial_zero_head_audit(
    model: EndToEndSpatialAdapter,
    dataset,
    config,
    device: torch.device,
) -> dict[str, float | bool]:
    loader = legacy.make_loader(dataset, config, False, 0)
    x, _ = next(iter(loader))
    x = x.to(device)
    model.eval()
    enabled = model.model(x)
    bypassed = model._backbone_only(x)
    max_abs = float((enabled - bypassed).abs().max().item())
    scale = float(bypassed.abs().max().item())
    tolerance = 1e-6 * max(scale, 1.0)
    return {
        "initial_zero_head_max_abs": max_abs,
        "initial_zero_head_tolerance": tolerance,
        "initial_zero_head_ok": bool(max_abs <= tolerance),
    }


def _prediction_path(output_root: Path, identity: str) -> Path:
    return output_root / "predictions" / f"{identity}.npz"


def train_model_with_locked_recipe(
    config,
    datasets: dict,
    metadata: dict,
    seed: int,
    output_root: Path,
    identity: str,
    device: torch.device,
    model_builder: Callable[[], nn.Module],
    *,
    input_channels: int,
    collect_spatial: bool,
) -> dict[str, Any]:
    """Literal shared implementation of the trainable-matched training loop."""
    if config.evaluation_split != "test":
        raise ValueError("正式评估划分必须为 test")
    legacy.set_seed(seed)
    model = model_builder().to(device)
    total_parameters, trainable_parameters = parameter_counts(model)
    if total_parameters != trainable_parameters:
        raise AssertionError("两臂都必须端到端训练全部参数")

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
    diagnostics: dict[str, Any] = {}
    if collect_spatial:
        diagnostics = legacy.collect_spatial_diagnostics(
            model.model, valid_loader, device
        )
    # This is the sole test-set traversal for the run.
    prediction, target, inference_seconds = legacy.predict(
        model, test_loader, device, metadata["center_station_idx"]
    )
    if not np.isfinite(prediction).all():
        raise FloatingPointError(f"{identity} 出现非有限测试预测")
    metrics = legacy.regression_metrics(
        target, prediction, metadata["center_mean"], metadata["center_std"]
    )

    log_path = output_root / "training_logs" / f"{identity}.csv"
    prediction_path = _prediction_path(output_root, identity)
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
    result = {
        "status": "completed",
        "best_epoch": best_epoch,
        "best_valid_loss": best_loss,
        "training_seconds": training_seconds,
        "test_inference_seconds": inference_seconds,
        "selection_split": "valid",
        "evaluation_split": "test",
        "test_evaluation_count": 1,
        "input_channels": int(input_channels),
        "selected_channel_index": int(metadata["center_station_idx"]),
        "total_parameter_count": total_parameters,
        "trainable_parameter_count": trainable_parameters,
        "parameter_count": total_parameters,
        "prediction_file": str(prediction_path.relative_to(output_root)),
        "checkpoint_file": str(checkpoint_path.relative_to(output_root)),
        "training_log_file": str(log_path.relative_to(output_root)),
        **metrics,
        **diagnostics,
    }
    if isinstance(model, EndToEndSpatialAdapter):
        backbone_count = sum(p.numel() for p in model.patch_tst.parameters())
        result.update(
            {
                "backbone_parameter_count": backbone_count,
                "spatial_head_parameter_count": spatial_parameter_count(model),
                "backbone_frozen": False,
            }
        )
    return result


def train_one_endtoend(
    config,
    datasets: dict,
    metadata: dict,
    family: str,
    hidden_size: int,
    seed: int,
    output_root: Path,
    identity: str,
    device: torch.device,
    *,
    spatial_enabled: bool = True,
) -> dict[str, Any]:
    audit_model = build_endtoend_model(
        config, metadata, family, hidden_size, seed, spatial_enabled=True
    ).to(device)
    initial_audit = initial_zero_head_audit(
        audit_model, datasets["valid"], config, device
    )
    del audit_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not initial_audit["initial_zero_head_ok"]:
        raise AssertionError(f"空间头初始不等价: {initial_audit}")

    result = train_model_with_locked_recipe(
        config,
        datasets,
        metadata,
        seed,
        output_root,
        identity,
        device,
        lambda: build_endtoend_model(
            config,
            metadata,
            family,
            hidden_size,
            seed,
            spatial_enabled=spatial_enabled,
        ),
        input_channels=len(metadata["station_ids"]),
        collect_spatial=spatial_enabled,
    )
    return {**result, **initial_audit}


def train_one_baseline_gate(
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
    return train_model_with_locked_recipe(
        config,
        datasets,
        metadata,
        seed,
        output_root,
        identity,
        device,
        lambda: build_single_station_model(
            family, hidden_size, config.horizon, metadata["center_station_idx"]
        ),
        input_channels=1,
        collect_spatial=False,
    )


def _is_oom(error: BaseException) -> bool:
    message = str(error).lower()
    return isinstance(error, torch.OutOfMemoryError) or any(
        phrase in message
        for phrase in (
            "out of memory",
            "hip error out of memory",
            "cuda error: out of memory",
        )
    )


def _failure_result(status: str, error: BaseException, retried: bool) -> dict[str, Any]:
    return {
        "status": status,
        "failure_reason": str(error),
        "learning_rate_retry": retried,
        "selection_split": "valid",
        "evaluation_split": "test",
        "test_evaluation_count": 0,
        "input_channels": EXPECTED_STATIONS,
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
    train_callable = train_fn or train_one_endtoend
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


def _parse_tasks(value: str | None) -> tuple[tuple[int, int], ...]:
    if not value:
        return tuple(TASK_GRID)
    tasks = tuple(
        tuple(int(part) for part in item.lower().split("x"))
        for item in value.split(",")
    )
    unknown = set(tasks) - set(TASK_GRID)
    if unknown:
        raise ValueError(f"任务不在预注册网格: {sorted(unknown)}")
    return tasks


def _parse_backbones(value: str) -> tuple[tuple[str, int], ...]:
    parsed = []
    for item in value.split(","):
        family, hidden = item.strip().split("_h")
        parsed.append((family, int(hidden)))
    result = tuple(parsed)
    unknown = set(result) - set(BACKBONES)
    if unknown:
        raise ValueError(f"主干不在预注册集合: {sorted(unknown)}")
    return result


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return pd.read_csv(path).to_dict("records") if path.is_file() else []


def _save_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _upsert(rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    rows[:] = [item for item in rows if item.get("run_id") != row.get("run_id")]
    rows.append(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default=None, help="逗号分隔 LxH")
    parser.add_argument("--seeds", default=None, help="显式覆盖种子")
    parser.add_argument(
        "--backbones",
        default=",".join(backbone_name(*item) for item in BACKBONES),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="24→1、168→24/seed2047/六主干，各两轮，写入 smoke",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backbones = _parse_backbones(args.backbones)
    tasks = ((24, 1), (168, 24)) if args.smoke else _parse_tasks(args.configs)
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
        assert_training_semantics(config, smoke=args.smoke)
        seeds = explicit_seeds or (
            HEADLINE_SEEDS if (history, horizon) in HEADLINE_TASKS else GRID_SEEDS
        )
        datasets, metadata = beijing.prepare_datasets_leakfree(config)
        if len(metadata["station_ids"]) != EXPECTED_STATIONS:
            raise AssertionError(
                f"{history}→{horizon} 应筛出{EXPECTED_STATIONS}站，"
                f"实际{len(metadata['station_ids'])}站"
            )
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
                    "spatial_variant": SPATIAL_VARIANT,
                    "backbones": [backbone_name(*item) for item in backbones],
                    "smoke": bool(args.smoke),
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
            for family, hidden_size in backbones:
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
                    "backbone": backbone_name(family, hidden_size),
                    "variant": f"endtoend_spatial_{backbone_name(family, hidden_size)}",
                    "spatial_variant": SPATIAL_VARIANT,
                    "smoke_test": bool(args.smoke),
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

    print("ENDTOEND_SPATIAL_SMOKE_DONE" if args.smoke else "ENDTOEND_SPATIAL_MATRIX_DONE")


if __name__ == "__main__":
    main()
