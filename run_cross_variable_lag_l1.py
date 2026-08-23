"""Run the frozen MC-CVLRA L1 train/validation ablation protocol.

Formal L1 uses seeds 2060-2062 and the fixed budgets encoded below.  ``--smoke``
uses one seed, one epoch, and optional sample limits, and always writes to a path
whose basename ends in ``_smoke``.  This runner never constructs test features,
targets, windows, predictions, caches, or metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cross_variable_lag_adapter import (
    ADAPTER_VARIANTS,
    BETA_INIT,
    BETA_MAX,
    CrossVariableLagAdapter,
    FlatResidualAdapter,
    assert_tensor_mappings_equal,
    clone_module_parameters_and_buffers,
    compare_tensor_mappings,
    controlled_token_input,
    expected_m_parameter_count,
    module_state_hash,
    select_flat_width,
    tensor_mapping_hash,
    trainable_parameter_count,
)
from cross_variable_lag_l0_data import AUXILIARY_COLUMNS, stable_hash
from cross_variable_lag_l1_data import (
    L1PreparedData,
    L1TaskConfig,
    L1TensorDataset,
    L1WindowArrays,
    prepare_l1_data,
)
from run_st_patchtst_ablation import ExperimentConfig as STExperimentConfig
from run_st_patchtst_ablation import build_model as build_st_model


FORMAL_SEEDS = (2060, 2061, 2062)
SHUFFLE_SEEDS = (3101, 3102, 3103, 3104, 3105)
BASE_VARIANTS = ("F00", "F10")
ALL_VARIANTS = ("F00", "F01", "F10", "F11", "B1", "B2", "B_flat", "B3")
LOCKED_ST_VARIANT = "st_sparse_station_bias_delta_forecast"
FORMAL_OUTPUTS = {
    (24, 1): Path(
        "experiments/results/cross_variable_lag_adapter/l1_selection_24h_1h"
    ),
    (168, 6): Path(
        "experiments/results/cross_variable_lag_adapter/l1_selection_168h_6h"
    ),
}
TASK_BUDGETS = {
    (24, 1): {"epochs": 40, "patience": 8, "batch_size": 256},
    (168, 6): {"epochs": 30, "patience": 6, "batch_size": 512},
}
KEY_L1_SOURCE_FILES = (
    "cross_variable_lag_adapter.py",
    "cross_variable_lag_l1_data.py",
    "run_cross_variable_lag_l1.py",
    "run_cross_variable_lag_l1_gpu_smoke.py",
    "summarize_cross_variable_lag_l1.py",
    "test_cross_variable_lag_l1.py",
)


@dataclass(frozen=True)
class L1RunConfig:
    history: int
    horizon: int
    epochs: int
    patience: int
    batch_size: int
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    minimum_improvement: float = 1e-7
    gradient_clip_norm: float = 1.0
    beta_max: float = BETA_MAX
    beta_init: float = BETA_INIT
    sparse_neighbor_top_k: int = 5
    selection_seeds: tuple[int, ...] = FORMAL_SEEDS
    shuffle_seeds: tuple[int, ...] = SHUFFLE_SEEDS
    variants: tuple[str, ...] = ALL_VARIANTS
    loss: str = "MSE"
    optimizer: str = "AdamW"
    evaluation_split: str = "validation"

    @property
    def task_name(self) -> str:
        return f"{self.history}h_{self.horizon}h"

    @property
    def config_hash(self) -> str:
        return stable_hash(asdict(self))


class CachedAdapterDataset(Dataset):
    def __init__(self, base_prediction: np.ndarray, arrays: L1WindowArrays) -> None:
        if len(base_prediction) != len(arrays.origin_rows):
            raise ValueError("Cached base predictions and adapter arrays differ in length")
        self.base_prediction = np.asarray(base_prediction, dtype=np.float32)
        self.arrays = arrays

    def __len__(self) -> int:
        return len(self.arrays.origin_rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return (
            torch.from_numpy(self.base_prediction[index]),
            torch.from_numpy(self.arrays.auxiliary_tokens[index]),
            torch.from_numpy(self.arrays.self_tokens[index]),
            torch.from_numpy(self.arrays.context[index]),
            torch.from_numpy(self.arrays.targets_scaled[index]),
            torch.tensor(index, dtype=torch.long),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("24:1", "168:6"), required=True)
    parser.add_argument(
        "--data-dir", default="tsai/data/stations_data", help="Read-only Beijing CSV directory."
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One seed/one epoch flow check; output basename is forced to end in _smoke.",
    )
    parser.add_argument("--smoke-train-samples", type=int, default=64)
    parser.add_argument("--smoke-validation-samples", type=int, default=48)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_git_provenance(repo_root: Path = Path(".")) -> dict[str, Any]:
    """Capture the exact source revision and dirty state used by a run."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    porcelain = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    source_hashes = {}
    for relative_path in KEY_L1_SOURCE_FILES:
        path = repo_root / relative_path
        if path.is_file():
            source_hashes[relative_path] = file_sha256(path)
    return {
        "git_commit": commit,
        "git_dirty": bool(porcelain),
        "git_status_porcelain": porcelain,
        "key_l1_source_sha256": source_hashes,
    }


def enforce_formal_clean_tree(provenance: dict[str, Any], smoke: bool) -> None:
    """Formal L1 is fail-closed on a dirty tree; engineering smoke is labelled."""
    if not smoke and bool(provenance["git_dirty"]):
        raise RuntimeError(
            "Formal L1 requires a clean git tree; run review/commit first. "
            "Dirty smoke is permitted only as pre-commit engineering evidence."
        )


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _st_config(config: L1RunConfig) -> STExperimentConfig:
    return STExperimentConfig(
        history=config.history,
        horizon=config.horizon,
        batch_size=config.batch_size,
        epochs=config.epochs,
        patience=config.patience,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        sparse_neighbor_top_k=config.sparse_neighbor_top_k,
        evaluation_split="valid",
        initialize_from_degraded=True,
        freeze_backbone=True,
    )


def build_l1_base_model(
    config: L1RunConfig,
    variant: str,
    num_stations: int,
    center_station_idx: int,
) -> nn.Module:
    if variant == "F00":
        st_variant = "degraded_patchtst"
    elif variant == "F10":
        st_variant = LOCKED_ST_VARIANT
    else:
        raise ValueError(f"Unknown base variant: {variant}")
    return build_st_model(
        _st_config(config),
        st_variant,
        num_stations=num_stations,
        center_idx=center_station_idx,
    )


def initialize_locked_st_from_f00(
    locked_model: nn.Module,
    f00_state: dict[str, torch.Tensor],
) -> None:
    incompatible = locked_model.load_state_dict(f00_state, strict=False)
    missing = [key for key in incompatible.missing_keys if key != "alpha_logit"]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "F00->F10 checkpoint incompatibility: "
            f"missing={missing}, unexpected={list(incompatible.unexpected_keys)}"
        )
    for parameter in locked_model.patch_tst.parameters():
        parameter.requires_grad = False
    locked_model.patch_tst.eval()


def patchtst_checkpoint_substate(
    full_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Clone the persistent PatchTST substate from an ST checkpoint."""
    prefix = "patch_tst."
    substate = {
        name[len(prefix) :]: value.detach().cpu().clone()
        for name, value in full_state.items()
        if name.startswith(prefix)
    }
    if not substate:
        raise AssertionError("Checkpoint has no patch_tst substate")
    return substate


def set_locked_spatial_train_mode(model: nn.Module) -> None:
    """Train F10 spatial modules while keeping every PatchTST module in eval."""
    model.train()
    model.patch_tst.eval()
    if model.patch_tst.training or any(
        module.training for module in model.patch_tst.modules()
    ):
        raise AssertionError("PatchTST entered train mode during locked spatial training")
    if any(parameter.requires_grad for parameter in model.patch_tst.parameters()):
        raise AssertionError("A PatchTST parameter is trainable in locked F10")


def _patchtst_audit_row(
    config: L1RunConfig,
    variant: str,
    seed: int,
    stage: str,
    model: nn.Module,
    expected_complete_state: dict[str, torch.Tensor] | None,
    expected_persistent_state: dict[str, torch.Tensor] | None,
    epoch: int | None = None,
) -> dict[str, Any]:
    complete_state = clone_module_parameters_and_buffers(model.patch_tst)
    persistent_state = {
        name: value.detach().cpu().clone()
        for name, value in model.patch_tst.state_dict().items()
    }
    complete_differences = (
        compare_tensor_mappings(complete_state, expected_complete_state)
        if expected_complete_state is not None
        else []
    )
    persistent_differences = (
        compare_tensor_mappings(persistent_state, expected_persistent_state)
        if expected_persistent_state is not None
        else []
    )
    return {
        "task": config.task_name,
        "variant": variant,
        "seed": seed,
        "epoch": "" if epoch is None else epoch,
        "stage": stage,
        "patchtst_persistent_state_sha256": tensor_mapping_hash(persistent_state),
        "patchtst_complete_state_sha256": tensor_mapping_hash(complete_state),
        "expected_persistent_state_sha256": (
            tensor_mapping_hash(expected_persistent_state)
            if expected_persistent_state is not None
            else ""
        ),
        "expected_complete_state_sha256": (
            tensor_mapping_hash(expected_complete_state)
            if expected_complete_state is not None
            else ""
        ),
        "persistent_tensor_count": len(persistent_state),
        "complete_parameter_buffer_count": len(complete_state),
        "persistent_changed_tensor_count": len(persistent_differences),
        "complete_changed_tensor_count": len(complete_differences),
        "persistent_changed_tensors_json": json.dumps(persistent_differences),
        "complete_changed_tensors_json": json.dumps(complete_differences),
        "persistent_state_equal": not persistent_differences,
        "all_parameters_and_buffers_equal": not complete_differences,
        "patchtst_eval": not model.patch_tst.training,
    }


def _assert_patchtst_matches(
    model: nn.Module,
    expected_complete_state: dict[str, torch.Tensor],
    expected_persistent_state: dict[str, torch.Tensor],
    label: str,
) -> None:
    assert_tensor_mappings_equal(
        clone_module_parameters_and_buffers(model.patch_tst),
        expected_complete_state,
        f"{label} complete PatchTST parameters/buffers",
    )
    assert_tensor_mappings_equal(
        model.patch_tst.state_dict(),
        expected_persistent_state,
        f"{label} persistent PatchTST state",
    )


def _assert_checkpoint_patchtst_matches(
    checkpoint_path: Path,
    expected_persistent_state: dict[str, torch.Tensor],
    label: str,
) -> str:
    checkpoint_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_patchtst = patchtst_checkpoint_substate(checkpoint_state)
    assert_tensor_mappings_equal(
        checkpoint_patchtst,
        expected_persistent_state,
        f"{label} checkpoint PatchTST substate",
    )
    return tensor_mapping_hash(checkpoint_patchtst)


@torch.no_grad()
def predict_base(
    model: nn.Module,
    arrays: L1WindowArrays,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    loader = make_loader(L1TensorDataset(arrays), batch_size, False, 0)
    predictions = []
    for x_st, _, _, _, _, _ in loader:
        predictions.append(model(x_st.to(device, non_blocking=True)).cpu())
    return torch.cat(predictions).numpy()


def _base_validation_loss(
    model: nn.Module,
    arrays: L1WindowArrays,
    batch_size: int,
    device: torch.device,
) -> float:
    prediction = predict_base(model, arrays, batch_size, device)
    return float(np.mean((prediction - arrays.targets_scaled) ** 2))


def _write_state_checkpoint(model: nn.Module, path: Path) -> tuple[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return file_sha256(path), module_state_hash(model)


def train_base_model(
    config: L1RunConfig,
    data: L1PreparedData,
    variant: str,
    seed: int,
    output_dir: Path,
    device: torch.device,
    f00_checkpoint: Path | None = None,
) -> tuple[nn.Module, dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    set_seed(seed)
    model = build_l1_base_model(
        config,
        variant,
        num_stations=len(data.metadata["selected_station_ids"]),
        center_station_idx=int(data.metadata["center_station_idx"]),
    ).to(device)
    initialized_from_hash = ""
    source_f00_patchtst_hash = ""
    if variant == "F10":
        if f00_checkpoint is None or not f00_checkpoint.is_file():
            raise FileNotFoundError("F10 requires the same-seed F00 best checkpoint")
        f00_state = torch.load(f00_checkpoint, map_location=device, weights_only=True)
        source_f00_patchtst = patchtst_checkpoint_substate(f00_state)
        source_f00_patchtst_hash = tensor_mapping_hash(source_f00_patchtst)
        initialize_locked_st_from_f00(model, f00_state)
        initialized_from_hash = file_sha256(f00_checkpoint)
        assert_tensor_mappings_equal(
            model.patch_tst.state_dict(),
            source_f00_patchtst,
            "F10 PatchTST immediately after F00 initialization",
        )
    initial_model_state_hash = module_state_hash(model)
    initial_patchtst_persistent = {
        name: value.detach().cpu().clone()
        for name, value in model.patch_tst.state_dict().items()
    }
    initial_patchtst_complete = clone_module_parameters_and_buffers(model.patch_tst)
    initial_patchtst_persistent_hash = tensor_mapping_hash(
        initial_patchtst_persistent
    )
    initial_patchtst_complete_hash = tensor_mapping_hash(initial_patchtst_complete)
    if variant == "F10" and initial_patchtst_persistent_hash != source_f00_patchtst_hash:
        raise AssertionError("F10 initialized PatchTST hash differs from F00 checkpoint")
    patchtst_audit_rows = [
        _patchtst_audit_row(
            config,
            variant,
            seed,
            (
                "f10_after_f00_checkpoint_initialization"
                if variant == "F10"
                else "f00_before_training"
            ),
            model,
            initial_patchtst_complete if variant == "F10" else None,
            initial_patchtst_persistent if variant == "F10" else None,
        )
    ]
    patchtst_audit_rows[0]["invariant_required"] = variant == "F10"
    patchtst_audit_rows[0]["source_f00_checkpoint_patchtst_sha256"] = (
        source_f00_patchtst_hash
    )
    patchtst_audit_rows[0]["checkpoint_patchtst_sha256"] = ""
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )
    loss_fn = nn.MSELoss()
    train_loader = make_loader(
        L1TensorDataset(data.train), config.batch_size, True, seed
    )
    checkpoint_path = output_dir / "checkpoints" / f"{variant}_seed{seed}.pt"
    history_rows: list[dict[str, Any]] = []
    best_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        if variant == "F10":
            set_locked_spatial_train_mode(model)
        else:
            model.train()
        train_loss_sum = 0.0
        train_count = 0
        for x_st, _, _, _, target, _ in train_loader:
            x_st = x_st.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x_st)
            loss = loss_fn(prediction, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{variant} seed={seed} non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, config.gradient_clip_norm)
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(x_st)
            train_count += len(x_st)
        if variant == "F10":
            _assert_patchtst_matches(
                model,
                initial_patchtst_complete,
                initial_patchtst_persistent,
                f"F10 seed={seed} epoch={epoch} after training batches",
            )
        validation_loss = _base_validation_loss(
            model, data.validation, config.batch_size, device
        )
        if variant == "F10":
            _assert_patchtst_matches(
                model,
                initial_patchtst_complete,
                initial_patchtst_persistent,
                f"F10 seed={seed} epoch={epoch} after validation",
            )
        scheduler.step(validation_loss)
        epoch_patchtst_audit = _patchtst_audit_row(
            config,
            variant,
            seed,
            "epoch_after_validation",
            model,
            initial_patchtst_complete if variant == "F10" else None,
            initial_patchtst_persistent if variant == "F10" else None,
            epoch=epoch,
        )
        epoch_patchtst_audit["invariant_required"] = variant == "F10"
        epoch_patchtst_audit["source_f00_checkpoint_patchtst_sha256"] = (
            source_f00_patchtst_hash
        )
        epoch_patchtst_audit["checkpoint_patchtst_sha256"] = ""
        patchtst_audit_rows.append(epoch_patchtst_audit)
        history_rows.append(
            {
                "task": config.task_name,
                "variant": variant,
                "seed": seed,
                "epoch": epoch,
                "train_loss": train_loss_sum / train_count,
                "validation_loss": validation_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "patchtst_eval_during_locked_training": (
                    not model.patch_tst.training if variant == "F10" else ""
                ),
                "patchtst_persistent_state_sha256": (
                    epoch_patchtst_audit["patchtst_persistent_state_sha256"]
                ),
                "patchtst_complete_state_sha256": (
                    epoch_patchtst_audit["patchtst_complete_state_sha256"]
                ),
                "patchtst_all_parameters_buffers_equal_to_initialization": (
                    epoch_patchtst_audit["all_parameters_and_buffers_equal"]
                    if variant == "F10"
                    else ""
                ),
            }
        )
        if validation_loss < best_loss - config.minimum_improvement:
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            _write_state_checkpoint(model, checkpoint_path)
            checkpoint_patchtst_hash = tensor_mapping_hash(
                patchtst_checkpoint_substate(
                    torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                )
            )
            if variant == "F10":
                _assert_patchtst_matches(
                    model,
                    initial_patchtst_complete,
                    initial_patchtst_persistent,
                    f"F10 seed={seed} epoch={epoch} best checkpoint live model",
                )
                independently_loaded_hash = _assert_checkpoint_patchtst_matches(
                    checkpoint_path,
                    initial_patchtst_persistent,
                    f"F10 seed={seed} epoch={epoch} best",
                )
                if independently_loaded_hash != checkpoint_patchtst_hash:
                    raise AssertionError("Independent best-checkpoint hashes disagree")
            checkpoint_audit = _patchtst_audit_row(
                config,
                variant,
                seed,
                "best_checkpoint_saved",
                model,
                initial_patchtst_complete if variant == "F10" else None,
                initial_patchtst_persistent if variant == "F10" else None,
                epoch=epoch,
            )
            checkpoint_audit["invariant_required"] = variant == "F10"
            checkpoint_audit["source_f00_checkpoint_patchtst_sha256"] = (
                source_f00_patchtst_hash
            )
            checkpoint_audit["checkpoint_patchtst_sha256"] = (
                checkpoint_patchtst_hash
            )
            patchtst_audit_rows.append(checkpoint_audit)
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break
    training_seconds = time.perf_counter() - started
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    if variant == "F10":
        _assert_patchtst_matches(
            model,
            initial_patchtst_complete,
            initial_patchtst_persistent,
            f"F10 seed={seed} final best-checkpoint load",
        )
        set_locked_spatial_train_mode(model)
        model.eval()
    final_audit = _patchtst_audit_row(
        config,
        variant,
        seed,
        "final_best_checkpoint_loaded",
        model,
        initial_patchtst_complete if variant == "F10" else None,
        initial_patchtst_persistent if variant == "F10" else None,
        epoch=best_epoch,
    )
    final_audit["invariant_required"] = variant == "F10"
    final_audit["source_f00_checkpoint_patchtst_sha256"] = (
        source_f00_patchtst_hash
    )
    final_audit["checkpoint_patchtst_sha256"] = _assert_checkpoint_patchtst_matches(
        checkpoint_path,
        initial_patchtst_persistent,
        f"F10 seed={seed} final checkpoint",
    ) if variant == "F10" else tensor_mapping_hash(
        patchtst_checkpoint_substate(
            torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        )
    )
    patchtst_audit_rows.append(final_audit)
    checkpoint_hash = file_sha256(checkpoint_path)
    state_hash = module_state_hash(model)
    if variant == "F10":
        for parameter in model.patch_tst.parameters():
            parameter.requires_grad = False
        model.patch_tst.eval()
    final_patchtst_persistent_hash = module_state_hash(model.patch_tst)
    final_patchtst_complete_hash = tensor_mapping_hash(
        clone_module_parameters_and_buffers(model.patch_tst)
    )
    checkpoint_row = {
        "task": config.task_name,
        "seed": seed,
        "variant": variant,
        "checkpoint_role": "best_validation_base",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "model_state_sha256": state_hash,
        "initialized_from_checkpoint": str(f00_checkpoint or ""),
        "initialized_from_sha256": initialized_from_hash,
        "patchtst_frozen": variant == "F10",
        "source_f00_checkpoint_patchtst_sha256": source_f00_patchtst_hash,
        "patchtst_initial_state_sha256": initial_patchtst_persistent_hash,
        "patchtst_initial_complete_state_sha256": initial_patchtst_complete_hash,
        "patchtst_checkpoint_state_sha256": final_audit[
            "checkpoint_patchtst_sha256"
        ],
        "patchtst_final_loaded_state_sha256": final_patchtst_persistent_hash,
        "patchtst_final_loaded_complete_state_sha256": (
            final_patchtst_complete_hash
        ),
        "patchtst_state_unchanged_from_f00_initialization": (
            variant == "F10"
            and source_f00_patchtst_hash
            == initial_patchtst_persistent_hash
            == final_audit["checkpoint_patchtst_sha256"]
            == final_patchtst_persistent_hash
            and initial_patchtst_complete_hash == final_patchtst_complete_hash
        ),
        "shared_by_adapter_variants": (
            "F01" if variant == "F00" else "F11,B1,B2,B_flat,B3"
        ),
    }
    model_fit_row = {
        "object_id": f"model_fit:{variant}:seed{seed}",
        "object_type": "training_model_fit",
        "task": config.task_name,
        "fold": "",
        "model": variant,
        "horizon_step": "",
        "fit_start": pd.Timestamp(data.train.origin_times.min()).isoformat(),
        "fit_end": pd.Timestamp(data.train.origin_times.max()).isoformat(),
        "row_count": len(data.train.origin_rows),
        "config_hash": config.config_hash,
        "state_hash": state_hash,
        "details": json.dumps(
            {
                "loss": config.loss,
                "optimizer": config.optimizer,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "checkpoint_sha256": checkpoint_hash,
                "source_f00_checkpoint_patchtst_sha256": (
                    source_f00_patchtst_hash
                ),
                "patchtst_initial_state_sha256": initial_patchtst_persistent_hash,
                "patchtst_checkpoint_state_sha256": final_audit[
                    "checkpoint_patchtst_sha256"
                ],
                "patchtst_final_loaded_state_sha256": (
                    final_patchtst_persistent_hash
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    }
    early_stopping_row = {
        "object_id": f"early_stopping:{variant}:seed{seed}",
        "object_type": "validation_early_stopping_selector",
        "task": config.task_name,
        "fold": "",
        "model": variant,
        "horizon_step": "",
        "fit_start": pd.Timestamp(data.validation.origin_times.min()).isoformat(),
        "fit_end": pd.Timestamp(data.validation.origin_times.max()).isoformat(),
        "row_count": len(data.validation.origin_rows),
        "config_hash": config.config_hash,
        "state_hash": stable_hash(
            {"best_epoch": best_epoch, "best_validation_loss": best_loss}
        ),
        "details": json.dumps(
            {
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
                "validation_samples": len(data.validation.origin_rows),
                "minimum_improvement": config.minimum_improvement,
                "patience": config.patience,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    }
    run_row = {
        "task": config.task_name,
        "seed": seed,
        "variant": variant,
        "base_variant": variant,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "training_seconds": training_seconds,
        "trainable_parameter_count": trainable_parameter_count(model),
        "checkpoint_sha256": checkpoint_hash,
        "base_state_before_sha256": initial_model_state_hash,
        "base_state_after_sha256": state_hash,
        "base_state_unchanged": initial_model_state_hash == state_hash,
        "source_f00_checkpoint_patchtst_sha256": source_f00_patchtst_hash,
        "patchtst_initial_state_sha256": initial_patchtst_persistent_hash,
        "patchtst_checkpoint_state_sha256": final_audit[
            "checkpoint_patchtst_sha256"
        ],
        "patchtst_final_loaded_state_sha256": final_patchtst_persistent_hash,
        "patchtst_initial_complete_state_sha256": initial_patchtst_complete_hash,
        "patchtst_final_loaded_complete_state_sha256": (
            final_patchtst_complete_hash
        ),
        "patchtst_state_unchanged": (
            initial_patchtst_persistent_hash == final_patchtst_persistent_hash
            and initial_patchtst_complete_hash == final_patchtst_complete_hash
        ),
        "patchtst_locked_eval": variant == "F10" and not model.patch_tst.training,
    }
    return model, run_row, history_rows, {
        "checkpoint": checkpoint_row,
        "fits": [model_fit_row, early_stopping_row],
        "patchtst_audits": patchtst_audit_rows,
    }


def build_adapter(config: L1RunConfig, variant: str) -> nn.Module:
    if variant not in ADAPTER_VARIANTS:
        raise ValueError(f"Unknown adapter variant: {variant}")
    if variant == "B_flat":
        width_info = select_flat_width(config.horizon)
        if not width_info["accepted"]:
            raise RuntimeError("B-flat parameter match exceeds the frozen 3% limit")
        return FlatResidualAdapter(
            config.horizon,
            width=int(width_info["selected_width"]),
            beta_max=config.beta_max,
            beta_init=config.beta_init,
        )
    return CrossVariableLagAdapter(
        config.horizon,
        beta_max=config.beta_max,
        beta_init=config.beta_init,
        uniform_gate=variant == "B3",
    )


def _gradient_norm(parameter: torch.Tensor | None) -> float:
    if parameter is None:
        return 0.0
    return float(parameter.detach().norm().item())


def gradient_contract(adapter: nn.Module) -> dict[str, float | bool]:
    named = dict(adapter.named_parameters())
    if isinstance(adapter, CrossVariableLagAdapter):
        output_names = [name for name in named if name.startswith("output_heads") and ".2." in name]
        beta_names = ["beta_logit"] if "beta_logit" in named else []
        upstream_names = [
            name for name in named if name not in set(output_names + beta_names)
        ]
        if adapter.uniform_gate:
            upstream_names = [
                name
                for name in upstream_names
                if not name.startswith("horizon_queries") and name != "score_bias"
            ]
    elif isinstance(adapter, FlatResidualAdapter):
        output_names = [name for name in named if name.startswith("network.4")]
        beta_names = ["beta_logit"] if "beta_logit" in named else []
        upstream_names = [
            name for name in named if name not in set(output_names + beta_names)
        ]
    else:
        raise TypeError("Unknown adapter class for gradient audit")
    output_norm = sum(_gradient_norm(named[name].grad) for name in output_names)
    upstream_norm = sum(_gradient_norm(named[name].grad) for name in upstream_names)
    beta_norm = sum(_gradient_norm(named[name].grad) for name in beta_names)
    return {
        "output_layer_gradient_norm_sum": output_norm,
        "upstream_gradient_norm_sum": upstream_norm,
        "beta_gradient_norm_sum": beta_norm,
        "output_layer_nonzero": output_norm > 0,
        "upstream_nonzero": upstream_norm > 0,
        "beta_nonzero": beta_norm > 0,
    }


def _adapter_forward(
    adapter: nn.Module,
    variant: str,
    base_prediction: torch.Tensor,
    auxiliary_tokens: torch.Tensor,
    self_tokens: torch.Tensor,
    context: torch.Tensor,
) -> torch.Tensor:
    tokens = controlled_token_input(auxiliary_tokens, self_tokens, variant)
    return adapter(base_prediction, tokens, context)


@torch.no_grad()
def predict_adapter(
    adapter: nn.Module,
    variant: str,
    base_prediction: np.ndarray,
    arrays: L1WindowArrays,
    batch_size: int,
    device: torch.device,
    auxiliary_override: np.ndarray | None = None,
) -> np.ndarray:
    adapter.eval()
    if auxiliary_override is not None:
        if auxiliary_override.shape != arrays.auxiliary_tokens.shape:
            raise ValueError("Auxiliary override shape mismatch")
        temporary = L1WindowArrays(
            **{
                **arrays.__dict__,
                "auxiliary_tokens": np.asarray(auxiliary_override, dtype=np.float32),
            }
        )
    else:
        temporary = arrays
    loader = make_loader(
        CachedAdapterDataset(base_prediction, temporary), batch_size, False, 0
    )
    predictions = []
    for base, auxiliary, self_tokens, context, _, _ in loader:
        predictions.append(
            _adapter_forward(
                adapter,
                variant,
                base.to(device, non_blocking=True),
                auxiliary.to(device, non_blocking=True),
                self_tokens.to(device, non_blocking=True),
                context.to(device, non_blocking=True),
            ).cpu()
        )
    return torch.cat(predictions).numpy()


def train_adapter(
    config: L1RunConfig,
    data: L1PreparedData,
    variant: str,
    seed: int,
    output_dir: Path,
    device: torch.device,
    base_model: nn.Module,
    base_variant: str,
    base_checkpoint_row: dict[str, Any],
    train_base_prediction: np.ndarray,
    validation_base_prediction: np.ndarray,
) -> tuple[nn.Module, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    set_seed(seed)
    adapter = build_adapter(config, variant).to(device)
    initial_adapter_hash = module_state_hash(adapter)
    expected_parameters = (
        select_flat_width(config.horizon)["selected_parameter_count"]
        if variant == "B_flat"
        else expected_m_parameter_count(config.horizon)
    )
    actual_parameters = trainable_parameter_count(adapter)
    if actual_parameters != expected_parameters:
        raise RuntimeError(
            f"{variant} parameter mismatch: actual={actual_parameters}, expected={expected_parameters}"
        )

    base_model.requires_grad_(False)
    base_model.eval()
    # A just-trained F00/F10 object can retain stale gradient tensors.  Clear
    # those tensors at the freeze boundary so the subsequent no-grad audit
    # distinguishes historical optimizer state from adapter backpropagation.
    for parameter in base_model.parameters():
        parameter.grad = None
    base_hash_before = module_state_hash(base_model)
    if base_hash_before != base_checkpoint_row["model_state_sha256"]:
        raise RuntimeError("Loaded base model state does not match checkpoint mapping")
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )
    loss_fn = nn.MSELoss()
    train_loader = make_loader(
        CachedAdapterDataset(train_base_prediction, data.train),
        config.batch_size,
        True,
        seed,
    )
    checkpoint_path = output_dir / "checkpoints" / f"{variant}_seed{seed}.pt"
    best_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    history_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    first_batch_seen = False
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        adapter.train()
        train_loss_sum = 0.0
        train_count = 0
        for base, auxiliary, self_tokens, context, target, _ in train_loader:
            base = base.to(device, non_blocking=True)
            auxiliary = auxiliary.to(device, non_blocking=True)
            self_tokens = self_tokens.to(device, non_blocking=True)
            context = context.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = _adapter_forward(
                adapter, variant, base, auxiliary, self_tokens, context
            )
            if not first_batch_seen and not torch.equal(prediction, base):
                raise AssertionError(f"{variant} does not start at exact base prediction")
            loss = loss_fn(prediction, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{variant} seed={seed} non-finite training loss")
            loss.backward()
            if not first_batch_seen:
                first_contract = gradient_contract(adapter)
                gradient_rows.append(
                    {
                        "task": config.task_name,
                        "seed": seed,
                        "variant": variant,
                        "backward_stage": "first_backward_before_update",
                        **first_contract,
                    }
                )
                if not (
                    first_contract["output_layer_nonzero"]
                    and not first_contract["upstream_nonzero"]
                    and not first_contract["beta_nonzero"]
                ):
                    raise AssertionError(f"{variant} first-backward gradient contract failed")
            nn.utils.clip_grad_norm_(adapter.parameters(), config.gradient_clip_norm)
            optimizer.step()
            if not first_batch_seen:
                optimizer.zero_grad(set_to_none=True)
                probe = _adapter_forward(
                    adapter, variant, base, auxiliary, self_tokens, context
                )
                loss_fn(probe, target).backward()
                later_contract = gradient_contract(adapter)
                gradient_rows.append(
                    {
                        "task": config.task_name,
                        "seed": seed,
                        "variant": variant,
                        "backward_stage": "after_first_update_probe_no_optimizer_step",
                        **later_contract,
                    }
                )
                if not (
                    later_contract["output_layer_nonzero"]
                    and later_contract["upstream_nonzero"]
                    and later_contract["beta_nonzero"]
                ):
                    raise AssertionError(f"{variant} later-backward gradient contract failed")
                optimizer.zero_grad(set_to_none=True)
                first_batch_seen = True
            train_loss_sum += float(loss.item()) * len(base)
            train_count += len(base)

        validation_prediction = predict_adapter(
            adapter,
            variant,
            validation_base_prediction,
            data.validation,
            config.batch_size,
            device,
        )
        validation_loss = float(
            np.mean((validation_prediction - data.validation.targets_scaled) ** 2)
        )
        scheduler.step(validation_loss)
        history_rows.append(
            {
                "task": config.task_name,
                "variant": variant,
                "seed": seed,
                "epoch": epoch,
                "train_loss": train_loss_sum / train_count,
                "validation_loss": validation_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if validation_loss < best_loss - config.minimum_improvement:
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            _write_state_checkpoint(adapter, checkpoint_path)
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break
    training_seconds = time.perf_counter() - started
    adapter.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    base_model.eval()
    base_hash_after = module_state_hash(base_model)
    if base_hash_after != base_hash_before:
        raise AssertionError(f"Frozen base state changed while training {variant}")
    if any(parameter.grad is not None for parameter in base_model.parameters()):
        raise AssertionError(f"Frozen base received gradients while training {variant}")
    checkpoint_hash = file_sha256(checkpoint_path)
    adapter_hash = module_state_hash(adapter)
    checkpoint_row = {
        "task": config.task_name,
        "seed": seed,
        "variant": variant,
        "checkpoint_role": "best_validation_adapter",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "model_state_sha256": adapter_hash,
        "initialized_from_checkpoint": base_checkpoint_row["checkpoint_path"],
        "initialized_from_sha256": base_checkpoint_row["checkpoint_sha256"],
        "base_state_sha256": base_hash_before,
        "patchtst_frozen": True,
        "shared_by_adapter_variants": "",
    }
    model_fit_row = {
        "object_id": f"model_fit:{variant}:seed{seed}",
        "object_type": "training_model_fit",
        "task": config.task_name,
        "fold": "",
        "model": variant,
        "horizon_step": "",
        "fit_start": pd.Timestamp(data.train.origin_times.min()).isoformat(),
        "fit_end": pd.Timestamp(data.train.origin_times.max()).isoformat(),
        "row_count": len(data.train.origin_rows),
        "config_hash": config.config_hash,
        "state_hash": adapter_hash,
        "details": json.dumps(
            {
                "loss": config.loss,
                "optimizer": config.optimizer,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "base_checkpoint_sha256": base_checkpoint_row["checkpoint_sha256"],
                "checkpoint_sha256": checkpoint_hash,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    }
    early_stopping_row = {
        "object_id": f"early_stopping:{variant}:seed{seed}",
        "object_type": "validation_early_stopping_selector",
        "task": config.task_name,
        "fold": "",
        "model": variant,
        "horizon_step": "",
        "fit_start": pd.Timestamp(data.validation.origin_times.min()).isoformat(),
        "fit_end": pd.Timestamp(data.validation.origin_times.max()).isoformat(),
        "row_count": len(data.validation.origin_rows),
        "config_hash": config.config_hash,
        "state_hash": stable_hash(
            {"best_epoch": best_epoch, "best_validation_loss": best_loss}
        ),
        "details": json.dumps(
            {
                "base_checkpoint_sha256": base_checkpoint_row["checkpoint_sha256"],
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
                "validation_samples": len(data.validation.origin_rows),
                "patience": config.patience,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    }
    run_row = {
        "task": config.task_name,
        "seed": seed,
        "variant": variant,
        "base_variant": base_variant,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "training_seconds": training_seconds,
        "trainable_parameter_count": actual_parameters,
        "initial_adapter_state_sha256": initial_adapter_hash,
        "checkpoint_sha256": checkpoint_hash,
        "base_checkpoint_sha256": base_checkpoint_row["checkpoint_sha256"],
        "base_state_before_sha256": base_hash_before,
        "base_state_after_sha256": base_hash_after,
        "base_state_unchanged": base_hash_before == base_hash_after,
        "base_eval_during_adapter": not base_model.training,
        "base_parameters_require_grad": any(
            parameter.requires_grad for parameter in base_model.parameters()
        ),
        "base_gradients_none": all(
            parameter.grad is None for parameter in base_model.parameters()
        ),
    }
    return adapter, run_row, history_rows, gradient_rows, {
        "checkpoint": checkpoint_row,
        "fits": [model_fit_row, early_stopping_row],
    }


def validation_prediction_rows(
    config: L1RunConfig,
    arrays: L1WindowArrays,
    seed: int,
    variant: str,
    base_variant: str,
    prediction_scaled: np.ndarray,
    center_mean: float,
    center_scale: float,
) -> list[dict[str, Any]]:
    if prediction_scaled.shape != arrays.targets_scaled.shape:
        raise ValueError("Prediction shape differs from validation target shape")
    prediction_physical = prediction_scaled[:, 0, :] * center_scale + center_mean
    rows = []
    for sample_index in range(len(arrays.origin_rows)):
        for horizon_index in range(config.horizon):
            rows.append(
                {
                    "task": config.task_name,
                    "seed": seed,
                    "variant": variant,
                    "base_variant": base_variant,
                    "split": "validation",
                    "sample_index": sample_index,
                    "origin_row": int(arrays.origin_rows[sample_index]),
                    "origin_time": pd.Timestamp(
                        arrays.origin_times[sample_index]
                    ).isoformat(),
                    "horizon_step": horizon_index + 1,
                    "target_row": int(arrays.target_rows[sample_index, horizon_index]),
                    "target_time": pd.Timestamp(
                        arrays.target_times[sample_index, horizon_index]
                    ).isoformat(),
                    "target_scaled": float(
                        arrays.targets_scaled[sample_index, 0, horizon_index]
                    ),
                    "prediction_scaled": float(
                        prediction_scaled[sample_index, 0, horizon_index]
                    ),
                    "target_ugm3": float(
                        arrays.targets_physical[sample_index, horizon_index]
                    ),
                    "prediction_ugm3": float(
                        prediction_physical[sample_index, horizon_index]
                    ),
                    "high_pollution": bool(
                        arrays.high_pollution[sample_index, horizon_index]
                    ),
                    "rapid_rise": bool(arrays.rapid_rise[sample_index, horizon_index]),
                    "decline": bool(arrays.decline[sample_index, horizon_index]),
                    "ordinary": bool(arrays.ordinary[sample_index, horizon_index]),
                }
            )
    return rows


@torch.no_grad()
def collect_gate_diagnostics(
    config: L1RunConfig,
    adapter: nn.Module,
    variant: str,
    base_prediction: np.ndarray,
    arrays: L1WindowArrays,
    device: torch.device,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(adapter, CrossVariableLagAdapter):
        return [], []
    adapter.eval()
    loader = make_loader(
        CachedAdapterDataset(base_prediction, arrays), config.batch_size, False, 0
    )
    weights = []
    pollution_mass = []
    meteorology_mass = []
    beta = None
    for base, auxiliary, self_tokens, context, _, _ in loader:
        tokens = controlled_token_input(
            auxiliary.to(device), self_tokens.to(device), variant
        )
        components = adapter.forward_components(
            base.to(device), tokens, context.to(device)
        )
        weights.append(components["gate_weights"].cpu())
        pollution_mass.append(components["pollutant_mass"].cpu())
        meteorology_mass.append(components["meteorology_mass"].cpu())
        beta = float(components["beta"].item())
    weight_array = torch.cat(weights).numpy()
    pollution_array = torch.cat(pollution_mass).numpy()
    meteorology_array = torch.cat(meteorology_mass).numpy()
    weight_rows = []
    for variable_index, variable in enumerate(AUXILIARY_COLUMNS):
        for lag_index, lag in enumerate((0, 1, 2, 3, 6, 12)):
            for horizon_index in range(config.horizon):
                values = weight_array[:, variable_index, lag_index, horizon_index]
                weight_rows.append(
                    {
                        "task": config.task_name,
                        "seed": seed,
                        "variant": variant,
                        "variable": variable,
                        "lag": lag,
                        "horizon_step": horizon_index + 1,
                        "mean_weight": float(values.mean()),
                        "std_weight": float(values.std()),
                    }
                )
    group_rows = []
    for horizon_index in range(config.horizon):
        poll = pollution_array[:, horizon_index]
        met = meteorology_array[:, horizon_index]
        group_rows.append(
            {
                "task": config.task_name,
                "seed": seed,
                "variant": variant,
                "horizon_step": horizon_index + 1,
                "pollutant_mass_mean": float(poll.mean()),
                "pollutant_mass_std": float(poll.std()),
                "meteorology_mass_mean": float(met.mean()),
                "meteorology_mass_std": float(met.std()),
                "mass_sum_max_abs_error": float(np.max(np.abs(poll + met - 1.0))),
                "beta": beta,
            }
        )
    return weight_rows, group_rows


def block_shuffle_source_indices(
    sample_count: int,
    permutation_seed: int,
    block_size: int = 24,
) -> np.ndarray:
    """Permute complete blocks while preserving position within each block."""
    if block_size != 24:
        raise ValueError("M-shuffle block size is frozen at 24 origins")
    usable = sample_count // block_size * block_size
    if usable < block_size:
        raise ValueError("M-shuffle requires at least one complete 24-origin block")
    blocks = np.arange(usable, dtype=np.int64).reshape(-1, block_size)
    generator = np.random.default_rng(permutation_seed)
    permutation = generator.permutation(len(blocks))
    return blocks[permutation].reshape(-1)


def tokens_from_auxiliary_history(
    auxiliary_history: np.ndarray,
    lags: tuple[int, ...] = (0, 1, 2, 3, 6, 12),
) -> np.ndarray:
    """Rebuild [B,7,6,2] tokens after a complete-history intervention."""
    history = np.asarray(auxiliary_history, dtype=np.float32)
    if history.ndim != 3 or history.shape[1] != 7:
        raise ValueError("auxiliary_history must have shape [B,7,L]")
    if tuple(lags) != (0, 1, 2, 3, 6, 12):
        raise ValueError("M-shuffle lags are frozen at 0,1,2,3,6,12")
    current_positions = np.asarray(
        [history.shape[-1] - 1 - lag for lag in lags], dtype=np.int64
    )
    previous_positions = current_positions - 1
    if previous_positions.min() < 0:
        raise ValueError("Auxiliary history is too short for the frozen lag tokens")
    values = history[:, :, current_positions]
    differences = values - history[:, :, previous_positions]
    return np.stack([values, differences], axis=-1).astype(np.float32)


def m_shuffle_prediction_rows(
    config: L1RunConfig,
    adapter: nn.Module,
    seed: int,
    validation_base_prediction: np.ndarray,
    arrays: L1WindowArrays,
    center_mean: float,
    center_scale: float,
    device: torch.device,
) -> list[dict[str, Any]]:
    usable = len(arrays.origin_rows) // 24 * 24
    diagnostic_arrays = arrays.subset(usable)
    base = validation_base_prediction[:usable]
    rows: list[dict[str, Any]] = []
    for permutation_seed in config.shuffle_seeds:
        source_indices = block_shuffle_source_indices(usable, permutation_seed)
        # Replace the complete U[B,7,L] history block first, then reconstruct the
        # frozen lag/value/difference tokens.  All seven variables share the same
        # block permutation and the receiver's within-block position.
        shuffled_history = diagnostic_arrays.auxiliary_history[source_indices]
        shuffled_tokens = tokens_from_auxiliary_history(shuffled_history)
        prediction = predict_adapter(
            adapter,
            "F11",
            base,
            diagnostic_arrays,
            config.batch_size,
            device,
            auxiliary_override=shuffled_tokens,
        )
        prediction_physical = prediction[:, 0, :] * center_scale + center_mean
        base_physical = base[:, 0, :] * center_scale + center_mean
        for sample_index in range(usable):
            for horizon_index in range(config.horizon):
                rows.append(
                    {
                        "task": config.task_name,
                        "train_seed": seed,
                        "permutation_seed": permutation_seed,
                        "diagnostic_sample_index": sample_index,
                        "source_sample_index": int(source_indices[sample_index]),
                        "origin_time": pd.Timestamp(
                            diagnostic_arrays.origin_times[sample_index]
                        ).isoformat(),
                        "horizon_step": horizon_index + 1,
                        "target_ugm3": float(
                            diagnostic_arrays.targets_physical[
                                sample_index, horizon_index
                            ]
                        ),
                        "base_prediction_ugm3": float(
                            base_physical[sample_index, horizon_index]
                        ),
                        "shuffled_prediction_ugm3": float(
                            prediction_physical[sample_index, horizon_index]
                        ),
                        "diagnostic_origin_count": usable,
                        "excluded_tail_origin_count": len(arrays.origin_rows) - usable,
                        "block_size_origins": 24,
                    }
                )
    return rows


def _write_frame(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _parameter_audit(config: L1RunConfig) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    flat_info = select_flat_width(config.horizon)
    rows = []
    for variant in ADAPTER_VARIANTS:
        set_seed(0)
        adapter = build_adapter(config, variant)
        rows.append(
            {
                "task": config.task_name,
                "variant": variant,
                "adapter_class": type(adapter).__name__,
                "trainable_parameter_count": trainable_parameter_count(adapter),
                "m_target_parameter_count": expected_m_parameter_count(config.horizon),
                "flat_width": getattr(adapter, "width", ""),
                "relative_to_m_percent": 100.0
                * abs(
                    trainable_parameter_count(adapter)
                    - expected_m_parameter_count(config.horizon)
                )
                / expected_m_parameter_count(config.horizon),
                "parameter_gate_pass": (
                    variant != "B_flat" or flat_info["accepted"]
                ),
            }
        )
    return rows, pd.DataFrame(flat_info["candidates"])


def run(config: L1RunConfig, args: argparse.Namespace, output_dir: Path) -> None:
    repo_root = Path(__file__).resolve().parent
    provenance = collect_git_provenance(repo_root)
    enforce_formal_clean_tree(provenance, args.smoke)
    if args.smoke:
        seeds = (FORMAL_SEEDS[0],)
        data = prepare_l1_data(
            args.data_dir,
            L1TaskConfig(config.history, config.horizon),
            train_sample_limit=args.smoke_train_samples,
            validation_sample_limit=args.smoke_validation_samples,
        )
    else:
        seeds = FORMAL_SEEDS
        data = prepare_l1_data(
            args.data_dir, L1TaskConfig(config.history, config.horizon)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_logs").mkdir(exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    run_manifest = {
        "scope": (
            "L1_SMOKE_FLOW_ONLY_NOT_SELECTION_EVIDENCE"
            if args.smoke
            else "L1_FORMAL_TRAIN_VALIDATION_SELECTION"
        ),
        "smoke": args.smoke,
        "formal_test_used": False,
        "formal_test_features_or_targets_loaded": False,
        "command": sys.argv,
        "task": config.task_name,
        "seeds": list(seeds),
        "formal_selection_seeds": list(FORMAL_SEEDS),
        "shuffle_seeds": list(SHUFFLE_SEEDS),
        "variants": list(ALL_VARIANTS),
        "output_dir": str(output_dir),
        "formal_output_dir": str(FORMAL_OUTPUTS[(config.history, config.horizon)]),
        "config_hash": config.config_hash,
        **provenance,
        "formal_clean_tree_required": True,
        "precommit_engineering_smoke": bool(args.smoke and provenance["git_dirty"]),
        "provenance_statement": (
            "PRE-COMMIT DIRTY-TREE ENGINEERING SMOKE; NOT FORMAL EVIDENCE"
            if args.smoke and provenance["git_dirty"]
            else "CLEAN-TREE RUN"
        ),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "experiment_config.json").write_text(
        json.dumps(
            {
                **asdict(config),
                "config_hash": config.config_hash,
                "smoke": args.smoke,
                "actual_seeds": list(seeds),
                "device": args.device,
                "smoke_train_samples": args.smoke_train_samples if args.smoke else None,
                "smoke_validation_samples": (
                    args.smoke_validation_samples if args.smoke else None
                ),
                "flat_width_selection": select_flat_width(config.horizon),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "dataset_metadata.json").write_text(
        json.dumps(data.metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(data.processing_ledger).to_csv(
        output_dir / "fit_ledger.csv", index=False
    )
    pd.DataFrame(data.station_selection_rows).to_csv(
        output_dir / "station_selection.csv", index=False
    )
    pd.DataFrame(data.data_quality_rows).to_csv(
        output_dir / "data_quality.csv", index=False
    )
    pd.DataFrame(data.target_time_rows).to_csv(
        output_dir / "target_time_sets.csv", index=False
    )
    pd.DataFrame(data.boundary_checks).to_csv(
        output_dir / "boundary_checks.csv", index=False
    )
    parameter_rows, flat_candidates = _parameter_audit(config)
    pd.DataFrame(parameter_rows).to_csv(output_dir / "parameter_audit.csv", index=False)
    flat_candidates.to_csv(output_dir / "flat_width_candidates.csv", index=False)

    device = torch.device(args.device)
    run_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    fit_rows = list(data.processing_ledger)
    prediction_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    shuffle_rows: list[dict[str, Any]] = []
    patchtst_audit_rows: list[dict[str, Any]] = []

    for seed in seeds:
        f00_model, f00_run, f00_history, f00_artifacts = train_base_model(
            config, data, "F00", seed, output_dir, device
        )
        run_rows.append(f00_run)
        checkpoint_rows.append(f00_artifacts["checkpoint"])
        fit_rows.extend(f00_artifacts["fits"])
        patchtst_audit_rows.extend(f00_artifacts["patchtst_audits"])
        pd.DataFrame(f00_history).to_csv(
            output_dir / "training_logs" / f"F00_seed{seed}.csv", index=False
        )
        f00_train_prediction = predict_base(
            f00_model, data.train, config.batch_size, device
        )
        f00_validation_prediction = predict_base(
            f00_model, data.validation, config.batch_size, device
        )
        prediction_rows.extend(
            validation_prediction_rows(
                config,
                data.validation,
                seed,
                "F00",
                "F00",
                f00_validation_prediction,
                data.metadata["center_mean"],
                data.metadata["center_scale"],
            )
        )

        f10_model, f10_run, f10_history, f10_artifacts = train_base_model(
            config,
            data,
            "F10",
            seed,
            output_dir,
            device,
            f00_checkpoint=Path(f00_artifacts["checkpoint"]["checkpoint_path"]),
        )
        run_rows.append(f10_run)
        checkpoint_rows.append(f10_artifacts["checkpoint"])
        fit_rows.extend(f10_artifacts["fits"])
        patchtst_audit_rows.extend(f10_artifacts["patchtst_audits"])
        pd.DataFrame(f10_history).to_csv(
            output_dir / "training_logs" / f"F10_seed{seed}.csv", index=False
        )
        f10_train_prediction = predict_base(
            f10_model, data.train, config.batch_size, device
        )
        f10_validation_prediction = predict_base(
            f10_model, data.validation, config.batch_size, device
        )
        prediction_rows.extend(
            validation_prediction_rows(
                config,
                data.validation,
                seed,
                "F10",
                "F10",
                f10_validation_prediction,
                data.metadata["center_mean"],
                data.metadata["center_scale"],
            )
        )

        base_context = {
            "F00": (
                f00_model,
                f00_artifacts["checkpoint"],
                f00_train_prediction,
                f00_validation_prediction,
            ),
            "F10": (
                f10_model,
                f10_artifacts["checkpoint"],
                f10_train_prediction,
                f10_validation_prediction,
            ),
        }
        trained_adapters: dict[str, nn.Module] = {}
        for variant in ADAPTER_VARIANTS:
            base_variant = "F00" if variant == "F01" else "F10"
            base_model, base_checkpoint, train_base_pred, validation_base_pred = base_context[
                base_variant
            ]
            adapter, run_row, history, gradients, artifacts = train_adapter(
                config,
                data,
                variant,
                seed,
                output_dir,
                device,
                base_model,
                base_variant,
                base_checkpoint,
                train_base_pred,
                validation_base_pred,
            )
            trained_adapters[variant] = adapter
            run_rows.append(run_row)
            checkpoint_rows.append(artifacts["checkpoint"])
            fit_rows.extend(artifacts["fits"])
            gradient_rows.extend(gradients)
            pd.DataFrame(history).to_csv(
                output_dir / "training_logs" / f"{variant}_seed{seed}.csv",
                index=False,
            )
            prediction = predict_adapter(
                adapter,
                variant,
                validation_base_pred,
                data.validation,
                config.batch_size,
                device,
            )
            prediction_rows.extend(
                validation_prediction_rows(
                    config,
                    data.validation,
                    seed,
                    variant,
                    base_variant,
                    prediction,
                    data.metadata["center_mean"],
                    data.metadata["center_scale"],
                )
            )
            weights, groups = collect_gate_diagnostics(
                config,
                adapter,
                variant,
                validation_base_pred,
                data.validation,
                device,
                seed,
            )
            gate_rows.extend(weights)
            group_rows.extend(groups)

        shuffle_rows.extend(
            m_shuffle_prediction_rows(
                config,
                trained_adapters["F11"],
                seed,
                f10_validation_prediction,
                data.validation,
                data.metadata["center_mean"],
                data.metadata["center_scale"],
                device,
            )
        )
        # Incremental artifacts make interrupted formal runs auditable.
        _write_frame(output_dir / "run_records.csv", run_rows)
        _write_frame(output_dir / "checkpoint_mapping.csv", checkpoint_rows)
        _write_frame(output_dir / "validation_predictions.csv", prediction_rows)
        _write_frame(output_dir / "gradient_audit.csv", gradient_rows)
        _write_frame(output_dir / "gate_weight_diagnostics.csv", gate_rows)
        _write_frame(output_dir / "group_mass_diagnostics.csv", group_rows)
        _write_frame(output_dir / "m_shuffle_predictions.csv", shuffle_rows)
        _write_frame(output_dir / "patchtst_state_audit.csv", patchtst_audit_rows)
        pd.DataFrame(fit_rows).to_csv(output_dir / "fit_ledger.csv", index=False)

    print(f"L1 artifacts written to: {output_dir.resolve()}")
    print("Run the independent summarizer before interpreting any validation metric.")


def main() -> None:
    args = parse_args()
    history, horizon = (int(value) for value in args.task.split(":"))
    budget = TASK_BUDGETS[(history, horizon)]
    config = L1RunConfig(
        history=history,
        horizon=horizon,
        epochs=1 if args.smoke else budget["epochs"],
        patience=1 if args.smoke else budget["patience"],
        batch_size=budget["batch_size"],
    )
    output_dir = Path(args.output_dir) if args.output_dir else FORMAL_OUTPUTS[(history, horizon)]
    if args.smoke and not output_dir.name.endswith("_smoke"):
        output_dir = output_dir.with_name(output_dir.name + "_smoke")
    if not args.smoke and output_dir.name.endswith("_smoke"):
        raise ValueError("Formal L1 output cannot use a _smoke directory")
    if args.smoke and (args.smoke_train_samples < 1 or args.smoke_validation_samples < 24):
        raise ValueError("Smoke needs positive train samples and at least 24 validation origins")
    run(config, args, output_dir)


if __name__ == "__main__":
    main()
