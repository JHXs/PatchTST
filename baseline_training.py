"""Training and evaluation paths for baseline comparison arms."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn

import run_st_patchtst_ablation as legacy
from baseline_traditional import fit_traditional_baseline


def _ensure_test_split(config) -> None:
    if config.evaluation_split != "test":
        raise ValueError(
            "Baseline protocol fixes evaluation_split='test'; validation is for selection only"
        )


def train_baseline(
    config,
    model: nn.Module | Callable[[], nn.Module],
    datasets: dict,
    metadata: dict,
    seed: int,
    device: torch.device,
    variant: str,
    output_dir: str | Path,
    registration: dict | None = None,
) -> dict:
    """Train one neural arm with legacy ``train_one_run`` semantics.

    A callable is preferred because legacy seeds before model construction.  A
    preconstructed module remains supported for the work-order API, but callers
    seeking paired determinism must construct it inside the callable.
    """
    _ensure_test_split(config)
    legacy.set_seed(seed)
    model_instance = model() if callable(model) else model
    model_instance = model_instance.to(device)
    trainable_parameters = [
        parameter for parameter in model_instance.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    loss_fn = nn.MSELoss()
    train_loader = legacy.make_loader(datasets["train"], config, True, seed)
    valid_loader = legacy.make_loader(datasets["valid"], config, False, seed)
    evaluation_loader = legacy.make_loader(
        datasets[config.evaluation_split], config, False, seed
    )

    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{variant}_seed{seed}.pt"
    best_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history_rows = []
    started = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        model_instance.train()
        train_loss_sum = 0.0
        train_count = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model_instance(x)
            loss = loss_fn(prediction, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{variant} seed={seed} has non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model_instance.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(x)
            train_count += len(x)

        valid_prediction, valid_target, _ = legacy.predict(
            model_instance,
            valid_loader,
            device,
            metadata["center_station_idx"],
        )
        train_loss = train_loss_sum / train_count
        valid_loss = float(np.mean((valid_prediction - valid_target) ** 2))
        scheduler.step(valid_loss)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if valid_loss < best_loss - 1e-7:
            best_loss = valid_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model_instance.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.patience:
            break

    training_seconds = time.perf_counter() - started
    model_instance.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    prediction, target, inference_seconds = legacy.predict(
        model_instance,
        evaluation_loader,
        device,
        metadata["center_station_idx"],
    )
    metrics = legacy.regression_metrics(
        target,
        prediction,
        metadata["center_mean"],
        metadata["center_std"],
    )

    logs_dir = output_dir / "training_logs"
    predictions_dir = output_dir / "predictions"
    logs_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history_rows).to_csv(
        logs_dir / f"{variant}_seed{seed}.csv", index=False
    )
    prediction_path = predictions_dir / f"{variant}_seed{seed}.npz"
    np.savez_compressed(
        prediction_path,
        prediction_scaled=prediction,
        target_scaled=target,
        prediction_ugm3=prediction * metadata["center_std"] + metadata["center_mean"],
        target_ugm3=target * metadata["center_std"] + metadata["center_mean"],
    )
    row = {
        "variant": variant,
        "seed": seed,
        "status": "completed",
        "best_epoch": best_epoch,
        "best_valid_loss": best_loss,
        "training_seconds": training_seconds,
        "test_inference_seconds": inference_seconds,
        "evaluation_split": config.evaluation_split,
        "parameter_count": sum(
            parameter.numel() for parameter in model_instance.parameters()
        ),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model_instance.parameters()
            if parameter.requires_grad
        ),
        "prediction_file": str(prediction_path.relative_to(output_dir)),
        **(registration or {}),
        **metrics,
    }
    print(
        f"[{variant} seed={seed}] RMSE={metrics['rmse_ugm3']:.4f} μg/m³, "
        f"MAE={metrics['mae_ugm3']:.4f}"
    )
    return row


def evaluate_traditional_baseline(
    config,
    arm: str,
    datasets: dict,
    metadata: dict,
    output_dir: str | Path,
) -> dict:
    """Fit on train, select on valid where applicable, and evaluate test once."""
    _ensure_test_split(config)
    output_dir = Path(output_dir)
    started = time.perf_counter()
    model = fit_traditional_baseline(arm, datasets, metadata)
    training_seconds = time.perf_counter() - started
    inference_started = time.perf_counter()
    prediction = model.predict(datasets[config.evaluation_split], metadata)
    inference_seconds = time.perf_counter() - inference_started
    _, target = _traditional_arrays(datasets[config.evaluation_split])
    metrics = legacy.regression_metrics(
        target,
        prediction,
        metadata["center_mean"],
        metadata["center_std"],
    )
    predictions_dir = output_dir / "predictions"
    logs_dir = output_dir / "training_logs"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = predictions_dir / f"{arm}.npz"
    np.savez_compressed(
        prediction_path,
        prediction_scaled=prediction,
        target_scaled=target,
        prediction_ugm3=prediction * metadata["center_std"] + metadata["center_mean"],
        target_ugm3=target * metadata["center_std"] + metadata["center_mean"],
    )
    pd.DataFrame(
        [
            {
                "fit": "train_only" if model.fit_sample_indices is not None else "none",
                "ridge_alpha_grid": "0.1,1,10,100" if model.selected_alpha is not None else "",
                "selected_alpha": model.selected_alpha,
            }
        ]
    ).to_csv(logs_dir / f"{arm}.csv", index=False)
    row = {
        "variant": arm,
        "seed": "deterministic",
        "status": "completed",
        "best_epoch": 0,
        "best_valid_loss": np.nan,
        "training_seconds": training_seconds,
        "test_inference_seconds": inference_seconds,
        "evaluation_split": config.evaluation_split,
        "parameter_count": 0,
        "trainable_parameter_count": 0,
        "layer": "A",
        "requested_capacity": "not_applicable",
        "capacity_status": "not_applicable",
        "input_channels": (
            len(metadata["station_ids"]) if arm == "trad_spatial_linear" else 1
        ),
        "selected_alpha": model.selected_alpha,
        "prediction_file": str(prediction_path.relative_to(output_dir)),
        **metrics,
    }
    print(f"[{arm}] RMSE={metrics['rmse_ugm3']:.4f} μg/m³")
    return row


def _traditional_arrays(dataset):
    # Local import keeps sklearn-dependent traditional code out of neural-only use.
    from baseline_traditional import dataset_arrays

    return dataset_arrays(dataset)
