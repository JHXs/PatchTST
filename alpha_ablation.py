"""Run validation-only alpha strategy ablations for ST-PatchTST.

The test split is loaded only as part of the existing split object and is never
indexed. Each completed condition/seed pair is appended to a CSV so interrupted
runs can resume without repeating finished experiments.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from fastai.callback.tracker import EarlyStoppingCallback, SaveModelCallback
from fastai.callback.training import GradientClip
from tsai.all import TSForecaster, load_object, set_seed, to_np

from ST_PatchTST_model import ST_PatchTST


POLLUTANT_NAMES = ("PM2.5", "PM10", "NO2", "CO", "O3", "SO2")
RECOMMENDED_ALPHA = (0.3124, 0.0139, 0.0357, 0.0054, 0.02, 0.0490)
CONDITIONS = (
    "alpha_zero",
    "uniform_alpha",
    "per_pollutant_alpha",
    "learnable_alpha",
)


def arch_config(params, alpha, learnable_alpha=False):
    return {
        "n_layers": 3,
        "n_heads": 4,
        "d_model": 16,
        "d_ff": 128,
        "attn_dropout": 0.0,
        "dropout": 0.2,
        "patch_len": 4,
        "stride": 2,
        "padding_patch": True,
        "num_stations": int(params["num_stations"]),
        "feat_size": int(params["feat_size"]),
        "center_station_idx": int(params["center_station_idx"]),
        "neighbor_d_model": 24,
        "neighbor_n_heads": 4,
        "neighbor_dropout": 0.3,
        "neighbor_attn_dropout": 0.1,
        "neighbor_temporal_kernel": 7,
        "alpha": list(alpha),
        "learnable_alpha": learnable_alpha,
        "alpha_max": 0.4,
    }


def condition_config(condition):
    if condition == "alpha_zero":
        return [0.0] * 6, False
    if condition == "uniform_alpha":
        return [0.02] * 6, False
    if condition == "per_pollutant_alpha":
        return list(RECOMMENDED_ALPHA), False
    if condition == "learnable_alpha":
        return list(RECOMMENDED_ALPHA), True
    raise ValueError(f"未知条件: {condition}")


def set_model_alpha(model, values):
    model.set_alpha(values)


def predict(learn, X_valid, alpha=None):
    if alpha is not None:
        set_model_alpha(learn.model, alpha)
    return to_np(learn.get_X_preds(X_valid)[0])


def calibrate_uniform(learn, X_valid, y_valid, alpha_max=0.4):
    pred_zero = predict(learn, X_valid, [0.0] * 6)
    pred_one = predict(learn, X_valid, [1.0] * 6)
    delta = (pred_one[:, :6] - pred_zero[:, :6]).reshape(-1).astype(np.float64)
    target = y_valid[:, :6].reshape(-1).astype(np.float64)
    base = pred_zero[:, :6].reshape(-1).astype(np.float64)
    denominator = float(np.dot(delta, delta))
    raw = 0.0 if denominator == 0 else float(np.dot(delta, target - base) / denominator)
    selected = float(np.clip(raw, 0.0, alpha_max))
    selected_alpha = [selected] * 6
    return selected_alpha, predict(learn, X_valid, selected_alpha), raw


def calibrate_per_pollutant(learn, X_valid, y_valid, alpha_max=0.4):
    pred_zero = predict(learn, X_valid, [0.0] * 6)
    pred_one = predict(learn, X_valid, [1.0] * 6)
    delta = pred_one - pred_zero
    selected_alpha = []
    raw_alpha = []
    for idx in range(6):
        target = y_valid[:, idx].reshape(-1).astype(np.float64)
        base = pred_zero[:, idx].reshape(-1).astype(np.float64)
        branch = delta[:, idx].reshape(-1).astype(np.float64)
        denominator = float(np.dot(branch, branch))
        raw = 0.0 if denominator == 0 else float(
            np.dot(branch, target - base) / denominator
        )
        raw_alpha.append(raw)
        selected_alpha.append(float(np.clip(raw, 0.0, alpha_max)))
    return selected_alpha, predict(learn, X_valid, selected_alpha), raw_alpha


def validation_metrics(y_valid, predictions):
    errors = y_valid.astype(np.float64) - predictions.astype(np.float64)
    return {
        "valid_mse": float(np.mean(errors ** 2)),
        "valid_rmse": float(np.sqrt(np.mean(errors ** 2))),
        "valid_mae": float(np.mean(np.abs(errors))),
        "pollutant_mse": {
            name: float(np.mean(errors[:, idx] ** 2))
            for idx, name in enumerate(POLLUTANT_NAMES)
        },
    }


def completed_pairs(results_path):
    if not results_path.exists():
        return set()
    with results_path.open(encoding="utf-8", newline="") as handle:
        return {
            (row["condition"], int(row["seed"]))
            for row in csv.DictReader(handle)
        }


def append_result(results_path, result):
    results_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_path.exists()
    with results_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=result.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(result)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260809, 20260810, 20260811])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=Path("tsai/models/alpha_ablation"))
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "ablation_results.csv"
    completed = completed_pairs(results_path)

    X = np.load("tsai/data/X.npz")["arr_0"]
    y = np.load("tsai/data/y.npz")["arr_0"]
    splits = load_object("tsai/data/splits.pkl")
    preproc_pipe = load_object("tsai/data/preproc_pipe.pkl")
    exp_pipe = load_object("tsai/data/exp_pipe.pkl")
    params = np.load("tsai/data/model_params.npz")
    X_valid, y_valid = X[splits[1]], y[splits[1]]

    for condition in args.conditions:
        initial_alpha, learnable = condition_config(condition)
        for seed in args.seeds:
            if (condition, seed) in completed:
                print(f"SKIP completed: {condition}, seed={seed}", flush=True)
                continue

            print(f"\n=== {condition}, seed={seed}, epochs={args.epochs} ===", flush=True)
            set_seed(seed, reproducible=True)
            checkpoint_name = f"{condition}_seed_{seed}"
            callbacks = [
                GradientClip(1.0),
                SaveModelCallback(monitor="valid_loss", fname=checkpoint_name, with_opt=False),
                EarlyStoppingCallback(monitor="valid_loss", patience=15),
            ]
            learn = TSForecaster(
                X,
                y,
                splits=splits,
                batch_size=args.batch_size,
                path=args.output_dir,
                pipelines=[preproc_pipe, exp_pipe],
                arch=ST_PatchTST,
                arch_config=arch_config(params, initial_alpha, learnable),
                metrics=[],
                cbs=callbacks,
            )
            total_params = sum(parameter.numel() for parameter in learn.model.parameters())
            trainable_params = sum(
                parameter.numel()
                for parameter in learn.model.parameters()
                if parameter.requires_grad
            )

            start = time.perf_counter()
            learn.fit_one_cycle(args.epochs, lr_max=args.learning_rate)
            elapsed = time.perf_counter() - start
            completed_epochs = len(learn.recorder.values)

            learned_alpha = learn.model.current_alpha().detach().cpu().tolist()
            raw_predictions = predict(learn, X_valid)
            raw_metrics = validation_metrics(y_valid, raw_predictions)

            if condition == "uniform_alpha":
                selected_alpha, selected_predictions, raw_solution = calibrate_uniform(
                    learn, X_valid, y_valid
                )
            elif condition == "per_pollutant_alpha":
                selected_alpha, selected_predictions, raw_solution = calibrate_per_pollutant(
                    learn, X_valid, y_valid
                )
            else:
                selected_alpha = learned_alpha
                selected_predictions = raw_predictions
                raw_solution = learned_alpha

            selected_metrics = validation_metrics(y_valid, selected_predictions)
            result = {
                "condition": condition,
                "seed": seed,
                "epochs_requested": args.epochs,
                "epochs_completed": completed_epochs,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "initial_alpha": json.dumps(initial_alpha),
                "learned_alpha": json.dumps(learned_alpha),
                "selection_raw_solution": json.dumps(raw_solution),
                "selected_alpha": json.dumps(selected_alpha),
                "raw_valid_mse": raw_metrics["valid_mse"],
                "valid_mse": selected_metrics["valid_mse"],
                "valid_rmse": selected_metrics["valid_rmse"],
                "valid_mae": selected_metrics["valid_mae"],
                "pollutant_mse": json.dumps(selected_metrics["pollutant_mse"]),
                "elapsed_seconds": elapsed,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "device": str(next(learn.model.parameters()).device),
            }
            append_result(results_path, result)
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

            learn.export(f"{checkpoint_name}.pt")
            del learn
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
