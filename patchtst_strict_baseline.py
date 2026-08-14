"""Train a strict center-station PatchTST baseline without rewriting data assets.

The script extracts the configured center station from the existing multi-site
array, keeps the same targets and chronological splits as ST-PatchTST, and only
evaluates the validation split.
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
from tsai.models.PatchTST import PatchTST as PatchTSTArch


POLLUTANT_NAMES = ("PM2.5", "PM10", "NO2", "CO", "O3", "SO2")


def arch_config():
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
    }


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


def completed_seeds(results_path):
    if not results_path.exists():
        return set()
    with results_path.open(encoding="utf-8", newline="") as handle:
        return {int(row["seed"]) for row in csv.DictReader(handle)}


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
    parser.add_argument(
        "--seeds", nargs="+", type=int,
        default=[20260809, 20260810, 20260811],
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("tsai/models/patchtst_strict_baseline"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "baseline_results.csv"
    finished = completed_seeds(results_path)

    X_multi = np.load("tsai/data/X.npz")["arr_0"]
    y = np.load("tsai/data/y.npz")["arr_0"]
    splits = load_object("tsai/data/splits.pkl")
    preproc_pipe = load_object("tsai/data/preproc_pipe.pkl")
    exp_pipe = load_object("tsai/data/exp_pipe.pkl")
    params = np.load("tsai/data/model_params.npz")

    num_stations = int(params["num_stations"])
    feat_size = int(params["feat_size"])
    center_station_idx = int(params["center_station_idx"])
    expected_channels = num_stations * feat_size
    if X_multi.shape[1] != expected_channels:
        raise ValueError(
            f"X channels={X_multi.shape[1]}, expected {expected_channels}"
        )
    X_center = X_multi.reshape(
        X_multi.shape[0], num_stations, feat_size, X_multi.shape[-1]
    )[:, center_station_idx]
    X_valid, y_valid = X_center[splits[1]], y[splits[1]]
    print(
        f"Strict baseline input={X_center.shape}, target={y.shape}, "
        f"center_index={center_station_idx}, valid={len(splits[1])}",
        flush=True,
    )

    for seed in args.seeds:
        if seed in finished:
            print(f"SKIP completed seed={seed}", flush=True)
            continue
        print(f"\n=== strict_patchtst seed={seed} ===", flush=True)
        set_seed(seed, reproducible=True)
        checkpoint_name = f"strict_patchtst_seed_{seed}"
        callbacks = [
            GradientClip(1.0),
            SaveModelCallback(
                monitor="valid_loss", fname=checkpoint_name, with_opt=False
            ),
            EarlyStoppingCallback(monitor="valid_loss", patience=15),
        ]
        learn = TSForecaster(
            X_center,
            y,
            splits=splits,
            batch_size=args.batch_size,
            path=args.output_dir,
            pipelines=[preproc_pipe, exp_pipe],
            arch=PatchTSTArch,
            arch_config=arch_config(),
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

        predictions = to_np(learn.get_X_preds(X_valid)[0])
        metrics = validation_metrics(y_valid, predictions)
        result = {
            "condition": "strict_patchtst",
            "seed": seed,
            "epochs_requested": args.epochs,
            "epochs_completed": completed_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "input_shape": json.dumps(list(X_center.shape)),
            "valid_mse": metrics["valid_mse"],
            "valid_rmse": metrics["valid_rmse"],
            "valid_mae": metrics["valid_mae"],
            "pollutant_mse": json.dumps(metrics["pollutant_mse"]),
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
