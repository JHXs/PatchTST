"""Validation-only iterative search for ST-PatchTST pollutant alpha values.

The ST-PatchTST output is affine in each fixed alpha once model weights are
fixed.  Each search round therefore trains one model, computes the exact
least-squares alpha on the validation split, and uses that calibrated vector
to initialize the next round.  The test split is intentionally untouched.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np
import torch
from fastai.callback.tracker import EarlyStoppingCallback, SaveModelCallback
from fastai.callback.training import GradientClip
from tsai.all import TSForecaster, load_object, set_seed, to_np

from ST_PatchTST_model import ST_PatchTST


POLLUTANT_NAMES = ("PM2.5", "PM10", "NO2", "CO", "O3", "SO2")
DEFAULT_ALPHA = (0.3124, 0.0139, 0.0357, 0.0054, 0.02, 0.0490)


def _arch_config(params, alpha):
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
    }


def _predict_with_alpha(learn, X_valid, alpha):
    with torch.no_grad():
        learn.model.set_alpha(alpha)
    return to_np(learn.get_X_preds(X_valid)[0])


def calibrate_alpha(learn, X_valid, y_valid, alpha_max):
    """Return exact per-pollutant validation-MSE minimizers for fixed weights."""
    pred_zero = _predict_with_alpha(learn, X_valid, [0.0] * 6)
    pred_one = _predict_with_alpha(learn, X_valid, [1.0] * 6)
    neighbor_delta = pred_one - pred_zero

    calibrated = []
    rows = []
    for idx, name in enumerate(POLLUTANT_NAMES):
        target = y_valid[:, idx].reshape(-1).astype(np.float64)
        base = pred_zero[:, idx].reshape(-1).astype(np.float64)
        delta = neighbor_delta[:, idx].reshape(-1).astype(np.float64)
        denominator = float(np.dot(delta, delta))
        raw_alpha = 0.0 if denominator == 0 else float(
            np.dot(delta, target - base) / denominator
        )
        selected_alpha = float(np.clip(raw_alpha, 0.0, alpha_max))
        calibrated.append(selected_alpha)
        rows.append(
            {
                "pollutant": name,
                "raw_alpha": raw_alpha,
                "selected_alpha": selected_alpha,
                "mse_alpha_0": float(np.mean((target - base) ** 2)),
                "mse_calibrated": float(
                    np.mean((target - (base + selected_alpha * delta)) ** 2)
                ),
            }
        )

    calibrated_pred = pred_zero.copy()
    for idx, alpha in enumerate(calibrated):
        calibrated_pred[:, idx] += alpha * neighbor_delta[:, idx]
    overall_mse = float(np.mean((y_valid - calibrated_pred) ** 2))
    return calibrated, overall_mse, rows


def train_round(
    X,
    y,
    splits,
    preproc_pipe,
    exp_pipe,
    params,
    alpha,
    seed,
    round_idx,
    epochs,
    learning_rate,
    output_dir,
):
    set_seed(seed, reproducible=True)
    checkpoint_name = f"alpha_search_round_{round_idx:02d}_seed_{seed}"
    callbacks = [
        GradientClip(1.0),
        SaveModelCallback(
            monitor="valid_loss", fname=checkpoint_name, with_opt=False
        ),
        EarlyStoppingCallback(monitor="valid_loss", patience=25),
    ]
    learn = TSForecaster(
        X,
        y,
        splits=splits,
        batch_size=16,
        path=output_dir,
        pipelines=[preproc_pipe, exp_pipe],
        arch=ST_PatchTST,
        arch_config=_arch_config(params, alpha),
        metrics=[],
        cbs=callbacks,
    )
    learn.fit_one_cycle(epochs, lr_max=learning_rate)
    return learn


def append_result(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=result.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(result)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--alpha-max", type=float, default=0.4)
    parser.add_argument("--tolerance", type=float, default=0.005)
    parser.add_argument("--start-alpha", type=float, nargs=6, default=DEFAULT_ALPHA)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("tsai/models/alpha_search")
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    X = np.load("tsai/data/X.npz")["arr_0"]
    y = np.load("tsai/data/y.npz")["arr_0"]
    splits = load_object("tsai/data/splits.pkl")
    preproc_pipe = load_object("tsai/data/preproc_pipe.pkl")
    exp_pipe = load_object("tsai/data/exp_pipe.pkl")
    params = np.load("tsai/data/model_params.npz")
    X_valid, y_valid = X[splits[1]], y[splits[1]]

    alpha = [float(value) for value in args.start_alpha]
    results_path = args.output_dir / "search_results.csv"
    for round_idx in range(1, args.rounds + 1):
        print(f"\n=== alpha search round {round_idx}: {alpha} ===", flush=True)
        learn = train_round(
            X,
            y,
            splits,
            preproc_pipe,
            exp_pipe,
            params,
            alpha,
            args.seed,
            round_idx,
            args.epochs,
            args.learning_rate,
            args.output_dir,
        )
        calibrated, valid_mse, pollutant_rows = calibrate_alpha(
            learn, X_valid, y_valid, args.alpha_max
        )
        max_change = float(np.max(np.abs(np.asarray(calibrated) - alpha)))
        result = {
            "round": round_idx,
            "seed": args.seed,
            "trained_alpha": json.dumps(alpha),
            "calibrated_alpha": json.dumps(calibrated),
            "calibrated_valid_mse": valid_mse,
            "max_alpha_change": max_change,
            "pollutant_details": json.dumps(pollutant_rows),
        }
        append_result(results_path, result)
        print(json.dumps(result, indent=2), flush=True)

        _predict_with_alpha(learn, X_valid[:1], calibrated)
        export_path = args.output_dir / f"round_{round_idx:02d}_calibrated.pt"
        learn.export(export_path.name)
        alpha = calibrated
        del learn
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if max_change <= args.tolerance:
            print(f"Converged: max alpha change {max_change:.6f}", flush=True)
            break


if __name__ == "__main__":
    main()
