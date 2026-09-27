"""Acceptance tests for trainable-parameter-matched single-station baselines."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset

import run_st_patchtst_ablation as legacy
from run_trainable_matched_baselines import (
    FAMILIES,
    HIDDEN_SIZES,
    SingleStationAdapter,
    assert_training_semantics,
    build_single_station_model,
    config_for,
    expected_identities,
    parameter_counts,
    run_one_with_failure_policy,
    train_one_baseline,
)
from summarize_trainable_matched import (
    compliance_checks,
    recompute_prediction_metrics,
)


def toy_data(history: int = 4, horizon: int = 2):
    generator = torch.Generator().manual_seed(771)
    x = torch.randn(18, 3, history, generator=generator)
    y = (0.6 * x[:, 1, -horizon:] + 0.1).unsqueeze(1)
    datasets = {
        "train": TensorDataset(x[:10], y[:10]),
        "valid": TensorDataset(x[10:14], y[10:14]),
        "test": TensorDataset(x[14:], y[14:]),
    }
    metadata = {
        "center_station_idx": 1,
        "center_mean": 25.0,
        "center_std": 7.0,
    }
    return datasets, metadata


def reference_training(config, datasets, metadata, seed: int):
    """Independent literal implementation of the locked two-epoch semantics."""
    legacy.set_seed(seed)
    model = build_single_station_model(
        "gru", 8, config.horizon, metadata["center_station_idx"]
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    loss_fn = nn.MSELoss()
    train_loader = legacy.make_loader(datasets["train"], config, True, seed)
    valid_loader = legacy.make_loader(datasets["valid"], config, False, seed)
    test_loader = legacy.make_loader(datasets["test"], config, False, seed)
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    best_state = None
    for epoch in range(1, config.epochs + 1):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        valid_prediction, valid_target, _ = legacy.predict(
            model, valid_loader, torch.device("cpu"), metadata["center_station_idx"]
        )
        valid_loss = float(np.mean((valid_prediction - valid_target) ** 2))
        scheduler.step(valid_loss)
        if valid_loss < best_loss - 1e-7:
            best_loss = valid_loss
            best_epoch = epoch
            stale = 0
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        if stale >= config.patience:
            break
    model.load_state_dict(best_state, strict=True)
    prediction, target, _ = legacy.predict(
        model, test_loader, torch.device("cpu"), metadata["center_station_idx"]
    )
    return best_loss, best_epoch, prediction, target, best_state


class TrainableMatchedTests(unittest.TestCase):
    def test_single_station_boundary_and_output_shape(self):
        torch.manual_seed(31)
        model = build_single_station_model("gru", 8, 3, center_station_idx=1)
        x = torch.randn(5, 4, 12)
        changed = x.clone()
        changed[:, 0] += 1000
        changed[:, 2:] -= 1000
        with torch.no_grad():
            actual = model(x)
            perturbed = model(changed)
        self.assertEqual(model.input_channels, 1)
        self.assertEqual(tuple(actual.shape), (5, 1, 3))
        self.assertTrue(torch.equal(actual, perturbed))

    def test_adapter_rejects_bad_shape(self):
        model = SingleStationAdapter(nn.Identity(), 0, 2)
        with self.assertRaisesRegex(ValueError, r"\[B,S,L\]"):
            model(torch.randn(3, 8))

    def test_parameter_registration_is_exact_and_fully_trainable(self):
        for family in FAMILIES:
            for hidden_size in HIDDEN_SIZES:
                for horizon in (1, 6, 24):
                    model = build_single_station_model(family, hidden_size, horizon, 0)
                    total, trainable = parameter_counts(model)
                    self.assertGreater(total, 0)
                    self.assertEqual(total, trainable)
                    self.assertEqual(
                        total, sum(parameter.numel() for parameter in model.parameters())
                    )

    def test_registered_matrix_has_768_identities(self):
        self.assertEqual(len(expected_identities()), 768)

    def test_training_recipe_manifest_exact(self):
        short = config_for(24, 1)
        long = config_for(168, 6)
        assert_training_semantics(short)
        assert_training_semantics(long)
        self.assertEqual((short.epochs, short.patience, short.batch_size), (40, 8, 256))
        self.assertEqual((long.epochs, long.patience, long.batch_size), (30, 6, 512))
        self.assertEqual(short.learning_rate, 1e-3)
        self.assertEqual(short.weight_decay, 1e-4)

    def test_training_semantics_bitwise_gate(self):
        datasets, metadata = toy_data()
        config = legacy.ExperimentConfig(
            history=4,
            horizon=2,
            batch_size=4,
            epochs=2,
            patience=2,
            learning_rate=1e-3,
            weight_decay=1e-4,
            evaluation_split="test",
        )
        seed = 2047
        reference = reference_training(config, datasets, metadata, seed)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = train_one_baseline(
                config,
                datasets,
                metadata,
                "gru",
                8,
                seed,
                root,
                "bitwise",
                torch.device("cpu"),
            )
            with np.load(root / result["prediction_file"]) as payload:
                prediction = payload["prediction_scaled"]
                target = payload["target_scaled"]
            state = torch.load(
                root / result["checkpoint_file"], map_location="cpu", weights_only=True
            )
        self.assertEqual(result["best_valid_loss"], reference[0])
        self.assertEqual(result["best_epoch"], reference[1])
        self.assertTrue(np.array_equal(prediction, reference[2]))
        self.assertTrue(np.array_equal(target, reference[3]))
        self.assertEqual(set(state), set(reference[4]))
        self.assertTrue(all(torch.equal(state[key], reference[4][key]) for key in state))

    def test_independent_prediction_recalculation(self):
        target = np.array([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=np.float64)
        prediction = target + np.array([[[0.25, -0.25]], [[0.25, -0.25]]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prediction.npz"
            np.savez_compressed(
                path, target_ugm3=target, prediction_ugm3=prediction
            )
            metrics = recompute_prediction_metrics(path)
        self.assertEqual(metrics["rmse_ugm3"], 0.25)
        self.assertEqual(metrics["mae_ugm3"], 0.25)

    def test_nonfinite_retries_once_then_records_terminal_status(self):
        datasets, metadata = toy_data()
        config = legacy.ExperimentConfig(history=4, horizon=2)
        calls = []

        def always_nonfinite(config_arg, *args):
            calls.append(config_arg.learning_rate)
            raise FloatingPointError("nan")

        result = run_one_with_failure_policy(
            config,
            datasets,
            metadata,
            "gru",
            8,
            2047,
            Path("unused"),
            "nonfinite",
            torch.device("cpu"),
            train_fn=always_nonfinite,
        )
        self.assertEqual(calls, [1e-3, 1e-4])
        self.assertEqual(result["status"], "nonfinite")
        self.assertTrue(result["learning_rate_retry"])

    def test_oom_records_terminal_status_without_retry(self):
        datasets, metadata = toy_data()
        config = legacy.ExperimentConfig(history=4, horizon=2)
        calls = []

        def oom(config_arg, *args):
            calls.append(config_arg.learning_rate)
            raise RuntimeError("HIP error out of memory")

        result = run_one_with_failure_policy(
            config,
            datasets,
            metadata,
            "lstm",
            8,
            2047,
            Path("unused"),
            "oom",
            torch.device("cpu"),
            train_fn=oom,
        )
        self.assertEqual(calls, [1e-3])
        self.assertEqual(result["status"], "infeasible_oom")
        self.assertFalse(result["learning_rate_retry"])

    def test_compliance_exposes_exceptional_terminal_rows(self):
        row = {
            "run_id": "exception",
            "status": "nonfinite",
            "input_channels": 1,
            "family": "gru",
            "hidden_size": 8,
            "horizon": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            checks = compliance_checks(
                pd.DataFrame([row]), Path(directory), pd.DataFrame()
            ).set_index("check")
        self.assertTrue(bool(checks.loc["all_runs_terminal", "pass"]))
        self.assertTrue(bool(checks.loc["single_station_input_channels", "pass"]))
        self.assertFalse(bool(checks.loc["exact_expected_run_identities", "pass"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
