"""T1--T7 acceptance tests for the frozen backbone-upgrade protocol."""

from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

import run_st_patchtst_ablation as legacy
from backbone_candidates import (
    BackboneAdapter,
    build_center_backbone,
    candidate_in_protocol_coverage,
    expected_candidates,
    load_candidate_adapter,
)
from run_backbone_upgrade import (
    probe_candidate_backbone,
    select_by_validation,
    selected_candidate,
    train_one_upgrade,
)
from summarize_backbone_upgrade import recompute_prediction_metrics


REAL_CHECKPOINT = Path(
    "/home/hansel/Documents/ITProject/Python/PatchTST/experiments/results/"
    "baselines/beijing/24h_1h/checkpoints/center_gru_default_seed2047.pt"
)


class TinyBackbone(nn.Module):
    def __init__(self, history: int, horizon: int):
        super().__init__()
        self.linear = nn.Linear(history, horizon)

    def forward(self, x):
        return self.linear(x[:, 0])


class NonfiniteBackbone(nn.Module):
    def __init__(self, horizon: int):
        super().__init__()
        self.horizon = horizon

    def forward(self, x):
        return torch.full((len(x), self.horizon), float("nan"), device=x.device)


def mounted_model(history=8, horizon=2):
    config = legacy.ExperimentConfig(
        history=history,
        horizon=horizon,
        n_layers=1,
        n_heads=1,
        d_model=4,
        d_ff=8,
        dropout=0.0,
        patch_len=4,
        stride=2,
        sparse_neighbor_top_k=5,
    )
    model = legacy.build_model(
        config,
        "st_sparse_station_bias_delta_forecast",
        num_stations=6,
        center_idx=0,
    )
    model.patch_tst = BackboneAdapter(TinyBackbone(history, horizon), horizon)
    for parameter in model.patch_tst.parameters():
        parameter.requires_grad = False
    return model


class BackboneUpgradeTests(unittest.TestCase):
    t1_max_abs = None
    t4_max_abs = None

    def test_t1_adapter_is_bitwise_identical_to_direct_checkpoint_forward(self):
        self.assertTrue(REAL_CHECKPOINT.is_file(), f"缺少真实检查点: {REAL_CHECKPOINT}")
        state = torch.load(REAL_CHECKPOINT, map_location="cpu", weights_only=True)
        direct = build_center_backbone(
            "center_gru", 24, 1, {"hidden_size": 100}
        )
        direct.load_state_dict(
            {key[len("model."):]: value for key, value in state.items() if key.startswith("model.")},
            strict=True,
        )
        wrapped = load_candidate_adapter(
            "center_gru",
            24,
            1,
            str(REAL_CHECKPOINT),
            {"hidden_size": 100},
        )
        x = torch.linspace(-2, 2, 96, dtype=torch.float32).reshape(4, 1, 24)
        direct.eval()
        wrapped.eval()
        with torch.no_grad():
            expected = direct(x).unsqueeze(1)
            actual = wrapped(x)
        type(self).t1_max_abs = float((actual - expected).abs().max())
        self.assertTrue(torch.equal(actual, expected))

    def test_t2_adapter_shape(self):
        adapter = BackboneAdapter(TinyBackbone(8, 3), 3)
        self.assertEqual(tuple(adapter(torch.randn(5, 1, 8)).shape), (5, 1, 3))

    def test_t3_backbone_is_frozen_and_unchanged_after_step(self):
        torch.manual_seed(1)
        model = mounted_model()
        before = {key: value.detach().clone() for key, value in model.patch_tst.state_dict().items()}
        self.assertTrue(all(not p.requires_grad for p in model.patch_tst.parameters()))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        prediction = model(torch.randn(4, 6, 8))
        loss = prediction.square().mean()
        loss.backward()
        optimizer.step()
        after = model.patch_tst.state_dict()
        self.assertTrue(all(torch.equal(before[key], after[key]) for key in before))

    def test_t4_zero_initialized_mount_equals_backbone(self):
        torch.manual_seed(2)
        model = mounted_model()
        model.eval()
        x = torch.randn(7, 6, 8)
        with torch.no_grad():
            mounted = model(x)
            backbone = model.patch_tst(x[:, :1])
        type(self).t4_max_abs = float((mounted - backbone).abs().max())
        self.assertLessEqual(type(self).t4_max_abs, 1e-12)

    def test_t5_selection_uses_validation_only(self):
        rows = [
            {
                "variant": "validation_winner",
                "candidate_status": "eligible",
                "best_valid_loss": 0.1,
                "test_rmse": 999.0,
                "selected": False,
            },
            {
                "variant": "test_winner",
                "candidate_status": "eligible",
                "best_valid_loss": 0.2,
                "test_rmse": 1.0,
                "selected": False,
            },
        ]
        selected = select_by_validation(deepcopy(rows))
        self.assertEqual([row["variant"] for row in selected if row["selected"]], ["validation_winner"])

    def test_t6_prediction_file_independent_recalculation(self):
        target = np.array([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=np.float64)
        prediction = target + 0.25
        backbone = target - 0.5
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prediction.npz"
            np.savez_compressed(
                path,
                target_ugm3=target,
                prediction_ugm3=prediction,
                backbone_prediction_ugm3=backbone,
            )
            metrics = recompute_prediction_metrics(path)
        self.assertLessEqual(abs(metrics["rmse_ugm3"] - 0.25), 1e-9)
        self.assertLessEqual(abs(metrics["backbone_rmse_ugm3"] - 0.5), 1e-9)

    def test_t7_insufficient_candidates_are_explicit(self):
        expected = expected_candidates("beijing", 24, 12)
        rows = [
            {
                "variant": spec.variant,
                "candidate_status": (
                    "eligible" if spec.variant == "center_gru_default"
                    else "checkpoint_missing" if candidate_in_protocol_coverage("beijing", 24, 12, spec)
                    else "not_in_protocol_coverage"
                ),
                "best_valid_loss": 0.1 if spec.variant == "center_gru_default" else float("nan"),
                "selected": False,
            }
            for spec in expected
        ]
        selected = select_by_validation(rows)
        self.assertEqual(len(selected), len(expected))
        self.assertTrue(any(row["candidate_status"] == "checkpoint_missing" for row in selected))
        resnet = next(row for row in selected if row["variant"] == "center_resnet_default")
        self.assertEqual(resnet["candidate_status"], "not_in_protocol_coverage")
        self.assertEqual(sum(bool(row["selected"]) for row in selected), 1)

    def test_p1_real_forward_rejects_nonfinite_backbone(self):
        config = legacy.ExperimentConfig(
            history=8, horizon=2, batch_size=4, n_layers=1, n_heads=1,
            d_model=4, d_ff=8, patch_len=4, stride=2,
        )
        dataset = TensorDataset(torch.randn(6, 3, 8), torch.randn(6, 1, 2))
        candidate = {
            "arm": "fake", "checkpoint_path": "fake.pt", "hyperparameters": "{}",
        }
        adapter = BackboneAdapter(NonfiniteBackbone(2), 2)
        with patch("run_backbone_upgrade.load_candidate_adapter", return_value=adapter):
            usable, reason = probe_candidate_backbone(
                candidate, config, dataset, {"center_station_idx": 1}, torch.device("cpu")
            )
        self.assertFalse(usable)
        self.assertIn("validation_forward_nonfinite", reason)
        rows = select_by_validation(
            [{"variant": "bad", "candidate_status": "backbone_nonfinite", "best_valid_loss": 0.01, "selected": False}]
        )
        self.assertIsNone(selected_candidate(rows))

    def test_runtime_nonfinite_backbone_returns_auditable_terminal_row(self):
        config = legacy.ExperimentConfig(
            history=8, horizon=2, batch_size=4, n_layers=1, n_heads=1,
            d_model=4, d_ff=8, dropout=0.0, patch_len=4, stride=2,
            sparse_neighbor_top_k=5,
        )
        model = mounted_model(config.history, config.horizon)
        model.patch_tst = BackboneAdapter(NonfiniteBackbone(config.horizon), config.horizon)
        for parameter in model.patch_tst.parameters():
            parameter.requires_grad = False
        dataset = TensorDataset(
            torch.randn(5, 6, config.history),
            torch.randn(5, 1, config.horizon),
        )
        winner = {
            "variant": "bad_variant", "arm": "bad_arm", "capacity": "bad_capacity",
            "best_valid_loss": 0.1, "checkpoint_path": "bad.pt",
        }
        with tempfile.TemporaryDirectory() as directory, patch(
            "run_backbone_upgrade.build_upgraded_model", return_value=model
        ):
            result = train_one_upgrade(
                config,
                {"train": dataset, "valid": dataset, "test": dataset},
                {"center_station_idx": 0},
                winner,
                2047,
                Path(directory),
                "nonfinite_test",
                torch.device("cpu"),
            )
        self.assertEqual(result["status"], "backbone_nonfinite")
        self.assertEqual(result["selected_variant"], "bad_variant")
        self.assertEqual(result["source_checkpoint"], "bad.pt")
        self.assertIn("NaN/Inf", result["failure_reason"])


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(BackboneUpgradeTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(f"T1 位级最大差值: {BackboneUpgradeTests.t1_max_abs}")
    print(f"T4 零初始化最大差值: {BackboneUpgradeTests.t4_max_abs}")
    raise SystemExit(0 if result.wasSuccessful() else 1)
