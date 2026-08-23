"""Tests for the P0 clean ST bridge and Gate F1 contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_causal_frequency_p0 import (
    P0ExperimentConfig,
    _load_checkpoint,
    _model_signature,
    _save_checkpoint,
    _state_sha256,
    build_clean_model,
    evaluate_f1_gate,
    freeze_patchtst_backbone,
)
from run_st_patchtst_ablation import set_seed


class CausalFrequencyP0Test(unittest.TestCase):
    def setUp(self) -> None:
        self.config = P0ExperimentConfig(
            history=8,
            horizon=2,
            batch_size=4,
            epochs=1,
            patience=1,
            n_layers=1,
            n_heads=2,
            d_model=8,
            d_ff=16,
            dropout=0.0,
            patch_len=4,
            stride=2,
        )
        self.metadata = {
            "station_ids": [1001, 1002, 1003, 1004, 1005, 1013],
            "center_station_idx": 5,
            "time_index_sha256": "synthetic-time-index",
            "target_timestamp_intersections_empty": True,
        }

    def test_clean_variants_share_locked_structure_and_top5_is_strict(self):
        degraded = build_clean_model(self.config, "degraded_patchtst_clean", 6, 5)
        locked = build_clean_model(self.config, "locked_st_clean", 6, 5)
        for model in (degraded, locked):
            self.assertEqual(model.fusion_stage, "forecast")
            self.assertEqual(model.neighbor_value_mode, "difference")
            self.assertTrue(model.use_station_gate_bias)
            self.assertEqual(model.neighbor_top_k, 5)
        self.assertIsNone(degraded.alpha_logit)
        self.assertIsNotNone(locked.alpha_logit)
        with self.assertRaisesRegex(ValueError, "至少 5 个邻站"):
            build_clean_model(self.config, "locked_st_clean", 5, 4)

    def test_degraded_is_neighbor_invariant(self):
        set_seed(9)
        model = build_clean_model(
            self.config, "degraded_patchtst_clean", 6, 5
        ).eval()
        x = torch.randn(3, 6, self.config.history)
        changed = x.clone()
        changed[:, :5] = torch.randn_like(changed[:, :5]) * 100
        with torch.no_grad():
            expected = model(x)
            actual = model(changed)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_degraded_checkpoint_initializes_locked_backbone_exactly(self):
        signature = _model_signature(self.config, self.metadata)
        set_seed(12)
        degraded = build_clean_model(
            self.config, "degraded_patchtst_clean", 6, 5
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "degraded.pt"
            _save_checkpoint(
                path, degraded, signature, "degraded_patchtst_clean", 12, 1, 0.5
            )
            set_seed(999)
            locked = build_clean_model(self.config, "locked_st_clean", 6, 5)
            payload = _load_checkpoint(
                path, locked, signature, 12, allow_missing_alpha=True
            )
        self.assertEqual(
            _state_sha256(locked.state_dict(), prefix="patch_tst."),
            payload["backbone_sha256"],
        )
        x = torch.randn(2, 6, self.config.history)
        degraded.eval()
        locked.eval()
        with torch.no_grad():
            degraded_prediction = degraded(x)
            locked_components = locked.forward_components(x)
        torch.testing.assert_close(
            locked_components["forecast_residual"],
            torch.zeros_like(locked_components["forecast_residual"]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            locked_components["prediction"], degraded_prediction, rtol=0, atol=0
        )

    def test_frozen_backbone_unchanged_after_optimizer_step(self):
        set_seed(14)
        model = build_clean_model(self.config, "locked_st_clean", 6, 5)
        freeze_patchtst_backbone(model)
        before_hash = _state_sha256(model.state_dict(), prefix="patch_tst.")
        before_spatial = model.spatial_forecast_out.weight.detach().clone()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        frozen_ids = {id(parameter) for parameter in model.patch_tst.parameters()}
        self.assertFalse(any(id(parameter) in frozen_ids for parameter in trainable))

        optimizer = torch.optim.AdamW(trainable, lr=1e-2)
        x = torch.randn(4, 6, self.config.history)
        y = torch.randn(4, 1, self.config.horizon)
        model.train()
        model.patch_tst.eval()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(
            _state_sha256(model.state_dict(), prefix="patch_tst."), before_hash
        )
        self.assertFalse(
            torch.equal(model.spatial_forecast_out.weight, before_spatial)
        )

    def test_f1_gate_requires_three_of_three_and_half_percent(self):
        gate_config = P0ExperimentConfig(
            history=168,
            horizon=6,
            n_layers=1,
            n_heads=2,
            d_model=8,
            d_ff=16,
        )
        rows = []
        paired = []
        for seed, improvement in zip((2052, 2053, 2054), (0.7, 0.8, 0.9)):
            rows.extend(
                [
                    {
                        "variant": "degraded_patchtst_clean",
                        "seed": seed,
                        "evaluation_split": "valid",
                        "rmse_ugm3": 10.0,
                        "mae_ugm3": 5.0,
                        "best_valid_loss": 0.2,
                        "backbone_frozen_unchanged": None,
                    },
                    {
                        "variant": "locked_st_clean",
                        "seed": seed,
                        "evaluation_split": "valid",
                        "rmse_ugm3": 9.9,
                        "mae_ugm3": 4.9,
                        "best_valid_loss": 0.19,
                        "backbone_frozen_unchanged": True,
                    },
                ]
            )
            paired.append(
                {
                    "seed": seed,
                    "rmse_relative_improvement_percent": improvement,
                }
            )
        status = evaluate_f1_gate(
            gate_config,
            self.metadata,
            pd.DataFrame(rows),
            pd.DataFrame(paired),
            smoke=False,
        )
        self.assertEqual(status["status"], "passed")
        paired[2]["rmse_relative_improvement_percent"] = -0.1
        failed = evaluate_f1_gate(
            gate_config,
            self.metadata,
            pd.DataFrame(rows),
            pd.DataFrame(paired),
            smoke=False,
        )
        self.assertEqual(failed["status"], "failed")


if __name__ == "__main__":
    unittest.main()
