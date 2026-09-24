"""Acceptance tests for the preregistered spatial-branch capacity search."""

from __future__ import annotations

import unittest

import torch
from torch import nn

import run_st_patchtst_ablation as legacy
from backbone_candidates import BackboneAdapter
from run_capacity_search import (
    CAPACITY_CANDIDATES,
    apply_capacity_candidate,
    select_capacity_by_validation,
)


class TinyBackbone(nn.Module):
    def __init__(self, history: int, horizon: int):
        super().__init__()
        self.linear = nn.Linear(history, horizon)

    def forward(self, x):
        return self.linear(x[:, 0])


def mounted_model(candidate_name: str = "cap128_b8_a20", history: int = 8, horizon: int = 2):
    base = legacy.ExperimentConfig(
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
    candidate = next(spec for spec in CAPACITY_CANDIDATES if spec.name == candidate_name)
    config = apply_capacity_candidate(base, candidate)
    model = legacy.build_model(
        config,
        "st_sparse_station_bias_delta_forecast",
        num_stations=6,
        center_idx=0,
    )
    model.patch_tst = BackboneAdapter(TinyBackbone(history, horizon), horizon)
    for parameter in model.patch_tst.parameters():
        parameter.requires_grad = False
    return config, model


class CapacitySearchTests(unittest.TestCase):
    def test_candidate_hyperparameters_reach_constructed_model(self):
        for candidate in CAPACITY_CANDIDATES:
            with self.subTest(candidate=candidate.name):
                config, model = mounted_model(candidate.name)
                self.assertEqual(config.neighbor_hidden_dim, candidate.neighbor_hidden_dim)
                self.assertEqual(config.spatial_pool_bins, candidate.spatial_pool_bins)
                self.assertEqual(model.neighbor_hidden_dim, candidate.neighbor_hidden_dim)
                self.assertEqual(model.spatial_pool_bins, candidate.spatial_pool_bins)
                self.assertEqual(model.spatial_forecast_out.in_features, candidate.neighbor_hidden_dim * (candidate.spatial_pool_bins + 1))
                alpha = model.max_alpha * torch.sigmoid(model.alpha_logit)
                self.assertAlmostEqual(float(alpha.item()), candidate.forecast_alpha_init, places=6)
                self.assertAlmostEqual(model.max_alpha, candidate.forecast_alpha_max, places=12)

    def test_selection_uses_validation_loss_not_test_rmse(self):
        rows = [
            {
                "capacity_candidate": "validation_winner",
                "status": "completed",
                "best_valid_loss": 0.1,
                "rmse_ugm3": 999.0,
            },
            {
                "capacity_candidate": "test_winner",
                "status": "completed",
                "best_valid_loss": 0.2,
                "rmse_ugm3": 1.0,
            },
        ]
        selected = select_capacity_by_validation(rows)
        winners = [row["capacity_candidate"] for row in selected if row["selected"]]
        self.assertEqual(winners, ["validation_winner"])

    def test_zero_initialized_branch_equals_frozen_backbone(self):
        torch.manual_seed(2)
        _, model = mounted_model()
        model.eval()
        x = torch.randn(7, 6, 8)
        with torch.no_grad():
            actual = model(x)
            expected = model.patch_tst(x[:, :1])
        self.assertTrue(torch.equal(actual, expected))

    def test_backbone_is_frozen_and_unchanged_after_optimizer_step(self):
        torch.manual_seed(3)
        _, model = mounted_model()
        before = {
            key: value.detach().clone() for key, value in model.patch_tst.state_dict().items()
        }
        self.assertTrue(all(not parameter.requires_grad for parameter in model.patch_tst.parameters()))
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=1e-3,
        )
        loss = model(torch.randn(4, 6, 8)).square().mean()
        loss.backward()
        optimizer.step()
        self.assertTrue(
            all(
                torch.equal(before[key], value)
                for key, value in model.patch_tst.state_dict().items()
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
