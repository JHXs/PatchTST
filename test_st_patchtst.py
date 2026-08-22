import unittest

import torch

from analyze_neighbor_lag_predictability import relative_rmse_improvement_percent
from run_st_patchtst_ablation import ExperimentConfig, build_model
from ST_PatchTST_model import ST_PatchTST


def build_test_model(
    alpha_max=0.2,
    alpha_init=0.02,
    value_mode="difference",
    fusion_stage="input",
    use_station_gate_bias=False,
    neighbor_top_k=None,
    use_forecast_confidence=False,
):
    return ST_PatchTST(
        c_in=5,
        c_out=1,
        seq_len=24,
        pred_dim=3,
        num_stations=5,
        feat_size=1,
        center_station_idx=2,
        alpha_max=alpha_max,
        alpha_init=alpha_init,
        gate_mode="pairwise_summary",
        use_null_neighbor=True,
        neighbor_value_mode=value_mode,
        fusion_stage=fusion_stage,
        use_station_gate_bias=use_station_gate_bias,
        neighbor_top_k=neighbor_top_k,
        use_forecast_confidence=use_forecast_confidence,
        n_layers=1,
        n_heads=4,
        d_model=16,
        d_ff=32,
        patch_len=4,
        stride=2,
    )


class STPatchTSTTest(unittest.TestCase):
    def test_pairwise_output_and_gate_shapes(self):
        model = build_test_model().eval()
        x = torch.randn(4, 5, 24)
        with torch.no_grad():
            output = model(x)
            components = model.spatial_components(x)
        self.assertEqual(tuple(output.shape), (4, 1, 3))
        self.assertEqual(tuple(components["gate_weights"].shape), (4, 4))
        gate_mass = components["gate_weights"].sum(1) + components["null_weight"]
        torch.testing.assert_close(gate_mass, torch.ones_like(gate_mass))

    def test_degraded_model_is_neighbor_invariant(self):
        model = build_test_model(alpha_max=0.0, alpha_init=0.0).eval()
        x = torch.randn(4, 5, 24)
        changed = x.clone()
        changed[:, [0, 1, 3, 4]] = torch.randn_like(changed[:, [0, 1, 3, 4]]) * 100
        with torch.no_grad():
            original_output = model(x)
            changed_output = model(changed)
        torch.testing.assert_close(original_output, changed_output, rtol=0, atol=0)

    def test_pairwise_branch_receives_finite_gradients(self):
        model = build_test_model()
        x = torch.randn(4, 5, 24)
        model(x).square().mean().backward()
        self.assertTrue(
            all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            )
        )

    def test_forecast_fusion_starts_from_exact_backbone_prediction(self):
        model = build_test_model(fusion_stage="forecast").eval()
        x = torch.randn(4, 5, 24)
        with torch.no_grad():
            components = model.forward_components(x)
        torch.testing.assert_close(
            components["prediction"],
            components["base_prediction"],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            components["forecast_residual"],
            torch.zeros_like(components["forecast_residual"]),
            rtol=0,
            atol=0,
        )

    def test_sparse_gate_limits_active_neighbors(self):
        model = build_test_model(
            fusion_stage="forecast",
            use_station_gate_bias=True,
            neighbor_top_k=2,
        ).eval()
        x = torch.randn(4, 5, 24)
        with torch.no_grad():
            components = model.spatial_components(x)
        active_neighbors = (components["gate_weights"] > 0).sum(dim=1)
        self.assertTrue(torch.equal(active_neighbors, torch.full_like(active_neighbors, 2)))

    def test_disable_spatial_bypasses_trained_forecast_head(self):
        model = build_test_model(fusion_stage="forecast").eval()
        torch.nn.init.normal_(model.spatial_forecast_out.weight)
        x = torch.randn(4, 5, 24)
        with torch.no_grad():
            enabled = model.forward_components(x)
            disabled = model.forward_components(x, disable_spatial=True)
        self.assertGreater(
            float((enabled["prediction"] - enabled["base_prediction"]).abs().max()),
            0.0,
        )
        torch.testing.assert_close(
            disabled["prediction"], disabled["base_prediction"], rtol=0, atol=0
        )

    def test_forecast_confidence_is_horizon_specific_and_bounded(self):
        model = build_test_model(
            fusion_stage="forecast",
            use_forecast_confidence=True,
        ).eval()
        x = torch.randn(4, 5, 24)
        with torch.no_grad():
            confidence = model.forward_components(x)["forecast_confidence"]
        self.assertEqual(tuple(confidence.shape), (4, 1, 3))
        self.assertTrue(bool(((confidence > 0) & (confidence < 1)).all()))

    def test_sparse_and_station_bias_are_independent_ablations(self):
        config = ExperimentConfig(
            history=24,
            horizon=1,
            sparse_neighbor_top_k=3,
            n_layers=1,
            n_heads=4,
            d_model=16,
            d_ff=32,
        )
        sparse = build_model(config, "st_sparse_delta_forecast", 5, 2)
        station_bias = build_model(
            config, "st_station_bias_delta_forecast", 5, 2
        )
        combined = build_model(
            config, "st_sparse_station_bias_delta_forecast", 5, 2
        )
        self.assertFalse(sparse.use_station_gate_bias)
        self.assertEqual(sparse.neighbor_top_k, 3)
        self.assertTrue(station_bias.use_station_gate_bias)
        self.assertIsNone(station_bias.neighbor_top_k)
        self.assertTrue(combined.use_station_gate_bias)
        self.assertEqual(combined.neighbor_top_k, 3)

    def test_relative_rmse_improvement_uses_baseline_denominator(self):
        self.assertAlmostEqual(relative_rmse_improvement_percent(10.0, 9.0), 10.0)

    def test_sparse_neighbor_top_k_must_be_positive(self):
        config = ExperimentConfig(
            history=24,
            horizon=1,
            sparse_neighbor_top_k=0,
            n_layers=1,
            n_heads=4,
            d_model=16,
            d_ff=32,
        )
        with self.assertRaisesRegex(ValueError, "必须为正整数"):
            build_model(config, "st_sparse_delta_forecast", 5, 2)


if __name__ == "__main__":
    unittest.main()
