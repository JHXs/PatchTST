import unittest

import torch

from ST_PatchTST_model import ST_PatchTST


def build_test_model(alpha_max=0.2, alpha_init=0.02, value_mode="difference"):
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


if __name__ == "__main__":
    unittest.main()
