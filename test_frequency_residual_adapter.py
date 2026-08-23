"""Correctness tests for the fixed-frequency MVP and time control."""

from __future__ import annotations

import unittest

import torch

from causal_local_filterbank_adapter import (
    CausalFilterBankResidualAdapter,
    CausalLocalFilterBank,
)
from frequency_residual_adapter import (
    FixedBandDecomposition,
    FixedFrequencyResidualAdapter,
    FrozenSTForecastResidual,
    TimeResidualAdapter,
    build_fixed_band_masks,
    trainable_parameter_count,
)
from run_causal_frequency_p0 import P0ExperimentConfig, build_clean_model
from run_st_patchtst_ablation import set_seed


class FrequencyResidualAdapterTest(unittest.TestCase):
    def test_masks_are_nonnegative_and_partition_unity_for_both_tasks(self):
        for seq_len in (24, 168, 25, 169):
            frequencies, masks = build_fixed_band_masks(seq_len)
            self.assertEqual(tuple(masks.shape), (3, seq_len // 2 + 1))
            self.assertEqual(len(frequencies), seq_len // 2 + 1)
            self.assertTrue(torch.isfinite(masks).all())
            self.assertTrue((masks >= 0).all())
            torch.testing.assert_close(
                masks.sum(dim=0), torch.ones_like(frequencies), rtol=0, atol=1e-6
            )
            torch.testing.assert_close(
                masks[:, 0], torch.tensor([1.0, 0.0, 0.0]), rtol=0, atol=0
            )

    def test_rfft_bands_reconstruct_even_and_odd_windows(self):
        for seq_len in (24, 25, 168, 169):
            decomposition = FixedBandDecomposition(seq_len)
            x = torch.randn(4, 1, seq_len, requires_grad=True)
            bands = decomposition(x)
            self.assertEqual(tuple(bands.shape), (4, 3, seq_len))
            torch.testing.assert_close(
                bands.sum(dim=1, keepdim=True), x, rtol=1e-5, atol=1e-5
            )
            bands.square().mean().backward()
            self.assertIsNotNone(x.grad)
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_time_and_frequency_adapters_have_identical_capacity(self):
        common = dict(
            seq_len=168,
            pred_len=6,
            hidden_dim=16,
            pool_bins=4,
            dropout=0.1,
            num_streams=3,
        )
        set_seed(77)
        time_adapter = TimeResidualAdapter(**common)
        set_seed(77)
        frequency_adapter = FixedFrequencyResidualAdapter(**common)
        set_seed(77)
        local_adapter = CausalFilterBankResidualAdapter(**common)
        self.assertEqual(
            trainable_parameter_count(time_adapter),
            trainable_parameter_count(frequency_adapter),
        )
        self.assertEqual(
            trainable_parameter_count(time_adapter),
            trainable_parameter_count(local_adapter),
        )
        time_state = time_adapter.state_dict()
        frequency_state = frequency_adapter.state_dict()
        for name in time_state:
            torch.testing.assert_close(time_state[name], frequency_state[name])
            torch.testing.assert_close(time_state[name], local_adapter.state_dict()[name])

    def test_causal_local_filterbank_reconstructs_and_uses_no_future(self):
        decomposition = CausalLocalFilterBank(seq_len=168)
        x = torch.randn(2, 1, 168, requires_grad=True)
        streams = decomposition(x)
        torch.testing.assert_close(
            streams.sum(dim=1, keepdim=True), x, rtol=1e-6, atol=1e-6
        )
        changed = x.detach().clone()
        changed[..., 100:] += 1000
        original_streams = decomposition(x.detach())
        changed_streams = decomposition(changed)
        torch.testing.assert_close(
            original_streams[..., :100],
            changed_streams[..., :100],
            rtol=0,
            atol=0,
        )
        streams.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_zero_initialization_and_disable_restore_exact_base(self):
        config = P0ExperimentConfig(
            history=24,
            horizon=1,
            n_layers=1,
            n_heads=2,
            d_model=8,
            d_ff=16,
            dropout=0.0,
            patch_len=4,
            stride=2,
        )
        set_seed(31)
        base = build_clean_model(config, "locked_st_clean", 6, 5)
        adapter = FixedFrequencyResidualAdapter(
            seq_len=24, pred_len=1, hidden_dim=8, pool_bins=2, dropout=0.0
        )
        model = FrozenSTForecastResidual(base, adapter).eval()
        x = torch.randn(3, 6, 24)
        with torch.no_grad():
            initial = model.forward_components(x)
            base_prediction = base(x)
        torch.testing.assert_close(initial["prediction"], base_prediction, rtol=0, atol=0)
        torch.testing.assert_close(
            initial["forecast_residual"],
            torch.zeros_like(initial["forecast_residual"]),
            rtol=0,
            atol=0,
        )

        with torch.no_grad():
            model.adapter.forecast_out.weight.fill_(0.2)
            model.adapter.forecast_out.bias.fill_(0.1)
            enabled = model(x)
            disabled = model.forward_components(x, disable_frequency=True)[
                "prediction"
            ]
        self.assertFalse(torch.equal(enabled, disabled))
        torch.testing.assert_close(disabled, base_prediction, rtol=0, atol=0)

    def test_only_adapter_and_beta_receive_finite_gradients(self):
        config = P0ExperimentConfig(
            history=24,
            horizon=1,
            n_layers=1,
            n_heads=2,
            d_model=8,
            d_ff=16,
            dropout=0.0,
            patch_len=4,
            stride=2,
        )
        set_seed(44)
        base = build_clean_model(config, "locked_st_clean", 6, 5)
        adapter = FixedFrequencyResidualAdapter(
            seq_len=24, pred_len=1, hidden_dim=8, pool_bins=2, dropout=0.0
        )
        model = FrozenSTForecastResidual(base, adapter)
        x = torch.randn(4, 6, 24)
        target = torch.randn(4, 1, 1)
        loss = torch.nn.functional.mse_loss(model(x), target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(parameter.grad is None for parameter in base.parameters()))
        output_parameters = list(adapter.forecast_out.parameters())
        self.assertTrue(all(parameter.grad is not None for parameter in output_parameters))
        self.assertTrue(
            all(torch.isfinite(parameter.grad).all() for parameter in output_parameters)
        )
        self.assertIsNotNone(model.beta_logit.grad)
        self.assertTrue(torch.isfinite(model.beta_logit.grad))

    def test_batch_size_one_and_invalid_shape(self):
        adapter = FixedFrequencyResidualAdapter(
            seq_len=24, pred_len=1, hidden_dim=4, pool_bins=2, dropout=0.0
        )
        output = adapter(torch.randn(1, 1, 24))
        self.assertEqual(tuple(output.shape), (1, 1, 1))
        with self.assertRaises(ValueError):
            adapter(torch.randn(1, 2, 24))


if __name__ == "__main__":
    unittest.main()
