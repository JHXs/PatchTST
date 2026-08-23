"""Unit tests for the frozen MC-CVLRA L1 implementation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

from cross_variable_lag_adapter import (
    CrossVariableLagAdapter,
    FlatResidualAdapter,
    FrozenBaseAdapterModel,
    clone_module_parameters_and_buffers,
    controlled_token_input,
    expected_m_parameter_count,
    module_state_hash,
    select_flat_width,
    trainable_parameter_count,
)
from cross_variable_lag_l0_data import AUXILIARY_COLUMNS, TARGET_COLUMN
from cross_variable_lag_l1_data import (
    BEIJING_STATION_POOL,
    L1TaskConfig,
    load_l1_selection_data,
    prepare_l1_data,
    select_training_stations,
)
from run_cross_variable_lag_l1 import (
    L1RunConfig,
    block_shuffle_source_indices,
    build_l1_base_model,
    enforce_formal_clean_tree,
    gradient_contract,
    initialize_locked_st_from_f00,
    set_locked_spatial_train_mode,
    tokens_from_auxiliary_history,
)
from summarize_cross_variable_lag_l1 import (
    compute_factorial_metrics,
    compute_l1_gate,
    recompute_shuffle_metrics,
)


def make_station_directory(root: Path, rows: int = 200) -> None:
    times = pd.date_range("2014-01-01", periods=rows, freq="h")
    index = np.arange(rows, dtype=np.float64)
    center = 60.0 + 0.05 * index + 8.0 * np.sin(index / 9.0)
    train_stop = rows * 7 // 10
    selection_stop = rows * 8 // 10
    for station_id in BEIJING_STATION_POOL:
        pm25 = center + (station_id - 1013) * 0.01
        if station_id == 1001:
            pm25 = pm25.copy()
            pm25[:train_stop] = center[:train_stop][::-1]
            pm25[train_stop:selection_stop] = center[train_stop:selection_stop]
        if station_id == 1002:
            pm25 = pm25.copy()
            pm25[:train_stop] = center[:train_stop]
            pm25[train_stop:selection_stop] = center[
                train_stop:selection_stop
            ][::-1]
        pm25[selection_stop:] = 999999.0 + station_id
        frame = pd.DataFrame(
            {
                "station_id": station_id,
                "time": times,
                TARGET_COLUMN: pm25,
                "PM10_Concentration": 100.0,
                "NO2_Concentration": 30.0 + 0.02 * index,
                "CO_Concentration": 1.0 + 0.001 * index,
                "O3_Concentration": 40.0 + 0.03 * index,
                "SO2_Concentration": 10.0 + 0.01 * index,
                "weather": 0,
                "temperature": 15.0 + 0.01 * index,
                "pressure": 750.0 + 0.01 * index,
                "humidity": 50.0 + 0.01 * index,
                "wind_speed": 3.0,
                "wind_direction": 2.0,
            }
        )
        frame.to_csv(root / f"df_station_{station_id}.csv", index=False)


class L1DataTests(unittest.TestCase):
    def test_station_selection_uses_raw_training_rows_only(self):
        rows = 30
        center = np.arange(rows, dtype=np.float64)
        values = {}
        for station_id in BEIJING_STATION_POOL:
            values[station_id] = center.copy()
        values[1001][:20] = center[:20][::-1]
        values[1001][20:] = center[20:]
        values[1002][:20] = center[:20]
        values[1002][20:] = center[20:][::-1]
        frame = pd.DataFrame(values)
        selected, ledger = select_training_stations(frame, train_row_stop=20)
        self.assertNotIn(1001, selected)
        self.assertIn(1002, selected)
        self.assertIn(1036, set(row["station_id"] for row in ledger))
        self.assertNotIn(1013, [
            row["station_id"] for row in ledger if row["selected_neighbor"]
        ])

    def test_loader_never_reads_test_features_and_pool_includes_1036(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_station_directory(root)
            original_read_csv = pd.read_csv
            calls = []

            def recording_read_csv(*args, **kwargs):
                calls.append(dict(kwargs))
                return original_read_csv(*args, **kwargs)

            with patch(
                "cross_variable_lag_l1_data.pd.read_csv", side_effect=recording_read_csv
            ):
                selection = load_l1_selection_data(root, L1TaskConfig(24, 1))
            self.assertEqual(calls[0]["usecols"], ["time"])
            self.assertNotIn("nrows", calls[0])
            self.assertTrue(all(call.get("nrows") == 160 for call in calls[1:]))
            self.assertEqual(selection.access_metadata["test_feature_rows_loaded"], 0)
            self.assertEqual(selection.access_metadata["test_target_values_loaded"], 0)
            self.assertEqual(selection.access_metadata["test_samples_constructed"], 0)
            self.assertEqual(selection.access_metadata["bfill_calls"], 0)
            self.assertIn(1036, selection.config.station_pool)
            self.assertLess(
                float(selection.center_clean[TARGET_COLUMN].max()), 999999.0
            )

    def test_p7_lag_tensors_and_official_target_sets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_station_directory(root)
            data = prepare_l1_data(root, L1TaskConfig(24, 1))
            self.assertEqual(tuple(data.train.auxiliary_tokens.shape[1:]), (7, 6, 2))
            self.assertEqual(tuple(data.train.auxiliary_history.shape[1:]), (7, 24))
            self.assertEqual(tuple(data.train.self_tokens.shape[1:]), (1, 6, 2))
            self.assertEqual(tuple(data.train.context.shape[1:]), (4,))
            self.assertEqual(tuple(data.train.targets_scaled.shape[1:]), (1, 1))
            self.assertEqual(
                data.metadata["auxiliary_variable_order"], list(AUXILIARY_COLUMNS)
            )
            self.assertEqual(data.metadata["lags"], [0, 1, 2, 3, 6, 12])
            origin = int(data.train.origin_rows[0])
            mean = data.metadata["auxiliary_mean"][0]
            scale = data.metadata["auxiliary_scale"][0]
            current = 30.0 + 0.02 * (origin - 1)
            previous = 30.0 + 0.02 * (origin - 2)
            self.assertAlmostEqual(
                float(data.train.auxiliary_tokens[0, 0, 0, 0]),
                (current - mean) / scale,
                places=5,
            )
            self.assertAlmostEqual(
                float(data.train.auxiliary_tokens[0, 0, 0, 1]),
                (current - previous) / scale,
                places=5,
            )
            self.assertTrue(data.metadata["all_target_time_sets_disjoint"])
            self.assertTrue(all(
                row["target_time_intersection_count"] == 0
                for row in data.boundary_checks
            ))
            train_targets = set(data.train.target_times.reshape(-1))
            validation_targets = set(data.validation.target_times.reshape(-1))
            self.assertFalse(train_targets & validation_targets)
            fitted = [
                row for row in data.processing_ledger
                if row["object_type"] in {
                    "fitted_selector",
                    "training_standardizer",
                    "training_quantile_thresholds",
                }
            ]
            train_end = pd.Timestamp(data.metadata["official_boundaries"]["train_end"])
            self.assertTrue(all(pd.Timestamp(row["fit_end"]) <= train_end for row in fitted))
            self.assertTrue(any(
                row["object_id"].startswith("station_pm25_cleaning:train:")
                for row in data.processing_ledger
            ))


class AdapterArchitectureTests(unittest.TestCase):
    def test_m_shapes_group_mass_and_exact_parameter_counts(self):
        for horizon, expected in ((1, 1532), (6, 6547)):
            model = CrossVariableLagAdapter(horizon)
            base = torch.randn(3, 1, horizon)
            tokens = torch.randn(3, 7, 6, 2)
            context = torch.randn(3, 4)
            components = model.forward_components(base, tokens, context)
            self.assertEqual(tuple(components["prediction"].shape), (3, 1, horizon))
            self.assertEqual(tuple(components["gate_weights"].shape), (3, 7, 6, horizon))
            torch.testing.assert_close(
                components["pollutant_mass"] + components["meteorology_mass"],
                torch.ones(3, horizon),
            )
            self.assertEqual(trainable_parameter_count(model), expected)
            self.assertEqual(expected_m_parameter_count(horizon), expected)

    def test_bflat_width_and_parameter_match_are_frozen(self):
        expected = {1: (15, 1532), 6: (47, 6540)}
        for horizon, (width, count) in expected.items():
            info = select_flat_width(horizon)
            model = FlatResidualAdapter(horizon, info["selected_width"])
            self.assertEqual(info["selected_width"], width)
            self.assertEqual(trainable_parameter_count(model), count)
            self.assertLessEqual(info["relative_difference_percent"], 3.0)

    def test_controlled_inputs_replace_only_the_frozen_axes(self):
        auxiliary = torch.arange(2 * 7 * 6 * 2, dtype=torch.float32).reshape(2, 7, 6, 2)
        self_tokens = -torch.arange(2 * 1 * 6 * 2, dtype=torch.float32).reshape(2, 1, 6, 2)
        b1 = controlled_token_input(auxiliary, self_tokens, "B1")
        b2 = controlled_token_input(auxiliary, self_tokens, "B2")
        self.assertTrue(torch.equal(b1[:, 0], self_tokens[:, 0]))
        self.assertTrue(torch.equal(b1[:, -1], self_tokens[:, 0]))
        for lag_index in range(6):
            self.assertTrue(torch.equal(b2[:, :, lag_index], auxiliary[:, :, 0]))
        self.assertTrue(
            torch.equal(controlled_token_input(auxiliary, self_tokens, "F11"), auxiliary)
        )

    def test_exact_bypasses_for_m_and_flat(self):
        base = torch.randn(4, 1, 6)
        tokens = torch.randn(4, 7, 6, 2)
        context = torch.randn(4, 4)
        for adapter in (
            CrossVariableLagAdapter(6),
            FlatResidualAdapter(6, 47),
        ):
            if isinstance(adapter, CrossVariableLagAdapter):
                for head in adapter.output_heads:
                    nn.init.normal_(head[-1].weight)
                    nn.init.normal_(head[-1].bias)
            else:
                nn.init.normal_(adapter.network[-1].weight)
                nn.init.normal_(adapter.network[-1].bias)
            disabled = adapter(base, tokens, context, disable_auxiliary=True)
            torch.testing.assert_close(disabled, base, rtol=0, atol=0)
        zero_m = CrossVariableLagAdapter(6, beta_max=0.0, beta_init=0.0)
        zero_flat = FlatResidualAdapter(6, 47, beta_max=0.0, beta_init=0.0)
        torch.testing.assert_close(zero_m(base, tokens, context), base, rtol=0, atol=0)
        torch.testing.assert_close(zero_flat(base, tokens, context), base, rtol=0, atol=0)

    def test_first_and_later_backward_gradient_contract(self):
        torch.manual_seed(7)
        adapter = CrossVariableLagAdapter(1)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-4)
        base = torch.randn(8, 1, 1)
        tokens = torch.randn(8, 7, 6, 2)
        context = torch.randn(8, 4)
        target = base + 1.0
        loss = (adapter(base, tokens, context) - target).square().mean()
        loss.backward()
        first = gradient_contract(adapter)
        self.assertTrue(first["output_layer_nonzero"])
        self.assertFalse(first["upstream_nonzero"])
        self.assertFalse(first["beta_nonzero"])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        (adapter(base, tokens, context) - target).square().mean().backward()
        later = gradient_contract(adapter)
        self.assertTrue(later["output_layer_nonzero"])
        self.assertTrue(later["upstream_nonzero"])
        self.assertTrue(later["beta_nonzero"])

    def test_frozen_base_remains_eval_no_grad_and_hash_stable(self):
        class DummyBase(nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout = nn.Dropout(0.5)
                self.linear = nn.Linear(5, 1)

            def forward(self, x):
                return self.linear(self.dropout(x[:, 0])).unsqueeze(1)

        base = DummyBase()
        for parameter in base.parameters():
            parameter.grad = torch.ones_like(parameter)
        wrapper = FrozenBaseAdapterModel(base, CrossVariableLagAdapter(1))
        before = module_state_hash(base)
        self.assertTrue(all(parameter.grad is None for parameter in base.parameters()))
        wrapper.train()
        self.assertFalse(base.training)
        prediction = wrapper(
            torch.randn(4, 1, 5),
            torch.randn(4, 7, 6, 2),
            torch.randn(4, 4),
        )
        prediction.square().mean().backward()
        self.assertTrue(all(parameter.grad is None for parameter in base.parameters()))
        self.assertEqual(module_state_hash(base), before)

    def test_locked_f10_keeps_real_patchtst_batchnorm_state_exact(self):
        """A real F10 step must not mutate PatchTST weights or BN buffers."""
        torch.manual_seed(123)
        config = L1RunConfig(
            history=24,
            horizon=1,
            epochs=1,
            patience=1,
            batch_size=8,
        )
        f00 = build_l1_base_model(
            config, "F00", num_stations=6, center_station_idx=2
        )
        f10 = build_l1_base_model(
            config, "F10", num_stations=6, center_station_idx=2
        )
        initialize_locked_st_from_f00(f10, f00.state_dict())

        batch_norms = [
            (name, module)
            for name, module in f10.patch_tst.named_modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
        ]
        self.assertTrue(batch_norms, "Regression fixture must contain real BatchNorm")
        patch_before = clone_module_parameters_and_buffers(f10.patch_tst)
        bn_before = {
            f"{name}.{field}": getattr(module, field).detach().clone()
            for name, module in batch_norms
            for field in ("weight", "running_mean", "running_var", "num_batches_tracked")
        }
        spatial_before = {
            name: parameter.detach().clone()
            for name, parameter in f10.named_parameters()
            if not name.startswith("patch_tst.") and parameter.requires_grad
        }
        optimizer = torch.optim.AdamW(
            [parameter for parameter in f10.parameters() if parameter.requires_grad],
            lr=1e-3,
            weight_decay=1e-4,
        )
        set_locked_spatial_train_mode(f10)
        self.assertFalse(f10.patch_tst.training)
        self.assertTrue(all(not module.training for _, module in batch_norms))
        x_st = torch.randn(8, 6, 24)
        optimizer.zero_grad(set_to_none=True)
        prediction = f10(x_st)
        target = prediction.detach() + 1.0
        (prediction - target).square().mean().backward()
        optimizer.step()

        patch_after = clone_module_parameters_and_buffers(f10.patch_tst)
        self.assertEqual(set(patch_before), set(patch_after))
        for name in patch_before:
            self.assertTrue(torch.equal(patch_before[name], patch_after[name]), name)
        for name, module in batch_norms:
            for field in ("weight", "running_mean", "running_var", "num_batches_tracked"):
                self.assertTrue(
                    torch.equal(bn_before[f"{name}.{field}"], getattr(module, field)),
                    f"{name}.{field}",
                )
        self.assertTrue(
            any(
                not torch.equal(spatial_before[name], parameter.detach())
                for name, parameter in f10.named_parameters()
                if name in spatial_before
            ),
            "At least one F10 spatial parameter must update",
        )

    def test_formal_run_rejects_dirty_tree_but_smoke_is_allowed(self):
        dirty = {"git_dirty": True}
        with self.assertRaises(RuntimeError):
            enforce_formal_clean_tree(dirty, smoke=False)
        enforce_formal_clean_tree(dirty, smoke=True)


class L1MetricAndGateTests(unittest.TestCase):
    def test_factorial_formulas(self):
        frame = pd.DataFrame(
            [
                {"task": "168h_6h", "seed": 1, "variant": variant, "rmse_ugm3": value,
                 "mae_ugm3": value, "smape_percent": value}
                for variant, value in (("F00", 10.0), ("F01", 8.0), ("F10", 9.0), ("F11", 7.5))
            ]
        )
        row = compute_factorial_metrics(frame).query("metric == 'rmse_ugm3'").iloc[0]
        self.assertEqual(row["G_A_given_S0"], 2.0)
        self.assertEqual(row["G_A_given_S1"], 1.5)
        self.assertEqual(row["G_S_given_A0"], 1.0)
        self.assertEqual(row["G_comb"], 2.5)
        self.assertEqual(row["I"], 0.5)
        self.assertEqual(row["R_attn"], 0.25)

    def test_block_shuffle_and_recompute_use_same_24_origin_subset(self):
        source = block_shuffle_source_indices(50, 3101)
        self.assertEqual(len(source), 48)
        self.assertTrue(np.array_equal(source % 24, np.arange(48) % 24))
        prediction_rows = []
        for sample_index in range(48):
            target = float(sample_index)
            for variant, offset in (("F10", 2.0), ("F11", 1.0)):
                prediction_rows.append(
                    {
                        "task": "168h_6h",
                        "seed": 2060,
                        "variant": variant,
                        "sample_index": sample_index,
                        "horizon_step": 1,
                        "target_ugm3": target,
                        "prediction_ugm3": target + offset,
                    }
                )
        shuffle_rows = []
        for permutation_seed in (3101, 3102, 3103, 3104, 3105):
            mapping = block_shuffle_source_indices(48, permutation_seed)
            for sample_index in range(48):
                target = float(sample_index)
                shuffle_rows.append(
                    {
                        "task": "168h_6h",
                        "train_seed": 2060,
                        "permutation_seed": permutation_seed,
                        "diagnostic_sample_index": sample_index,
                        "source_sample_index": int(mapping[sample_index]),
                        "horizon_step": 1,
                        "target_ugm3": target,
                        "base_prediction_ugm3": target + 2.0,
                        "shuffled_prediction_ugm3": target + 1.8,
                        "diagnostic_origin_count": 48,
                        "excluded_tail_origin_count": 2,
                    }
                )
        permutation, summary = recompute_shuffle_metrics(
            pd.DataFrame(prediction_rows), pd.DataFrame(shuffle_rows)
        )
        self.assertEqual(len(permutation), 5)
        self.assertAlmostEqual(float(summary.iloc[0]["G_orig"]), 1.0)
        self.assertAlmostEqual(float(summary.iloc[0]["L_s"]), 0.8)

    def test_shuffle_rebuilds_tokens_from_complete_auxiliary_history(self):
        history = np.arange(2 * 7 * 24, dtype=np.float32).reshape(2, 7, 24)
        tokens = tokens_from_auxiliary_history(history)
        self.assertEqual(tuple(tokens.shape), (2, 7, 6, 2))
        self.assertTrue(np.array_equal(tokens[:, :, 0, 0], history[:, :, -1]))
        self.assertTrue(
            np.array_equal(tokens[:, :, 0, 1], history[:, :, -1] - history[:, :, -2])
        )
        self.assertTrue(np.array_equal(tokens[:, :, 5, 0], history[:, :, -13]))

    def test_l1_hard_gate_and_smoke_suppression(self):
        overall_rows = []
        variants_168 = {
            "F00": 102.0,
            "F01": 101.0,
            "F10": 100.0,
            "F11": 99.0,
            "B1": 100.0,
            "B2": 100.0,
            "B_flat": 100.0,
            "B3": 99.5,
        }
        variants_24 = {**variants_168, "F10": 100.0, "F11": 100.1}
        for task, variants in (("168h_6h", variants_168), ("24h_1h", variants_24)):
            for seed in (2060, 2061, 2062):
                for variant, rmse in variants.items():
                    overall_rows.append(
                        {
                            "task": task,
                            "seed": seed,
                            "variant": variant,
                            "rmse_ugm3": rmse,
                        }
                    )
        horizon_rows = [
            {"task": "168h_6h", "seed": seed, "variant": variant,
             "horizon_step": horizon_step, "rmse_ugm3": rmse}
            for seed in (2060, 2061, 2062)
            for variant, rmse in (("F10", 100.0), ("F11", 99.0))
            for horizon_step in range(1, 7)
        ]
        scenario_rows = [
            {"task": "168h_6h", "seed": seed, "variant": variant,
             "scenario": "high_pollution", "rmse_ugm3": rmse}
            for seed in (2060, 2061, 2062)
            for variant, rmse in (("F10", 100.0), ("F11", 100.5))
        ]
        shuffle_rows = [
            {"task": "168h_6h", "train_seed": seed, "G_orig": 1.0,
             "L_s": 0.6, "G_orig_positive": True,
             "shuffle_loss_at_least_50_percent": True}
            for seed in (2060, 2061, 2062)
        ]
        gate = compute_l1_gate(
            pd.DataFrame(overall_rows),
            pd.DataFrame(horizon_rows),
            pd.DataFrame(scenario_rows),
            pd.DataFrame(shuffle_rows),
            smoke=False,
        )
        self.assertEqual(gate["status"], "PASS")
        smoke_gate = compute_l1_gate(
            pd.DataFrame(overall_rows),
            pd.DataFrame(horizon_rows),
            pd.DataFrame(scenario_rows),
            pd.DataFrame(shuffle_rows),
            smoke=True,
        )
        self.assertEqual(smoke_gate["status"], "NOT_EVALUABLE_SMOKE")
        self.assertFalse(smoke_gate["formal_selection_performed"])


if __name__ == "__main__":
    unittest.main()
