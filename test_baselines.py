"""Acceptance tests T1--T9 for baseline-comparison P1."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_st_patchtst_ablation as legacy
from baseline_models import (
    CAPACITY_TIERS,
    MATCHED_LOWER,
    MATCHED_UPPER,
    NEURAL_ARMS,
    build_baseline_model,
    channel_indices_for_arm,
)
from baseline_traditional import (
    TRADITIONAL_ARMS,
    dataset_arrays,
    fit_traditional_baseline,
)
from baseline_training import evaluate_traditional_baseline, train_baseline
from summarize_baselines import recompute_directory


def synthetic_problem(history=24, horizon=3, stations=8):
    row_count = 120
    time = np.arange(row_count, dtype=np.float32)
    values = np.stack(
        [0.02 * time + station + np.sin(time / (4 + station)) for station in range(stations)],
        axis=1,
    ).astype(np.float32)
    train_indices = np.arange(0, 48, dtype=np.int64)
    valid_indices = np.arange(55, 70, dtype=np.int64)
    test_indices = np.arange(75, row_count - history - horizon + 1, dtype=np.int64)
    center_idx = 2
    datasets = {
        "train": legacy.ForecastWindowDataset(values, train_indices, history, horizon, center_idx),
        "valid": legacy.ForecastWindowDataset(values, valid_indices, history, horizon, center_idx),
        "test": legacy.ForecastWindowDataset(values, test_indices, history, horizon, center_idx),
    }
    station_ids = list(range(1001, 1001 + stations))
    metadata = {
        "station_ids": station_ids,
        "center_station_idx": center_idx,
        "center_mean": 20.0,
        "center_std": 5.0,
        "start_time": "2020-01-01 00:00:00",
        "station_correlations": {
            str(station_id): 1.0 - 0.02 * index
            for index, station_id in enumerate(station_ids)
        },
    }
    config = legacy.ExperimentConfig(
        history=history,
        horizon=horizon,
        batch_size=8,
        epochs=2,
        patience=2,
        evaluation_split="test",
    )
    return config, datasets, metadata


class BaselineAcceptanceTests(unittest.TestCase):
    def test_t1_bit_level_legacy_reproduction(self):
        config, datasets, metadata = synthetic_problem(history=8, horizon=2, stations=3)
        config = replace(config, batch_size=4, epochs=2, patience=2)
        seed = 2047
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy_row = legacy.train_one_run(
                config,
                datasets,
                metadata,
                "degraded_patchtst",
                seed,
                root / "legacy",
                torch.device("cpu"),
            )

            def factory():
                return legacy.build_model(
                    config,
                    "degraded_patchtst",
                    len(metadata["station_ids"]),
                    metadata["center_station_idx"],
                )

            replica_row = train_baseline(
                config,
                factory,
                datasets,
                metadata,
                seed,
                torch.device("cpu"),
                "degraded_patchtst_replica",
                root / "replica",
            )
            relative = abs(replica_row["rmse_ugm3"] - legacy_row["rmse_ugm3"]) / abs(
                legacy_row["rmse_ugm3"]
            )
            self.assertLessEqual(relative, 1e-12)
            with np.load(root / "legacy/predictions/degraded_patchtst_seed2047.npz") as left:
                with np.load(
                    root / "replica/predictions/degraded_patchtst_replica_seed2047.npz"
                ) as right:
                    np.testing.assert_array_equal(left["prediction_scaled"], right["prediction_scaled"])

    def test_t2_all_arm_shapes(self):
        config, datasets, metadata = synthetic_problem()
        x, _ = next(iter(torch.utils.data.DataLoader(datasets["train"], batch_size=4)))
        for arm in NEURAL_ARMS:
            for capacity in CAPACITY_TIERS:
                with self.subTest(arm=arm, capacity=capacity):
                    model, _ = build_baseline_model(arm, capacity, config, metadata)
                    model = model.to("cpu")
                    self.assertEqual(tuple(model(x).shape), (4, 1, config.horizon))
        for arm in TRADITIONAL_ARMS:
            with self.subTest(arm=arm):
                model = fit_traditional_baseline(arm, datasets, metadata)
                self.assertEqual(
                    tuple(model.predict(datasets["test"], metadata).shape),
                    (len(datasets["test"]), 1, config.horizon),
                )

    def test_t3_train_only_fitting(self):
        config, datasets, metadata = synthetic_problem()
        train_indices = set(datasets["train"].sample_indices.tolist())
        for arm in ("trad_climatology", "trad_ar", "trad_ridge", "trad_spatial_linear"):
            with self.subTest(arm=arm):
                model = fit_traditional_baseline(arm, datasets, metadata)
                self.assertTrue(set(model.fit_sample_indices.tolist()) <= train_indices)
                before = model.predict(datasets["valid"], metadata)
                original = datasets["test"].values
                datasets["test"].values = original.copy()
                datasets["test"].values[:] = 1_000_000
                after = model.predict(datasets["valid"], metadata)
                datasets["test"].values = original
                np.testing.assert_array_equal(before, after)

    def test_t4_information_sets(self):
        config, _, metadata = synthetic_problem(stations=8)
        del config
        self.assertEqual(channel_indices_for_arm("center_gru", metadata), (2,))
        all_indices = channel_indices_for_arm("plain_mix_patchtst_all", metadata)
        self.assertEqual(len(all_indices), len(metadata["station_ids"]))
        top5 = channel_indices_for_arm("plain_mix_patchtst_top5", metadata)
        self.assertEqual(len(top5), 6)
        self.assertIn(metadata["center_station_idx"], top5)

    def test_t5_capacity_registration(self):
        config, _, metadata = synthetic_problem()
        for arm in NEURAL_ARMS:
            with self.subTest(arm=arm):
                _, registration = build_baseline_model(arm, "matched", config, metadata)
                self.assertGreater(registration.parameter_count, 0)
                if registration.capacity_status == "matched":
                    self.assertGreaterEqual(registration.parameter_count, MATCHED_LOWER)
                    self.assertLessEqual(registration.parameter_count, MATCHED_UPPER)
                else:
                    self.assertEqual(registration.capacity_status, "matched_nearest")

    def test_t6_seed_determinism(self):
        config, datasets, metadata = synthetic_problem(history=8, horizon=2, stations=3)
        config = replace(config, epochs=1, patience=1, batch_size=4)

        def factory():
            return build_baseline_model("center_gru", "matched", config, metadata)[0]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_baseline(
                config, factory, datasets, metadata, 77, torch.device("cpu"), "gru", root / "a"
            )
            train_baseline(
                config, factory, datasets, metadata, 77, torch.device("cpu"), "gru", root / "b"
            )
            with np.load(root / "a/predictions/gru_seed77.npz") as left:
                with np.load(root / "b/predictions/gru_seed77.npz") as right:
                    np.testing.assert_array_equal(left["prediction_scaled"], right["prediction_scaled"])

    def test_t7_evaluation_split_guard(self):
        config, datasets, metadata = synthetic_problem()
        config = replace(config, evaluation_split="valid")
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                evaluate_traditional_baseline(
                    config,
                    "trad_persistence",
                    datasets,
                    metadata,
                    temporary,
                )

    def test_t8_traditional_naive_correctness(self):
        history, horizon = 24, 3
        values = np.arange(80, dtype=np.float32)[:, None]
        dataset = legacy.ForecastWindowDataset(values, [5, 6], history, horizon, 0)
        datasets = {"train": dataset, "valid": dataset, "test": dataset}
        metadata = {
            "station_ids": [1013],
            "center_station_idx": 0,
            "start_time": "2020-01-01 00:00:00",
        }
        persistence = fit_traditional_baseline("trad_persistence", datasets, metadata)
        daily = fit_traditional_baseline("trad_daily_naive", datasets, metadata)
        x, _ = dataset_arrays(dataset)
        expected_persistence = np.repeat(x[:, :, -1:], horizon, axis=2)
        expected_daily = x[:, :, [0, 1, 2]]
        np.testing.assert_array_equal(persistence.predict(dataset, metadata), expected_persistence)
        np.testing.assert_array_equal(daily.predict(dataset, metadata), expected_daily)

        one_step_config, one_step_datasets, one_step_metadata = synthetic_problem(horizon=1)
        del one_step_config
        for arm in ("trad_ar", "trad_ridge", "trad_spatial_linear"):
            fitted = fit_traditional_baseline(arm, one_step_datasets, one_step_metadata)
            self.assertEqual(
                fitted.predict(one_step_datasets["test"], one_step_metadata).shape,
                (len(one_step_datasets["test"]), 1, 1),
            )

    def test_t9_independent_summary_recalculation(self):
        prediction = np.array([[[1.0], [2.0]]], dtype=np.float32).reshape(2, 1, 1)
        target = np.array([[[0.0], [1.5]]], dtype=np.float32).reshape(2, 1, 1)
        metrics = legacy.regression_metrics(target, prediction, 10.0, 2.0)
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary) / "beijing/24h_1h"
            (result / "predictions").mkdir(parents=True)
            np.savez_compressed(
                result / "predictions/trad_persistence.npz",
                prediction_scaled=prediction,
                target_scaled=target,
                prediction_ugm3=prediction * 2 + 10,
                target_ugm3=target * 2 + 10,
            )
            pd.DataFrame(
                [
                    {
                        "variant": "trad_persistence",
                        "seed": "deterministic",
                        "layer": "A",
                        "evaluation_split": "test",
                        "parameter_count": 0,
                        "requested_capacity": "not_applicable",
                        "capacity_status": "not_applicable",
                        "prediction_file": "predictions/trad_persistence.npz",
                        **metrics,
                    }
                ]
            ).to_csv(result / "raw_metrics.csv", index=False)
            (result / "experiment_config.json").write_text(
                '{"history": 24, "horizon": 1, "smoke": true}', encoding="utf-8"
            )
            _, comparisons = recompute_directory(temporary)
            self.assertLessEqual(comparisons["relative_difference"].max(), 1e-9)

    def test_t10_neighbor_information_flow(self):
        config, _, metadata = synthetic_problem(history=24, horizon=3, stations=8)
        generator = torch.Generator().manual_seed(9102)
        x = torch.randn(4, len(metadata["station_ids"]), config.history, generator=generator)
        perturbed = x.clone()
        neighbor_mask = torch.ones(x.shape[1], dtype=torch.bool)
        neighbor_mask[metadata["center_station_idx"]] = False
        perturbed[:, neighbor_mask] += 5.0

        flowing_arms = (
            "plain_mix_patchtst_all",
            "plain_mix_patchtst_top5",
            "concat_patchtst_all",
            "multi_gru",
            "multi_tst",
        )
        channel_independent_arms = ("patchtst_ci_all", "patchtst_ci_top5")
        for index, arm in enumerate((*flowing_arms, *channel_independent_arms)):
            legacy.set_seed(12000 + index)
            model, _ = build_baseline_model(arm, "default", config, metadata)
            model = model.to("cpu").eval()
            with torch.no_grad():
                change = float((model(perturbed) - model(x)).abs().max().item())
            print(f"T10 {arm} max_abs_change={change:.12g}")
            with self.subTest(arm=arm):
                if arm in flowing_arms:
                    self.assertGreater(change, 1e-8)
                else:
                    self.assertEqual(change, 0.0)


if __name__ == "__main__":
    unittest.main()
