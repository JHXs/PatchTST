"""T1-T8 acceptance tests for the single-station baseline implementation."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

import run_st_patchtst_ablation as legacy
from run_single_station_baselines import (
    RunKey,
    train_neural_baseline,
    validate_ingest_row,
)
from single_station_models import CAPACITY_TIERS, NEURAL_ARMS, build_single_station_model
from single_station_traditional import (
    TRADITIONAL_ARMS,
    dataset_arrays,
    fit_traditional_baseline,
)
from summarize_single_station import build_s3_tables, build_s7, recompute_directory


def synthetic_problem(history=24, horizon=3, stations=4):
    row_count = 140
    time_axis = np.arange(row_count, dtype=np.float32)
    values = np.stack(
        [
            0.02 * time_axis + station + np.sin(time_axis / (4 + station))
            for station in range(stations)
        ],
        axis=1,
    ).astype(np.float32)
    center_idx = min(2, stations - 1)
    datasets = {
        "train": legacy.ForecastWindowDataset(
            values, np.arange(0, 55), history, horizon, center_idx
        ),
        "valid": legacy.ForecastWindowDataset(
            values, np.arange(60, 75), history, horizon, center_idx
        ),
        "test": legacy.ForecastWindowDataset(
            values,
            np.arange(80, row_count - history - horizon + 1),
            history,
            horizon,
            center_idx,
        ),
    }
    metadata = {
        "station_ids": list(range(1001, 1001 + stations)),
        "center_station_idx": center_idx,
        "center_mean": 20.0,
        "center_std": 5.0,
        "start_time": "2020-01-01 00:00:00",
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


def copy_datasets(datasets):
    return {
        name: legacy.ForecastWindowDataset(
            dataset.values.copy(),
            dataset.sample_indices.copy(),
            dataset.history,
            dataset.horizon,
            dataset.center_idx,
        )
        for name, dataset in datasets.items()
    }


class AccessForbiddenDataset:
    def __init__(self, name):
        self.name = name

    def __getattr__(self, attribute):
        raise AssertionError(f"fit illegally accessed {self.name}.{attribute}")


class SingleStationAcceptanceTests(unittest.TestCase):
    def test_t1_single_channel_construction_and_ingest(self):
        config, datasets, metadata = synthetic_problem()
        x, _ = next(iter(torch.utils.data.DataLoader(datasets["train"], batch_size=4)))
        perturbed = x.clone()
        mask = torch.ones(x.shape[1], dtype=torch.bool)
        mask[metadata["center_station_idx"]] = False
        perturbed[:, mask] += 1000
        for arm in NEURAL_ARMS:
            for capacity in CAPACITY_TIERS:
                with self.subTest(arm=arm, capacity=capacity):
                    model, registration = build_single_station_model(
                        arm, capacity, config, metadata
                    )
                    model = model.to("cpu")
                    self.assertEqual(registration.input_channels, 1)
                    model.eval()
                    with torch.no_grad():
                        np.testing.assert_array_equal(
                            model(x).numpy(), model(perturbed).numpy()
                        )
        for arm in TRADITIONAL_ARMS:
            fitted = fit_traditional_baseline(arm, datasets, metadata)
            self.assertEqual(fitted.input_channels, 1)
        key = RunKey("beijing", 24, 1, None, "center_gru", "2047", "default")
        row = {
            "variant": "center_gru_default",
            "arm": "center_gru",
            "seed": 2047,
            "requested_capacity": "default",
            "input_channels": 1,
            "evaluation_split": "test",
        }
        validate_ingest_row(row, key, ("beijing", 24, 1, None))

    def test_t2_all_arm_output_shapes(self):
        config, datasets, metadata = synthetic_problem()
        x, _ = next(iter(torch.utils.data.DataLoader(datasets["train"], batch_size=4)))
        for arm in NEURAL_ARMS:
            with self.subTest(neural_arm=arm):
                model, _ = build_single_station_model(arm, "default", config, metadata)
                model = model.to("cpu")
                self.assertEqual(tuple(model(x).shape), (4, 1, config.horizon))
        for arm in TRADITIONAL_ARMS:
            with self.subTest(traditional_arm=arm):
                model = fit_traditional_baseline(arm, datasets, metadata)
                prediction = model.predict(datasets["test"], metadata)
                self.assertEqual(
                    tuple(prediction.shape), (len(datasets["test"]), 1, config.horizon)
                )

    def test_t3_no_test_fit_and_ridge_validation_only_selection(self):
        _, source, metadata = synthetic_problem()
        clean = copy_datasets(source)
        poisoned = copy_datasets(source)
        poisoned["test"].values[:] = 1_000_000
        train_indices = set(clean["train"].sample_indices.tolist())
        for arm in ("trad_climatology", "trad_ar", "trad_ridge"):
            with self.subTest(arm=arm):
                left = fit_traditional_baseline(arm, clean, metadata)
                right = fit_traditional_baseline(arm, poisoned, metadata)
                self.assertTrue(set(left.fit_sample_indices.tolist()) <= train_indices)
                self.assertEqual(left.selected_alpha, right.selected_alpha)
                np.testing.assert_array_equal(
                    left.predict(clean["valid"], metadata),
                    right.predict(clean["valid"], metadata),
                )

        for arm in ("trad_climatology", "trad_ar", "trad_ridge"):
            guarded = copy_datasets(source)
            guarded["test"] = AccessForbiddenDataset("test")
            if arm != "trad_ridge":
                guarded["valid"] = AccessForbiddenDataset("valid")
            fit_traditional_baseline(arm, guarded, metadata)

        train_x, _ = dataset_arrays(clean["train"])
        ridge_fit_inputs = []
        original_fit = Ridge.fit

        def guarded_fit(estimator, x, y, *args, **kwargs):
            ridge_fit_inputs.append(np.asarray(x).copy())
            return original_fit(estimator, x, y, *args, **kwargs)

        with patch.object(Ridge, "fit", new=guarded_fit):
            fit_traditional_baseline("trad_ridge", clean, metadata)
        self.assertEqual(len(ridge_fit_inputs), 5)
        for fitted_x in ridge_fit_inputs:
            np.testing.assert_array_equal(fitted_x, train_x[:, 0, :])

    def test_t4_bit_level_legacy_training_loop(self):
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

            replica_row = train_neural_baseline(
                config,
                factory,
                datasets,
                metadata,
                seed,
                torch.device("cpu"),
                "degraded_patchtst_replica",
                root / "replica",
            )
            difference = abs(replica_row["rmse_ugm3"] - legacy_row["rmse_ugm3"])
            print(f"T4 bit_level_rmse_abs_diff={difference:.17g}")
            self.assertEqual(difference, 0.0)
            with np.load(
                root / "legacy/predictions/degraded_patchtst_seed2047.npz"
            ) as left:
                with np.load(
                    root
                    / "replica/predictions/degraded_patchtst_replica_seed2047.npz"
                ) as right:
                    np.testing.assert_array_equal(
                        left["prediction_scaled"], right["prediction_scaled"]
                    )

    def test_t5_ingest_rejects_channel_or_config_mismatch(self):
        key = RunKey("beijing", 24, 1, None, "center_gru", "2047", "default")
        valid = {
            "variant": "center_gru_default",
            "arm": "center_gru",
            "seed": 2047,
            "requested_capacity": "default",
            "input_channels": 1,
            "evaluation_split": "test",
        }
        invalid_channel = {**valid, "input_channels": 2}
        with self.assertRaisesRegex(ValueError, "input_channels=2"):
            validate_ingest_row(invalid_channel, key, ("beijing", 24, 1, None))
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            validate_ingest_row(valid, key, ("beijing", 48, 1, None))

    def test_t6_independent_prediction_recalculation(self):
        prediction = np.array([1.0, 2.0], dtype=np.float32).reshape(2, 1, 1)
        target = np.array([0.0, 1.5], dtype=np.float32).reshape(2, 1, 1)
        metrics = legacy.regression_metrics(target, prediction, 10.0, 2.0)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = root / "beijing/24h_1h"
            (result / "predictions").mkdir(parents=True)
            path = result / "predictions/trad_persistence.npz"
            np.savez_compressed(
                path,
                prediction_scaled=prediction,
                target_scaled=target,
                prediction_ugm3=prediction * 2 + 10,
                target_ugm3=target * 2 + 10,
            )
            pd.DataFrame(
                [
                    {
                        "variant": "trad_persistence",
                        "arm": "trad_persistence",
                        "seed": "deterministic",
                        "status": "completed",
                        "evaluation_split": "test",
                        "input_channels": 1,
                        "capacity_tier": "not_applicable",
                        "provenance": "same arm definition",
                        "prediction_file": str(path),
                        **metrics,
                    }
                ]
            ).to_csv(result / "raw_metrics.csv", index=False)
            _, comparisons = recompute_directory(root)
            self.assertLessEqual(comparisons["absolute_difference"].max(), 1e-9)

    def test_t7_best_single_station_and_pair_direction(self):
        baselines = []
        for seed in range(2047, 2052):
            baselines.extend(
                [
                    {
                        "city": "beijing",
                        "history": 24,
                        "horizon": 1,
                        "variant": "center_gru_default",
                        "arm": "center_gru",
                        "capacity_tier": "default",
                        "seed": seed,
                        "status": "completed",
                        "rmse_ugm3": 10.0,
                    },
                    {
                        "city": "beijing",
                        "history": 24,
                        "horizon": 1,
                        "variant": "center_lstm_default",
                        "arm": "center_lstm",
                        "capacity_tier": "default",
                        "seed": seed,
                        "status": "completed",
                        "rmse_ugm3": 12.0,
                    },
                ]
            )
        models = pd.DataFrame(
            [
                {
                    "city": "beijing",
                    "history": 24,
                    "horizon": 1,
                    "arm": "st",
                    "seed": seed,
                    "rmse_ugm3": 9.0 if seed < 2051 else 11.0,
                }
                for seed in range(2047, 2052)
            ]
        )
        _, paired, summary = build_s3_tables(pd.DataFrame(baselines), models)
        self.assertEqual(summary.loc[0, "best_single_station_arm"], "center_gru_default")
        self.assertEqual(summary.loc[0, "better_direction_count"], "4/5")
        self.assertEqual(len(paired), 5)
        self.assertAlmostEqual(summary.loc[0, "mean_reduction_percent"], 6.0)

    def test_t8_exception_rows_excluded_but_counted(self):
        baselines = []
        for seed in range(2047, 2052):
            baselines.append(
                {
                    "city": "beijing",
                    "history": 24,
                    "horizon": 1,
                    "variant": "center_gru_default",
                    "arm": "center_gru",
                    "capacity_tier": "default",
                    "seed": seed,
                    "status": "completed",
                    "rmse_ugm3": 10.0,
                }
            )
        baselines.extend(
            [
                {
                    "city": "beijing",
                    "history": 24,
                    "horizon": 1,
                    "variant": "center_tst_default",
                    "arm": "center_tst",
                    "capacity_tier": "default",
                    "seed": 2047,
                    "status": "nonfinite",
                    "rmse_ugm3": np.nan,
                },
                {
                    "city": "beijing",
                    "history": 24,
                    "horizon": 1,
                    "variant": "center_mlp_default",
                    "arm": "center_mlp",
                    "capacity_tier": "default",
                    "seed": 2047,
                    "status": "infeasible_oom",
                    "rmse_ugm3": np.nan,
                },
            ]
        )
        model_rows = pd.DataFrame(
            [
                {
                    "city": "beijing",
                    "history": 24,
                    "horizon": 1,
                    "arm": "st",
                    "seed": seed,
                    "rmse_ugm3": 9.0,
                }
                for seed in range(2047, 2052)
            ]
        )
        _, paired, _ = build_s3_tables(pd.DataFrame(baselines), model_rows)
        self.assertEqual(set(paired["best_single_station_arm"]), {"center_gru_default"})
        s7 = build_s7(pd.DataFrame(baselines), pd.DataFrame())
        counts = dict(zip(s7["item"], s7["count"]))
        self.assertEqual(counts["nonfinite"], 1)
        self.assertEqual(counts["infeasible_oom"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
