"""Acceptance and independent-review regression tests for baseline comparison."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression, Ridge

import baseline_traditional
import run_baseline_comparison as baseline_runner
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
from summarize_baselines import (
    ST_VARIANT,
    build_best_of_baselines_tables,
    recompute_directory,
    write_best_of_markdown,
)


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


def independent_dataset_copies(datasets):
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
    """Fail loudly if fitting code touches a forbidden split."""

    def __init__(self, split_name):
        self.split_name = split_name

    def __getattr__(self, attribute):
        raise AssertionError(
            f"fit illegally accessed {self.split_name}.{attribute}"
        )


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
        _, source_datasets, metadata = synthetic_problem()
        clean_datasets = independent_dataset_copies(source_datasets)
        poisoned_datasets = independent_dataset_copies(source_datasets)
        poisoned_datasets["test"].values[:] = 1_000_000
        train_indices = set(clean_datasets["train"].sample_indices.tolist())
        for arm in ("trad_climatology", "trad_ar", "trad_ridge", "trad_spatial_linear"):
            with self.subTest(arm=arm):
                clean = fit_traditional_baseline(arm, clean_datasets, metadata)
                poisoned = fit_traditional_baseline(arm, poisoned_datasets, metadata)
                self.assertTrue(set(clean.fit_sample_indices.tolist()) <= train_indices)
                self.assertEqual(clean.selected_alpha, poisoned.selected_alpha)
                np.testing.assert_allclose(
                    clean.predict(clean_datasets["valid"], metadata),
                    poisoned.predict(clean_datasets["valid"], metadata),
                    rtol=0,
                    atol=0,
                )
                if clean.climatology_by_hour is not None:
                    np.testing.assert_array_equal(
                        clean.climatology_by_hour, poisoned.climatology_by_hour
                    )
                if clean.estimator is not None:
                    np.testing.assert_array_equal(
                        clean.estimator.coef_, poisoned.estimator.coef_
                    )
                    np.testing.assert_array_equal(
                        np.asarray(clean.estimator.intercept_),
                        np.asarray(poisoned.estimator.intercept_),
                    )

        # No arm may inspect test during fit. Climatology and AR also have no
        # validation-selection step, so valid is forbidden for those two.
        for arm in ("trad_climatology", "trad_ar", "trad_ridge", "trad_spatial_linear"):
            guarded = independent_dataset_copies(source_datasets)
            guarded["test"] = AccessForbiddenDataset("test")
            if arm in {"trad_climatology", "trad_ar"}:
                guarded["valid"] = AccessForbiddenDataset("valid")
            with self.subTest(guarded_arm=arm):
                fit_traditional_baseline(arm, guarded, metadata)

        # Estimator.fit must receive train features only. Ridge may read valid
        # solely to score the preregistered alpha grid, never to refit.
        train_x, _ = dataset_arrays(clean_datasets["train"])
        expected_ar = train_x[:, metadata["center_station_idx"], -24:]
        original_linear_fit = LinearRegression.fit
        linear_fit_inputs = []

        def guarded_linear_fit(estimator, x, y, *args, **kwargs):
            linear_fit_inputs.append(np.asarray(x).copy())
            return original_linear_fit(estimator, x, y, *args, **kwargs)

        with patch.object(LinearRegression, "fit", new=guarded_linear_fit):
            fit_traditional_baseline("trad_ar", clean_datasets, metadata)
        self.assertTrue(linear_fit_inputs)
        for fitted_x in linear_fit_inputs:
            np.testing.assert_array_equal(fitted_x, expected_ar)

        original_ridge_fit = Ridge.fit
        for arm, expected in (
            ("trad_ridge", train_x[:, metadata["center_station_idx"], :]),
            ("trad_spatial_linear", train_x.reshape(len(train_x), -1)),
        ):
            ridge_fit_inputs = []

            def guarded_ridge_fit(estimator, x, y, *args, **kwargs):
                ridge_fit_inputs.append(np.asarray(x).copy())
                return original_ridge_fit(estimator, x, y, *args, **kwargs)

            with patch.object(Ridge, "fit", new=guarded_ridge_fit):
                fit_traditional_baseline(arm, clean_datasets, metadata)
            self.assertEqual(len(ridge_fit_inputs), 5)
            for fitted_x in ridge_fit_inputs:
                np.testing.assert_array_equal(fitted_x, expected)

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

        # Climatology: each target hour appears on days 2 and 3, so the exact
        # train-only hourly mean is (24+h + 48+h) / 2 = 36+h.
        hourly_values = np.arange(72, dtype=np.float32)[:, None]
        hourly_train = legacy.ForecastWindowDataset(
            hourly_values, np.arange(48), 24, 1, 0
        )
        hourly_datasets = {
            "train": hourly_train,
            "valid": legacy.ForecastWindowDataset(hourly_values, [0], 24, 1, 0),
            "test": legacy.ForecastWindowDataset(hourly_values, [1], 24, 1, 0),
        }
        climatology = fit_traditional_baseline(
            "trad_climatology", hourly_datasets, metadata
        )
        np.testing.assert_array_equal(
            climatology.climatology_by_hour,
            36 + np.arange(24, dtype=np.float32),
        )

        # Independent non-overlapping windows with a known exact linear target.
        rng = np.random.default_rng(8675309)
        block_count = 80
        linear_values = np.zeros((block_count * 4, 1), dtype=np.float32)
        starts = np.arange(0, block_count * 4, 4, dtype=np.int64)
        for start in starts:
            features = rng.normal(0, 10, size=3).astype(np.float32)
            linear_values[start:start + 3, 0] = features
            linear_values[start + 3, 0] = (
                1.0 + 0.5 * features[0] - 2.0 * features[1] + 3.0 * features[2]
            )
        linear_datasets = {
            "train": legacy.ForecastWindowDataset(linear_values, starts[:50], 3, 1, 0),
            "valid": legacy.ForecastWindowDataset(linear_values, starts[50:65], 3, 1, 0),
            "test": legacy.ForecastWindowDataset(linear_values, starts[65:], 3, 1, 0),
        }
        linear_metadata = {
            "station_ids": [1013],
            "center_station_idx": 0,
            "start_time": "2020-01-01 00:00:00",
        }
        _, linear_target = dataset_arrays(linear_datasets["test"])
        for arm, tolerance in (("trad_ar", 1e-5), ("trad_ridge", 2e-3)):
            fitted = fit_traditional_baseline(arm, linear_datasets, linear_metadata)
            prediction = fitted.predict(linear_datasets["test"], linear_metadata)
            rmse = float(np.sqrt(np.mean((prediction - linear_target) ** 2)))
            self.assertLess(rmse, tolerance, f"{arm} rmse={rmse}")

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

    def test_review_oom_resume_is_idempotent(self):
        config, datasets, metadata = synthetic_problem(history=8, horizon=1, stations=3)
        config = replace(config, epochs=1, patience=1, batch_size=4)
        stable_state = {"commit": "test", "hashes": {}, "git_status": ""}
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "beijing/8h_1h"
            oom = torch.OutOfMemoryError("HIP out of memory")
            with (
                patch.object(baseline_runner, "train_baseline", side_effect=oom) as train_mock,
                patch.object(baseline_runner, "code_state", return_value=stable_state),
                patch.object(torch.cuda, "empty_cache", return_value=None),
                patch.object(torch.cuda, "get_device_name", return_value="Synthetic GPU"),
            ):
                arguments = (
                    config,
                    datasets,
                    metadata,
                    output_dir,
                    ("concat_patchtst_all",),
                    (),
                    ("default",),
                    (2047,),
                    torch.device("cuda"),
                    stable_state,
                    True,
                )
                baseline_runner.run_one_dataset(*arguments)
                baseline_runner.run_one_dataset(*arguments)
                self.assertEqual(train_mock.call_count, 1)
            raw = pd.read_csv(output_dir / "raw_metrics.csv")
            self.assertEqual(len(raw), 1)
            self.assertEqual(raw.loc[0, "status"], "infeasible_oom")
            self.assertFalse(raw.duplicated(["variant", "seed", "requested_capacity"]).any())

    def test_review_best_of_baselines_and_missing_st(self):
        records = []
        values = {
            "plain_mix_patchtst_all_default": {1: 10.0, 2: 7.0, 3: 8.0},
            "multi_gru_default": {1: 9.0, 2: 9.0, 3: 9.0},
        }
        base_arms = {
            "plain_mix_patchtst_all_default": "plain_mix_patchtst_all",
            "multi_gru_default": "multi_gru",
        }
        for variant, seed_values in values.items():
            for seed, rmse in seed_values.items():
                records.append(
                    {
                        "city": "beijing",
                        "history": 24,
                        "horizon": 1,
                        "variant": variant,
                        "arm": base_arms[variant],
                        "seed": seed,
                        "status": "completed",
                        "layer": "C",
                        "rmse_ugm3": rmse,
                    }
                )
        records.append(
            {
                "city": "beijing",
                "history": 24,
                "horizon": 1,
                "variant": "trad_spatial_linear",
                "arm": "trad_spatial_linear",
                "seed": "deterministic",
                "status": "completed",
                "layer": "A",
                "rmse_ugm3": 20.0,
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            st_dir = root / "beijing_st/24h_1h"
            st_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {"variant": ST_VARIANT, "seed": 1, "rmse_ugm3": 8.0},
                    {"variant": ST_VARIANT, "seed": 2, "rmse_ugm3": 9.0},
                ]
            ).to_csv(st_dir / "raw_metrics.csv", index=False)
            details, paired, summary = build_best_of_baselines_tables(
                pd.DataFrame(records), root / "beijing_st", root / "guangzhou_st"
            )
            paired = paired.set_index("seed")
            self.assertEqual(paired.loc[1, "best_seed_arm"], "multi_gru_default")
            self.assertEqual(paired.loc[1, "best_baseline_rmse_ugm3"], 9.0)
            self.assertEqual(paired.loc[2, "best_seed_arm"], "plain_mix_patchtst_all_default")
            self.assertEqual(paired.loc[2, "best_baseline_rmse_ugm3"], 7.0)
            self.assertEqual(paired.loc[1, "st_minus_best_baseline_rmse_ugm3"], -1.0)
            self.assertEqual(paired.loc[2, "st_minus_best_baseline_rmse_ugm3"], 2.0)
            self.assertEqual(paired.loc[3, "st_status"], "missing")
            self.assertTrue(np.isnan(paired.loc[3, "st_rmse_ugm3"]))
            self.assertEqual(paired.loc[1, "st_better_direction_count"], "1/2")
            self.assertEqual(summary.loc[0, "winner_arm"], "plain_mix_patchtst_all_default")
            arm_mean = details[
                details["candidate_arm"] == "plain_mix_patchtst_all_default"
            ]["arm_mean_rmse_ugm3"].iloc[0]
            self.assertAlmostEqual(arm_mean, 25 / 3)
            markdown_path = root / "B3.md"
            write_best_of_markdown(summary, markdown_path)
            markdown = markdown_path.read_text(encoding="utf-8")
            self.assertIn("plain_mix_patchtst_all_default", markdown)
            self.assertIn("missing", markdown)

    def test_review_traditional_arm_is_never_missing(self):
        config, datasets, metadata = synthetic_problem(history=24, horizon=1)
        with tempfile.TemporaryDirectory() as temporary:
            row = evaluate_traditional_baseline(
                config, "trad_persistence", datasets, metadata, temporary
            )
            frame = pd.DataFrame([row])
            self.assertFalse(frame["arm"].isna().any())
            self.assertEqual(frame.loc[0, "arm"], "trad_persistence")


if __name__ == "__main__":
    unittest.main()
