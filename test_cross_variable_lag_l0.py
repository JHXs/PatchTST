"""Unit tests for the preregistered L0 selection-interval implementation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from cross_variable_lag_l0_data import (
    AUXILIARY_COLUMNS,
    PROHIBITED_COLUMNS,
    SOURCE_AUXILIARY_COLUMNS,
    TARGET_COLUMN,
    L0TaskConfig,
    build_l0_splits,
    build_window_data,
    causal_ffill,
    clean_selection_frame,
    convert_pressure_to_hpa,
    establish_official_boundaries,
    load_selection_data,
)
from run_cross_variable_lag_l0 import (
    compute_gate,
    fit_standardizer,
    regression_metrics,
    select_alpha,
)
from summarize_cross_variable_lag_l0 import recompute_metric_frames


def make_clean_frame(rows: int) -> pd.DataFrame:
    index = pd.date_range("2014-01-01", periods=rows, freq="h")
    base = np.arange(rows, dtype=np.float64)
    return pd.DataFrame(
        {
            TARGET_COLUMN: 20.0 + base,
            "NO2": 30.0 + base * 0.1,
            "CO": 1.0 + base * 0.01,
            "O3": 40.0 + base * 0.2,
            "SO2": 10.0 + base * 0.05,
            "temperature": 15.0 + base * 0.01,
            "pressure_hPa_fixed": 1000.0 + base * 0.001,
            "humidity_clean": 50.0 + base * 0.01,
        },
        index=index,
    )


def make_raw_csv(path: Path, rows: int = 100) -> None:
    times = pd.date_range("2014-01-01", periods=rows, freq="h")
    values = np.arange(rows, dtype=np.float64)
    frame = pd.DataFrame(
        {
            "station_id": 1013,
            "time": times,
            TARGET_COLUMN: 20.0 + values,
            "PM10_Concentration": 100.0 + values,
            "NO2_Concentration": 30.0 + values,
            "CO_Concentration": 1.0 + values * 0.01,
            "O3_Concentration": 40.0 + values,
            "SO2_Concentration": 10.0 + values,
            "weather": 0,
            "temperature": 15.0 + values * 0.01,
            "pressure": 1000.0 + values * 0.01,
            "humidity": 50.0,
            "wind_speed": 3.0,
            "wind_direction": 2.0,
        }
    )
    selection_stop = int(rows * 0.8)
    frame.loc[selection_stop:, TARGET_COLUMN] = 999999.0
    frame.to_csv(path, index=False)


class L0CleaningTests(unittest.TestCase):
    def test_pressure_conversion_is_exact_and_thresholded(self):
        converted = convert_pressure_to_hpa(pd.Series([750.0, 900.0, np.inf]))
        self.assertAlmostEqual(converted.iloc[0], 750.0 * 1.33322, places=10)
        self.assertEqual(converted.iloc[1], 900.0)
        self.assertTrue(np.isnan(converted.iloc[2]))

    def test_official_8760_boundaries_are_exact_70_10_20(self):
        timestamps = pd.date_range("2014-01-01", periods=8760, freq="h")
        boundaries = establish_official_boundaries(timestamps)
        self.assertEqual(boundaries.train_row_stop, 6132)
        self.assertEqual(boundaries.selection_row_stop, 7008)
        self.assertEqual(boundaries.selection_row_stop - boundaries.train_row_stop, 876)
        self.assertEqual(boundaries.total_timestamp_rows - boundaries.selection_row_stop, 1752)

    def test_causal_ffill_is_limited_to_six_and_never_bfills(self):
        source = pd.Series([np.nan, 1.0] + [np.nan] * 7 + [2.0])
        filled = causal_ffill(source, limit=6)
        self.assertTrue(np.isnan(filled.iloc[0]))
        self.assertTrue((filled.iloc[2:8] == 1.0).all())
        self.assertTrue(np.isnan(filled.iloc[8]))
        self.assertEqual(filled.iloc[9], 2.0)

    def test_p7_cleaning_rules_order_and_split_isolation(self):
        timestamps = pd.date_range("2014-01-01", periods=100, freq="h")
        boundaries = establish_official_boundaries(timestamps)
        raw = pd.DataFrame(
            {
                TARGET_COLUMN: np.full(80, 25.0),
                "NO2_Concentration": np.full(80, 30.0),
                "CO_Concentration": np.full(80, 1.0),
                "O3_Concentration": np.full(80, 40.0),
                "SO2_Concentration": np.full(80, 10.0),
                "temperature": np.full(80, 20.0),
                "pressure": np.full(80, 1000.0),
                "humidity": np.full(80, 50.0),
            },
            index=timestamps[:80],
        )
        raw.iloc[5, raw.columns.get_loc("NO2_Concentration")] = 0.0
        raw.iloc[6, raw.columns.get_loc("CO_Concentration")] = -1.0
        raw.iloc[7, raw.columns.get_loc("O3_Concentration")] = 0.0
        raw.iloc[8, raw.columns.get_loc("SO2_Concentration")] = -1.0
        raw.iloc[9, raw.columns.get_loc("temperature")] = 61.0
        raw.iloc[10, raw.columns.get_loc("pressure")] = 750.0
        raw.iloc[11, raw.columns.get_loc("humidity")] = 101.0
        raw.iloc[0, raw.columns.get_loc("pressure")] = 600.0
        raw.iloc[boundaries.train_row_stop, raw.columns.get_loc("humidity")] = 0.0
        cleaned, ledger, quality = clean_selection_frame(
            raw, boundaries, L0TaskConfig(history=24, horizon=1)
        )
        self.assertEqual(tuple(cleaned.columns), (TARGET_COLUMN, *AUXILIARY_COLUMNS))
        self.assertEqual(cleaned.iloc[5]["NO2"], 30.0)
        self.assertEqual(cleaned.iloc[6]["CO"], 1.0)
        self.assertEqual(cleaned.iloc[7]["O3"], 40.0)
        self.assertEqual(cleaned.iloc[8]["SO2"], 10.0)
        self.assertEqual(cleaned.iloc[9]["temperature"], 20.0)
        self.assertAlmostEqual(
            cleaned.iloc[10]["pressure_hPa_fixed"], 750.0 * 1.33322
        )
        self.assertEqual(cleaned.iloc[11]["humidity_clean"], 50.0)
        self.assertTrue(np.isnan(cleaned.iloc[0]["pressure_hPa_fixed"]))
        self.assertTrue(
            np.isnan(cleaned.iloc[boundaries.train_row_stop]["humidity_clean"])
        )
        self.assertEqual(len(ledger), 14)
        self.assertEqual(
            list(
                row["variable"]
                for row in quality
                if row["split"] == "train" and row["variable"] in AUXILIARY_COLUMNS
            ),
            list(AUXILIARY_COLUMNS),
        )

    def test_loader_does_not_load_or_construct_test_features(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "station.csv"
            make_raw_csv(path)
            original_read_csv = pd.read_csv
            calls = []

            def recording_read_csv(*args, **kwargs):
                calls.append(dict(kwargs))
                return original_read_csv(*args, **kwargs)

            with patch("cross_variable_lag_l0_data.pd.read_csv", side_effect=recording_read_csv):
                selection = load_selection_data(
                    path, L0TaskConfig(history=24, horizon=1)
                )
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0]["usecols"], ["time"])
            self.assertEqual(calls[1]["nrows"], 80)
            self.assertEqual(selection.access_metadata["test_feature_rows_loaded"], 0)
            self.assertEqual(selection.access_metadata["test_samples_constructed"], 0)
            self.assertLess(selection.frame[TARGET_COLUMN].max(), 999999.0)
            self.assertTrue(set(PROHIBITED_COLUMNS).isdisjoint(selection.frame.columns))


class L0SplitAndFeatureTests(unittest.TestCase):
    def test_feature_shapes_and_lag_zero_means_t_minus_one(self):
        frame = make_clean_frame(300)
        window = build_window_data(frame, L0TaskConfig(history=24, horizon=1))
        self.assertEqual(window.features["R_self"].shape[1], 12)
        self.assertEqual(window.features["R_recent"].shape[1], 26)
        self.assertEqual(window.features["R_lag"].shape[1], 96)
        origin = int(window.origin_rows[0])
        self.assertEqual(window.features["R_self"][0, 0], frame.iloc[origin - 1][TARGET_COLUMN])
        self.assertEqual(
            window.features["R_self"][0, 1],
            frame.iloc[origin - 1][TARGET_COLUMN] - frame.iloc[origin - 2][TARGET_COLUMN],
        )
        self.assertEqual(window.features["R_lag"][0, 12], frame.iloc[origin - 1]["NO2"])
        self.assertEqual(
            window.features["R_lag"][0, 13],
            frame.iloc[origin - 1]["NO2"] - frame.iloc[origin - 2]["NO2"],
        )

    def test_outer_inner_target_sets_disjoint_and_embargo_exact(self):
        frame = make_clean_frame(900)
        for history, horizon, expected in ((168, 6, 5), (24, 1, 0)):
            window = build_window_data(
                frame, L0TaskConfig(history=history, horizon=horizon)
            )
            splits = build_l0_splits(window)
            self.assertEqual(len(splits.boundary_checks), 17)
            self.assertTrue(
                all(
                    row["target_time_intersection_count"] == 0
                    and row["assertion_passed"]
                    for row in splits.boundary_checks
                )
            )
            self.assertTrue(
                all(
                    row["embargo_origin_count"] == expected
                    for row in splits.boundary_checks
                )
            )
            for fold in splits.folds:
                self.assertEqual(len(fold.outer_embargo_origin_rows), expected)
                self.assertEqual(len(fold.inner_embargo_indices), expected)

    def test_standardizer_is_fit_on_training_slice_only(self):
        training = np.array([[0.0], [2.0]])
        validation = np.array([[1000.0]])
        scaler = fit_standardizer(training)
        self.assertEqual(float(scaler.mean[0]), 1.0)
        self.assertEqual(float(scaler.scale[0]), 1.0)
        self.assertEqual(float(scaler.transform(validation)[0, 0]), 999.0)


class L0GateAndRecomputeTests(unittest.TestCase):
    def test_alpha_tie_uses_larger_value(self):
        self.assertEqual(select_alpha({0.1: 2.0, 1.0: 2.0, 10.0: 3.0}), 1.0)

    def test_gate_calculation_pass_and_stop(self):
        self_values = np.full(5, 10.0)
        recent_values = np.full(5, 9.95)
        passing_lag = np.full(5, 9.90)
        passed = compute_gate(self_values, recent_values, passing_lag)
        self.assertEqual(passed["status"], "PASS")
        self.assertAlmostEqual(
            passed["observed"]["lag_vs_self_mean_improvement_percent"], 1.0
        )
        stopped = compute_gate(self_values, recent_values, np.full(5, 9.94))
        self.assertEqual(stopped["status"], "STOP")

    def test_result_metrics_are_recomputed_from_predictions(self):
        rows = []
        for fold in range(1, 6):
            for model, offset in (("R_self", 1.0), ("R_recent", 0.5), ("R_lag", 0.0)):
                for origin in range(3):
                    for horizon_step in (1, 2):
                        actual = float(origin + horizon_step)
                        rows.append(
                            {
                                "task": "168h_6h",
                                "fold": fold,
                                "model": model,
                                "selected_alpha": 1.0,
                                "origin_time": f"2014-01-{origin + 1:02d}",
                                "horizon_step": horizon_step,
                                "actual": actual,
                                "prediction": actual + offset,
                            }
                        )
        horizon, all_metrics = recompute_metric_frames(pd.DataFrame(rows))
        self.assertEqual(len(horizon), 5 * 3 * 2)
        self.assertEqual(len(all_metrics), 5 * 3)
        lag_rmse = all_metrics.loc[all_metrics["model"] == "R_lag", "rmse_all"]
        self.assertTrue((lag_rmse == 0.0).all())
        self.assertAlmostEqual(
            regression_metrics(np.array([1.0]), np.array([2.0]))["rmse"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
