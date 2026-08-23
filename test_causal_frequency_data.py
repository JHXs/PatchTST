"""Tests for the leak-safe causal-frequency data contract."""

from __future__ import annotations

import hashlib
import json
import unittest

import numpy as np
import pandas as pd

from causal_frequency_data import (
    CausalFrequencyDataConfig,
    causal_forward_fill_and_trim,
    prepare_causal_datasets,
    split_target_origins,
)


def make_synthetic_frame(row_count: int = 30) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=row_count, freq="h")
    center = np.arange(row_count, dtype=np.float64)
    train_boundary = int(np.floor(0.7 * row_count))
    future_only_neighbor = np.resize(
        np.array([3.0, -2.0, 4.0, -1.0, 0.5]), row_count
    )
    future_only_neighbor[train_boundary:] = center[train_boundary:]
    return pd.DataFrame(
        {
            1013: center,
            1002: future_only_neighbor,
            1001: center * 2 + 1,
        },
        index=index,
    )


class CausalFrequencyDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = CausalFrequencyDataConfig(
            center_station_id=1013,
            history=4,
            horizon=3,
            train_fraction=0.7,
            valid_fraction=0.1,
            correlation_threshold=0.85,
        )

    def test_target_start_split_exact_boundaries(self):
        origins = split_target_origins(30, 4, 3, 21, 24)
        np.testing.assert_array_equal(origins["train"], np.arange(4, 19))
        np.testing.assert_array_equal(origins["valid"], np.array([21]))
        np.testing.assert_array_equal(origins["test"], np.arange(24, 28))

    def test_target_timestamps_are_disjoint_and_windows_are_aligned(self):
        datasets, metadata = prepare_causal_datasets(
            self.config, make_synthetic_frame()
        )
        target_sets = {
            split: set(dataset.target_timestamp_matrix_ns().reshape(-1).tolist())
            for split, dataset in datasets.items()
        }
        self.assertFalse(target_sets["train"] & target_sets["valid"])
        self.assertFalse(target_sets["train"] & target_sets["test"])
        self.assertFalse(target_sets["valid"] & target_sets["test"])
        self.assertTrue(metadata["target_timestamp_intersections_empty"])

        valid_x, valid_y = datasets["valid"][0]
        self.assertEqual(tuple(valid_x.shape), (2, 4))
        self.assertEqual(tuple(valid_y.shape), (1, 3))
        center_idx = metadata["center_station_idx"]
        restored_x = (
            valid_x[center_idx].numpy() * metadata["center_std"]
            + metadata["center_mean"]
        )
        restored_y = (
            valid_y[0].numpy() * metadata["center_std"]
            + metadata["center_mean"]
        )
        np.testing.assert_allclose(restored_x, np.arange(17, 21), atol=1e-5)
        np.testing.assert_allclose(restored_y, np.arange(21, 24), atol=1e-5)

    def test_station_selection_scaler_and_q90_use_train_only(self):
        frame = make_synthetic_frame()
        datasets, metadata = prepare_causal_datasets(self.config, frame)
        mutated = frame.copy()
        mutated.iloc[21:, mutated.columns.get_loc(1013)] += 10000
        mutated.iloc[21:, mutated.columns.get_loc(1001)] -= 5000
        mutated.iloc[21:, mutated.columns.get_loc(1002)] += 8000
        mutated_datasets, mutated_metadata = prepare_causal_datasets(
            self.config, mutated
        )

        self.assertEqual(metadata["station_ids"], [1001, 1013])
        self.assertEqual(mutated_metadata["station_ids"], metadata["station_ids"])
        np.testing.assert_allclose(
            mutated_metadata["station_means"], metadata["station_means"]
        )
        np.testing.assert_allclose(
            mutated_metadata["station_stds"], metadata["station_stds"]
        )
        self.assertEqual(
            mutated_metadata["peak_q90_ugm3"], metadata["peak_q90_ugm3"]
        )
        expected_q90 = float(np.quantile(np.arange(4, 21), 0.9))
        self.assertAlmostEqual(
            metadata["peak_q90_ugm3"], expected_q90, delta=1e-5
        )

        original_train_x = datasets["train"][0][0].numpy()
        mutated_train_x = mutated_datasets["train"][0][0].numpy()
        np.testing.assert_array_equal(original_train_x, mutated_train_x)

    def test_causal_ffill_never_uses_future_and_trims_leading_prefix(self):
        index = pd.date_range("2021-01-01", periods=7, freq="h")
        frame = pd.DataFrame(
            {
                1001: [np.nan, np.nan, 2.0, 3.0, 4.0, 5.0, 6.0],
                1013: [0.0, 1.0, 2.0, np.nan, 100.0, 5.0, 6.0],
            },
            index=index,
        )
        filled, metadata = causal_forward_fill_and_trim(frame)
        self.assertEqual(metadata["leading_rows_trimmed"], 2)
        self.assertEqual(filled.index[0], index[2])
        self.assertEqual(filled.loc[index[3], 1013], 2.0)

        changed_future = frame.copy()
        changed_future.loc[index[4]:, 1013] += 9999
        changed, _ = causal_forward_fill_and_trim(changed_future)
        pd.testing.assert_frame_equal(
            filled.loc[:index[3]], changed.loc[:index[3]]
        )

    def test_deduplicate_sort_and_hourly_alignment_contract(self):
        index = pd.to_datetime(
            [
                "2022-01-01 02:00",
                "2022-01-01 00:00",
                "2022-01-01 02:00",
            ]
        )
        frame = pd.DataFrame(
            {1013: [2.0, 0.0, 999.0], 1001: [12.0, 10.0, 999.0]},
            index=index,
        )
        filled, metadata = causal_forward_fill_and_trim(frame)
        self.assertTrue(filled.index.is_monotonic_increasing)
        self.assertTrue(filled.index.is_unique)
        self.assertEqual(len(filled), 3)
        self.assertEqual(filled.iloc[1][1013], 0.0)
        self.assertEqual(filled.iloc[2][1013], 2.0)
        self.assertEqual(metadata["deduplicated_time_rows"], 1)
        self.assertEqual(metadata["inserted_missing_hour_rows"], 1)

    def test_timestamp_hash_matches_independent_little_endian_digest(self):
        datasets, metadata = prepare_causal_datasets(
            self.config, make_synthetic_frame()
        )
        matrix = datasets["train"].target_timestamp_matrix_ns()
        unique_ns = np.unique(matrix.reshape(-1))
        expected = hashlib.sha256(
            np.asarray(unique_ns, dtype="<i8").tobytes(order="C")
        ).hexdigest()
        self.assertEqual(
            metadata["target_timestamps"]["train"][
                "unique_timestamp_sha256"
            ],
            expected,
        )

    def test_metadata_is_json_finite_and_complete(self):
        _, metadata = prepare_causal_datasets(self.config, make_synthetic_frame())
        json.dumps(metadata, allow_nan=False)
        required = {
            "protocol_version",
            "train_boundary_row",
            "valid_boundary_row",
            "station_ids",
            "candidate_train_correlations",
            "station_means",
            "station_raw_stds_ddof0",
            "station_stds",
            "peak_q90_ugm3",
            "target_timestamps",
            "target_timestamp_intersections_empty",
        }
        self.assertTrue(required.issubset(metadata))
        self.assertEqual(metadata["standardization_ddof"], 0)
        self.assertIn("sorted unique train target", metadata["peak_q90_definition"])


if __name__ == "__main__":
    unittest.main()
