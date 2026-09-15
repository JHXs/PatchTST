"""Unit tests for the Stage 0 synthetic witness (no station data involved)."""

from __future__ import annotations

import unittest

import numpy as np
import torch

import run_wcs_stage0 as stage0


class Stage0ProtocolTests(unittest.TestCase):
    def test_dgp_signs_are_opposite_and_targets_have_correct_monotonicity(self) -> None:
        state, wind = stage0.generate_series(1234, length=4000)
        idx = np.arange(stage0.LOOKBACK, len(state))
        monotone = stage0.true_conditional_mean(state, wind, idx, sign=+1.0)[:, 0]
        anti = stage0.true_conditional_mean(state, wind, idx, sign=-1.0)[:, 0]
        # Within a narrow wind band, compare conditional means at fixed context by construction:
        # the difference between the two DGPs equals 2 * A(x) * g(w) >= 0 pointwise.
        x = np.stack([state[idx], state[idx - 1], wind[idx - 1]], axis=1)
        amplitude = np.log1p(np.exp(x @ stage0.GAMMA))
        shape = 1.0 - np.exp(-stage0.LAMBDA * wind[idx])
        self.assertTrue(np.allclose(anti - monotone, 2 * amplitude * shape))
        self.assertTrue(np.all(amplitude * shape >= 0))

    def test_candidate_is_strictly_decreasing_and_saturating(self) -> None:
        model = stage0.build_arm("wcs_neg", 1, seed=7)
        rng = np.random.default_rng(0)
        x = torch.tensor(rng.normal(0, 1, size=(1, stage0.LOOKBACK * 2)), dtype=torch.float32)
        grid = torch.tensor(np.linspace(0, 12, 40, dtype=np.float32)).unsqueeze(1)
        with torch.no_grad():
            y = np.array([float(model(x, w)) for w in grid])
        diffs = np.diff(y)
        self.assertTrue(np.all(diffs < 0), "candidate must be strictly decreasing in wind speed")
        self.assertTrue(np.all(np.abs(diffs)[1:] <= np.abs(diffs)[:-1] + 1e-9), "marginal effect must shrink (saturation)")

    def test_wrong_sign_arm_increases_in_wind(self) -> None:
        model = stage0.build_arm("wcs_pos", 1, seed=11)
        x = torch.zeros(1, stage0.LOOKBACK * 2)
        grid = torch.tensor(np.linspace(0, 12, 25, dtype=np.float32)).unsqueeze(1)
        with torch.no_grad():
            y = np.array([float(model(x, w)) for w in grid])
        self.assertTrue(np.all(np.diff(y) > 0))

    def test_control_has_at_least_as_many_parameters(self) -> None:
        for horizon in stage0.HORIZONS:
            control = stage0.count_parameters(stage0.build_arm("uncon_add", horizon, 0))
            candidate = stage0.count_parameters(stage0.build_arm("wcs_neg", horizon, 0))
            self.assertGreaterEqual(control, candidate, f"H={horizon}: control must not be smaller")

    def test_time_split_prevents_leakage(self) -> None:
        data = stage0.build_dataset(3, +1.0, 1)
        self.assertEqual(data.train_windows.shape[1], data.test_windows.shape[1])
        tr_idx, te_idx = stage0.setting_indices(data.train_w, data.test_w, "S-shift", 3)
        self.assertTrue(np.all(data.train_w[tr_idx] <= stage0.SHIFT_TRAIN_MAX_W))
        self.assertTrue(np.all(data.test_w[te_idx] > stage0.SHIFT_TEST_MIN_W))
        self.assertGreater(int(data.test_end_indices.min()), int(data.test_end_indices.min()) - 1)
        # the test segment lies strictly after the training segment in time
        train_last_end = stage0.LOOKBACK + int(stage0.TRAIN_FRACTION * (20000 - stage0.LOOKBACK)) - 1
        self.assertTrue(np.all(data.test_end_indices > train_last_end - 1))

    def test_small_setting_uses_a_subset_of_the_training_segment(self) -> None:
        data = stage0.build_dataset(5, +1.0, 1)
        tr_idx, te_idx = stage0.setting_indices(data.train_w, data.test_w, "S-small", 5)
        self.assertLess(len(tr_idx), len(data.train_w))
        self.assertEqual(len(te_idx), len(data.test_w))

    def test_short_run_reproduces_metric_from_predictions(self) -> None:
        store: dict = {}
        record = stage0.run_single(0, +1.0, 1, "S-small", "all", prediction_store=store)
        data = stage0.build_dataset(0, +1.0, 1)
        picks = [int(np.where(np.isclose(data.test_end_indices, end))[0][0]) for end in record["test_end_indices"]]
        targets = data.test_y[np.asarray(picks, dtype=int)]
        pred = store["S-small|wcs_neg"]
        rmse = float(np.sqrt(np.mean((pred - targets) ** 2)))
        self.assertAlmostEqual(rmse, record["arms"]["wcs_neg"]["rmse"], places=6)


if __name__ == "__main__":
    unittest.main()
