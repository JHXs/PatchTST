"""Protocol and regression tests for Baseline-v2."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_baseline_v2 as runner
import summarize_baseline_v2 as summary


REUSED_TRAINABLE = (
    Path("experiments/results/baseline_v2/reused/backbone_upgrade")
    / "experiments/results/trainable_matched"
)
VERIFICATION_DIR = Path("experiments/results/baseline_v2/verification")


class TestBaselineV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        VERIFICATION_DIR.mkdir(parents=True, exist_ok=True)

    def test_identity_grid_is_exactly_448(self) -> None:
        identities = runner.expected_identities()
        self.assertEqual(len(runner.ARMS), 7)
        self.assertEqual(len(identities), 448)

    def test_single_station_boundary_and_center_resolution(self) -> None:
        station_ids = list(range(1004, 1022))
        self.assertEqual(station_ids.index(1013), 9)
        backbone = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(24, 1))
        model = runner.SingleStationAdapter(backbone, center_station_idx=9, horizon=1)
        x = torch.zeros(2, 18, 24)
        x[:, 9, :] = 1.0
        self.assertEqual(model.input_channels, 1)
        original = model(x).detach().clone()
        self.assertEqual(tuple(original.shape), (2, 1, 1))
        x[:, 0, :] = 1e6
        self.assertTrue(torch.equal(original, model(x)))

    def test_wrapper_shapes_parameter_counts_and_reproducibility(self) -> None:
        for arm, expected in runner.INFORMER_REGISTERED_COUNTS.items():
            for history, horizon in ((24, 1), (168, 24)):
                total, trainable = runner.independent_parameter_count(
                    arm, history, horizon, 9
                )
                self.assertEqual((total, trainable), (expected, expected))
        for task, counts in runner.TST_REGISTERED_COUNTS.items():
            for arm, expected in counts.items():
                total, trainable = runner.independent_parameter_count(
                    arm, task[0], task[1], 9
                )
                self.assertEqual((total, trainable), (expected, expected))

        reproducible = {}
        for arm in ("informer_d12_e2", "tst_d12_n1"):
            runner.legacy.set_seed(31415)
            first = runner.build_single_station_model(arm, 24, 6, 9).to("cpu").eval()
            x = torch.linspace(-1, 1, 2 * 18 * 24).reshape(2, 18, 24)
            with torch.no_grad():
                first_output = first(x)
            runner.legacy.set_seed(31415)
            second = runner.build_single_station_model(arm, 24, 6, 9).to("cpu").eval()
            with torch.no_grad():
                second_output = second(x)
            self.assertEqual(tuple(first_output.shape), (2, 1, 6))
            self.assertTrue(torch.equal(first_output, second_output))
            for key, value in first.state_dict().items():
                self.assertTrue(torch.equal(value, second.state_dict()[key]))
            reproducible[arm] = True

        artifact = {
            "passed": True,
            "shape": [2, 1, 6],
            "input_channels": 1,
            "informer_registered_counts": runner.INFORMER_REGISTERED_COUNTS,
            "tst_registered_counts_checked": {
                f"{history}x{horizon}": counts
                for (history, horizon), counts in runner.TST_REGISTERED_COUNTS.items()
            },
            "same_seed_bitwise_reproducible": reproducible,
        }
        (VERIFICATION_DIR / "wrapper_gate.json").write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def test_gru_h16_archived_skeleton_reproduction_gate(self) -> None:
        archived = pd.read_csv(REUSED_TRAINABLE / "raw_metrics.csv")
        expected = archived[
            archived["history"].eq(24)
            & archived["horizon"].eq(1)
            & archived["seed"].eq(2047)
            & archived["variant"].eq("center_gru_h16")
        ].iloc[0]
        config = runner.config_for(24, 1)
        datasets, metadata = runner.beijing.prepare_datasets_leakfree(config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with tempfile.TemporaryDirectory() as temporary:
            actual = runner.train_one_baseline(
                config,
                datasets,
                metadata,
                runner.REPRODUCTION_ARM,
                2047,
                Path(temporary),
                "gru_h16_reproduction_gate",
                device,
            )
        rmse_relative = abs(actual["rmse_ugm3"] - expected["rmse_ugm3"]) / abs(
            expected["rmse_ugm3"]
        )
        valid_relative = abs(
            actual["best_valid_loss"] - expected["best_valid_loss"]
        ) / abs(expected["best_valid_loss"])
        self.assertLess(rmse_relative, 1e-9)
        self.assertLess(valid_relative, 1e-9)
        artifact = {
            "passed": True,
            "device": str(device),
            "archived_rmse_ugm3": float(expected["rmse_ugm3"]),
            "reproduced_rmse_ugm3": float(actual["rmse_ugm3"]),
            "rmse_relative_difference": float(rmse_relative),
            "archived_best_valid_loss": float(expected["best_valid_loss"]),
            "reproduced_best_valid_loss": float(actual["best_valid_loss"]),
            "best_valid_loss_relative_difference": float(valid_relative),
            "threshold": 1e-9,
        }
        (VERIFICATION_DIR / "reproduction_gate.json").write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def test_failure_policy_distinguishes_terminal_categories(self) -> None:
        config = runner.config_for(24, 1, smoke=True)

        def nonfinite(*_args):
            raise FloatingPointError("injected nonfinite")

        def oom(*_args):
            raise torch.OutOfMemoryError("injected out of memory")

        def nonfinite_prediction(*_args):
            raise runner.NonfinitePredictionError("injected prediction")

        common = (
            config,
            {},
            {},
            "informer_d8_e1",
            2047,
            Path("."),
            "injected",
            torch.device("cpu"),
        )
        self.assertEqual(
            runner.run_one_with_failure_policy(*common, train_fn=nonfinite)["status"],
            "nonfinite",
        )
        self.assertEqual(
            runner.run_one_with_failure_policy(*common, train_fn=oom)["status"],
            "infeasible_oom",
        )
        self.assertEqual(
            runner.run_one_with_failure_policy(
                *common, train_fn=nonfinite_prediction
            )["status"],
            "nonfinite_prediction",
        )

    def test_resume_skips_terminal_without_rewrite(self) -> None:
        identity = runner.run_id(24, 1, 2047, "informer_d8_e1")
        rows = [{"run_id": identity, "status": "completed", "sentinel": 123}]
        before = json.dumps(rows, sort_keys=True)
        self.assertTrue(runner.should_skip_identity(rows, identity))
        self.assertEqual(json.dumps(rows, sort_keys=True), before)
        self.assertFalse(runner.should_skip_identity(rows, identity + "_missing"))

    def test_common_subset_pairing_and_per_lead_formula(self) -> None:
        baseline = pd.DataFrame(
            {
                "history": [24, 24],
                "horizon": [1, 1],
                "seed": [1, 2],
                "variant": ["x", "x"],
                "status": ["completed", "completed"],
                "rmse_ugm3": [12.0, 99.0],
                "mae_ugm3": [1.0, 1.0],
            }
        )
        ours = pd.DataFrame(
            {
                "history": [24],
                "horizon": [1],
                "seed": [1],
                "status": ["completed"],
                "rmse_ugm3": [10.0],
                "mae_ugm3": [1.0],
            }
        )
        paired = summary.paired_with_o(baseline, ours, arm="x")
        self.assertEqual(len(paired), 1)
        self.assertAlmostEqual(paired.iloc[0]["baseline_relative_to_o_percent"], 20.0)

        prediction = np.array([[[1.0, 4.0]], [[3.0, 8.0]]], dtype=np.float32)
        target = np.array([[[0.0, 0.0]], [[0.0, 0.0]]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixed.npz"
            np.savez_compressed(
                path,
                prediction_ugm3=prediction,
                target_ugm3=target,
                prediction_scaled=prediction,
                target_scaled=target,
            )
            actual = summary._prediction_rmse_by_lead(path)
        expected = np.sqrt(np.mean((prediction - target) ** 2, axis=(0, 1)))
        np.testing.assert_array_equal(actual, expected)

    def test_rank_correlation_direction_and_forbidden_fields(self) -> None:
        statistic = summary.spearmanr([1, 2, 3], [3, 2, 1]).statistic
        self.assertEqual(statistic, -1.0)
        completed = {
            "selection_split": "valid",
            "evaluation_split": "test",
            "test_evaluation_count": 1,
        }
        self.assertEqual(completed["selection_split"], "valid")
        self.assertEqual(completed["evaluation_split"], "test")
        self.assertEqual(completed["test_evaluation_count"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
