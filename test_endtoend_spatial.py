"""Acceptance and preflight fairness tests for the end-to-end comparison."""

from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import run_beijing_leakfree_coverage as beijing
import run_st_patchtst_ablation as legacy
from run_endtoend_spatial import (
    BACKBONES,
    EXPECTED_STATIONS,
    build_endtoend_model,
    expected_identities,
    run_one_with_failure_policy,
    spatial_parameter_count,
    train_one_baseline_gate,
    train_one_endtoend,
)
from run_trainable_matched_baselines import (
    TRAINING_SEMANTICS,
    build_single_station_model,
    config_for,
    parameter_counts,
    training_semantics_manifest,
)
from summarize_endtoend_spatial import recompute_prediction_metrics


PREFLIGHT_OUTPUT = Path("tables/endtoend_spatial/preflight_fairness_gates.json")
RUN_INTEGRATION = os.environ.get("RUN_ENDTOEND_FAIRNESS_INTEGRATION") == "1"


def relative_difference(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(abs(expected), 1e-12)


def compare_predictions(left: Path, right: Path) -> tuple[float, bool]:
    with np.load(left) as left_payload, np.load(right) as right_payload:
        left_prediction = left_payload["prediction_scaled"]
        right_prediction = right_payload["prediction_scaled"]
        left_target = left_payload["target_scaled"]
        right_target = right_payload["target_scaled"]
    scale = max(float(np.max(np.abs(left_prediction))), 1.0)
    relative = float(np.max(np.abs(left_prediction - right_prediction))) / scale
    return relative, bool(np.array_equal(left_target, right_target))


class EndToEndSpatialTests(unittest.TestCase):
    def test_registered_matrix_has_384_identities(self):
        self.assertEqual(len(expected_identities()), 384)

    def test_parameter_registration_is_exact(self):
        metadata = {"station_ids": list(range(1000, 1018)), "center_station_idx": 9}
        for horizon in (1, 6, 24):
            config = config_for(24, horizon)
            for family, hidden_size in BACKBONES:
                legacy.set_seed(2047)
                baseline = build_single_station_model(family, hidden_size, horizon, 9)
                baseline_total, baseline_trainable = parameter_counts(baseline)
                legacy.set_seed(2047)
                ours = build_endtoend_model(
                    config, metadata, family, hidden_size, 2047
                )
                ours_total, ours_trainable = parameter_counts(ours)
                backbone_count = sum(p.numel() for p in ours.patch_tst.parameters())
                head_count = spatial_parameter_count(ours)
                self.assertEqual(baseline_total, baseline_trainable)
                self.assertEqual(ours_total, ours_trainable)
                self.assertEqual(backbone_count, baseline_trainable)
                self.assertEqual(ours_trainable, baseline_trainable + head_count)

    def test_input_channel_boundary(self):
        torch.manual_seed(44)
        baseline = build_single_station_model("gru", 8, 3, center_station_idx=9)
        x = torch.randn(4, EXPECTED_STATIONS, 24)
        changed = x.clone()
        changed[:, :9] += 100.0
        changed[:, 10:] -= 100.0
        with torch.no_grad():
            original = baseline(x)
            perturbed = baseline(changed)
        self.assertEqual(baseline.input_channels, 1)
        self.assertTrue(torch.equal(original, perturbed))

        config = config_for(24, 3)
        metadata = {
            "station_ids": list(range(1000, 1018)),
            "center_station_idx": 9,
        }
        ours = build_endtoend_model(config, metadata, "gru", 8, 2047)
        self.assertEqual(ours.input_channels, EXPECTED_STATIONS)
        self.assertEqual(ours.model.num_stations, EXPECTED_STATIONS)
        self.assertEqual(tuple(ours(x).shape), (4, 1, 3))

    def test_no_test_peeking_manifest(self):
        for history, horizon in ((24, 1), (168, 24)):
            config = config_for(history, horizon, smoke=True)
            manifest = training_semantics_manifest(config, smoke=True)
            self.assertEqual(manifest["selection_split"], "valid")
            self.assertEqual(manifest["evaluation_split"], "test")
            self.assertEqual(manifest["optimizer"], TRAINING_SEMANTICS["optimizer"])
            self.assertEqual(config.evaluation_split, "test")

    def test_independent_recalculation_exact(self):
        target = np.array([[[2.0, 5.0]], [[7.0, 11.0]]], dtype=np.float32)
        prediction = target + np.float32(0.125)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prediction.npz"
            np.savez_compressed(
                path, target_ugm3=target, prediction_ugm3=prediction
            )
            metrics = recompute_prediction_metrics(path)
        self.assertEqual(metrics["rmse_ugm3"], 0.125)
        self.assertEqual(metrics["mae_ugm3"], 0.125)

    def test_nonfinite_retry_policy(self):
        config = legacy.ExperimentConfig(history=4, horizon=2)
        calls = []

        def always_nonfinite(config_arg, *args):
            calls.append(config_arg.learning_rate)
            raise FloatingPointError("nan")

        result = run_one_with_failure_policy(
            config,
            {},
            {},
            "gru",
            8,
            2047,
            Path("unused"),
            "nonfinite",
            torch.device("cpu"),
            train_fn=always_nonfinite,
        )
        self.assertEqual(result["status"], "nonfinite")
        self.assertEqual(calls, [config.learning_rate, config.learning_rate * 0.1])

    @unittest.skipUnless(
        RUN_INTEGRATION,
        "set RUN_ENDTOEND_FAIRNESS_INTEGRATION=1 for two-config preflight",
    )
    def test_five_preflight_fairness_gates(self):
        # The equivalence gate audits implementation semantics, so keep it on
        # deterministic CPU kernels.  ROCm RNN kernels can diverge across two
        # otherwise identical independent trainings and would confound this gate.
        device = torch.device("cpu")
        zero_details = []
        recalculation_differences = []
        channel_checks = []
        parameter_rows = []
        no_peeking_checks = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for history, horizon in ((24, 1), (168, 24)):
                config = config_for(history, horizon, smoke=True)
                datasets, metadata = beijing.prepare_datasets_leakfree(config)
                self.assertEqual(len(metadata["station_ids"]), EXPECTED_STATIONS)
                baseline_identity = f"gate_{history}h_{horizon}h_b"
                ours_identity = f"gate_{history}h_{horizon}h_o_zero"
                baseline = train_one_baseline_gate(
                    config,
                    datasets,
                    metadata,
                    "gru",
                    8,
                    2047,
                    root,
                    baseline_identity,
                    device,
                )
                ours = train_one_endtoend(
                    config,
                    datasets,
                    metadata,
                    "gru",
                    8,
                    2047,
                    root,
                    ours_identity,
                    device,
                    spatial_enabled=False,
                )
                prediction_relative, targets_equal = compare_predictions(
                    root / baseline["prediction_file"],
                    root / ours["prediction_file"],
                )
                valid_relative = relative_difference(
                    ours["best_valid_loss"], baseline["best_valid_loss"]
                )
                rmse_relative = relative_difference(
                    ours["rmse_ugm3"], baseline["rmse_ugm3"]
                )
                b_state = torch.load(
                    root / baseline["checkpoint_file"],
                    map_location="cpu",
                    weights_only=True,
                )
                o_state = torch.load(
                    root / ours["checkpoint_file"],
                    map_location="cpu",
                    weights_only=True,
                )
                o_backbone = {
                    key[len("model.patch_tst."):]: value
                    for key, value in o_state.items()
                    if key.startswith("model.patch_tst.")
                }
                states_equal = set(b_state) == set(o_backbone) and all(
                    torch.equal(value, o_backbone[key])
                    for key, value in b_state.items()
                )
                passed = bool(
                    targets_equal
                    and states_equal
                    and prediction_relative <= 1e-6
                    and valid_relative <= 1e-6
                    and rmse_relative <= 1e-6
                )
                zero_details.append(
                    {
                        "history": history,
                        "horizon": horizon,
                        "prediction_relative_difference": prediction_relative,
                        "valid_loss_relative_difference": valid_relative,
                        "rmse_relative_difference": rmse_relative,
                        "targets_equal": targets_equal,
                        "backbone_states_bitwise_equal": states_equal,
                        "passed": passed,
                    }
                )
                self.assertTrue(passed, zero_details[-1])

                for result in (baseline, ours):
                    metrics = recompute_prediction_metrics(
                        root / result["prediction_file"]
                    )
                    difference = abs(metrics["rmse_ugm3"] - result["rmse_ugm3"])
                    recalculation_differences.append(difference)
                    self.assertLess(difference, 1e-9)
                channel_checks.append(
                    baseline["input_channels"] == 1
                    and ours["input_channels"] == EXPECTED_STATIONS
                )
                parameter_rows.append(
                    ours["trainable_parameter_count"]
                    == baseline["trainable_parameter_count"]
                    + ours["spatial_head_parameter_count"]
                )
                no_peeking_checks.append(
                    baseline["selection_split"] == "valid"
                    and ours["selection_split"] == "valid"
                    and baseline["evaluation_split"] == "test"
                    and ours["evaluation_split"] == "test"
                    and baseline["test_evaluation_count"] == 1
                    and ours["test_evaluation_count"] == 1
                )
                del datasets
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        gates = [
            {
                "gate": "zero_spatial_equivalence",
                "passed": all(row["passed"] for row in zero_details),
                "detail": json.dumps(zero_details, ensure_ascii=False),
            },
            {
                "gate": "parameter_registration",
                "passed": all(parameter_rows),
                "detail": f"checks={len(parameter_rows)}",
            },
            {
                "gate": "input_channel_boundary",
                "passed": all(channel_checks),
                "detail": "B=1; O=18",
            },
            {
                "gate": "no_test_peeking",
                "passed": all(no_peeking_checks),
                "detail": "selection=valid; test traversal count=1",
            },
            {
                "gate": "independent_recalculation",
                "passed": max(recalculation_differences, default=math.inf) < 1e-9,
                "detail": (
                    f"max_rmse_abs_diff={max(recalculation_differences, default=math.inf):.3e}"
                ),
            },
        ]
        PREFLIGHT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        PREFLIGHT_OUTPUT.write_text(
            json.dumps(
                {
                    "device": str(device),
                    "configs": ["24x1", "168x24"],
                    "seed": 2047,
                    "backbone": "gru_h8",
                    "relative_tolerance": 1e-6,
                    "gates": gates,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self.assertTrue(all(gate["passed"] for gate in gates), gates)


if __name__ == "__main__":
    unittest.main()
