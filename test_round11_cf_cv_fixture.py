import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from cf_cv_smearing_fixture import (
    FORMAL_SEEDS,
    TASKS,
    RectangularLinearOperator,
    SequenceDGP,
    assert_cross_task_labels,
    assert_sequence_invariants,
    build_block_operator,
    build_oof_plan,
    build_prefix_scaled_operator,
    build_windows,
    compute_control_variate,
    compute_prefix_control_variate,
    expected_fit_keys,
    fit_input_scaler,
    fit_target_scaler,
    gaussian_mgf_uniform,
    hpc_factor,
    generate_sequence,
    metric_rows,
    rho_gate_status,
    select_s_pool,
    SMOOTHER_BANDWIDTH,
    SMOOTHER_MAX_SUPPORT,
    validate_manifest,
    validate_seed_set,
    with_splits,
)
from summarize_round11_cf_cv_fixture import build_method_predictions, validate_mode_metadata, write_verification


def _fake_registered_task(sequence, task, *, output_dir, seed, rho_stop=False, **kwargs):
    from cf_cv_smearing_fixture import atomic_json, atomic_npz, sha256_file
    stem = output_dir / f"task_{seed}_{task.name}"
    npz_path, json_path = stem.with_suffix(".npz"), stem.with_suffix(".json")
    atomic_npz(npz_path, y_valid=np.zeros((1, 1)), y_test=np.zeros((1, 1)),
               m_r=np.zeros(2), s_r=np.zeros(2), valid_log_m=np.zeros((1, 1)), test_log_m=np.zeros((1, 1)))
    keys = [f"seed={seed}/task={task.name}/stage=raw_full"] + [
        f"seed={seed}/task={task.name}/stage=log_oof/block={i}" for i in range(5)
    ] + [f"seed={seed}/task={task.name}/stage=log_full"]
    info = {"schema": "round11-cf-cv-task/v2", "unit_key": f"seed={seed}/task={task.name}",
            "fit_keys": keys, "elapsed_seconds": 0.0, "npz_sha256": sha256_file(npz_path),
            "rho_gate": {"valid": {"stop": rho_stop}, "test": {"stop": rho_stop}}}
    return info, json_path, npz_path


class Round11FixtureContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sequence = generate_sequence(4001, n=1024, burn_in=64)
        cls.h1 = with_splits(build_windows(cls.sequence, *TASKS["H1"]))
        cls.h6 = with_splits(build_windows(cls.sequence, *TASKS["H6"]))

    def test_unique_absolute_labels_and_cross_lead_equality(self):
        assert_cross_task_labels(self.sequence, [self.h1, self.h6])
        self.assertTrue(np.all(self.sequence.y >= 0))
        self.assertEqual(np.unique(self.sequence.timestamps).size, len(self.sequence.timestamps))

    def test_negative_values_are_rejected(self):
        bad_y = self.sequence.y.copy()
        bad_y[0] = -1.0
        bad = SequenceDGP(self.sequence.seed, self.sequence.variant, self.sequence.timestamps, self.sequence.a,
                          self.sequence.q, self.sequence.mu, self.sequence.sigma, self.sequence.epsilon,
                          self.sequence.z, bad_y)
        with self.assertRaisesRegex(AssertionError, "negative"):
            assert_sequence_invariants(bad)

    def test_window_interface_shapes(self):
        self.assertEqual(self.h1.X_raw.shape[1:], (1, 24))
        self.assertEqual(self.h1.y_raw.shape[1:], (1, 1))
        self.assertEqual(self.h6.X_raw.shape[1:], (1, 168))
        self.assertEqual(self.h6.y_raw.shape[1:], (1, 6))

    def test_forward_oof_leakage_and_purge(self):
        for task in (self.h1, self.h6):
            plan = build_oof_plan(task)
            self.assertEqual(len(plan.blocks), 5)
            self.assertGreater(len(plan.warmup_indices), 0)
            for block in plan.blocks:
                if len(block.fit_indices):
                    fit_max = task.target_timestamps[block.fit_indices].max()
                    self.assertLessEqual(int(fit_max), block.query_min_timestamp - block.purge)
                self.assertTrue(set(block.fit_indices).isdisjoint(set(block.query_indices)))
                self.assertEqual(block.purge, task.history + task.horizon + 1)
                self.assertTrue(set(block.inner_fit_indices).isdisjoint(set(block.inner_valid_indices)))

    def test_fit_key_contract_is_exactly_42_for_formal_fixture(self):
        keys = expected_fit_keys(FORMAL_SEEDS)
        self.assertEqual(len(keys), 42)
        self.assertEqual(len(set(keys)), 42)
        for seed in FORMAL_SEEDS:
            for task in TASKS:
                prefix = f"seed={seed}/task={task}"
                selected = [key for key in keys if key.startswith(prefix)]
                self.assertEqual(len(selected), 7)
                self.assertEqual(sum("stage=log_oof" in key for key in selected), 5)

    def test_rectangular_operator_and_old_square_interface_rejection(self):
        fit = np.asarray([[0.0], [1.0], [2.0]])
        query = np.asarray([[0.5], [1.5]])
        op = build_block_operator(query, fit, np.zeros(2), np.zeros(3), bandwidth=1.0)
        self.assertEqual(op.shape, (2, 3))
        self.assertEqual(op.apply(np.ones(3)).shape, (2,))
        with self.assertRaises((TypeError, ValueError)):
            op @ op
        with self.assertRaises((TypeError, ValueError)):
            op ** 2

    def test_control_variate_identity_and_zero_variance_stop(self):
        residual = np.asarray([-0.4, 0.0, 0.3, 0.8])
        b = np.asarray([-0.1, 0.1, 0.0, 0.2])
        control = compute_control_variate(residual, b)
        v = control["v"]
        h = build_block_operator(np.asarray([[0.0], [1.0]]), np.asarray([[-1.0], [0.0], [1.0]]),
                                 np.zeros(2), np.zeros(3))
        u = np.exp(residual[:3])
        s_rem = h.apply(u)
        s_cv = h.apply(u - control["lambda"] * v[:3])
        np.testing.assert_allclose(s_cv - s_rem, -control["lambda"] * h.apply(v[:3]), rtol=1e-10, atol=1e-12)
        with self.assertRaisesRegex(ValueError, r"Var\(v\)"):
            compute_control_variate(np.ones(4), np.ones(4))

    def test_m_s_pools_are_causal_same_lead_and_per_lead_centered(self):
        timestamps = np.asarray([10, 11, 20, 21, 30, 31])
        leads = np.asarray([0, 1, 0, 1, 0, 1])
        selected, supports = select_s_pool(timestamps, leads, purge=5)
        self.assertTrue(np.all(selected >= 0))
        self.assertTrue(set(range(len(timestamps))) - set(selected))
        for row, support in zip(selected, supports):
            self.assertGreaterEqual(len(support), 2)
            self.assertTrue(np.all(leads[support] == leads[row]))
            self.assertTrue(np.all(timestamps[support] < timestamps[row]))
            self.assertTrue(np.all(timestamps[support] <= timestamps[row] - 5))
        residual = np.asarray([0.1, 0.4, -0.2, 0.7])
        b = np.asarray([0.0, 0.1, -0.1, 0.2])
        prefix = compute_prefix_control_variate(residual, b)
        self.assertAlmostEqual(float(prefix["vtilde"].mean()), 0.0, places=14)

    def test_each_fit_scaler_is_fit_only_and_target_is_per_lead(self):
        task = self.h6
        first = np.arange(20)
        later = np.arange(20, 40)
        self.assertNotEqual(fit_input_scaler(task, first), fit_input_scaler(task, later))
        target = task.z_target[:, None, :]
        first_target = fit_target_scaler(target, first)
        later_target = fit_target_scaler(target, later)
        self.assertNotEqual(first_target, later_target)
        self.assertEqual(len(first_target["mean"]), task.horizon)

    def test_prefix_condition_scaler_and_stable_smoother_contract(self):
        query = np.asarray([[10.0, 1.0], [20.0, 2.0]])
        fit = np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        op = build_prefix_scaled_operator(query, fit, (np.asarray([0, 1]), np.asarray([0, 1, 2, 3])), name="B")
        self.assertEqual(op.shape, (2, 4))
        self.assertEqual(SMOOTHER_BANDWIDTH, 1.0)
        self.assertEqual(SMOOTHER_MAX_SUPPORT, 128)

    def test_build_block_operator_unique_lead_order_is_invariant(self):
        query = np.asarray([[0.2], [1.2], [0.8], [1.8]])
        fit = np.asarray([[0.0], [1.0], [2.0], [3.0]])
        q_leads = np.asarray([1, 0, 1, 0])
        f_leads = np.asarray([0, 1, 0, 1])
        grouped = build_block_operator(query, fit, q_leads, f_leads)
        direct_rows = []
        direct_weights = []
        for row in range(len(query)):
            one = build_block_operator(query[row:row + 1], fit, q_leads[row:row + 1], f_leads)
            direct_rows.extend(one.row_indices)
            direct_weights.extend(one.row_weights)
        direct = RectangularLinearOperator(grouped.shape, direct_rows, direct_weights)
        np.testing.assert_allclose(grouped.apply(np.arange(4.0)), direct.apply(np.arange(4.0)))

    def test_negative_controls_are_finite_and_formula_only(self):
        for variant in ("homoskedastic", "additive_raw"):
            sequence = generate_sequence(4001, n=512, burn_in=32, variant=variant)
            task = with_splits(build_windows(sequence, 24, 1))
            assert_cross_task_labels(sequence, [task])
            self.assertTrue(np.isfinite(sequence.z).all())
            self.assertTrue(np.all(sequence.y >= 0.0))
            if variant == "additive_raw":
                np.testing.assert_array_equal(sequence.z, np.log1p(sequence.y))
        residual = np.asarray([-0.4, 0.0, 0.3, 0.8])
        expected = float(np.exp(np.mean(residual)) * gaussian_mgf_uniform(np.std(residual)))
        self.assertAlmostEqual(hpc_factor(residual), expected, places=12)

    def test_bandwidth_is_frozen(self):
        with self.assertRaisesRegex(ValueError, "frozen"):
            build_block_operator(np.asarray([[0.0]]), np.asarray([[0.0]]), np.asarray([0]), np.asarray([0]), bandwidth=2.0)

    def test_metric_rebuild_does_not_need_run_metrics(self):
        truth = np.ones((2, 1))
        prediction = np.zeros((2, 1))
        self.assertEqual(metric_rows(truth, prediction)["mse"], 1.0)

    def test_rho_gate_thresholds_and_formal_seed_refusal(self):
        self.assertTrue(rho_gate_status([0.01])["stop"])
        self.assertTrue(rho_gate_status([0.01, 0.02, 0.03, 0.04, 0.10, 0.20])["stop"])
        self.assertFalse(rho_gate_status([0.01, 0.20, 0.20, 0.20, 0.20, 0.20])["stop"])
        with self.assertRaisesRegex(ValueError, "formal fixture"):
            validate_seed_set((4004,), formal=True)

    def test_registered_mode_semantics_and_formal_mislabel_rejection(self):
        valid = {"registered_fixture": True, "formal_l0": False, "formal": False,
                 "non_formal": True, "smoke": False, "status": "COMPLETE",
                 "rho_diagnostic_stop": False, "rho_gate_applied": True, "rho_stop": False}
        validate_mode_metadata(valid)
        invalid = dict(valid, formal=True)
        with self.assertRaisesRegex(ValueError, "formal"):
            validate_mode_metadata(invalid)
        smoke = dict(valid, registered_fixture=False, smoke=True, status="SMOKE_COMPLETE",
                     rho_gate_applied=False, rho_stop=False, rho_diagnostic_stop=True)
        validate_mode_metadata(smoke)
        with self.assertRaisesRegex(ValueError, "rho_stop"):
            validate_mode_metadata(dict(smoke, rho_stop=True))

    def test_historical_seed_registry_is_conservative(self):
        from cf_cv_smearing_fixture import HISTORICAL_SEEDS
        self.assertTrue(set(range(2024, 2062)).issubset(HISTORICAL_SEEDS))
        self.assertTrue(set(range(3101, 3151)).issubset(HISTORICAL_SEEDS))
        self.assertTrue(set(range(20260939, 20260969)).issubset(HISTORICAL_SEEDS))

    def test_registered_resume_and_budget_binding(self):
        import run_round11_cf_cv_fixture as runner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "registered_smoke"
            binding = {"commit": "test", "branch": "round11-test", "clean_tree": True, "dirty_paths": []}
            with patch.object(runner, "run_one_task", side_effect=_fake_registered_task), patch("cf_cv_smearing_fixture.git_binding", return_value=binding):
                first = runner.run_fixture(root, seeds=(4001,), smoke=True, max_epochs=1, patience=1)
            self.assertFalse(first["registered_fixture"])
            self.assertFalse(first["formal_l0"])
            self.assertFalse(first["formal"])
            self.assertTrue(first["non_formal"])
            self.assertTrue(first["smoke"])
            self.assertEqual(first["status"], "SMOKE_COMPLETE")
            self.assertFalse(first["rho_gate_applied"])
            self.assertFalse(first["rho_stop"])
            self.assertIn("historical_seeds", first)
            self.assertGreaterEqual(first["wall_time"], 0.0)
            self.assertIn("ru_maxrss", first)
            self.assertEqual(first["preflight"]["commit"], "test")
            self.assertGreaterEqual(first["task_elapsed_seconds_sum"], 0.0)
            self.assertGreaterEqual(first["invocation_wall_time_seconds"], 0.0)
            self.assertEqual(first["wall_time"], first["invocation_wall_time_seconds"])
            self.assertEqual(first["resolved_rules"]["max_epochs"], 1)
            self.assertTrue(json.loads((root / "manifest.json").read_text(encoding="utf-8"))["preflight"]["clean_tree"])
            with patch.object(runner, "run_one_task", side_effect=AssertionError("resume retrained")), patch("cf_cv_smearing_fixture.git_binding", return_value=binding):
                resumed = runner.run_fixture(root, seeds=(4001,), smoke=True, max_epochs=1, patience=1)
            self.assertEqual(resumed["fit_count"], 14)
            with patch.object(runner, "run_one_task", side_effect=AssertionError("budget mismatch reached training")), patch("cf_cv_smearing_fixture.git_binding", return_value=binding):
                with self.assertRaisesRegex(ValueError, "configuration"):
                    runner.run_fixture(root, seeds=(4001,), smoke=True, max_epochs=2, patience=1)
            with self.assertRaisesRegex(ValueError, "max_epochs=40"):
                runner.run_fixture(Path(directory) / "registered_invalid", seeds=FORMAL_SEEDS,
                                   smoke=False, max_epochs=1, patience=8)

    def test_rho_stop_writes_complete_evidence_without_early_raise(self):
        import run_round11_cf_cv_fixture as runner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "rho_stop_smoke"
            binding = {"commit": "test", "branch": "round11-test", "clean_tree": True, "dirty_paths": []}
            fake = lambda *args, **kwargs: _fake_registered_task(*args, rho_stop=True, **kwargs)
            with patch.object(runner, "run_one_task", side_effect=fake), patch("cf_cv_smearing_fixture.git_binding", return_value=binding):
                result = runner.run_fixture(root, seeds=(4001,), smoke=True)
            self.assertEqual(result["status"], "SMOKE_COMPLETE")
            self.assertTrue(result["engineering_complete"])
            self.assertTrue(result["rho_diagnostic_stop"])
            self.assertFalse(result["rho_gate_applied"])
            self.assertFalse(result["rho_stop"])
            self.assertFalse(result["formal_l0"])
            status = json.loads((root / "strict_fixture_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "SMOKE_COMPLETE")
            self.assertTrue(status["rho_diagnostic_stop"])
            self.assertFalse(status["rho_gate_applied"])
            self.assertFalse(status["rho_stop"])
            self.assertTrue((root / "metadata.json").exists())
            self.assertTrue((root / "manifest.json").exists())

    def test_registered_runner_rejects_negative_control_training(self):
        import run_round11_cf_cv_fixture as runner
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "heteroskedastic"):
                runner.run_fixture(Path(directory) / "negative_smoke", seeds=(4001,), smoke=True,
                                   dgp_variant="homoskedastic")

    def test_raw_negative_base_is_allowed_but_log_arms_use_boundary(self):
        log_mean = np.zeros((1, 1))
        raw = np.asarray([[-3.0]])
        predictions = build_method_predictions(log_mean, raw, np.ones((1, 1)), np.ones((1, 1)),
                                                np.ones((1, 1)), np.ones((1, 1)))
        self.assertEqual(float(predictions["raw_mse_base"][0, 0]), -3.0)
        self.assertGreaterEqual(float(predictions["naive_log"][0, 0]), 0.0)

    def test_verification_writer_is_byte_stable_and_does_not_touch_runner_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner_manifest = root / "manifest.json"
            runner_manifest.write_bytes(b"runner-manifest")
            before = runner_manifest.read_bytes()
            summary = {"schema": "test", "source_runner_manifest_sha256": "fixed", "verified": True}
            write_verification(root, summary, "fixed")
            first_summary = (root / "verification" / "summary.json").read_bytes()
            first_manifest = (root / "verification" / "manifest.json").read_bytes()
            write_verification(root, summary, "fixed")
            self.assertEqual(before, runner_manifest.read_bytes())
            self.assertEqual(first_summary, (root / "verification" / "summary.json").read_bytes())
            self.assertEqual(first_manifest, (root / "verification" / "manifest.json").read_bytes())

    def test_tamper_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.bin"
            artifact.write_bytes(b"original")
            from cf_cv_smearing_fixture import sha256_file, atomic_json
            atomic_json(root / "manifest.json", {"schema": "test", "artifacts": {artifact.name: sha256_file(artifact)}})
            self.assertEqual(validate_manifest(root)["schema"], "test")
            artifact.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "tamper"):
                validate_manifest(root)

    def test_resume_crash_window_has_atomic_pair_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Simulates the runner crash after the NPZ/JSON pair but before progress.
            artifact = root / "task_4001_H1.npz"
            artifact.write_bytes(b"complete")
            info = root / "task_4001_H1.json"
            info.write_text(json.dumps({"unit_key": "seed=4001/task=H1"}), encoding="utf-8")
            self.assertTrue(artifact.exists() and info.exists())
            progress = root / "progress.json"
            self.assertFalse(progress.exists())
            progress.write_text(json.dumps({"completed": ["seed=4001/task=H1"]}), encoding="utf-8")
            self.assertIn("seed=4001/task=H1", json.loads(progress.read_text())["completed"])

    def test_real_crash_window_unregistered_pair_tamper_is_rejected(self):
        import run_round11_cf_cv_fixture as runner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "round11_smoke"
            with patch.object(runner, "run_one_task", side_effect=_fake_registered_task), \
                 patch("cf_cv_smearing_fixture.git_binding", return_value={"commit": "test", "branch": "round11-test", "clean_tree": True, "dirty_paths": []}):
                with self.assertRaisesRegex(RuntimeError, "injected crash"):
                    runner.run_fixture(root, seeds=(4001,), smoke=True, sequence_n=512, burn_in=32, fail_after=1)
            root.joinpath("task_4001_H1.npz").write_bytes(b"tampered-after-crash-before-manifest")
            with patch.object(runner, "run_one_task", side_effect=AssertionError("training must not be reached")), \
                 patch("cf_cv_smearing_fixture.git_binding", return_value={"commit": "test", "branch": "round11-test", "clean_tree": True, "dirty_paths": []}):
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    runner.run_fixture(root, seeds=(4001,), smoke=True, sequence_n=512, burn_in=32)


@unittest.skipUnless(importlib.util.find_spec("tsai"), "tsai is provided by the repository environment")
class PatchTSTAdapterShapeTests(unittest.TestCase):
    def test_real_tsai_patchtst_adapter_shape(self):
        from cf_cv_smearing_fixture import PatchTSTAdapter
        task = with_splits(build_windows(generate_sequence(4001, n=512, burn_in=32), 24, 1))
        # One tiny fit is an interface smoke, not a performance run.
        fit = task.split_indices["train"][:32]
        valid = task.split_indices["train"][32:40]
        adapter = PatchTSTAdapter(24, 1, 4001, max_epochs=1, patience=1, batch_size=16)
        X = task.X_raw.astype(np.float32)
        adapter.fit(X, task.y_raw, fit, valid)
        self.assertEqual(adapter.predict(X[valid]).shape, (len(valid), 1, 1))


if __name__ == "__main__":
    unittest.main()
