import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from ccsprf_l0 import (
    ARMS,
    ModelConfig,
    SyntheticConfig,
    TAU_L0,
    ForecastArm,
    build_prior_arrays,
    calendar_key,
    fit_calendar_prior,
    fit_global_prior,
    generate_synthetic_series,
    paired_bootstrap_median,
    prepare_experiment,
    set_seed,
    source_tree_violations,
    train_one_arm,
    training_rng_seed,
    verify_prediction_artifact,
)
from summarize_ccsprf_l0 import ARMS as SUMMARY_ARMS, CONDITIONS as SUMMARY_CONDITIONS, SEEDS as SUMMARY_SEEDS, TASKS as SUMMARY_TASKS, gates, summarize, validate_progress


class CCSPRFL0Test(unittest.TestCase):
    def test_dgp_is_hourly_and_calendar_off_changes_only_calendar_condition(self):
        on_config = SyntheticConfig(seed=991001, num_points=24 * 7 * 3, calendar_on=True)
        off_config = SyntheticConfig(seed=991001, num_points=24 * 7 * 3, calendar_on=False)
        on_values, on_times, on_meta = generate_synthetic_series(on_config)
        off_values, off_times, off_meta = generate_synthetic_series(off_config)
        self.assertTrue(np.all(np.diff(on_times) == 3_600_000_000_000))
        np.testing.assert_array_equal(on_times, off_times)
        self.assertGreater(float(np.max(np.abs(on_meta["hour_effect_centered"]))), 0.0)
        self.assertEqual(max(map(abs, off_meta["hour_effect_centered"])), 0.0)
        self.assertFalse(np.array_equal(on_values, off_values))

    def test_split_target_timestamps_do_not_overlap(self):
        prepared = prepare_experiment(SyntheticConfig(history=24, horizon=6, seed=991002, num_points=24 * 7 * 8))
        target_sets = []
        for split, starts in prepared.split_indices.items():
            target_sets.append({
                int(value)
                for start in starts
                for value in prepared.timestamps_ns[start + 24 : start + 24 + 6]
            })
        self.assertTrue(target_sets[0].isdisjoint(target_sets[1]))
        self.assertTrue(target_sets[1].isdisjoint(target_sets[2]))
        self.assertEqual(prepared.metadata["train_fraction"], 0.70)

    def test_calendar_loo_uses_unique_timestamp_not_window_count(self):
        # The same physical timestamp is deliberately repeated in the window view;
        # the prior fitter must reject that view instead of counting it twice.
        timestamps = np.asarray([0, 3_600_000_000_000, 7_200_000_000_000], dtype=np.int64)
        values = np.asarray([1.0, 3.0, 5.0])
        table = fit_calendar_prior(timestamps, values, tau=TAU_L0)
        self.assertEqual(sum(record[0] for record in table.groups.values()), 3)
        expected_mu0_loo = (table.total_count * table.mu0 - values[0]) / (table.total_count - 1)
        self.assertAlmostEqual(table.loo_prior(timestamps[0], values[0]), expected_mu0_loo)
        with self.assertRaisesRegex(ValueError, "重复"):
            fit_calendar_prior(np.asarray([0, 0], dtype=np.int64), [1.0, 2.0])
        global_table = fit_global_prior(timestamps, values, tau=TAU_L0)
        expected_global_loo = (global_table.total_count * global_table.mu0 - values[0]) / 2
        # Global LOO is exact and must not be shrunk again by tau.
        self.assertAlmostEqual(global_table.loo_prior(timestamps[0], values[0]), expected_global_loo)

    def test_calendar_loo_excludes_self_from_global_baseline(self):
        # All three timestamps share one calendar key.  The old self-inclusive
        # baseline produces a different value even though the group mean is also
        # leave-one-out; this exact decimal check catches that implementation.
        timestamps = np.asarray([0, 7 * 24 * 3_600_000_000_000, 14 * 24 * 3_600_000_000_000], dtype=np.int64)
        values = np.asarray([1.0, 2.0, 9.0])
        table = fit_calendar_prior(timestamps, values, tau=50.0)
        expected_mu0_loo = (3 * table.mu0 - values[2]) / 2
        expected_group_mean_loo = (values[0] + values[1]) / 2
        expected = (2 / 52) * expected_group_mean_loo + (50 / 52) * expected_mu0_loo
        self.assertAlmostEqual(table.loo_prior(timestamps[2], values[2]), expected, places=12)
        old_wrong = (2 / 52) * expected_group_mean_loo + (50 / 52) * table.mu0
        self.assertNotAlmostEqual(table.loo_prior(timestamps[2], values[2]), old_wrong, places=8)

    def test_prior_arrays_use_loo_only_for_train(self):
        prepared = prepare_experiment(SyntheticConfig(history=24, horizon=6, seed=991003, num_points=24 * 7 * 8))
        priors = build_prior_arrays(prepared)
        self.assertEqual(
            priors["arrays"]["train"]["calendar"].shape[1],
            6,
        )
        self.assertFalse(
            np.array_equal(
                priors["arrays"]["train"]["calendar"][0],
                priors["arrays"]["train"]["values_scaled"][0],
            )
        )
        self.assertTrue(np.all(np.isfinite(priors["arrays"]["test"]["calendar"])))

    def test_embedding_parameter_increment_is_exactly_124(self):
        config = ModelConfig(n_layers=1, d_ff=32)
        base = ForecastArm(24, 1, config, "BASE")
        emb = ForecastArm(24, 1, config, "EMB")
        self.assertEqual(sum(p.numel() for p in emb.parameters()) - sum(p.numel() for p in base.parameters()), 124)
        x = torch.randn(3, 24)
        times = torch.as_tensor([[24 * 3_600_000_000_000], [48 * 3_600_000_000_000], [72 * 3_600_000_000_000]], dtype=torch.int64)
        output, bias = emb(x, times)
        self.assertEqual(tuple(output.shape), (3, 1))
        self.assertEqual(tuple(bias.shape), (3, 1))

    def test_backbone_hash_is_equal_across_arms_for_same_seed(self):
        from ccsprf_l0 import expected_backbone_hashes

        hashes = expected_backbone_hashes(991005, 24, 1, ModelConfig(n_layers=1, d_ff=32))
        self.assertEqual(len(set(hashes.values())), 1)

    def test_training_rng_is_reset_after_every_arm_constructor(self):
        # EMB allocates 124 parameters after the backbone.  The reset must make
        # the subsequent loader/dropout stream identical for every arm.
        config = ModelConfig(n_layers=1, d_ff=32)
        states = []
        for arm in ARMS:
            set_seed(991005)
            ForecastArm(24, 1, config, arm)
            set_seed(training_rng_seed(991005))
            states.append(torch.get_rng_state().clone())
        for state in states[1:]:
            self.assertTrue(torch.equal(states[0], state))
        self.assertEqual(training_rng_seed(991005), 1991008)

    def test_resume_source_tree_allows_only_output_paths(self):
        repo = Path("/repo"); output = repo / "experiments" / "results" / "round9" / "l0_formal"
        self.assertEqual(source_tree_violations(repo, output, "?? experiments/results/round9/l0_formal/\n"), [])
        self.assertEqual(source_tree_violations(repo, output, " M ccsprf_l0.py\n"), [" M ccsprf_l0.py"])
        self.assertEqual(source_tree_violations(repo, output, "?? experiments/results/round9/other/\n"), ["?? experiments/results/round9/other/"])

    def test_single_arm_fixture_writes_reconstructable_artifact_and_tamper_is_rejected(self):
        prepared = prepare_experiment(SyntheticConfig(history=24, horizon=1, seed=991004, num_points=24 * 7 * 4))
        priors = build_prior_arrays(prepared)
        config = ModelConfig(n_layers=1, d_ff=32, epochs=1, patience=1, batch_size=32)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = train_one_arm(prepared, priors, "CCSPRF", 991004, config, output_dir=root)
            artifact = root / result["prediction_artifact"]
            verified = verify_prediction_artifact(artifact)
            np.testing.assert_allclose(
                verified["arrays"]["prediction_scaled"],
                verified["arrays"]["prior_scaled"] + verified["arrays"]["residual_scaled"],
                atol=1e-6,
            )
            tampered = root / "tampered.npz"
            tampered_manifest = root / "tampered.json"
            shutil.copy2(artifact, tampered)
            shutil.copy2(artifact.with_suffix(".json"), tampered_manifest)
            with np.load(tampered, allow_pickle=False) as loaded:
                arrays = {key: loaded[key] for key in loaded.files}
            arrays["prediction_scaled"] = arrays["prediction_scaled"].copy()
            arrays["prediction_scaled"].flat[0] += 1.0
            np.savez_compressed(tampered, **arrays)
            with self.assertRaisesRegex(RuntimeError, "hash 不匹配"):
                verify_prediction_artifact(tampered)

    def test_bootstrap_is_reproducible_and_reports_median(self):
        values = [0.01, 0.02, 0.03, 0.04]
        first = paired_bootstrap_median(values, seed=20260969, replicates=200)
        second = paired_bootstrap_median(values, seed=20260969, replicates=200)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first[0], 0.025)

    def test_gate_booleans_are_python_json_scalars(self):
        rows = {}
        for condition in SUMMARY_CONDITIONS:
            for history, horizon in SUMMARY_TASKS:
                task = f"{history}h_{horizon}h"
                for seed in SUMMARY_SEEDS:
                    for arm in SUMMARY_ARMS:
                        rows[(condition, task, seed, arm)] = {
                            "rmse_raw": 1.0,
                            "prior_only_rmse_raw": 2.0,
                            "per_step_rmse_raw": [1.0] * horizon,
                        }
        result = gates(rows)
        json.dumps(result)
        self.assertIs(type(result["pass"]), bool)
        for task_result in result["tasks"].values():
            self.assertIs(type(task_result["pass"]), bool)
            self.assertTrue(all(type(value) is bool for value in task_result["gates"].values()))

    def test_artifact_records_final_emb_weights_and_tau_semantics(self):
        prepared = prepare_experiment(SyntheticConfig(history=24, horizon=1, seed=991009, num_points=24 * 7 * 4))
        priors = build_prior_arrays(prepared)
        config = ModelConfig(n_layers=1, d_ff=32, epochs=1, patience=1, batch_size=32)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            emb = train_one_arm(prepared, priors, "EMB", 991009, config, output_dir=root)
            emb_verified = verify_prediction_artifact(root / emb["prediction_artifact"])
            self.assertEqual(set(emb_verified["arrays"]) & {"emb_hour_weight", "emb_weekday_weight"}, {"emb_hour_weight", "emb_weekday_weight"})
            self.assertEqual(emb_verified["arrays"]["emb_hour_weight"].shape, (24, 4))
            self.assertEqual(emb_verified["arrays"]["emb_weekday_weight"].shape, (7, 4))
            self.assertIsNone(emb_verified["manifest"]["tau"])
            self.assertFalse(emb_verified["manifest"]["tau_applied"])
            base = train_one_arm(prepared, priors, "BASE", 991009, config, output_dir=root)
            base_verified = verify_prediction_artifact(root / base["prediction_artifact"])
            self.assertNotIn("emb_hour_weight", base_verified["arrays"])
            self.assertNotIn("emb_weekday_weight", base_verified["arrays"])
            self.assertIsNone(base_verified["manifest"]["tau"])
            self.assertFalse(base_verified["manifest"]["tau_applied"])
            cc = train_one_arm(prepared, priors, "CCSPRF", 991009, config, output_dir=root)
            cc_manifest = verify_prediction_artifact(root / cc["prediction_artifact"])["manifest"]
            self.assertEqual(cc_manifest["tau"], TAU_L0)
            self.assertTrue(cc_manifest["tau_applied"])

    def test_formal_expected_key_count(self):
        from ccsprf_l0 import CONDITIONS, FORMAL_SEEDS, TASKS, expected_keys

        keys = expected_keys(FORMAL_SEEDS, TASKS, CONDITIONS, ARMS)
        self.assertEqual(len(keys), 480)
        self.assertEqual(len(keys), len(set(keys)))

    def test_formal_progress_validator_requires_exact_ordered_completion(self):
        manifest = {"expected_keys": ["a", "b"], "expected_keys_sha256": "keys", "canonical_payload_sha256": "payload"}
        validate_progress(manifest, {"expected_keys": ["a", "b"], "expected_keys_sha256": "keys", "canonical_payload_sha256": "payload", "completed_keys": ["a", "b"]}, True)
        with self.assertRaisesRegex(RuntimeError, "480"):
            validate_progress(manifest, {"expected_keys": ["a", "b"], "expected_keys_sha256": "keys", "canonical_payload_sha256": "payload", "completed_keys": ["a"]}, True)

    def test_resume_skips_complete_key_and_rejects_tamper(self):
        from ccsprf_l0 import run_experiment

        config = ModelConfig(n_layers=1, d_ff=32, epochs=1, patience=1, batch_size=32)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "resume"
            args = dict(
                output_dir=root,
                seeds=[991106],
                tasks=[(24, 1)],
                conditions=["calendar_on"],
                arms=["BASE"],
                config=config,
                num_points=672,
                device="cpu",
                formal=True,
            )
            # A one-key formal-shaped run is only a resume-engineering test.
            with mock.patch("ccsprf_l0.check_clean_tree"), mock.patch("ccsprf_l0.check_source_tree"):
                first = run_experiment(**args)
                second = run_experiment(resume=True, **args)
            self.assertEqual(len(first), 1)
            self.assertEqual(len(second), 1)
            artifact = root / first[0]["prediction_artifact"]
            for relative, message in (
                (Path("datasets/calendar_on/24h_1h/seed991106.npz"), "dataset/prior"),
                (Path("prior_tables/calendar_on/24h_1h/seed991106.json"), "dataset/prior"),
                (Path("training_logs/calendar_on/24h_1h/seed991106_BASE.csv"), "training log"),
            ):
                incomplete = Path(temp) / f"missing-{relative.name}"
                shutil.copytree(root, incomplete)
                (incomplete / relative).unlink()
                with mock.patch("ccsprf_l0.check_source_tree"):
                    with self.assertRaisesRegex(RuntimeError, message):
                        run_experiment(resume=True, output_dir=incomplete, **{key: value for key, value in args.items() if key != "output_dir"})
                self.assertFalse((incomplete / relative).exists())
            progress_path = root / "progress.json"
            progress = json.loads(progress_path.read_text())
            progress["completed_keys"] = []
            progress_path.write_text(json.dumps(progress), encoding="utf-8")
            with mock.patch("ccsprf_l0.check_source_tree"), mock.patch("ccsprf_l0.train_one_arm", side_effect=AssertionError("reconciliation 不得重训")):
                reconciled = run_experiment(resume=True, **args)
            self.assertEqual(len(reconciled), 1)
            self.assertEqual(json.loads(progress_path.read_text())["completed_keys"], progress["expected_keys"])

            crash_window = Path(temp) / "crash-window"
            shutil.copytree(root, crash_window)
            crash_progress = json.loads((crash_window / "progress.json").read_text())
            crash_progress["completed_keys"] = []
            (crash_window / "progress.json").write_text(json.dumps(crash_progress), encoding="utf-8")
            crash_artifact = crash_window / first[0]["prediction_artifact"]
            crash_artifact.unlink()
            crash_artifact.with_suffix(".json").unlink()
            with mock.patch("ccsprf_l0.check_source_tree"):
                with self.assertRaisesRegex(RuntimeError, "reconciliation key 文件不完整"):
                    run_experiment(resume=True, output_dir=crash_window, **{key: value for key, value in args.items() if key != "output_dir"})
            self.assertEqual(json.loads((crash_window / "progress.json").read_text())["completed_keys"], [])

            residue = Path(temp) / "residue"
            shutil.copytree(root, residue)
            residue_progress = json.loads((residue / "progress.json").read_text())
            residue_progress["completed_keys"] = []
            (residue / "progress.json").write_text(json.dumps(residue_progress), encoding="utf-8")
            (residue / "raw_metrics.csv").write_text((residue / "raw_metrics.csv").read_text().splitlines()[0] + "\n", encoding="utf-8")
            with mock.patch("ccsprf_l0.check_source_tree"):
                with self.assertRaisesRegex(RuntimeError, "未完成 key 存在 artifact/sidecar/log/raw 残留"):
                    run_experiment(resume=True, output_dir=residue, **{key: value for key, value in args.items() if key != "output_dir"})
            with artifact.open("ab") as handle:
                handle.write(b"tamper")
            with mock.patch("ccsprf_l0.check_source_tree"):
                with self.assertRaisesRegex(RuntimeError, "hash 不匹配"):
                    run_experiment(resume=True, **args)

    def test_complete_fixture_summary_and_independent_integrity_rejections(self):
        from ccsprf_l0 import run_experiment

        config = ModelConfig(n_layers=1, d_ff=32, epochs=1, patience=1, batch_size=64)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "fixture"
            with mock.patch("ccsprf_l0.check_clean_tree"):
                run_experiment(root, [991107], [(24, 1), (168, 6)], ["calendar_on", "calendar_off"], list(ARMS), config, 672, "cpu", formal=False)
            result = summarize(root)
            self.assertEqual(result["status"], "NOT_APPLICABLE_FIXTURE")
            self.assertEqual(json.loads((root / "strict_release_status.json").read_text())["status"], "NOT_APPLICABLE_FIXTURE")

            missing = Path(temp) / "missing"
            shutil.copytree(root, missing)
            missing_artifact = missing / "predictions" / "calendar_on" / "24h_1h" / "seed991107" / "EMB.npz"
            missing_artifact.unlink()
            missing_artifact.with_suffix(".json").unlink()
            with self.assertRaisesRegex(RuntimeError, "ENGINEERING_FAILURE"):
                summarize(missing)

            extra = Path(temp) / "extra"
            shutil.copytree(root, extra)
            extra_dir = extra / "predictions" / "calendar_on" / "24h_1h" / "seed991107"
            shutil.copy2(extra_dir / "BASE.npz", extra_dir / "EXTRA.npz")
            shutil.copy2(extra_dir / "BASE.json", extra_dir / "EXTRA.json")
            with self.assertRaisesRegex(RuntimeError, "ENGINEERING_FAILURE"):
                summarize(extra)

            tampered = Path(temp) / "tampered"
            shutil.copytree(root, tampered)
            target = tampered / "predictions" / "calendar_on" / "24h_1h" / "seed991107" / "BASE.npz"
            with np.load(target, allow_pickle=False) as loaded:
                arrays = {key: loaded[key] for key in loaded.files}
            arrays["target_scaled"] = arrays["target_scaled"].copy()
            arrays["target_scaled"].flat[0] += 0.25
            np.savez_compressed(target, **arrays)
            sidecar = target.with_suffix(".json")
            sidecar_value = json.loads(sidecar.read_text())
            sidecar_value["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
            sidecar.write_text(json.dumps(sidecar_value), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "ENGINEERING_FAILURE"):
                summarize(tampered)

            bad_sidecar = Path(temp) / "bad-sidecar"
            shutil.copytree(root, bad_sidecar)
            emb_sidecar = bad_sidecar / "predictions" / "calendar_on" / "24h_1h" / "seed991107" / "EMB.json"
            emb_metadata = json.loads(emb_sidecar.read_text())
            emb_metadata["center_std"] += 1.0
            emb_sidecar.write_text(json.dumps(emb_metadata), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "center_mean/std"):
                summarize(bad_sidecar)

            broken_progress = Path(temp) / "broken-progress"
            shutil.copytree(root, broken_progress)
            progress_path = broken_progress / "progress.json"
            progress = json.loads(progress_path.read_text())
            progress["completed_keys"] = progress["completed_keys"][:-1]
            progress_path.write_text(json.dumps(progress), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "progress"):
                summarize(broken_progress)


if __name__ == "__main__":
    unittest.main()
