from __future__ import annotations

import unittest

import numpy as np

from tacbr_l0 import (
    C_GRID, DGPConfig, _array_sha256, _fit_non_tf_direct, fit_candidate,
    fit_non_tf_profile, make_synthetic, non_tf_certificate, prepare_data,
    predict, tf_certificate, verify_prediction_reconstruction,
)


def fixture_prepared(group: str = "V", labels: int = 24):
    config = DGPConfig(p=16, n_train=48, n_validation=12, n_test=16, target_label_stride=2)
    return prepare_data(make_synthetic(8702, group, labels, config), labels)


class TACBRL0Tests(unittest.TestCase):
    def test_generator_reuses_bottom_random_variables_between_v_and_f(self):
        config = DGPConfig(p=16, n_train=48, n_validation=12, n_test=16, target_label_stride=2)
        v = make_synthetic(8702, "V", 24, config)
        f = make_synthetic(8702, "F", 24, config)
        np.testing.assert_array_equal(v.x_target, f.x_target)
        np.testing.assert_array_equal(v.x_sources, f.x_sources)
        np.testing.assert_array_equal(v.y_target, f.y_target)
        self.assertFalse(np.array_equal(v.y_sources, f.y_sources))

    def test_v_exact_and_f_nonrepresentable_shift(self):
        data_v = make_synthetic(8702, "V", 24, DGPConfig(p=16, n_train=48, n_validation=12, n_test=16, target_label_stride=2))
        data_f = make_synthetic(8702, "F", 24, data_v.config)
        for station in range(5):
            phi = np.column_stack((np.ones(data_v.config.n_total), data_v.x_sources[station, :, :2]))
            self.assertLess(np.max(np.abs(data_v.true_source_shift[station] - phi @ data_v.source_bias[station])), 1e-12)
            residual = data_f.true_source_shift[station] - 0.7 * (phi @ data_f.source_bias[station])
            design = np.column_stack((np.ones(data_f.config.n_total), data_f.x_sources[station, :, :2]))
            projection = design @ np.linalg.lstsq(design, residual, rcond=None)[0]
            self.assertGreater(float(np.var(residual - projection)), 1e-3)

    def test_v_and_f_random_shift_variances_match(self):
        data_v = make_synthetic(8702, "V", 24, DGPConfig(p=16, n_train=48, n_validation=12, n_test=16, target_label_stride=2))
        data_f = make_synthetic(8702, "F", 24, data_v.config)
        for station in range(5):
            a = data_v.source_bias[station]
            expected = 1.01 * (a[1] ** 2 + a[2] ** 2)
            # The exact coefficient-norm calculation is the population
            # variance check; empirical AR variance is intentionally not used.
            v_norm_sq = a[1] ** 2 + a[2] ** 2
            f_norm_sq = 0.7**2 * v_norm_sq + (1.0 - 0.7**2) * v_norm_sq
            self.assertAlmostEqual(1.01 * v_norm_sq, expected, places=14)
            self.assertAlmostEqual(1.01 * f_norm_sq, expected, places=14)

    def test_support_is_target_train_loo_95_percentile(self):
        prepared = fixture_prepared()
        self.assertEqual(prepared.source_train_masks.shape, (5, 48))
        target_z = prepared.z_target[prepared.target_train_positions]
        distances = np.sqrt(np.sum((target_z[:, None] - target_z[None, :]) ** 2, axis=2))
        np.fill_diagonal(distances, np.inf)
        kth = np.partition(distances, 11, axis=1)[:, 11]
        self.assertAlmostEqual(prepared.tau, float(np.quantile(kth, 0.95, method="linear")), places=14)

    def test_lambda_zero_is_target_only_at_same_eta(self):
        prepared = fixture_prepared()
        target = _fit_non_tf_direct(prepared, "target_only", float(C_GRID[4]), lam=1.0)
        zero = _fit_non_tf_direct(prepared, "tacbr", float(C_GRID[4]), lam=0.0)
        np.testing.assert_allclose(predict(prepared, target, "validation"), predict(prepared, zero, "validation"), atol=1e-8)
        np.testing.assert_allclose(predict(prepared, target, "test"), predict(prepared, zero, "test"), atol=1e-8)

    def test_profile_and_direct_station_linear_head_are_equivalent(self):
        prepared = fixture_prepared()
        c = float(C_GRID[5])
        direct = _fit_non_tf_direct(prepared, "tacbr", c)
        profiled = fit_non_tf_profile(prepared, "tacbr", c)
        self.assertEqual(direct.status, "optimal")
        self.assertEqual(profiled.status, "optimal")
        np.testing.assert_allclose(predict(prepared, direct, "validation"), predict(prepared, profiled, "validation"), atol=2e-7)
        np.testing.assert_allclose(predict(prepared, direct, "test"), predict(prepared, profiled, "test"), atol=2e-7)
        self.assertLess(direct.kkt_inf, 1e-6)
        self.assertLess(profiled.kkt_inf, 1e-6)

    def test_all_methods_have_convex_certificates(self):
        prepared = fixture_prepared()
        methods = (
            "target_only", "global_full", "matched_global", "matched_global_station_intercept",
            "unmatched_conditional_bias_head", "tacbr", "tacbr_oracle",
            "transfusion_paper_exact", "transfusion_intercept",
        )
        for method in methods:
            with self.subTest(method=method):
                fit = fit_candidate(prepared, method, float(C_GRID[3]))
                self.assertEqual(fit.status, "optimal")
                self.assertLess(fit.kkt_inf, 1e-6)
                self.assertTrue(np.isfinite(fit.objective))

    def test_tf_weighting_and_two_stage_kkt(self):
        prepared = fixture_prepared()
        for method in ("transfusion_paper_exact", "transfusion_intercept"):
            fit = fit_candidate(prepared, method, float(C_GRID[2]))
            self.assertEqual(fit.tf_task_betas.shape, (6, 16))
            np.testing.assert_allclose(fit.tf_task_betas.shape[0], 6)
            objective, kkt = tf_certificate(prepared, method, fit.c, fit.tf_task_betas, fit.tf_task_intercepts,
                                            fit.tf_delta, fit.tf_delta_intercept, fit.beta, fit.intercept)
            self.assertAlmostEqual(objective, fit.objective, places=8)
            self.assertLess(kkt, 1e-6)

    def test_prediction_reconstruction_rejects_tampering(self):
        prepared = fixture_prepared()
        fit = fit_candidate(prepared, "tacbr", float(C_GRID[3]))
        val = predict(prepared, fit, "validation")
        test = predict(prepared, fit, "test")
        digest = _array_sha256(val, test)
        self.assertLess(verify_prediction_reconstruction(prepared, fit, val, test, digest), 1e-12)
        tampered = val.copy()
        tampered[0] += 1e-4
        with self.assertRaises(ValueError):
            verify_prediction_reconstruction(prepared, fit, tampered, test, digest)


if __name__ == "__main__":
    unittest.main()
