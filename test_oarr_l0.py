"""Synthetic-only tests: never read a project station file."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import oarr_l0 as o
import summarize_oarr_l0 as summary


class OarrMathTests(unittest.TestCase):
    def test_overlap_psd_and_exact_entries(self):
        origins = np.array([0, 1, 2, 6, 50])
        O = o.overlap_kernel(origins)
        self.assertAlmostEqual(O[0, 1], 5/6)
        self.assertEqual(O[0, 3], 0)
        self.assertTrue(np.allclose(np.diag(O), 1))
        self.assertGreaterEqual(np.linalg.eigvalsh(O).min(), -1e-12)

    def test_h1_is_identity_shrinkage(self):
        p = np.array([.6, .3, .1])
        O = o.overlap_kernel([1, 2, 3], horizon=1)
        w, audit = o.solve_weights(p, O)
        np.testing.assert_allclose(w, (p+1/3)/2, atol=1e-8)
        self.assertLessEqual(audit['projected_gradient_inf'], 1e-6)

    def test_all_same_timestamps_constant_penalty(self):
        p = np.array([.6, .3, .1])
        w, _ = o.solve_weights(p, np.ones((3, 3)))
        np.testing.assert_allclose(w, p, atol=1e-8)

    def test_bad_kernel_rejected(self):
        with self.assertRaises(ValueError):
            o.solve_weights(np.array([.5, .5]), np.array([[1., 2.], [2., 1.]]))

    def test_simplex_projection_independent(self):
        rng = np.random.default_rng(43)
        for _ in range(10):
            v = rng.normal(size=16)
            np.testing.assert_allclose(o.project_simplex(v), summary.simplex_projection(v), atol=1e-12)

    def test_permutation_query_specific_and_repeatable(self):
        np.testing.assert_array_equal(o.permutation(5101, 4400), o.permutation(5101, 4400))
        self.assertFalse(np.array_equal(o.permutation(5101, 4400), o.permutation(5101, 4401)))
        self.assertEqual(set(o.permutation(5101, 4400)), set(range(16)))

    def test_mmr_zero_and_ties(self):
        keys = np.zeros((16, 168)); keys[-1] = 1
        C = o.cosine_matrix(keys)
        self.assertEqual(C[0, 1], 1)
        self.assertEqual(C[0, -1], 0)
        got = o.mmr_indices(keys, np.zeros(16), np.arange(16))
        self.assertEqual(got[0], 0)
        self.assertEqual(got[1], 15)
        self.assertEqual(len(set(got)), 8)

    def test_all_methods_aligned_and_simplex(self):
        rng = np.random.default_rng(9)
        keys = rng.normal(size=(16, 168))
        d, origins = np.arange(16)/20, np.arange(16)
        weights, audits, O = o.all_weights(keys, d, origins, 200)
        self.assertEqual(set(weights), set(o.METHODS)-{'BASE'})
        for name, w in weights.items():
            self.assertAlmostEqual(w.sum(), 1)
            self.assertGreaterEqual(w.min(), -1e-8)
        self.assertEqual(np.flatnonzero(weights['SEP']).tolist(), [0, 6, 12])
        self.assertEqual(np.flatnonzero(weights['BLOCK']).tolist(), [0, 6, 12])
        self.assertEqual(len(audits), 6)


class OarrDataTests(unittest.TestCase):
    def test_ffill_limit_and_segment_reset(self):
        raw = np.ones(300)
        raw[20:27] = np.nan
        raw[100] = np.nan
        original, filled, rates = o.clean_segments(raw, [0, 100, 200, 300])
        self.assertTrue(np.isnan(original[20:27]).all())
        np.testing.assert_allclose(filled[20:26], 1)
        self.assertTrue(np.isnan(filled[26]))
        self.assertTrue(np.isnan(filled[100]))
        self.assertAlmostEqual(rates['A'], .07)

    def test_missing_rate_before_fill(self):
        raw = np.ones(300); raw[10:21] = np.nan
        with self.assertRaises(o.DataGate):
            o.clean_segments(raw, [0, 100, 200, 300])

    def test_window_target_boundaries_and_original_mask(self):
        raw = np.arange(1., 1001.)
        z = (raw-raw.mean())/raw.std()
        raw[203] = np.nan
        origins, x, y = o.windows(z, raw, 200, 250)
        self.assertGreaterEqual(origins.min(), 204)
        self.assertLess((origins+o.H-1).max(), 250)
        np.testing.assert_allclose(x[0], z[origins[0]-168:origins[0]])
        self.assertTrue(np.isfinite(y).all())

    def test_full_timestamp_only_then_feature_prefix(self):
        n = 8760
        times = pd.Series(pd.date_range('2000-01-01', periods=n, freq='h').astype(str))
        frames = [pd.DataFrame({'time': times}),
                  pd.DataFrame({'time': times[:5256], 'PM25_Concentration': 10+np.arange(5256)%31})]
        with patch.object(o.pd, 'read_csv', side_effect=frames) as read_csv:
            _, _, z, edges, meta = o.load_prefix(o.SOURCE)
        self.assertEqual(read_csv.call_args_list[0].kwargs, {'usecols': ['time']})
        self.assertEqual(read_csv.call_args_list[1].kwargs['nrows'], 5256)
        self.assertEqual(edges, [0, 3504, 4380, 5256])
        self.assertEqual(len(z), 5256)
        self.assertEqual(meta['unread_feature_rows'], 3504)

    def test_dirty_gate_before_read_and_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'run'
            with patch.object(o, 'git_provenance', side_effect=RuntimeError('dirty')):
                with patch.object(o, 'load_prefix') as load:
                    with self.assertRaises(RuntimeError):
                        o.run(out)
                    load.assert_not_called()
            self.assertFalse(out.exists())

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(o, 'load_prefix') as load:
                with self.assertRaises(RuntimeError):
                    o.run(tmp)
                load.assert_not_called()

    def test_maturity_excludes_recent_targets(self):
        query = 100
        origins = np.array([87, 88, 89, 94])
        np.testing.assert_array_equal(origins[origins+o.H <= query-o.H], [87, 88])


class EndToEndTests(unittest.TestCase):
    def test_synthetic_pipeline_and_tampering(self):
        rng = np.random.default_rng(8701)
        times = pd.Series(pd.date_range('2000-01-01', periods=8760, freq='h').astype(str))
        frames = [pd.DataFrame({'time': times}), pd.DataFrame({
            'time': times[:5256], 'PM25_Concentration': 20+rng.normal(size=5256)})]
        with patch.object(o.pd, 'read_csv', side_effect=frames):
            synthetic = o.load_prefix(o.SOURCE)
        with tempfile.TemporaryDirectory(prefix='oarr_synthetic_') as tmp:
            out = Path(tmp)/'run'
            with patch.object(o, 'load_prefix', return_value=synthetic), patch.object(
                o, 'git_provenance', return_value={'dirty': False, 'commit': 'SYNTHETIC_TEST_ONLY'}
            ), contextlib.redirect_stdout(io.StringIO()):
                o.run(out)
                _, verification = summary.summarize(out)
            self.assertTrue(verification['passed'])
            with self.assertRaises(ValueError):
                summary.summarize(out)
            p = pd.read_csv(out/'predictions.csv')
            p.loc[0, 'SIM'] += 1
            p.to_csv(out/'predictions.csv', index=False)
            with self.assertRaises(AssertionError):
                summary.summarize(out, out/'tampered_verification')
            self.assertFalse((out/'tampered_verification').exists())
            self.assertTrue((out/'verification/gate_status.json').exists())


class GateTests(unittest.TestCase):
    def data(self):
        rows = []
        for b in range(4):
            for h in range(6):
                row = {'block': b, 'h': h+1, 'truth': 0., **{n: 1.5 for n in o.METHODS}}
                row['BASE'] = 2.; row['OARR'] = 1.
                for n in summary.PERMS:
                    row[n] = 1.1
                rows.append(row)
        return pd.DataFrame(rows)

    def test_pass_and_separate_permutation_scores(self):
        f = self.data()
        g = summary.evaluate_gates(f)
        self.assertEqual(g['status'], 'PASS')
        self.assertAlmostEqual(g['permutation_gap_pct'], 10)

    def test_one_strong_control_failure_stops(self):
        f = self.data(); f['ISO'] = .9
        self.assertEqual(summary.evaluate_gates(f)['status'], 'STOP')

    def test_direction_gate(self):
        f = self.data(); f.loc[f.block.isin([0, 1]), 'ISO'] = .9
        g = summary.evaluate_gates(f)
        self.assertEqual(g['paired']['ISO']['positive_blocks'], 2)
        self.assertEqual(g['status'], 'STOP')

    def test_shuffle_gate(self):
        f = self.data()
        for n in summary.PERMS:
            f[n] = 1.
        self.assertEqual(summary.evaluate_gates(f)['status'], 'STOP')

    def test_zero_denominator_invalid(self):
        f = self.data(); f['BASE'] = 0.
        with self.assertRaises(ValueError):
            summary.evaluate_gates(f)


if __name__ == '__main__':
    unittest.main()
