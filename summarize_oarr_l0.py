"""Independently rebuild OARR L0 predictions and gates from persisted artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CONTROLS = ('BASE', 'UNIFORM', 'SIM', 'ISO', 'SEP', 'BLOCK', 'MMR')
PERMS = tuple(f'PERM{s}' for s in range(5101, 5106))
METHODS = (*CONTROLS, 'OARR', *PERMS)


def evaluate_gates(frame):
    metrics = {}
    for group in ('all', 0, 1, 2, 3):
        subset = frame if group == 'all' else frame.loc[frame.block == group]
        if subset.empty:
            raise ValueError('empty scoring block')
        metrics[str(group)] = {name: float(np.sqrt(np.mean((subset[name] - subset.truth)**2)))
                               for name in METHODS}
    all_rmse = metrics['all']
    if min(all_rmse.values()) <= 1e-12:
        raise ValueError('RMSE too close to zero for relative gates')
    paired = {}
    for name in CONTROLS:
        gain = 100 * (all_rmse[name] - all_rmse['OARR']) / all_rmse[name]
        positive = sum(metrics[str(b)]['OARR'] < metrics[str(b)][name] for b in range(4))
        threshold = .5 if name == 'BASE' else .2
        paired[name] = {'gain_pct': gain, 'positive_blocks': positive,
                        'threshold_pct': threshold, 'passed': bool(gain >= threshold and positive >= 3)}
    perm_gap = 100 * (np.mean([all_rmse[name] for name in PERMS]) - all_rmse['OARR']) / all_rmse['OARR']
    passed = all(v['passed'] for v in paired.values()) and perm_gap >= .1
    return {'status': 'PASS' if passed else 'STOP', 'passed': bool(passed),
            'paired': paired, 'permutation_gap_pct': float(perm_gap),
            'permutation_passed': bool(perm_gap >= .1), 'rmse': metrics,
            'claim_boundary': 'ridge residual proxy only; not locked ST or test performance'}


def simplex_projection(v):
    # Independent bounded bisection implementation for optimality verification.
    lo, hi = float(v.min()-1), float(v.max())
    for _ in range(100):
        mid = (lo+hi)/2
        if np.maximum(v-mid, 0).sum() > 1:
            lo = mid
        else:
            hi = mid
    return np.maximum(v-(lo+hi)/2, 0)


def summarize(directory, output=None):
    directory = Path(directory)
    output = Path(output) if output is not None else directory / 'verification'
    if output.exists():
        raise ValueError('verification output exists; use a new directory, never overwrite')
    if (directory / 'failure.json').exists():
        raise ValueError('failed/incomplete execution cannot generate a performance gate')
    manifest = json.loads((directory / 'manifest.json').read_text())
    integrity = json.loads((directory / 'integrity.json').read_text())
    frame = pd.read_csv(directory / 'predictions.csv')
    memory = pd.read_csv(directory / 'memory_predictions.csv')
    assert manifest['dirty'] is False and integrity['passed'] is True
    assert integrity['edges'] == [0, 3504, 4380, 5256]
    assert integrity['feature_rows_read'] == 5256 and integrity['fit_calls'] == 1
    assert integrity['scale_fit_slice'] == [0, 3504]
    assert integrity['baseline_state_before'] == integrity['baseline_state_after']
    assert all(integrity[k] == 0 for k in ('post60_feature_access_count', 'post60_target_access_count', 'post60_prediction_count'))
    assert max(integrity['original_missing_rates'].values()) <= .1
    assert not frame.duplicated(['origin', 'h']).any() and not memory.duplicated(['origin', 'h']).any()
    assert np.isfinite(frame[['truth', *METHODS]].to_numpy()).all()
    assert np.isfinite(memory[['truth', 'base', 'residual']].to_numpy()).all()
    assert set(frame.block) == {0, 1, 2, 3}
    assert frame.origin.min() >= 4380 and frame.target_index.max() < 5256
    assert memory.origin.min() >= 3504 and memory.target_index.max() < 4380
    assert (frame.target_index == frame.origin + frame.h - 1).all()
    assert (memory.target_index == memory.origin + memory.h - 1).all()
    assert np.allclose(memory.residual, memory.truth-memory.base, atol=1e-10, rtol=1e-12)
    for tab in (frame, memory):
        assert (tab.groupby('origin').size() == 6).all()
        assert tab.groupby('origin').h.apply(lambda x: set(x) == set(range(1, 7))).all()
        repeated = tab.groupby('target_index').truth.agg(['min', 'max'])
        assert np.allclose(repeated['min'], repeated['max'], atol=1e-10, rtol=1e-12)
    for i in range(4):
        ids = set(frame.loc[frame.block == i, 'target_index'])
        for j in range(i):
            assert not ids & set(frame.loc[frame.block == j, 'target_index'])
        spec = integrity['blocks'][i]
        assert frame.loc[frame.block == i, 'origin'].nunique() == spec['evaluated_queries']
        assert spec['evaluated_queries'] >= .8*spec['legal_queries']
        assert min(ids) >= spec['start'] and max(ids) < spec['stop']
    mem = memory.pivot(index='origin', columns='h', values='residual').sort_index(axis=1)
    by_origin = {int(t): rows.sort_values('h') for t, rows in frame.groupby('origin')}
    seen, overlap, errors = set(), [], []
    max_kkt = 0.
    for line in (directory / 'retrieval_audit.jsonl').read_text().splitlines():
        audit = json.loads(line)
        t = audit['origin']
        assert t not in seen and t in by_origin
        seen.add(t)
        origins = np.array(audit['candidate_origins'], dtype=np.int64)
        d = np.asarray(audit['distance'], dtype=np.float64)
        assert len(origins) == len(set(origins)) == 16
        assert np.isfinite(d).all() and (d >= 0).all()
        assert np.array_equal(np.lexsort((origins, d)), np.arange(16))
        assert (origins+6 <= t-6).all() and set(origins) <= set(mem.index)
        p = np.exp(-d + d.min()); p /= p.sum()
        O = np.maximum(0, 6-np.abs(origins[:, None]-origins[None, :]))/6
        overlap.append(float(O.sum()-16))
        assert abs(overlap[-1]-audit['overlap_offdiag_sum']) < 1e-10
        local = by_origin[t]
        assert (local.block == audit['block']).all()
        weights = audit['weights']
        assert set(weights) == set(METHODS)-{'BASE'}
        for name, raw in weights.items():
            w = np.asarray(raw)
            assert len(w) == 16 and np.isfinite(w).all() and w.min() >= -1e-8 and abs(w.sum()-1) <= 1e-8
            prediction = local.BASE.to_numpy() + w @ mem.loc[origins].to_numpy()
            error = float(np.max(np.abs(prediction-local[name].to_numpy())))
            errors.append(error)
            assert error <= 1e-8
            if name in ('UNIFORM', 'SIM', 'ISO'):
                expected = {'UNIFORM': np.ones(16)/16, 'SIM': p, 'ISO': (p+1/16)/2}[name]
                assert np.allclose(w, expected, atol=1e-10, rtol=0)
            if name == 'SEP':
                selected = []
                for idx, origin in enumerate(origins):
                    if all(abs(origin-origins[j]) >= 6 for j in selected):
                        selected.append(idx)
                expected = np.zeros(16); expected[selected] = p[selected]/p[selected].sum()
                assert np.allclose(w, expected, atol=1e-10)
            if name == 'BLOCK':
                selected, bins = [], set()
                for idx, origin in enumerate(origins):
                    if origin//6 not in bins:
                        bins.add(origin//6); selected.append(idx)
                expected = np.zeros(16); expected[selected] = p[selected]/p[selected].sum()
                assert np.allclose(w, expected, atol=1e-10)
            if name == 'OARR' or name.startswith('PERM'):
                kernel = O
                if name.startswith('PERM'):
                    s = int(name[4:])
                    order = np.random.Generator(np.random.PCG64(np.random.SeedSequence([s, t]))).permutation(16)
                    kernel = O[np.ix_(order, order)]
                grad = 2*(w-p)+2*kernel@w
                kkt = float(np.max(np.abs(w-simplex_projection(w-grad))))
                max_kkt = max(max_kkt, kkt)
                assert kkt <= 1e-6 and audit['solvers'][name]['success'] is True
    assert seen == set(by_origin)
    gate = evaluate_gates(frame)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (directory/'predictions.csv', directory/'memory_predictions.csv', directory/'retrieval_audit.jsonl')}
    verification = {'passed': True, 'query_count': len(seen), 'max_reconstruction_error': max(errors),
                    'max_independent_kkt': max_kkt, 'mean_overlap_offdiag_sum': float(np.mean(overlap)),
                    'fraction_queries_with_overlap': float(np.mean(np.array(overlap) > 1e-12)),
                    'artifact_sha256': hashes, 'gate_status': gate['status'],
                    'summarizer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    'versions': {'numpy': np.__version__, 'pandas': pd.__version__},
                    'limits': 'no original station values reread; input-distance/MMR key verification relies on runner tests and source audit'}
    output.mkdir(parents=True, exist_ok=False)
    (output/'gate_status.json').write_text(json.dumps(gate, indent=2, allow_nan=False))
    (output/'independent_verification.json').write_text(json.dumps(verification, indent=2, allow_nan=False))
    pd.DataFrame(gate['rmse']).T.to_csv(output/'rmse_recomputed.csv', index_label='block')
    print(json.dumps({'status': gate['status'], 'paired': gate['paired'],
                      'permutation_gap_pct': gate['permutation_gap_pct'], 'integrity': True}, indent=2))
    return gate, verification


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    summarize(args.directory, args.output_dir)
