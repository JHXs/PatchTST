"""Frozen, prefix-only OARR L0. Protocol: theory commit a920250 (ridge proxy only)."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.linear_model import Ridge

L, H, K = 168, 6, 16
PERM_SEEDS = tuple(range(5101, 5106))
CONTROLS = ('BASE', 'UNIFORM', 'SIM', 'ISO', 'SEP', 'BLOCK', 'MMR')
METHODS = (*CONTROLS, 'OARR', *(f'PERM{s}' for s in PERM_SEEDS))
VERSIONS = {'python': '3.12.13', 'numpy': '2.4.4', 'scipy': '1.18.0',
            'sklearn': '1.7.2', 'pandas': '2.3.3'}
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'tsai/data/stations_data_Guangzhou/df_station_9027.csv'
PROTOCOL = ROOT / 'docs/重叠感知历史检索/02_L0冻结协议与资源.md'


class DataGate(RuntimeError):
    """A preregistered data prerequisite failed, not a performance STOP."""


def sha_array(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def project_simplex(x):
    x = np.asarray(x, dtype=np.float64)
    u = np.sort(x)[::-1]
    css = np.cumsum(u) - 1
    idx = np.arange(1, len(x) + 1)
    rho = np.flatnonzero(u - css / idx > 0)[-1]
    return np.maximum(x - css[rho] / (rho + 1), 0)


def overlap_kernel(origins, horizon=H):
    origins = np.asarray(origins, dtype=np.int64)
    return np.maximum(0, horizon - np.abs(origins[:, None] - origins[None, :])) / horizon


def solve_weights(p, kernel):
    p, kernel = np.asarray(p, dtype=np.float64), np.asarray(kernel, dtype=np.float64)
    n = len(p)
    if kernel.shape != (n, n) or not np.isfinite(kernel).all():
        raise ValueError('invalid kernel')
    if not np.allclose(kernel, kernel.T, atol=1e-12) or np.linalg.eigvalsh(kernel).min() < -1e-10:
        raise ValueError('kernel must be symmetric PSD')
    if not np.isfinite(p).all() or p.min() < 0 or abs(p.sum() - 1) > 1e-10:
        raise ValueError('invalid probability vector')

    def fun(w):
        return float((w - p) @ (w - p) + w @ kernel @ w)

    def jac(w):
        return 2 * (w - p) + 2 * kernel @ w

    result = minimize(fun, p.copy(), jac=jac, method='SLSQP', bounds=[(0., 1.)] * n,
                      constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1,
                                    'jac': lambda w: np.ones(n)}],
                      options={'ftol': 1e-14, 'maxiter': 1000})
    w = result.x
    constraint = max(abs(w.sum() - 1), max(0., -float(w.min())))
    kkt = float(np.max(np.abs(w - project_simplex(w - jac(w)))))
    if not result.success or constraint > 1e-8 or kkt > 1e-6:
        raise RuntimeError(f'QP failure: {result.message}; constraint={constraint}, kkt={kkt}')
    return w, {'constraint_error': float(constraint), 'projected_gradient_inf': kkt,
               'iterations': int(result.nit), 'success': True}


def permutation(seed, origin, n=K):
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, int(origin)]))).permutation(n)


def cosine_matrix(keys):
    norms = np.linalg.norm(keys, axis=1)
    both_zero = (norms[:, None] <= 1e-12) & (norms[None, :] <= 1e-12)
    denom = norms[:, None] * norms[None, :]
    valid = (norms[:, None] > 1e-12) & (norms[None, :] > 1e-12)
    cos = np.zeros_like(denom)
    np.divide(keys @ keys.T, denom, out=cos, where=valid)
    cos[both_zero] = 1
    return np.clip(cos, -1, 1)


def mmr_indices(keys, distances, origins, count=8):
    span = float(np.ptp(distances))
    relevance = (distances.max() - distances) / span if span > 1e-12 else np.ones(len(distances))
    redundancy = (1 + cosine_matrix(keys)) / 2
    selected = [int(np.lexsort((origins, distances))[0])]
    while len(selected) < min(count, len(origins)):
        available = np.array([i for i in range(len(origins)) if i not in selected])
        score = .5 * relevance[available] - .5 * redundancy[np.ix_(available, selected)].max(axis=1)
        selected.append(int(available[np.lexsort((origins[available], -score))[0]]))
    return selected


def restricted_weights(p, selected):
    w = np.zeros_like(p)
    mass = p[selected].sum()
    if mass <= 0 or not np.isfinite(mass):
        raise RuntimeError('zero selected probability mass')
    w[selected] = p[selected] / mass
    return w


def all_weights(keys, distances, origins, query_origin):
    # All arrays follow the same distance/origin-ordered Top-K list.
    p = softmax(-distances)
    kernel = overlap_kernel(origins)
    weights = {'UNIFORM': np.ones(K) / K, 'SIM': p, 'ISO': (p + 1 / K) / 2}
    sep, block, seen = [], [], set()
    for i, origin in enumerate(origins):
        if all(abs(int(origin) - int(origins[j])) >= H for j in sep):
            sep.append(i)
        bucket = int(origin) // H
        if bucket not in seen:
            seen.add(bucket)
            block.append(i)
    weights['SEP'] = restricted_weights(p, sep)
    weights['BLOCK'] = restricted_weights(p, block)
    weights['MMR'] = restricted_weights(p, mmr_indices(keys, distances, origins))
    weights['OARR'], audit = solve_weights(p, kernel)
    solvers = {'OARR': audit}
    for seed in PERM_SEEDS:
        order = permutation(seed, query_origin)
        name = f'PERM{seed}'
        weights[name], solvers[name] = solve_weights(p, kernel[np.ix_(order, order)])
    return weights, solvers, kernel


def clean_segments(raw, edges):
    raw = np.asarray(raw, dtype=np.float64).copy()
    raw[~np.isfinite(raw) | (raw <= 0)] = np.nan
    filled = raw.copy()
    missing = {}
    for name, lo, hi in zip(('A', 'B', 'Q'), edges[:-1], edges[1:]):
        rate = float(np.isnan(raw[lo:hi]).mean())
        missing[name] = rate
        if rate > .1:
            raise DataGate(f'{name} original missing rate {rate} > 10%')
        # State resets at every segment boundary. Original targets remain separate.
        filled[lo:hi] = pd.Series(raw[lo:hi]).ffill(limit=6).to_numpy()
    return raw, filled, missing


def load_prefix(path=SOURCE):
    times = pd.read_csv(path, usecols=['time'])['time']
    parsed = pd.DatetimeIndex(pd.to_datetime(times, errors='raise'))
    n = len(parsed)
    if n != 8760 or parsed.has_duplicates or parsed.hasnans or not np.all(np.diff(parsed.asi8) == 3600 * 10**9):
        raise DataGate('require original 8760 unique consecutive hourly timestamps')
    edges = [0, 2 * n // 5, n // 2, 3 * n // 5]
    prefix = pd.read_csv(path, usecols=['time', 'PM25_Concentration'], nrows=edges[-1])
    if len(prefix) != edges[-1] or not pd.DatetimeIndex(pd.to_datetime(prefix['time'])).equals(parsed[:edges[-1]]):
        raise DataGate('feature prefix does not match timestamp boundary')
    raw = pd.to_numeric(prefix['PM25_Concentration'], errors='coerce').to_numpy(dtype=np.float64)
    original, filled, missing = clean_segments(raw, edges)
    observed_a = original[:edges[1]]
    mu, sd = float(np.nanmean(observed_a)), float(np.nanstd(observed_a, ddof=0))
    if not np.isfinite(sd) or sd <= 1e-12:
        raise DataGate('degenerate A scale')
    meta = {'source': str(path.relative_to(ROOT)), 'total_timestamp_rows': n,
            'feature_rows_read': len(prefix), 'feature_stop_exclusive': edges[-1], 'edges': edges,
            'unread_feature_rows': n - edges[-1], 'post60_feature_access_count': 0,
            'post60_target_access_count': 0, 'post60_prediction_count': 0,
            'columns': ['time', 'PM25_Concentration'], 'original_missing_rates': missing,
            'scale_fit_slice': [0, edges[1]], 'mean': mu, 'std_ddof0': sd,
            'timestamp_sha256': sha_array(parsed.asi8), 'prefix_numeric_sha256': sha_array(original)}
    return parsed, original, (filled - mu) / sd, edges, meta


def windows(z, original, lo, hi):
    origins = []
    for t in range(max(L, lo), hi - H + 1):
        if np.isfinite(z[t-L:t]).all() and np.isfinite(original[t:t+H]).all():
            origins.append(t)
    origins = np.asarray(origins, dtype=np.int64)
    if not len(origins):
        raise DataGate(f'no legal windows in {lo}:{hi}')
    # X [N,L]; Y [N,H], targets never replaced with imputed values.
    return origins, np.stack([z[t-L:t] for t in origins]), np.stack([original[t:t+H] for t in origins])


def state_hash(model):
    return sha_array(np.concatenate([model.coef_.ravel(), np.atleast_1d(model.intercept_).ravel()]))


def version_info():
    return {'python': sys.version.split()[0], 'numpy': np.__version__, 'scipy': scipy.__version__,
            'sklearn': sklearn.__version__, 'pandas': pd.__version__}


def git_provenance():
    dirty = subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True)
    if dirty.strip():
        raise RuntimeError('formal execution requires a clean tree before reading any station data')
    return {'commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
            'protocol_theory_commit': 'a920250', 'protocol_sha256': hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'implementation_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                      for name in ('oarr_l0.py', 'summarize_oarr_l0.py', 'test_oarr_l0.py')},
            'dirty': False}


def run(output):
    output = Path(output)
    if output.exists():
        raise RuntimeError('output exists; never overwrite a formal run')
    provenance = git_provenance()
    if version_info() != VERSIONS:
        raise RuntimeError(f'dependency mismatch: {version_info()}')
    output.mkdir(parents=True, exist_ok=False)
    provenance['versions'] = version_info()
    (output / 'manifest.json').write_text(json.dumps(provenance, indent=2))
    try:
        times, original, z, edges, meta = load_prefix()
        a_origins, a_x, a_y = windows(z, original, 0, edges[1])
        b_origins, b_x, b_y = windows(z, original, edges[1], edges[2])
        model = Ridge(alpha=100., solver='svd', fit_intercept=True).fit(a_x, a_y)
        before = state_hash(model)
        b_base = model.predict(b_x)
        residuals = b_y - b_base
        rows, counts, memory_rows = [], [], []
        for origin, pred, truth in zip(b_origins, b_base, b_y):
            for h in range(H):
                memory_rows.append({'origin': int(origin), 'h': h+1, 'target_index': int(origin+h),
                                    'truth': float(truth[h]), 'base': float(pred[h]),
                                    'residual': float(truth[h]-pred[h])})
        blocks = np.array_split(np.arange(edges[2], edges[3]), 4)
        max_kkt, max_constraint = 0., 0.
        with (output / 'retrieval_audit.jsonl').open('w') as audit_file:
            for block_id, block in enumerate(blocks):
                q_origins, q_x, q_y = windows(z, original, int(block[0]), int(block[-1])+1)
                q_base = model.predict(q_x)
                evaluated = 0
                for t, query, truth, base in zip(q_origins, q_x, q_y, q_base):
                    eligible = np.flatnonzero(b_origins + H <= t - H)
                    if len(eligible) < K:
                        continue
                    distances = np.mean((b_x[eligible] - query)**2, axis=1)
                    order = np.lexsort((b_origins[eligible], distances))[:K]
                    selected = eligible[order]
                    d = distances[order]
                    origins = b_origins[selected]
                    weights, solvers, kernel = all_weights(b_x[selected], d, origins, int(t))
                    predictions = {'BASE': base}
                    predictions.update({name: base + w @ residuals[selected] for name, w in weights.items()})
                    if not all(np.isfinite(v).all() for v in predictions.values()):
                        raise RuntimeError('nonfinite prediction')
                    for h in range(H):
                        rows.append({'origin': int(t), 'block': block_id, 'h': h+1,
                                     'target_index': int(t+h), 'target_time': str(times[t+h]),
                                     'truth': float(truth[h]), **{name: float(predictions[name][h]) for name in METHODS}})
                    max_kkt = max(max_kkt, *(a['projected_gradient_inf'] for a in solvers.values()))
                    max_constraint = max(max_constraint, *(a['constraint_error'] for a in solvers.values()))
                    diagnostic = {'origin': int(t), 'block': block_id,
                                  'candidate_origins': origins.tolist(), 'distance': d.tolist(),
                                  'weights': {name: w.tolist() for name, w in weights.items()},
                                  'solvers': solvers,
                                  'effective_count': {name: float(1 / (w @ w)) for name, w in weights.items()},
                                  'support_count': {name: int((w > 1e-12).sum()) for name, w in weights.items()},
                                  'overlap_offdiag_sum': float(kernel.sum() - K)}
                    audit_file.write(json.dumps(diagnostic, allow_nan=False) + '\n')
                    evaluated += 1
                if evaluated < .8 * len(q_origins):
                    raise DataGate(f'block {block_id}: evaluated fewer than 80% legal Q queries')
                counts.append({'block': block_id, 'start': int(block[0]), 'stop': int(block[-1])+1,
                               'legal_queries': len(q_origins), 'evaluated_queries': evaluated})
                print(f'block {block_id+1}/4 complete: {evaluated}', flush=True)
        after = state_hash(model)
        assert before == after
        assert max(a_origins + H - 1) < edges[1] <= min(b_origins)
        assert max(b_origins + H - 1) < edges[2]
        pred_frame = pd.DataFrame(rows)
        assert pred_frame.target_index.max() < edges[3]
        target_sets = [set(pred_frame.loc[pred_frame.block == i, 'target_index']) for i in range(4)]
        assert all(not target_sets[i] & target_sets[j] for i in range(4) for j in range(i))
        meta.update({'model': 'Ridge(alpha=100,solver=svd,fit_intercept=True)', 'fit_calls': 1,
                     'train_windows': len(a_origins), 'memory_windows': len(b_origins),
                     'train_target_max': int(max(a_origins+H-1)),
                     'memory_target_min': int(min(b_origins)), 'memory_target_max': int(max(b_origins+H-1)),
                     'blocks': counts, 'baseline_state_before': before, 'baseline_state_after': after,
                     'max_kkt': max_kkt, 'max_constraint': max_constraint, 'passed': True})
        pred_frame.to_csv(output / 'predictions.csv', index=False)
        pd.DataFrame(memory_rows).to_csv(output / 'memory_predictions.csv', index=False)
        (output / 'integrity.json').write_text(json.dumps(meta, indent=2, allow_nan=False))
        return output
    except Exception as exc:
        status = 'DATA_GATE' if isinstance(exc, DataGate) else 'ENGINEERING_FAILURE'
        (output / 'failure.json').write_text(json.dumps({'status': status, 'error': str(exc)}, indent=2))
        raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'experiments/results/oarr/l0_9027')
    args = parser.parse_args()
    run(args.output_dir)
