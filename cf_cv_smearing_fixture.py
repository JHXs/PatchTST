"""Round 11 CF-CV-CS synthetic fixture.

This module is deliberately self contained.  It never reads project data, model
checkpoints, or previous experiment outputs.  The neural adapter uses the
``tsai`` PatchTST implementation only when a fit is requested.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


FORMAL_SEEDS = (4001, 4002, 4003)
HISTORICAL_SEEDS = (
    tuple(range(2024, 2062))
    + tuple(range(3101, 3151))
    + tuple(range(20260939, 20260969))
)
TASKS = {"H1": (24, 1), "H6": (168, 6)}
METHODS = (
    "raw_mse_base",
    "naive_log",
    "global_smearing",
    "hpc",
    "strongest_existing_conditional_smearing",
    "cf_cv_cs_removal",
    "cf_cv_cs",
)
FIT_STAGES = ("raw_full", "log_oof", "log_full")
PATCHTST_CONFIG = {
    "model": "PatchTST",
    "input_channels": 1,
    "output_channels": 1,
    "n_layers": 2,
    "n_heads": 2,
    "d_model": 16,
    "d_ff": 64,
    "attn_dropout": 0.0,
    "dropout": 0.10,
    "patch_len": 8,
    "stride": 4,
    "padding_patch": True,
    "optimizer": "AdamW",
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "loss": "MSE",
    "max_epochs": 40,
    "early_stopping_patience": 8,
    "inner_valid_fraction": 0.20,
}
SMOOTHER_BANDWIDTH = 1.0
SMOOTHER_MAX_SUPPORT = 128  # frozen by docs/round11-正值目标估计/12_主代理工程修复授权.md


@dataclass(frozen=True)
class SequenceDGP:
    seed: int
    variant: str
    timestamps: np.ndarray
    a: np.ndarray
    q: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    epsilon: np.ndarray
    z: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class WindowTask:
    name: str
    history: int
    horizon: int
    X_raw: np.ndarray
    y_raw: np.ndarray
    z_target: np.ndarray
    target_timestamps: np.ndarray
    origins: np.ndarray
    split_indices: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class OOFBlock:
    block: int
    query_indices: np.ndarray
    fit_indices: np.ndarray
    inner_fit_indices: np.ndarray
    inner_valid_indices: np.ndarray
    query_min_timestamp: int
    fit_max_timestamp: int | None
    purge: int
    excluded_indices: np.ndarray
    exclusion_reason: str | None = None


@dataclass(frozen=True)
class OOFPlan:
    task: str
    history: int
    horizon: int
    train_indices: np.ndarray
    warmup_indices: np.ndarray
    blocks: tuple[OOFBlock, ...]
    exclusions: Mapping[str, np.ndarray]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_sequence(seed: int, *, n: int = 4096, burn_in: int = 512, variant: str = "heteroskedastic") -> SequenceDGP:
    """Generate exactly one absolute-time state/target sequence."""
    if variant not in {"heteroskedastic", "homoskedastic", "additive_raw"}:
        raise ValueError(f"unknown DGP variant: {variant}")
    if n < 32 or burn_in < 0:
        raise ValueError("n must be at least 32 and burn_in must be non-negative")
    streams = np.random.SeedSequence(int(seed)).spawn(4)
    rng_a, rng_q, rng_e, rng_eta = (np.random.default_rng(s) for s in streams)
    total = n + burn_in
    abs_s = np.arange(1, total + 1, dtype=np.int64)
    a = np.empty(total, dtype=np.float64)
    q = np.empty(total, dtype=np.float64)
    previous_a = 0.0
    previous_q = 0.0
    noise_bound = np.sqrt(3.0)
    for i in range(total):
        xi_a = rng_a.uniform(-noise_bound, noise_bound)
        xi_q = rng_q.uniform(-noise_bound, noise_bound)
        a[i] = np.clip(0.92 * previous_a + 0.18 * xi_a, -2.0, 2.0)
        q[i] = np.clip(0.80 * previous_q + 0.25 * xi_q, -2.0, 2.0)
        previous_a, previous_q = a[i], q[i]
    mu = 2.40 + 0.35 * a + 0.18 * q + 0.10 * np.sin(2.0 * np.pi * abs_s / 24.0)
    if variant == "heteroskedastic":
        sigma = 0.12 + 0.18 * _sigmoid(1.10 * a + 0.50 * q)
        epsilon = rng_e.uniform(-noise_bound, noise_bound, total)
        z = mu + sigma * epsilon
        y = np.exp(z) - 1.0
    elif variant == "homoskedastic":
        sigma = np.full(total, 0.24, dtype=np.float64)
        epsilon = rng_e.uniform(-noise_bound, noise_bound, total)
        z = mu + sigma * epsilon
        y = np.exp(z) - 1.0
    else:
        sigma = np.ones(total, dtype=np.float64)
        epsilon = rng_eta.uniform(-1.0, 1.0, total)
        nu = 6.0 + 0.8 * a + 0.4 * q
        y = nu + epsilon
        z = np.log1p(y)
    retained = slice(burn_in, None)
    result = SequenceDGP(
        seed=int(seed), variant=variant, timestamps=abs_s[retained], a=a[retained], q=q[retained],
        mu=mu[retained], sigma=sigma[retained], epsilon=epsilon[retained], z=z[retained], y=y[retained],
    )
    assert_sequence_invariants(result)
    return result


def assert_sequence_invariants(sequence: SequenceDGP) -> None:
    if not np.all(sequence.y >= 0.0):
        raise AssertionError("DGP generated a negative positive-target value")
    if not np.all(np.isfinite(sequence.z)):
        raise AssertionError("DGP generated a non-finite log1p target")
    if np.unique(sequence.timestamps).size != sequence.timestamps.size:
        raise AssertionError("absolute timestamps are not unique")
    if sequence.variant == "heteroskedastic" and not np.all(sequence.sigma < 0.30):
        raise AssertionError("heteroskedastic sigma violates the frozen bound")


def _task_name(history: int, horizon: int) -> str:
    for name, value in TASKS.items():
        if value == (history, horizon):
            return name
    return f"H{horizon}_L{history}"


def build_windows(sequence: SequenceDGP, history: int, horizon: int) -> WindowTask:
    """Cut windows from the one sequence; target labels retain absolute timestamps."""
    if history <= 0 or horizon <= 0 or history + horizon > len(sequence.y):
        raise ValueError("invalid history/horizon")
    origins = np.arange(history - 1, len(sequence.y) - horizon, dtype=np.int64)
    x = np.stack([sequence.y[o - history + 1 : o + 1] for o in origins], axis=0)[:, None, :]
    y = np.stack([sequence.y[o + 1 : o + horizon + 1] for o in origins], axis=0)[:, None, :]
    z = np.stack([sequence.z[o + 1 : o + horizon + 1] for o in origins], axis=0)
    timestamps = np.stack([sequence.timestamps[o + 1 : o + horizon + 1] for o in origins], axis=0)
    task = WindowTask(
        name=_task_name(history, horizon), history=history, horizon=horizon, X_raw=x, y_raw=y,
        z_target=z, target_timestamps=timestamps, origins=origins, split_indices={}
    )
    assert_window_labels(sequence, task)
    return task


def build_split_indices(task: WindowTask, train_fraction: float = 0.70, valid_fraction: float = 0.10) -> dict[str, np.ndarray]:
    retained_start = int(task.target_timestamps.min()) - task.history
    # Split in retained-sequence coordinates, not in the number of overlapping windows.
    total = int(task.target_timestamps.max() - retained_start + 1)
    train_end = retained_start + int(np.floor(total * train_fraction))
    valid_end = retained_start + int(np.floor(total * (train_fraction + valid_fraction)))
    target = task.target_timestamps
    train = np.flatnonzero(np.all(target < train_end, axis=1))
    valid = np.flatnonzero(np.all((target >= train_end) & (target < valid_end), axis=1))
    test = np.flatnonzero(np.all(target >= valid_end, axis=1))
    return {"train": train, "valid": valid, "test": test}


def with_splits(task: WindowTask, splits: Mapping[str, np.ndarray] | None = None) -> WindowTask:
    return WindowTask(**{**asdict(task), "split_indices": dict(splits or build_split_indices(task))})


def assert_window_labels(sequence: SequenceDGP, task: WindowTask) -> None:
    relative = task.target_timestamps - sequence.timestamps[0]
    expected_y = sequence.y[relative]
    if task.name.startswith("H") and not np.array_equal(task.y_raw[:, 0, :], expected_y):
        raise AssertionError("window Y does not equal the unique sequence target")
    if not np.array_equal(task.z_target, sequence.z[relative]):
        raise AssertionError("window Z does not equal the unique sequence target")
    labels: dict[int, list[float]] = {}
    for stamp, value in zip(task.target_timestamps.ravel(), task.y_raw[:, 0, :].ravel()):
        labels.setdefault(int(stamp), []).append(float(value))
    for stamp, values in labels.items():
        if not np.array_equal(np.asarray(values), np.repeat(sequence.y[stamp - sequence.timestamps[0]], len(values))):
            raise AssertionError(f"duplicate absolute timestamp {stamp} has inconsistent labels")


def assert_cross_task_labels(sequence: SequenceDGP, tasks: Iterable[WindowTask]) -> None:
    """Check all repeated target timestamps across H=1/H=6 and across leads."""
    all_labels: dict[int, list[float]] = {}
    all_logs: dict[int, list[float]] = {}
    for task in tasks:
        assert_window_labels(sequence, task)
        for stamp, value, log_value in zip(task.target_timestamps.ravel(), task.y_raw[:, 0, :].ravel(), task.z_target.ravel()):
            all_labels.setdefault(int(stamp), []).append(float(value))
            if np.isfinite(log_value):
                all_logs.setdefault(int(stamp), []).append(float(log_value))
    for stamp, values in all_labels.items():
        expected = sequence.y[stamp - sequence.timestamps[0]]
        if not np.array_equal(np.asarray(values), np.repeat(expected, len(values))):
            raise AssertionError(f"cross-task Y mismatch at absolute timestamp {stamp}")
    for stamp, values in all_logs.items():
        expected = sequence.z[stamp - sequence.timestamps[0]]
        if not np.array_equal(np.asarray(values), np.repeat(expected, len(values))):
            raise AssertionError(f"cross-task Z mismatch at absolute timestamp {stamp}")


def fit_input_scaler(task: WindowTask, train_indices: Sequence[int]) -> dict[str, float]:
    values = np.asarray(task.X_raw)[np.asarray(train_indices)]
    mean = float(values.mean())
    scale = float(values.std())
    return {"mean": mean, "scale": scale if scale > 0.0 else 1.0}


def fit_target_scaler(target: np.ndarray, fit_indices: Sequence[int]) -> dict[str, list[float]]:
    """Fit per-lead target normalization on the current fit prefix only."""
    values = np.asarray(target, dtype=np.float64)[np.asarray(fit_indices)]
    values = values[:, 0, :] if values.ndim == 3 else values
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale == 0.0] = 1.0
    return {"mean": mean.tolist(), "scale": scale.tolist()}


def transform_target(target: np.ndarray, scaler: Mapping[str, Sequence[float]]) -> np.ndarray:
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    scale = np.asarray(scaler["scale"], dtype=np.float64)
    return ((np.asarray(target, dtype=np.float64) - mean.reshape(1, 1, -1)) / scale.reshape(1, 1, -1)).astype(np.float32)


def inverse_target(prediction: np.ndarray, scaler: Mapping[str, Sequence[float]]) -> np.ndarray:
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    scale = np.asarray(scaler["scale"], dtype=np.float64)
    return np.asarray(prediction, dtype=np.float64) * scale.reshape(1, 1, -1) + mean.reshape(1, 1, -1)


def transform_inputs(task: WindowTask, scaler: Mapping[str, float]) -> np.ndarray:
    return ((task.X_raw - float(scaler["mean"])) / float(scaler["scale"])).astype(np.float32)


def split_train_inner(indices: Sequence[int], target_timestamps: np.ndarray, purge: int, valid_fraction: float = 0.20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train-inner fit, validation, and excluded indices using forward purge."""
    ordered = np.asarray(indices, dtype=np.int64)
    if ordered.size < 4:
        raise ValueError("not enough records for train-inner blocked validation")
    ordered = ordered[np.argsort(target_timestamps[ordered].max(axis=1), kind="stable")]
    valid_count = max(1, int(np.ceil(ordered.size * valid_fraction)))
    valid = ordered[-valid_count:]
    valid_min = int(target_timestamps[valid].min())
    fit = ordered[target_timestamps[ordered].max(axis=1) <= valid_min - purge]
    excluded = np.setdiff1d(ordered, np.concatenate([fit, valid]), assume_unique=False)
    if fit.size < 2 or valid.size < 1:
        raise ValueError("train-inner blocked validation has no legal fit prefix")
    return fit, valid, excluded


def build_oof_plan(task: WindowTask, *, n_blocks: int = 5, warmup_fraction: float = 0.20) -> OOFPlan:
    if not task.split_indices:
        task = with_splits(task)
    train = np.asarray(task.split_indices["train"], dtype=np.int64)
    ordered = train[np.argsort(task.target_timestamps[train].max(axis=1), kind="stable")]
    warmup_count = max(8, int(np.ceil(len(ordered) * warmup_fraction)))
    if warmup_count >= len(ordered) - n_blocks:
        raise ValueError("warm-up leaves too few OOF query records")
    warmup = ordered[:warmup_count]
    query_chunks = tuple(np.asarray(chunk, dtype=np.int64) for chunk in np.array_split(ordered[warmup_count:], n_blocks))
    purge = task.history + task.horizon + 1
    blocks: list[OOFBlock] = []
    excluded_warmup = warmup.copy()
    for block_id, query in enumerate(query_chunks):
        q_min = int(task.target_timestamps[query].min())
        legal = ordered[task.target_timestamps[ordered].max(axis=1) <= q_min - purge]
        try:
            inner_fit, inner_valid, inner_excluded = split_train_inner(legal, task.target_timestamps, purge)
            reason = None
        except ValueError:
            inner_fit = np.empty(0, dtype=np.int64)
            inner_valid = np.empty(0, dtype=np.int64)
            inner_excluded = legal.copy()
            reason = "no_legal_train_inner_prefix"
        blocks.append(OOFBlock(
            block=block_id, query_indices=query, fit_indices=legal,
            inner_fit_indices=inner_fit, inner_valid_indices=inner_valid,
            query_min_timestamp=q_min,
            fit_max_timestamp=int(task.target_timestamps[legal].max()) if legal.size else None,
            purge=purge, excluded_indices=inner_excluded, exclusion_reason=reason,
        ))
    exclusions = {"warmup": excluded_warmup, "train_inner": np.concatenate([b.excluded_indices for b in blocks])}
    return OOFPlan(task=task.name, history=task.history, horizon=task.horizon, train_indices=train,
                   warmup_indices=warmup, blocks=tuple(blocks), exclusions=exclusions)


def gaussian_mgf_uniform(u: np.ndarray | float) -> np.ndarray | float:
    """MGF of Uniform[-sqrt(3), sqrt(3)], with a stable zero continuation."""
    values = np.asarray(u, dtype=np.float64)
    x = np.sqrt(3.0) * values
    out = np.ones_like(values)
    nonzero = np.abs(x) > 1e-8
    out[nonzero] = np.sinh(x[nonzero]) / x[nonzero]
    out[~nonzero] = 1.0 + (x[~nonzero] ** 2) / 6.0 + (x[~nonzero] ** 4) / 120.0
    return float(out) if np.ndim(u) == 0 else out


uniform_mgf = gaussian_mgf_uniform


def apply_raw_boundary(log_mean: np.ndarray, factor: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, np.exp(np.asarray(log_mean, dtype=np.float64)) * np.asarray(factor, dtype=np.float64) - 1.0)


def weighted_stats(values: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float]:
    x = np.asarray(values, dtype=np.float64).ravel()
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    w = w / w.sum()
    mean = float(np.dot(w, x))
    return mean, float(np.dot(w, (x - mean) ** 2))


def hpc_factor(residual: np.ndarray, weights: np.ndarray | None = None) -> float:
    delta, variance = weighted_stats(residual, weights)
    return float(np.exp(delta) * gaussian_mgf_uniform(np.sqrt(max(variance, 0.0))))


class RectangularLinearOperator:
    """A sparse-row linear operator with explicit rectangular dimensions."""

    def __init__(self, shape: tuple[int, int], row_indices: Sequence[np.ndarray], row_weights: Sequence[np.ndarray], *, name: str = "H_E"):
        if len(shape) != 2 or shape[0] < 0 or shape[1] < 0:
            raise ValueError("operator shape must be two-dimensional")
        if len(row_indices) != shape[0] or len(row_weights) != shape[0]:
            raise ValueError("operator rows do not match shape")
        self.shape = (int(shape[0]), int(shape[1]))
        self.row_indices = tuple(np.asarray(v, dtype=np.int64) for v in row_indices)
        self.row_weights = tuple(np.asarray(v, dtype=np.float64) for v in row_weights)
        self.name = name
        for ids, weights in zip(self.row_indices, self.row_weights):
            if ids.ndim != 1 or weights.ndim != 1 or ids.size != weights.size:
                raise ValueError("operator row index/weight mismatch")
            if ids.size and (ids.min() < 0 or ids.max() >= self.shape[1]):
                raise ValueError("operator row references an invalid column")

    def apply(self, vector: np.ndarray) -> np.ndarray:
        v = np.asarray(vector, dtype=np.float64).ravel()
        if v.size != self.shape[1]:
            raise ValueError(f"{self.name} expects vector length {self.shape[1]}, got {v.size}")
        return np.asarray([float(np.dot(w, v[i])) if i.size else 0.0 for i, w in zip(self.row_indices, self.row_weights)], dtype=np.float64)

    def to_dense(self, *, max_entries: int = 2_000_000) -> np.ndarray:
        if self.shape[0] * self.shape[1] > max_entries:
            raise MemoryError("dense materialization is disabled for this operator")
        matrix = np.zeros(self.shape, dtype=np.float64)
        for row, (ids, weights) in enumerate(zip(self.row_indices, self.row_weights)):
            matrix[row, ids] = weights
        return matrix

    def compose(self, right: "RectangularLinearOperator") -> "RectangularLinearOperator":
        if not isinstance(right, RectangularLinearOperator) or self.shape[1] != right.shape[0]:
            raise ValueError("rectangular operator composition has incompatible dimensions")
        raise NotImplementedError("dense composition is intentionally not part of CF-CV-CS")

    def __matmul__(self, other: Any) -> Any:
        raise TypeError("矩形 H_E 不支持 H_E²；请使用 apply(vector) 或显式 compose")

    def __pow__(self, exponent: int) -> Any:
        raise TypeError("矩形 H_E 不支持 H_E²；旧方阵接口已删除")


def _local_linear_row(query: np.ndarray, fit: np.ndarray, bandwidth: float, fit_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if fit.shape[0] == 0:
        raise ValueError("smoother has no legal fit records")
    if bandwidth <= 0.0:
        raise ValueError("bandwidth must be positive")
    original_indices = np.arange(fit.shape[0], dtype=np.int64)
    delta = (fit - query[None, :]) / float(bandwidth)
    distance = np.sum(delta * delta, axis=1)
    if fit.shape[0] > SMOOTHER_MAX_SUPPORT:
        keep = np.argsort(distance, kind="stable")[:SMOOTHER_MAX_SUPPORT]
        fit = fit[keep]
        fit_weights = np.asarray(fit_weights)[keep]
        original_indices = original_indices[keep]
        delta = (fit - query[None, :]) / float(bandwidth)
        distance = np.sum(delta * delta, axis=1)
    kernel = np.exp(-0.5 * np.minimum(distance, 700.0)) * np.asarray(fit_weights, dtype=np.float64)
    design = np.column_stack([np.ones(fit.shape[0]), delta])
    gram = design.T @ (kernel[:, None] * design)
    gram[1:, 1:] += 1e-10 * np.eye(fit.shape[1])
    rhs = design.T * kernel[None, :]
    try:
        row = np.linalg.solve(gram, rhs)[0]
    except np.linalg.LinAlgError:
        row = kernel / max(float(kernel.sum()), 1e-300)
    total = float(row.sum())
    if not np.isfinite(total) or abs(total) < 1e-14:
        row = kernel / max(float(kernel.sum()), 1e-300)
    else:
        row = row / total
    return original_indices, row.astype(np.float64)


def build_smoother_operator(query_c: np.ndarray, fit_c: np.ndarray, *, bandwidth: float = 1.0, fit_weights: np.ndarray | None = None, column_indices: np.ndarray | None = None, name: str = "H_E") -> RectangularLinearOperator:
    if float(bandwidth) != SMOOTHER_BANDWIDTH:
        raise ValueError("bandwidth selection is frozen at 1.0")
    query = np.asarray(query_c, dtype=np.float64)
    fit = np.asarray(fit_c, dtype=np.float64)
    if query.ndim != 2 or fit.ndim != 2 or query.shape[1] != fit.shape[1]:
        raise ValueError("query and fit condition matrices must have matching dimensions")
    fw = np.ones(fit.shape[0], dtype=np.float64) if fit_weights is None else np.asarray(fit_weights, dtype=np.float64)
    if fw.shape != (fit.shape[0],):
        raise ValueError("fit weights have the wrong shape")
    columns = np.arange(fit.shape[0], dtype=np.int64) if column_indices is None else np.asarray(column_indices, dtype=np.int64)
    if columns.shape != (fit.shape[0],):
        raise ValueError("column_indices have the wrong shape")
    rows: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for q in query:
        local, row = _local_linear_row(q, fit, bandwidth, fw)
        rows.append(columns[local])
        weights.append(row)
    return RectangularLinearOperator((query.shape[0], fit.shape[0] if column_indices is None else int(columns.max() + 1) if columns.size else 0), rows, weights, name=name)


def build_block_operator(query_c: np.ndarray, fit_c: np.ndarray, query_leads: np.ndarray, fit_leads: np.ndarray, *, bandwidth: float = 1.0, name: str = "H_E") -> RectangularLinearOperator:
    """Build H_E over the fixed, all-lead training-record column order."""
    if float(bandwidth) != SMOOTHER_BANDWIDTH:
        raise ValueError("bandwidth selection is frozen at 1.0")
    query = np.asarray(query_c, dtype=np.float64)
    fit = np.asarray(fit_c, dtype=np.float64)
    q_leads = np.asarray(query_leads, dtype=np.int64).ravel()
    f_leads = np.asarray(fit_leads, dtype=np.int64).ravel()
    if query.shape[0] != q_leads.size or fit.shape[0] != f_leads.size:
        raise ValueError("lead ids do not match condition rows")
    all_rows: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    for lead in np.unique(q_leads):
        q_mask = np.flatnonzero(q_leads == lead)
        f_mask = np.flatnonzero(f_leads == lead)
        if f_mask.size == 0:
            raise ValueError(f"no legal training fit records for lead {lead}")
        for row in q_mask:
            local, weights = _local_linear_row(query[row], fit[f_mask], bandwidth, np.ones(f_mask.size))
            all_rows.append(f_mask[local])
            all_weights.append(weights)
    # q_mask iteration is lead-grouped; restore original query row order.
    order = np.concatenate([np.flatnonzero(q_leads == lead) for lead in np.unique(q_leads)]) if q_leads.size else np.empty(0, dtype=np.int64)
    if order.size:
        restored_rows = [None] * order.size
        restored_weights = [None] * order.size
        for grouped_pos, original_pos in enumerate(order):
            restored_rows[original_pos] = all_rows[grouped_pos]
            restored_weights[original_pos] = all_weights[grouped_pos]
        all_rows, all_weights = restored_rows, restored_weights
    return RectangularLinearOperator((query.shape[0], fit.shape[0]), all_rows, all_weights, name=name)


def build_causal_operator(query_c: np.ndarray, fit_c: np.ndarray, query_leads: np.ndarray, fit_leads: np.ndarray,
                          query_timestamps: np.ndarray, fit_timestamps: np.ndarray, *, purge: int,
                          query_block_min: np.ndarray | None = None, bandwidth: float = 1.0,
                          name: str = "B") -> RectangularLinearOperator:
    """Build a block-aware causal operator without filling query rows into fit columns."""
    if float(bandwidth) != SMOOTHER_BANDWIDTH:
        raise ValueError("bandwidth selection is frozen at 1.0")
    query = np.asarray(query_c, dtype=np.float64)
    fit = np.asarray(fit_c, dtype=np.float64)
    q_leads = np.asarray(query_leads, dtype=np.int64).ravel()
    f_leads = np.asarray(fit_leads, dtype=np.int64).ravel()
    qt = np.asarray(query_timestamps, dtype=np.int64).ravel()
    ft = np.asarray(fit_timestamps, dtype=np.int64).ravel()
    if query.shape[0] != q_leads.size or query.shape[0] != qt.size:
        raise ValueError("query operator metadata has inconsistent lengths")
    if fit.shape[0] != f_leads.size or fit.shape[0] != ft.size:
        raise ValueError("fit operator metadata has inconsistent lengths")
    if query_block_min is None:
        q_min = qt
    else:
        q_min = np.asarray(query_block_min, dtype=np.int64).ravel()
        if q_min.size != query.shape[0]:
            raise ValueError("query block minima have the wrong length")
    rows: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for row in range(query.shape[0]):
        allowed = np.flatnonzero((f_leads == q_leads[row]) & (ft <= q_min[row] - purge))
        if allowed.size == 0:
            rows.append(np.empty(0, dtype=np.int64))
            weights.append(np.empty(0, dtype=np.float64))
            continue
        local, local_weights = _local_linear_row(query[row], fit[allowed], bandwidth, np.ones(allowed.size))
        rows.append(allowed[local])
        weights.append(local_weights)
    return RectangularLinearOperator((query.shape[0], fit.shape[0]), rows, weights, name=name)


def build_prefix_scaled_operator(query_c_raw: np.ndarray, fit_c_raw: np.ndarray,
                                 allowed_columns: Sequence[np.ndarray], *, name: str) -> RectangularLinearOperator:
    """Build rows whose condition scaler is fit independently on that row's legal prefix."""
    query = np.asarray(query_c_raw, dtype=np.float64)
    fit = np.asarray(fit_c_raw, dtype=np.float64)
    if query.ndim != 2 or fit.ndim != 2 or query.shape[1] != fit.shape[1] or len(allowed_columns) != query.shape[0]:
        raise ValueError("prefix-scaled operator inputs have inconsistent shapes")
    rows: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for row, columns in enumerate(allowed_columns):
        ids = np.asarray(columns, dtype=np.int64)
        if ids.size == 0:
            raise ValueError(f"{name} row {row} has no legal fit prefix")
        scaler = fit_condition_scaler(fit[ids])
        q_scaled = transform_conditions(query[row : row + 1], scaler)[0]
        fit_scaled = transform_conditions(fit[ids], scaler)
        local, local_weights = _local_linear_row(q_scaled, fit_scaled, SMOOTHER_BANDWIDTH, np.ones(ids.size))
        rows.append(ids[local])
        weights.append(local_weights)
    return RectangularLinearOperator((query.shape[0], fit.shape[0]), rows, weights, name=name)


def build_s_prefix_operator(query_c_raw: np.ndarray, fit_c_raw: np.ndarray,
                            query_leads: np.ndarray, fit_leads: np.ndarray,
                            allowed_columns: Sequence[np.ndarray], *, name: str) -> RectangularLinearOperator:
    """H rows with per-lead, fit-prefix-only C scaling."""
    query = np.asarray(query_c_raw, dtype=np.float64)
    fit = np.asarray(fit_c_raw, dtype=np.float64)
    q_leads = np.asarray(query_leads, dtype=np.int64).ravel()
    f_leads = np.asarray(fit_leads, dtype=np.int64).ravel()
    if query.shape[0] != q_leads.size or len(allowed_columns) != query.shape[0]:
        raise ValueError("S prefix operator inputs have inconsistent shapes")
    rows: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for row, columns in enumerate(allowed_columns):
        ids = np.asarray(columns, dtype=np.int64)
        ids = ids[f_leads[ids] == q_leads[row]]
        if ids.size == 0:
            raise ValueError(f"{name} row {row} has no same-lead legal fit prefix")
        scaler = fit_condition_scaler(fit[ids])
        q_scaled = transform_conditions(query[row : row + 1], scaler)[0]
        fit_scaled = transform_conditions(fit[ids], scaler)
        local, local_weights = _local_linear_row(q_scaled, fit_scaled, SMOOTHER_BANDWIDTH, np.ones(ids.size))
        rows.append(ids[local])
        weights.append(local_weights)
    return RectangularLinearOperator((query.shape[0], fit.shape[0]), rows, weights, name=name)


def build_conditions(task: WindowTask, log_prediction: np.ndarray, row_indices: Sequence[int] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the frozen C=(h,m,mean,std,last,slope) from visible history only."""
    prediction = np.asarray(log_prediction, dtype=np.float64)
    selected = np.arange(len(task.X_raw), dtype=np.int64) if row_indices is None else np.asarray(row_indices, dtype=np.int64)
    if prediction.shape != (len(selected), task.horizon):
        raise ValueError(f"log prediction must have shape {(len(selected), task.horizon)}, got {prediction.shape}")
    history = np.asarray(task.X_raw[selected, 0, :], dtype=np.float64)
    mean = history.mean(axis=1)
    std = history.std(axis=1)
    last = history[:, -1]
    centered_time = np.arange(task.history, dtype=np.float64) - (task.history - 1) / 2.0
    denominator = float(np.dot(centered_time, centered_time))
    slope = (history - mean[:, None]) @ centered_time / denominator
    rows = []
    leads = []
    timestamps = []
    for row in range(len(history)):
        for lead in range(task.horizon):
            rows.append([lead + 1.0, prediction[row, lead], mean[row], std[row], last[row], slope[row]])
            leads.append(lead)
            timestamps.append(task.target_timestamps[selected[row], lead])
    return np.asarray(rows, dtype=np.float64), np.asarray(leads, dtype=np.int64), np.asarray(timestamps, dtype=np.int64)


def fit_condition_scaler(condition: np.ndarray) -> dict[str, list[float]]:
    c = np.asarray(condition, dtype=np.float64)
    mean = c.mean(axis=0)
    scale = c.std(axis=0)
    scale[scale == 0.0] = 1.0
    return {"mean": mean.tolist(), "scale": scale.tolist()}


def transform_conditions(condition: np.ndarray, scaler: Mapping[str, Sequence[float]]) -> np.ndarray:
    return (np.asarray(condition, dtype=np.float64) - np.asarray(scaler["mean"], dtype=np.float64)) / np.asarray(scaler["scale"], dtype=np.float64)


def compute_control_variate(residual: np.ndarray, b_prediction: np.ndarray, *, weights: np.ndarray | None = None) -> dict[str, Any]:
    r = np.asarray(residual, dtype=np.float64).ravel()
    b = np.asarray(b_prediction, dtype=np.float64).ravel()
    if r.shape != b.shape:
        raise ValueError("residual and b prediction shapes differ")
    w = np.ones_like(r) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    if w.shape != r.shape:
        raise ValueError("control-variate weights have the wrong shape")
    w = w / w.sum()
    v = r - b
    v = v - np.dot(w, v)
    u = np.exp(r)
    u_mean = np.dot(w, u)
    variance_v = float(np.dot(w, v * v))
    covariance_uv = float(np.dot(w, (u - u_mean) * v))
    if variance_v <= 0.0 or not np.isfinite(variance_v):
        raise ValueError("STOP: control-variate Var(v) is zero or non-finite")
    lam = covariance_uv / variance_v
    return {
        "lambda": float(lam), "v": v, "u": u, "u_mean": float(u_mean),
        "var_v": variance_v, "cov_uv": covariance_uv,
        "variance_ratio": float(np.dot(w, (u - lam * v - np.dot(w, u - lam * v)) ** 2) / max(np.dot(w, (u - u_mean) ** 2), 1e-300)),
    }


def compute_prefix_control_variate(residual: np.ndarray, b_prediction: np.ndarray) -> dict[str, Any]:
    """Compute v0, same-prefix-centered vtilde, and lambda on one lead prefix."""
    r = np.asarray(residual, dtype=np.float64).ravel()
    b = np.asarray(b_prediction, dtype=np.float64).ravel()
    if r.size != b.size or r.size < 2:
        raise ValueError("control-variate prefix requires at least two records")
    v0 = r - b
    vtilde = v0 - v0.mean()
    u = np.exp(r)
    u_mean = float(u.mean())
    var_v = float(np.mean(vtilde ** 2))
    cov_uv = float(np.mean((u - u_mean) * vtilde))
    if var_v <= 0.0 or not np.isfinite(var_v):
        raise ValueError("STOP: prefix Var(vtilde) is zero or non-finite")
    lam = cov_uv / var_v
    pseudo = u - lam * vtilde
    variance_ratio = float(np.var(pseudo) / max(np.var(u), 1e-300))
    return {"v0": v0, "vtilde": vtilde, "u": u, "lambda": float(lam),
            "var_v": var_v, "cov_uv": cov_uv, "variance_ratio": variance_ratio,
            "mean_v0": float(v0.mean())}


def select_s_pool(m_timestamps: np.ndarray, m_leads: np.ndarray, purge: int) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Return immutable S<-M row selection and each S row's legal M support."""
    timestamps = np.asarray(m_timestamps, dtype=np.int64).ravel()
    leads = np.asarray(m_leads, dtype=np.int64).ravel()
    if timestamps.size != leads.size:
        raise ValueError("M timestamps/leads mismatch")
    supports: list[np.ndarray] = []
    selected: list[int] = []
    for row in range(timestamps.size):
        support = np.flatnonzero((leads == leads[row]) & (timestamps < timestamps[row]) & (timestamps <= timestamps[row] - purge))
        if support.size >= 2:
            selected.append(row)
            supports.append(support)
    return np.asarray(selected, dtype=np.int64), tuple(supports)


def verify_operator_identity(s_rem: np.ndarray, s_cv: np.ndarray, lam: float, h_e_v: np.ndarray, *, rtol: float = 1e-10, atol: float = 1e-12) -> None:
    lhs = np.asarray(s_cv, dtype=np.float64) - np.asarray(s_rem, dtype=np.float64)
    rhs = -float(lam) * np.asarray(h_e_v, dtype=np.float64)
    np.testing.assert_allclose(lhs, rhs, rtol=rtol, atol=atol)


def rho_diagnostic(h_e_u: np.ndarray, h_e_v: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(h_e_v, dtype=np.float64)) / (np.linalg.norm(np.asarray(h_e_u, dtype=np.float64)) + 1e-12))


def rho_gate_status(rho_by_lead: Sequence[float], *, threshold: float = 0.05) -> dict[str, Any]:
    values = np.asarray(rho_by_lead, dtype=np.float64)
    near_zero = values <= threshold
    required = 1 if values.size == 1 else int(np.ceil(values.size * 4.0 / 6.0))
    return {"threshold": threshold, "near_zero_by_lead": near_zero.tolist(),
            "near_zero_count": int(near_zero.sum()), "required_stop_count": required,
            "stop": bool(int(near_zero.sum()) >= required)}


class PatchTSTAdapter:
    """Small direct adapter around the real tsai PatchTST class."""

    def __init__(self, history: int, horizon: int, seed: int, *, device: str = "cpu", max_epochs: int = 40, patience: int = 8, batch_size: int | None = None):
        self.history, self.horizon, self.seed = int(history), int(horizon), int(seed)
        self.device_name, self.max_epochs, self.patience = device, int(max_epochs), int(patience)
        self.batch_size = int(batch_size or (128 if horizon == 1 else 256))
        self.model: Any = None
        self.fit_info: dict[str, Any] = {}

    def fit(self, X: np.ndarray, y: np.ndarray, fit_indices: Sequence[int], valid_indices: Sequence[int]) -> "PatchTSTAdapter":
        try:
            import torch
            from tsai.models.PatchTST import PatchTST
        except ImportError as exc:
            raise RuntimeError("Round11 fixture requires the existing tsai/PyTorch environment") from exc
        torch.manual_seed(self.seed)
        if torch.cuda.is_available() and self.device_name.startswith("cuda"):
            torch.cuda.manual_seed_all(self.seed)
        device = torch.device(self.device_name)
        self.model = PatchTST(c_in=1, c_out=1, seq_len=self.history, pred_dim=self.horizon,
                              n_layers=2, n_heads=2, d_model=16, d_ff=64, attn_dropout=0.0,
                              dropout=0.10, patch_len=8, stride=4, padding_patch=True).to(device)
        x_tensor = torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)
        y_tensor = torch.as_tensor(np.asarray(y, dtype=np.float32), device=device)
        fit_ids = np.asarray(fit_indices, dtype=np.int64)
        valid_ids = np.asarray(valid_indices, dtype=np.int64)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.0001)
        best_loss = float("inf")
        best_state: dict[str, Any] | None = None
        stale = 0
        epochs_run = 0
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        for epoch in range(self.max_epochs):
            self.model.train()
            order = torch.randperm(fit_ids.size, generator=generator).numpy()
            for start in range(0, fit_ids.size, self.batch_size):
                ids = torch.as_tensor(fit_ids[order[start : start + self.batch_size]], device=device)
                optimizer.zero_grad(set_to_none=True)
                prediction = self.model(x_tensor[ids])
                if prediction.shape != y_tensor[ids].shape:
                    raise AssertionError(f"PatchTST output {tuple(prediction.shape)} != target {tuple(y_tensor[ids].shape)}")
                loss = torch.mean((prediction - y_tensor[ids]) ** 2)
                loss.backward()
                optimizer.step()
            self.model.eval()
            with torch.no_grad():
                valid_loss = float(torch.mean((self.model(x_tensor[valid_ids]) - y_tensor[valid_ids]) ** 2).cpu())
            epochs_run = epoch + 1
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_state = {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.fit_info = {"epochs_run": epochs_run, "inner_valid_mse": best_loss, "seed": self.seed, "device": str(device)}
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("adapter is not fitted")
        import torch
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            output = self.model(torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)).detach().cpu().numpy()
        if output.ndim != 3 or output.shape[1:] != (1, self.horizon):
            raise AssertionError(f"unexpected PatchTST prediction shape {output.shape}")
        return output.astype(np.float64)


class FitLedger:
    def __init__(self) -> None:
        self.keys: list[str] = []

    def add(self, seed: int, task: str, stage: str, block: int | None = None) -> str:
        if stage not in FIT_STAGES:
            raise ValueError(f"unknown fit stage {stage}")
        key = f"seed={int(seed)}/task={task}/stage={stage}" + (f"/block={int(block)}" if block is not None else "")
        self.keys.append(key)
        return key

    def validate(self, seeds: Sequence[int]) -> None:
        expected = []
        for seed in seeds:
            for task in TASKS:
                expected.extend([f"seed={seed}/task={task}/stage=raw_full"])
                expected.extend([f"seed={seed}/task={task}/stage=log_oof/block={i}" for i in range(5)])
                expected.extend([f"seed={seed}/task={task}/stage=log_full"])
        if sorted(self.keys) != sorted(expected):
            raise AssertionError(f"fit ledger mismatch: expected {len(expected)}, got {len(self.keys)}")


def expected_fit_keys(seeds: Sequence[int] = FORMAL_SEEDS) -> list[str]:
    ledger = FitLedger()
    for seed in seeds:
        for task in TASKS:
            ledger.add(seed, task, "raw_full")
            for block in range(5):
                ledger.add(seed, task, "log_oof", block)
            ledger.add(seed, task, "log_full")
    return ledger.keys


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    os.replace(temp, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temp, path)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def git_binding(repo: Path | str = ".", *, exclude_path: Path | str | None = None) -> dict[str, Any]:
    root = Path(repo)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip()
        raw_status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, text=True)
        excluded = None if exclude_path is None else Path(exclude_path).resolve()
        dirty_paths: list[str] = []
        for line in raw_status.splitlines():
            if len(line) < 4:
                continue
            path_text = line[3:]
            if " -> " in path_text:
                path_text = path_text.split(" -> ", 1)[1]
            candidate = (root / path_text).resolve()
            if excluded is not None and (candidate == excluded or excluded in candidate.parents):
                continue
            dirty_paths.append(path_text)
        clean = not dirty_paths
    except (OSError, subprocess.CalledProcessError):
        commit, branch, clean, dirty_paths = "unknown", "unknown", False, ["<git-status-unavailable>"]
    return {"commit": commit, "branch": branch, "clean_tree": bool(clean), "dirty_paths": dirty_paths}


def validate_seed_set(seeds: Sequence[int], *, formal: bool) -> tuple[int, ...]:
    normalized = tuple(int(seed) for seed in seeds)
    if len(set(normalized)) != len(normalized):
        raise ValueError("seed set contains duplicates")
    if set(normalized) & set(HISTORICAL_SEEDS):
        raise ValueError("seed set overlaps registered historical seeds")
    if formal and normalized != FORMAL_SEEDS:
        raise ValueError("formal fixture only accepts seeds 4001,4002,4003")
    return normalized


def validate_manifest(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, digest in manifest.get("artifacts", {}).items():
        artifact = output_dir / name
        if not artifact.exists() or sha256_file(artifact) != digest:
            raise ValueError(f"tamper detected for artifact {name}")
    return manifest


def metric_rows(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return {"mse": float(np.mean(error ** 2)), "rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error)))}


def task_metadata(task: WindowTask, plan: OOFPlan) -> dict[str, Any]:
    return {
        "task": task.name, "history": task.history, "horizon": task.horizon,
        "shapes": {"X": [1, 1, task.history], "y": [1, 1, task.horizon]},
        "purge": task.history + task.horizon + 1,
        "train_indices": task.split_indices["train"].tolist(),
        "valid_indices": task.split_indices["valid"].tolist(),
        "test_indices": task.split_indices["test"].tolist(),
        "warmup_indices": plan.warmup_indices.tolist(),
        "blocks": [{"block": b.block, "query_indices": b.query_indices.tolist(), "fit_indices": b.fit_indices.tolist(),
                    "inner_fit_indices": b.inner_fit_indices.tolist(), "inner_valid_indices": b.inner_valid_indices.tolist(),
                    "excluded_indices": b.excluded_indices.tolist(), "query_min_timestamp": b.query_min_timestamp,
                    "fit_max_timestamp": b.fit_max_timestamp, "purge": b.purge, "exclusion_reason": b.exclusion_reason} for b in plan.blocks],
        "exclusions": {key: value.tolist() for key, value in plan.exclusions.items()},
    }


def safe_task_name(task: str) -> str:
    return task.replace("/", "_")
