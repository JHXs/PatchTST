"""Pure-synthetic TACBR L0 data generation, convex fits, and certificates.

This module deliberately has no project-data imports.  It is usable with an
isolated environment containing numpy, cvxpy, and osqp.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SEEDS = tuple(range(20260909, 20260939))
METHODS = (
    "target_only",
    "global_full",
    "matched_global",
    "matched_global_station_intercept",
    "unmatched_conditional_bias_head",
    "tacbr",
    "tacbr_oracle",
    "transfusion_paper_exact",
    "transfusion_intercept",
)
TF_METHODS = {"transfusion_paper_exact", "transfusion_intercept"}
C_GRID = np.logspace(-2.0, 1.0, 12, dtype=np.float64)


@dataclass(frozen=True)
class DGPConfig:
    p: int = 128
    n_train: int = 768
    n_validation: int = 128
    n_test: int = 256
    n_stations: int = 5
    horizon: int = 1
    target_label_stride: int = 8
    k_neighbors: int = 12
    ar: float = 0.7

    @property
    def n_total(self) -> int:
        return self.n_train + self.n_validation + self.n_test

    def target_indices(self, label_count: int) -> np.ndarray:
        if label_count == self.n_train:
            return np.arange(self.n_train, dtype=int)
        if label_count == self.n_train // self.target_label_stride:
            return np.arange(0, self.n_train, self.target_label_stride, dtype=int)
        raise ValueError("label_count must be 96 or 768 for the formal DGP")


@dataclass
class SyntheticData:
    seed: int
    group: str
    config: DGPConfig
    x_target: np.ndarray
    y_target: np.ndarray
    x_sources: np.ndarray
    y_sources: np.ndarray
    true_source_shift: np.ndarray
    target_train_indices: np.ndarray
    beta_true: np.ndarray
    source_bias: np.ndarray

    @property
    def target_train_end(self) -> int:
        return self.config.n_train

    @property
    def validation_slice(self) -> slice:
        return slice(self.config.n_train, self.config.n_train + self.config.n_validation)

    @property
    def test_slice(self) -> slice:
        return slice(self.config.n_train + self.config.n_validation, self.config.n_total)


def _formal_bias(p: int) -> np.ndarray:
    base = np.array(
        [[0.3, 0.8, -0.4], [-0.2, 0.6, -0.2], [0.4, 0.5, -0.5],
         [-0.3, 0.7, -0.1], [0.1, 0.4, -0.6]], dtype=np.float64
    )
    if p < 10:
        raise ValueError("the DGP needs p >= 10")
    return base


def _station_offsets(p: int, n_stations: int) -> tuple[np.ndarray, np.ndarray]:
    # Frozen formula uses (s+1)*(j+1) with mathematical coordinates j=1..p.
    coords = np.arange(2, p + 2, dtype=np.float64)
    stations = np.arange(1, n_stations + 1, dtype=np.float64)
    c = np.stack([0.3 * np.sin((s + 1) * coords) for s in stations])
    d = np.stack([np.sin((s + 1) * coords) for s in stations])
    return c, d


def make_synthetic(seed: int, group: str, label_count: int, config: DGPConfig | None = None) -> SyntheticData:
    """Generate one frozen V/F seed without any project data access."""
    config = config or DGPConfig()
    if group not in {"V", "F"}:
        raise ValueError("group must be V or F")
    if config.n_stations != 5 and config.p == 128:
        raise ValueError("formal station count is fixed at five")
    if label_count not in {config.n_train, config.n_train // config.target_label_stride}:
        raise ValueError("unsupported target label count")

    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    # Required call order: X initial value, u, xi, epsilon0, epsilon_s.
    x_initial = rng.standard_normal(config.p)
    innovations = rng.standard_normal((config.n_total, config.p))
    xi = rng.standard_normal((config.n_stations, config.n_total, config.p))
    epsilon_target = rng.standard_normal(config.n_total)
    epsilon_sources = rng.standard_normal((config.n_stations, config.n_total))

    x_target = np.empty((config.n_total, config.p), dtype=np.float64)
    previous = x_initial
    ar_scale = np.sqrt(1.0 - config.ar**2)
    for t in range(config.n_total):
        previous = config.ar * previous + ar_scale * innovations[t]
        x_target[t] = previous

    offsets, directions = _station_offsets(config.p, config.n_stations)
    x_sources = x_target[None, :, :] + offsets[:, None, :] + 0.1 * xi
    beta_true = np.zeros(config.p, dtype=np.float64)
    beta_true[:6] = (1.2, -1.0, 0.8, -0.6, 0.4, -0.3)
    source_bias = _formal_bias(config.p)[:config.n_stations]
    phi_target = np.column_stack((np.ones(config.n_total), x_target[:, :2]))
    y_target = x_target @ beta_true + epsilon_target
    source_shift = np.empty((config.n_stations, config.n_total), dtype=np.float64)
    y_sources = np.empty_like(source_shift)
    for station in range(config.n_stations):
        phi_source = np.column_stack((np.ones(config.n_total), x_sources[station, :, :2]))
        linear_bias = phi_source @ source_bias[station]
        if group == "V":
            shift = linear_bias
        else:
            # Mathematical coordinates X_3,...,X_10 are Python [2:10].
            d = directions[station, 2:10]
            omitted = np.sqrt(source_bias[station, 1] ** 2 + source_bias[station, 2] ** 2)
            omitted *= ((x_sources[station, :, 2:10] - offsets[station, 2:10]) @ d)
            omitted /= np.sqrt(np.sum(d**2))
            shift = 0.7 * linear_bias + np.sqrt(1.0 - 0.7**2) * omitted
        source_shift[station] = shift
        y_sources[station] = x_sources[station] @ beta_true + shift + epsilon_sources[station]
    return SyntheticData(
        seed=seed, group=group, config=config, x_target=x_target, y_target=y_target,
        x_sources=x_sources, y_sources=y_sources, true_source_shift=source_shift,
        target_train_indices=config.target_indices(label_count), beta_true=beta_true,
        source_bias=source_bias,
    )


@dataclass
class PreparedData:
    data: SyntheticData
    label_count: int
    x_target: np.ndarray
    y_target: np.ndarray
    x_sources: np.ndarray
    y_sources: np.ndarray
    mu: np.ndarray
    sd: np.ndarray
    z_target: np.ndarray
    z_sources: np.ndarray
    target_train_positions: np.ndarray
    source_train_masks: np.ndarray
    source_validation_masks: np.ndarray
    tau: float

    @property
    def n0(self) -> int:
        return self.target_train_positions.size

    @property
    def n_source(self) -> int:
        return self.data.config.n_train


def _support_mask(source_z: np.ndarray, target_z: np.ndarray, tau: float, k: int) -> np.ndarray:
    distances = np.sqrt(np.sum((source_z[:, None, :] - target_z[None, :, :]) ** 2, axis=2))
    kth = np.partition(distances, min(k - 1, target_z.shape[0] - 1), axis=1)[:, min(k - 1, target_z.shape[0] - 1)]
    return kth <= tau


def prepare_data(data: SyntheticData, label_count: int | None = None) -> PreparedData:
    label_count = label_count or data.target_train_indices.size
    indices = data.target_train_indices if label_count == data.target_train_indices.size else data.config.target_indices(label_count)
    x_train = data.x_target[indices]
    mu = np.mean(x_train, axis=0, dtype=np.float64)
    sd = np.std(x_train, axis=0, ddof=0, dtype=np.float64)
    if np.any(sd <= 1e-12):
        raise ValueError("standard deviation <= 1e-12")
    x_target = (data.x_target - mu) / sd
    x_sources = (data.x_sources - mu[None, None, :]) / sd[None, None, :]
    target_z_train = x_target[indices, :2]
    pairwise = np.sqrt(np.sum((target_z_train[:, None, :] - target_z_train[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(pairwise, np.inf)
    k = min(data.config.k_neighbors, target_z_train.shape[0] - 1)
    loo_kth = np.partition(pairwise, k - 1, axis=1)[:, k - 1]
    tau = float(np.quantile(loo_kth, 0.95, method="linear"))
    source_train_masks = np.stack([
        _support_mask(x_sources[s, :data.config.n_train, :2], target_z_train, tau, data.config.k_neighbors)
        for s in range(data.config.n_stations)
    ])
    val_slice = data.validation_slice
    source_validation_masks = np.stack([
        _support_mask(x_sources[s, val_slice, :2], target_z_train, tau, data.config.k_neighbors)
        for s in range(data.config.n_stations)
    ])
    return PreparedData(
        data=data, label_count=label_count, x_target=x_target, y_target=data.y_target,
        x_sources=x_sources, y_sources=data.y_sources, mu=mu, sd=sd,
        z_target=x_target[:, :2], z_sources=x_sources[:, :, :2],
        target_train_positions=indices, source_train_masks=source_train_masks,
        source_validation_masks=source_validation_masks, tau=tau,
    )


def phi_from_z(z: np.ndarray) -> np.ndarray:
    return np.column_stack((np.ones(z.shape[0], dtype=np.float64), z[:, 0], z[:, 1]))


def c_to_eta(c: float, n0: int, p: int) -> float:
    return float(c * np.sqrt(np.log(p) / n0))


def _cvxpy():
    try:
        import cvxpy as cp
    except ImportError as exc:  # pragma: no cover - exercised in non-isolated project env
        raise RuntimeError("TACBR L0 requires cvxpy==1.7.2 in an isolated environment") from exc
    return cp


def _solver_kwargs(cp: Any) -> dict[str, Any]:
    # OSQP is the locked solver.  A caller can override via environment only by
    # changing this source; this prevents silent solver substitution.
    if "OSQP" not in cp.installed_solvers():
        raise RuntimeError("OSQP is not installed; refusing a different formal solver")
    return dict(solver=cp.OSQP, eps_abs=1e-9, eps_rel=1e-9, max_iter=300_000, polish=True, verbose=False)


def _status_ok(problem: Any) -> bool:
    return str(problem.status).lower() == "optimal"


def _solver_residuals(problem: Any) -> tuple[float | None, float | None, int | None]:
    stats = problem.solver_stats
    extra = getattr(stats, "extra_stats", None)
    info = getattr(extra, "info", None)
    primal = getattr(info, "prim_res", None)
    dual = getattr(info, "dual_res", None)
    iters = getattr(info, "iter", None)
    return (None if primal is None else float(primal), None if dual is None else float(dual),
            None if iters is None else int(iters))


def _array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
        digest.update(str(a.shape).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def _lasso_subgradient(coef: np.ndarray, grad: np.ndarray, penalty: float) -> tuple[np.ndarray, float]:
    u = np.zeros_like(coef)
    nonzero = np.abs(coef) > 1e-7
    u[nonzero] = np.sign(coef[nonzero])
    if penalty > 0:
        u[~nonzero] = np.clip(-grad[~nonzero] / penalty, -1.0, 1.0)
    residual = grad + penalty * u
    return u, float(np.max(np.abs(residual)))


def _non_tf_spec(method: str, prepared: PreparedData) -> tuple[np.ndarray, int, bool, bool]:
    if method not in {"target_only", "global_full", "matched_global", "matched_global_station_intercept",
                      "unmatched_conditional_bias_head", "tacbr", "tacbr_oracle"}:
        raise ValueError(f"not a non-TF method: {method}")
    active = np.ones_like(prepared.source_train_masks, dtype=bool)
    use_sources = method != "target_only"
    if method in {"matched_global", "matched_global_station_intercept", "tacbr", "tacbr_oracle"}:
        active = prepared.source_train_masks.copy()
    if method == "tacbr_oracle":
        active = prepared.source_train_masks.copy()
    nuisance_dim = 0
    if method in {"matched_global_station_intercept",}:
        nuisance_dim = 1
    elif method in {"unmatched_conditional_bias_head", "tacbr"}:
        nuisance_dim = 3
    return active, nuisance_dim, use_sources, method == "tacbr_oracle"


@dataclass
class FitResult:
    method: str
    c: float
    eta: float | None
    status: str
    beta: np.ndarray
    intercept: float
    nuisance: np.ndarray
    tf_task_betas: np.ndarray | None
    tf_task_intercepts: np.ndarray | None
    tf_delta: np.ndarray | None
    tf_delta_intercept: float | None
    objective: float
    reported_objective: float | None
    kkt_inf: float
    raw_primal_residual: float | None
    raw_dual_residual: float | None
    iterations: int | None
    prediction_sha256: str = ""
    validation_mse: float | None = None
    test_mse: float | None = None
    tf_objective_first: float | None = None
    tf_objective_second: float | None = None


def _fit_non_tf_direct(prepared: PreparedData, method: str, c: float, lam: float = 1.0,
                       nuisance_override: np.ndarray | None = None) -> FitResult:
    cp = _cvxpy()
    p = prepared.x_target.shape[1]
    eta = c_to_eta(c, prepared.n0, p)
    active, nuisance_dim, use_sources, oracle = _non_tf_spec(method, prepared)
    x0 = prepared.x_target[prepared.target_train_positions]
    y0 = prepared.y_target[prepared.target_train_positions]
    beta = cp.Variable(p)
    intercept = cp.Variable()
    nuisance = cp.Variable((prepared.data.config.n_stations, nuisance_dim)) if nuisance_dim else None
    terms = [cp.sum_squares(x0 @ beta + intercept - y0) / prepared.n0]
    if use_sources:
        for station in range(prepared.data.config.n_stations):
            mask = active[station]
            if not np.any(mask):
                continue
            xs = prepared.x_sources[station, :prepared.data.config.n_train][mask]
            ys = prepared.y_sources[station, :prepared.data.config.n_train][mask]
            residual = xs @ beta + intercept - ys
            if nuisance_dim:
                ph = phi_from_z(prepared.z_sources[station, :prepared.data.config.n_train][mask])
                if nuisance_dim == 1:
                    ph = ph[:, :1]
                # Source objective is ||Y - f - Phi a||^2; with the
                # prediction-minus-label residual this is f + Phi a - Y.
                residual = residual + ph @ nuisance[station]
            if oracle:
                ys = ys - prepared.data.true_source_shift[station, :prepared.data.config.n_train][mask]
                residual = xs @ beta + intercept - ys
            terms.append(lam / prepared.data.config.n_stations * cp.sum_squares(residual) / int(np.sum(mask)))
    if nuisance_dim:
        terms.append(1e-3 / prepared.data.config.n_stations * cp.sum_squares(nuisance))
    objective = cp.Minimize(sum(terms) + eta * cp.norm1(beta))
    problem = cp.Problem(objective)
    problem.solve(**_solver_kwargs(cp))
    status = str(problem.status).lower()
    b = np.asarray(beta.value, dtype=np.float64).reshape(-1) if beta.value is not None else np.full(p, np.nan)
    alpha = float(intercept.value) if intercept.value is not None else np.nan
    a = np.asarray(nuisance.value, dtype=np.float64) if nuisance_dim and nuisance.value is not None else np.zeros((prepared.data.config.n_stations, nuisance_dim))
    raw_objective, kkt = non_tf_certificate(prepared, method, b, alpha, a, c, lam=lam)
    primal, dual, iterations = _solver_residuals(problem)
    return FitResult(method, c, eta, status, b, alpha, a, None, None, None, None,
                     raw_objective, None if problem.value is None else float(problem.value), kkt,
                     primal, dual, iterations)


def fit_non_tf_profile(prepared: PreparedData, method: str, c: float, lam: float = 1.0) -> FitResult:
    """Independent profile implementation for the low-dimensional nuisance.

    The nuisance is analytically profiled from each source's Gram matrix.  It
    is used only for the station-linear-head equivalence check and does not
    replace the direct implementation used by the experiment.
    """
    if method not in {"matched_global_station_intercept", "unmatched_conditional_bias_head", "tacbr"}:
        raise ValueError("profile is only defined for nuisance-head methods")
    cp = _cvxpy()
    p = prepared.x_target.shape[1]
    eta = c_to_eta(c, prepared.n0, p)
    active, nuisance_dim, _, _ = _non_tf_spec(method, prepared)
    x0 = prepared.x_target[prepared.target_train_positions]
    y0 = prepared.y_target[prepared.target_train_positions]
    beta = cp.Variable(p)
    intercept = cp.Variable()
    terms = [cp.sum_squares(x0 @ beta + intercept - y0) / prepared.n0]
    for station in range(prepared.data.config.n_stations):
        mask = active[station]
        if not np.any(mask):
            continue
        xs = prepared.x_sources[station, :prepared.data.config.n_train][mask]
        ys = prepared.y_sources[station, :prepared.data.config.n_train][mask]
        ph = phi_from_z(prepared.z_sources[station, :prepared.data.config.n_train][mask])[:, :nuisance_dim]
        t = lam / int(np.sum(mask))
        gram = ph.T @ ph
        inverse = np.linalg.inv(t * gram + 1e-3 * np.eye(nuisance_dim, dtype=np.float64))
        residual = xs @ beta + intercept - ys
        # Write the profiled quadratic as one PSD matrix so CVXPY's DCP
        # verifier can certify it without relying on a negative quad_form.
        q_matrix = t * np.eye(len(ys), dtype=np.float64) - t**2 * ph @ inverse @ ph.T
        q_matrix = (q_matrix + q_matrix.T) / 2.0
        eigenvalues, eigenvectors = np.linalg.eigh(q_matrix)
        q_matrix = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
        profiled = cp.quad_form(residual, q_matrix)
        terms.append(profiled / prepared.data.config.n_stations)
    problem = cp.Problem(cp.Minimize(sum(terms) + eta * cp.norm1(beta)))
    problem.solve(**_solver_kwargs(cp))
    b = np.asarray(beta.value, dtype=np.float64).reshape(-1) if beta.value is not None else np.full(p, np.nan)
    alpha = float(intercept.value) if intercept.value is not None else np.nan
    # Recover each profiled nuisance at the returned beta.
    a = np.zeros((prepared.data.config.n_stations, nuisance_dim), dtype=np.float64)
    for station in range(prepared.data.config.n_stations):
        mask = active[station]
        if np.any(mask):
            xs = prepared.x_sources[station, :prepared.data.config.n_train][mask]
            ys = prepared.y_sources[station, :prepared.data.config.n_train][mask]
            ph = phi_from_z(prepared.z_sources[station, :prepared.data.config.n_train][mask])[:, :nuisance_dim]
            t = lam / int(np.sum(mask))
            a[station] = -np.linalg.solve(t * (ph.T @ ph) + 1e-3 * np.eye(nuisance_dim), t * ph.T @ (xs @ b + alpha - ys))
    raw_objective, kkt = non_tf_certificate(prepared, method, b, alpha, a, c, lam=lam)
    primal, dual, iterations = _solver_residuals(problem)
    return FitResult(method, c, eta, str(problem.status).lower(), b, alpha, a, None, None, None, None,
                     raw_objective, None if problem.value is None else float(problem.value), kkt,
                     primal, dual, iterations)


def non_tf_certificate(prepared: PreparedData, method: str, beta: np.ndarray, intercept: float,
                       nuisance: np.ndarray, c: float, lam: float = 1.0) -> tuple[float, float]:
    """Recompute the non-TF objective and KKT residual from raw coefficients."""
    p = prepared.x_target.shape[1]
    eta = c_to_eta(c, prepared.n0, p)
    active, nuisance_dim, use_sources, oracle = _non_tf_spec(method, prepared)
    x0 = prepared.x_target[prepared.target_train_positions]
    y0 = prepared.y_target[prepared.target_train_positions]
    r0 = x0 @ beta + intercept - y0
    objective = float(np.mean(r0**2) + eta * np.sum(np.abs(beta)))
    grad_beta = 2 * x0.T @ r0 / prepared.n0
    grad_alpha = 2 * np.mean(r0)
    grad_a = np.zeros_like(nuisance)
    for station in range(prepared.data.config.n_stations):
        mask = active[station]
        if not use_sources or not np.any(mask):
            continue
        xs = prepared.x_sources[station, :prepared.data.config.n_train][mask]
        ys = prepared.y_sources[station, :prepared.data.config.n_train][mask]
        if oracle:
            ys = ys - prepared.data.true_source_shift[station, :prepared.data.config.n_train][mask]
        residual = xs @ beta + intercept - ys
        ph = np.empty((len(ys), 0))
        if nuisance_dim:
            ph = phi_from_z(prepared.z_sources[station, :prepared.data.config.n_train][mask])[:, :nuisance_dim]
            residual = residual + ph @ nuisance[station]
            grad_a[station] = 2 * lam / prepared.data.config.n_stations * ph.T @ residual / len(ys) + 2e-3 / prepared.data.config.n_stations * nuisance[station]
        grad_beta += 2 * lam / prepared.data.config.n_stations * xs.T @ residual / len(ys)
        grad_alpha += 2 * lam / prepared.data.config.n_stations * np.mean(residual)
        objective += float(lam / prepared.data.config.n_stations * np.mean(residual**2) + 1e-3 / prepared.data.config.n_stations * np.sum(nuisance[station] ** 2)) if nuisance_dim else float(lam / prepared.data.config.n_stations * np.mean(residual**2))
    _, beta_kkt = _lasso_subgradient(beta, grad_beta, eta)
    kkt = max(beta_kkt, float(abs(grad_alpha)), float(np.max(np.abs(grad_a))) if grad_a.size else 0.0)
    return objective, kkt


def _fit_tf(prepared: PreparedData, method: str, c: float) -> FitResult:
    cp = _cvxpy()
    intercept_version = method == "transfusion_intercept"
    p = prepared.x_target.shape[1]
    x0 = prepared.x_target[prepared.target_train_positions]
    y0 = prepared.y_target[prepared.target_train_positions]
    xs = [x0] + [prepared.x_sources[s, :prepared.data.config.n_train] for s in range(prepared.data.config.n_stations)]
    ys = [y0] + [prepared.y_sources[s, :prepared.data.config.n_train] for s in range(prepared.data.config.n_stations)]
    nks = np.array([x.shape[0] for x in xs], dtype=np.float64)
    total_n = int(np.sum(nks))
    lam0 = c * np.sqrt(np.log(p) / total_n)
    tf_weight = 8.0 * np.sqrt(nks[1] / total_n)
    lamt = c * np.sqrt(np.log(p) / prepared.n0)
    task_beta = cp.Variable((len(xs), p))
    task_alpha = cp.Variable(len(xs)) if intercept_version else None
    terms = []
    for task, (x, y) in enumerate(zip(xs, ys)):
        pred = x @ task_beta[task]
        if intercept_version:
            pred = pred + task_alpha[task]
        terms.append(cp.sum_squares(pred - y) / (2.0 * total_n))
    fused = sum(tf_weight * cp.norm1(task_beta[s] - task_beta[0]) for s in range(1, len(xs)))
    problem = cp.Problem(cp.Minimize(sum(terms) + lam0 * cp.norm1(task_beta[0]) + lam0 * fused))
    problem.solve(**_solver_kwargs(cp))
    status = str(problem.status).lower()
    task_b = np.asarray(task_beta.value, dtype=np.float64) if task_beta.value is not None else np.full((len(xs), p), np.nan)
    task_a = np.asarray(task_alpha.value, dtype=np.float64) if intercept_version and task_alpha.value is not None else np.zeros(len(xs), dtype=np.float64)
    weights = nks / total_n
    w = weights @ task_b
    w_alpha = float(weights @ task_a)
    residual_first = y0 - x0 @ w - w_alpha
    delta = cp.Variable(p)
    delta_alpha = cp.Variable() if intercept_version else None
    pred_delta = x0 @ (w + delta)
    if intercept_version:
        pred_delta = pred_delta + w_alpha + delta_alpha
    problem_two = cp.Problem(cp.Minimize(cp.sum_squares(pred_delta - y0) / (2.0 * prepared.n0) + lamt * cp.norm1(delta)))
    problem_two.solve(**_solver_kwargs(cp))
    d = np.asarray(delta.value, dtype=np.float64).reshape(-1) if delta.value is not None else np.full(p, np.nan)
    da = float(delta_alpha.value) if intercept_version and delta_alpha.value is not None else 0.0
    final_beta = w + d
    final_intercept = w_alpha + da
    raw_objective, kkt = tf_certificate(prepared, method, c, task_b, task_a, d, da, final_beta, final_intercept)
    p1, d1, i1 = _solver_residuals(problem)
    p2, d2, i2 = _solver_residuals(problem_two)
    primal = max(v for v in (p1, p2) if v is not None) if p1 is not None or p2 is not None else None
    dual = max(v for v in (d1, d2) if v is not None) if d1 is not None or d2 is not None else None
    iterations = sum(v for v in (i1, i2) if v is not None) if i1 is not None or i2 is not None else None
    if not _status_ok(problem) or not _status_ok(problem_two):
        status = "engineering_failure"
    result = FitResult(method, c, None, status, final_beta, final_intercept, np.zeros((prepared.data.config.n_stations, 0)),
                     task_b, task_a, d, da, raw_objective,
                     None if problem.value is None or problem_two.value is None else float(problem.value + problem_two.value),
                     kkt, primal, dual, iterations)
    first_objective = 0.0
    for task, (x, y) in enumerate(zip(xs, ys)):
        residual = x @ task_b[task] + (task_a[task] if intercept_version else 0.0) - y
        first_objective += float(np.sum(residual**2) / (2.0 * total_n))
    first_objective += float(lam0 * np.sum(np.abs(task_b[0])) + lam0 * tf_weight * np.sum(np.abs(task_b[1:] - task_b[0])))
    second_residual = x0 @ (w + d) + w_alpha + (da if intercept_version else 0.0) - y0
    result.tf_objective_first = first_objective
    result.tf_objective_second = float(np.sum(second_residual**2) / (2.0 * prepared.n0) + lamt * np.sum(np.abs(d)))
    return result


def tf_certificate(prepared: PreparedData, method: str, c: float, task_b: np.ndarray, task_a: np.ndarray,
                   delta: np.ndarray, delta_alpha: float, final_beta: np.ndarray,
                   final_intercept: float) -> tuple[float, float]:
    intercept_version = method == "transfusion_intercept"
    p = prepared.x_target.shape[1]
    x0 = prepared.x_target[prepared.target_train_positions]
    y0 = prepared.y_target[prepared.target_train_positions]
    xs = [x0] + [prepared.x_sources[s, :prepared.data.config.n_train] for s in range(prepared.data.config.n_stations)]
    ys = [y0] + [prepared.y_sources[s, :prepared.data.config.n_train] for s in range(prepared.data.config.n_stations)]
    nks = np.array([x.shape[0] for x in xs], dtype=np.float64)
    total_n = int(np.sum(nks))
    lam0 = c * np.sqrt(np.log(p) / total_n)
    fused_weight = 8.0 * np.sqrt(nks[1] / total_n)
    lamt = c * np.sqrt(np.log(p) / prepared.n0)
    objective = 0.0
    grads = []
    intercept_grad = []
    for k, (x, y) in enumerate(zip(xs, ys)):
        residual = x @ task_b[k] + (task_a[k] if intercept_version else 0.0) - y
        objective += float(np.sum(residual**2) / (2 * total_n))
        grads.append(x.T @ residual / total_n)
        intercept_grad.append(float(np.sum(residual) / total_n))
    diffs = task_b[1:] - task_b[0]
    objective += float(lam0 * np.sum(np.abs(task_b[0])) + lam0 * fused_weight * np.sum(np.abs(diffs)))
    u0 = np.zeros(p)
    nz = np.abs(task_b[0]) > 1e-7
    u0[nz] = np.sign(task_b[0, nz])
    uk = np.zeros_like(diffs)
    nz_diff = np.abs(diffs) > 1e-7
    uk[nz_diff] = np.sign(diffs[nz_diff])
    for idx in range(len(uk)):
        uk[idx, ~nz_diff[idx]] = np.clip(-grads[idx + 1][~nz_diff[idx]] / (lam0 * fused_weight), -1, 1)
    uk_residuals = [grads[k] + lam0 * fused_weight * uk[k - 1] for k in range(1, len(xs))]
    u0[~nz] = np.clip(-(grads[0] - lam0 * fused_weight * np.sum(uk, axis=0))[~nz] / lam0, -1, 1)
    first_residual = grads[0] + lam0 * u0 - lam0 * fused_weight * np.sum(uk, axis=0)
    kkt_first = max(float(np.max(np.abs(first_residual))), *(float(np.max(np.abs(r))) for r in uk_residuals),
                    *(float(abs(v)) for v in intercept_grad) if intercept_version else (0.0,))
    weights = nks / total_n
    w = weights @ task_b
    w_alpha = float(weights @ task_a) if intercept_version else 0.0
    residual_two = x0 @ (w + delta) + w_alpha + (delta_alpha if intercept_version else 0.0) - y0
    objective += float(np.sum(residual_two**2) / (2 * prepared.n0) + lamt * np.sum(np.abs(delta)))
    gdelta = x0.T @ residual_two / prepared.n0
    _, kkt_delta = _lasso_subgradient(delta, gdelta, lamt)
    kkt = max(kkt_first, kkt_delta, float(abs(np.mean(residual_two))) if intercept_version else 0.0)
    return objective, kkt


def fit_candidate(prepared: PreparedData, method: str, c: float) -> FitResult:
    if method in TF_METHODS:
        return _fit_tf(prepared, method, c)
    return _fit_non_tf_direct(prepared, method, c)


def predict(prepared: PreparedData, fit: FitResult, split: str = "validation") -> np.ndarray:
    if split == "validation":
        x = prepared.x_target[prepared.data.validation_slice]
    elif split == "test":
        x = prepared.x_target[prepared.data.test_slice]
    else:
        raise ValueError("split must be validation or test")
    return x @ fit.beta + fit.intercept


def compute_metrics(prepared: PreparedData, fit: FitResult) -> dict[str, Any]:
    val_pred = predict(prepared, fit, "validation")
    test_pred = predict(prepared, fit, "test")
    val_y = prepared.y_target[prepared.data.validation_slice]
    test_y = prepared.y_target[prepared.data.test_slice]
    fit.validation_mse = float(np.mean((val_pred - val_y) ** 2))
    fit.test_mse = float(np.mean((test_pred - test_y) ** 2))
    fit.prediction_sha256 = _array_sha256(val_pred, test_pred)
    return {"validation_mse": fit.validation_mse, "test_mse": fit.test_mse,
            "prediction_sha256": fit.prediction_sha256}


def verify_prediction_reconstruction(prepared: PreparedData, fit: FitResult,
                                     validation_prediction: np.ndarray,
                                     test_prediction: np.ndarray,
                                     expected_sha256: str | None = None) -> float:
    """Reject tampered stored predictions and return the reconstruction error."""
    rebuilt_validation = predict(prepared, fit, "validation")
    rebuilt_test = predict(prepared, fit, "test")
    validation_prediction = np.asarray(validation_prediction, dtype=np.float64)
    test_prediction = np.asarray(test_prediction, dtype=np.float64)
    error = max(float(np.max(np.abs(rebuilt_validation - validation_prediction))),
                float(np.max(np.abs(rebuilt_test - test_prediction))))
    if error > 1e-8:
        raise ValueError(f"prediction reconstruction mismatch: {error:.3e}")
    if expected_sha256 is not None and _array_sha256(validation_prediction, test_prediction) != expected_sha256:
        raise ValueError("prediction SHA-256 mismatch")
    return error


def fit_grid(prepared: PreparedData, method: str, c_grid: Iterable[float] = C_GRID) -> list[FitResult]:
    fits = []
    for c in c_grid:
        fit = fit_candidate(prepared, method, float(c))
        compute_metrics(prepared, fit)
        fits.append(fit)
    return fits


def select_fit(fits: list[FitResult]) -> FitResult | None:
    if not fits or any(f.status != "optimal" or not np.isfinite(f.validation_mse or np.nan) or f.kkt_inf > 1e-6 for f in fits):
        return None
    return min(fits, key=lambda f: (float(f.validation_mse), -float(f.c)))


def method_bias_mse(prepared: PreparedData, fit: FitResult, method: str) -> float:
    if method == "tacbr_oracle":
        return 0.0
    if method in TF_METHODS or method in {"target_only", "global_full", "matched_global"}:
        nuisance = np.zeros((prepared.data.config.n_stations, 0))
    else:
        nuisance = fit.nuisance
    values = []
    for station in range(prepared.data.config.n_stations):
        mask = prepared.source_validation_masks[station]
        if not np.any(mask):
            continue
        true = prepared.data.true_source_shift[station, prepared.data.validation_slice][mask]
        if nuisance.shape[1] == 0:
            estimated = np.zeros_like(true)
        else:
            ph = phi_from_z(prepared.z_sources[station, prepared.data.validation_slice][mask])[:, :nuisance.shape[1]]
            estimated = ph @ nuisance[station]
        values.extend((true - estimated) ** 2)
    return float(np.mean(values)) if values else float("nan")


def fit_to_record(prepared: PreparedData, fit: FitResult) -> dict[str, Any]:
    coefficient_arrays = [fit.beta, np.asarray([fit.intercept]), fit.nuisance]
    if fit.tf_task_betas is not None:
        coefficient_arrays.extend([fit.tf_task_betas, fit.tf_task_intercepts, fit.tf_delta,
                                    np.asarray([fit.tf_delta_intercept])])
    n_tf = prepared.n0 + prepared.data.config.n_stations * prepared.n_source
    return {
        "seed": prepared.data.seed, "group": prepared.data.group, "label_count": prepared.label_count,
        "method": fit.method, "c": fit.c, "n0": prepared.n0, "n_source": prepared.n_source,
        "N_tf": n_tf,
        "tau": prepared.tau, "eta": fit.eta, "status": fit.status,
        "lambda_non_tf": 1.0 if fit.method not in TF_METHODS else None,
        "rho": 1e-3 if fit.method not in TF_METHODS else None,
        "tf_lambda0": fit.c * np.sqrt(np.log(prepared.x_target.shape[1]) / n_tf) if fit.method in TF_METHODS else None,
        "tf_fused_weight": 8.0 * np.sqrt(prepared.n_source / n_tf) if fit.method in TF_METHODS else None,
        "tf_lambda_tilde": fit.c * np.sqrt(np.log(prepared.x_target.shape[1]) / prepared.n0) if fit.method in TF_METHODS else None,
        "objective": fit.objective, "reported_objective": fit.reported_objective,
        "tf_objective_first": fit.tf_objective_first, "tf_objective_second": fit.tf_objective_second,
        "kkt_inf": fit.kkt_inf, "raw_primal_residual": fit.raw_primal_residual,
        "raw_dual_residual": fit.raw_dual_residual, "iterations": fit.iterations,
        "validation_mse": fit.validation_mse, "test_mse": fit.test_mse,
        "prediction_sha256": fit.prediction_sha256,
        "coefficient_sha256": _array_sha256(*coefficient_arrays),
        "source_train_acceptance": prepared.source_train_masks.mean(axis=1).tolist(),
        "source_validation_acceptance": prepared.source_validation_masks.mean(axis=1).tolist(),
    }


def save_fit_bundle(path: Path, fits: list[FitResult], prepared: PreparedData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {"c": np.array([f.c for f in fits], dtype=np.float64)}
    arrays["beta"] = np.stack([f.beta for f in fits])
    arrays["intercept"] = np.array([f.intercept for f in fits], dtype=np.float64)
    arrays["nuisance"] = np.stack([f.nuisance for f in fits])
    arrays["val_prediction"] = np.stack([predict(prepared, f, "validation") for f in fits])
    arrays["test_prediction"] = np.stack([predict(prepared, f, "test") for f in fits])
    if any(f.tf_task_betas is not None for f in fits):
        arrays["tf_task_betas"] = np.stack([f.tf_task_betas for f in fits])
        arrays["tf_task_intercepts"] = np.stack([f.tf_task_intercepts for f in fits])
        arrays["tf_delta"] = np.stack([f.tf_delta for f in fits])
        arrays["tf_delta_intercept"] = np.array([f.tf_delta_intercept for f in fits], dtype=np.float64)
    np.savez_compressed(path, **arrays)


def load_environment_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "scipy", "sklearn", "cvxpy", "osqp"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[name] = "not-installed"
    return versions


def config_to_json(config: DGPConfig) -> dict[str, Any]:
    return {k: getattr(config, k) for k in ("p", "n_train", "n_validation", "n_test", "n_stations", "horizon", "target_label_stride", "k_neighbors", "ar")}


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
