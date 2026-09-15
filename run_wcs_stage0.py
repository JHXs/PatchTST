"""Round 16 Stage 0: pure-synthetic two-sided witness for the wind-response shape constraint.

Implements exactly the frozen protocol in
`docs/气象条件化机制筛查/04_Stage0协议.md`. No station data is read anywhere in this module.

Arms (identical backbone, only the final use of w_t differs):
  uncon_add : y = f(x) + h([window, w_t])                     (same-information strong baseline)
  wcs_neg   : y = f(x) - softplus(g(x)) * (1 - exp(-softplus(k) * w_t))   (candidate)
  wcs_pos   : same as wcs_neg with a plus sign                (wrong-sign control)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

# Tiny tensors on many threads thrash badly here (measured ~100x slowdown with 16 threads).
# Set the limits before importing torch so spawned workers inherit them as well.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

LOOKBACK = 24
BETA = np.array([0.8, -0.5, 0.3], dtype=np.float64)
GAMMA = np.array([0.5, 0.3, 1.0], dtype=np.float64)
LAMBDA = 0.15
NOISE_STD = 0.5
LEAD_COEFFICIENTS = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float64)
HORIZONS = (1, 6)
TRAIN_FRACTION = 0.7
EARLY_STOP_FRACTION = 0.1
SHIFT_TRAIN_MAX_W = 4.5
SHIFT_TEST_MIN_W = 5.5
SMALL_TRAIN_FRACTION = 0.25
MAX_EPOCHS = 60
PATIENCE = 8
BATCH_SIZE = 256
LEARNING_RATE = 1e-3


@dataclass
class SyntheticData:
    """Time-ordered windows; the test segment always follows the training segment."""

    train_windows: np.ndarray        # [n_train, L, 2]
    train_w: np.ndarray              # [n_train]
    train_y: np.ndarray              # [n_train, H]
    test_windows: np.ndarray         # [n_test, L, 2]
    test_w: np.ndarray               # [n_test]
    test_y: np.ndarray               # [n_test, H]
    test_end_indices: np.ndarray     # absolute window end index for each test row


def generate_series(seed: int, length: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    wind = np.empty(length)
    state = np.empty(length)
    wind[0], state[0] = 5.0, 0.0
    shocks_w = rng.normal(0.0, 1.0, size=length)
    shocks_s = rng.normal(0.0, 1.0, size=length)
    for t in range(1, length):
        wind[t] = 0.8 * wind[t - 1] + 0.2 * 5.0 + shocks_w[t]
        state[t] = 0.8 * state[t - 1] + 0.2 * shocks_s[t]
    return state, np.clip(wind, 0.0, 14.0)


def true_conditional_mean(state: np.ndarray, wind: np.ndarray, idx: np.ndarray, sign: float) -> np.ndarray:
    """y_h = c_h * (f(x) - sign * A(x) * g(w)); sign=+1 gives the monotone-decreasing DGP-M."""
    x = np.stack([state[idx], state[idx - 1], wind[idx - 1]], axis=1)
    f = x @ BETA
    amplitude = np.log1p(np.exp(x @ GAMMA))
    shape = 1.0 - np.exp(-LAMBDA * wind[idx])
    base = f - sign * amplitude * shape
    return np.stack([c * base for c in LEAD_COEFFICIENTS[:6]], axis=1)


def _window_view(state: np.ndarray, wind: np.ndarray) -> np.ndarray:
    """[n, L, 2] windows for every valid end index, built without copying the series."""
    arr = np.stack([state, wind], axis=1)
    view = np.lib.stride_tricks.sliding_window_view(arr, LOOKBACK, axis=0)
    return np.ascontiguousarray(np.transpose(view, (0, 2, 1)))


def build_dataset(seed: int, sign: float, horizon: int, length: int = 20000) -> SyntheticData:
    state, wind = generate_series(seed, length)
    windows = _window_view(state, wind)
    ends = np.arange(LOOKBACK, length)
    rng = np.random.default_rng(seed + 99991)
    noise = rng.normal(0.0, NOISE_STD, size=(ends.size, 1))
    targets = true_conditional_mean(state, wind, ends, sign) + noise
    n_train = int(TRAIN_FRACTION * ends.size)
    train_slice, test_slice = slice(0, n_train), slice(n_train, None)
    return SyntheticData(
        train_windows=windows[train_slice],
        train_w=wind[ends[train_slice]],
        train_y=targets[train_slice, :horizon],
        test_windows=windows[test_slice],
        test_w=wind[ends[test_slice]],
        test_y=targets[test_slice, :horizon],
        test_end_indices=ends[test_slice],
    )


class Backbone(nn.Module):
    """Two-layer MLP over the flattened [state, wind] window; shared by every arm."""

    def __init__(self, horizon: int, hidden: int = 64):
        super().__init__()
        self.input_dim = LOOKBACK * 2
        self.trunk = nn.Sequential(
            nn.Linear(self.input_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.head = nn.Linear(hidden, horizon)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)


class UnconstrainedAdditive(nn.Module):
    """Same-information strong baseline: reads the window and w_t through a free MLP."""

    def __init__(self, horizon: int, hidden: int = 64):
        super().__init__()
        self.base = Backbone(horizon, hidden)
        self.condition = nn.Sequential(
            nn.Linear(self.base.input_dim + 1, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, horizon),
        )

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return self.base.head(self.base.features(x)) + self.condition(torch.cat([x, w], dim=1))


class ShapeConstrained(nn.Module):
    """Candidate: y = f(x) -/+ softplus(g(x)) * (1 - exp(-softplus(k) * w))."""

    def __init__(self, horizon: int, sign: float, hidden: int = 64):
        super().__init__()
        self.sign = float(sign)
        self.base = Backbone(horizon, hidden)
        self.amplitude = nn.Sequential(
            nn.Linear(self.base.input_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, horizon),
        )
        self.kappa = nn.Parameter(torch.tensor(-1.0))

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        amplitude = torch.nn.functional.softplus(self.amplitude(x))
        shape = 1.0 - torch.exp(-torch.nn.functional.softplus(self.kappa) * w)
        return self.base.head(self.base.features(x)) - self.sign * amplitude * shape


def build_arm(name: str, horizon: int, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    if name == "uncon_add":
        return UnconstrainedAdditive(horizon)
    if name == "wcs_neg":
        return ShapeConstrained(horizon, sign=+1.0)
    if name == "wcs_pos":
        return ShapeConstrained(horizon, sign=-1.0)
    raise ValueError(f"unknown arm {name}")


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _flatten(windows: np.ndarray) -> np.ndarray:
    return windows.reshape(windows.shape[0], -1)


def train_arm(model: nn.Module, windows: np.ndarray, w_now: np.ndarray, targets: np.ndarray,
              seed: int, verbose: bool = False) -> dict:
    generator = torch.Generator().manual_seed(seed)
    x = torch.tensor(_flatten(windows), dtype=torch.float32)
    w = torch.tensor(w_now, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(targets, dtype=torch.float32)
    n = x.shape[0]
    n_val = max(64, int(EARLY_STOP_FRACTION * n))
    x_tr, w_tr, y_tr = x[:-n_val], w[:-n_val], y[:-n_val]
    x_va, w_va, y_va = x[-n_val:], w[-n_val:], y[-n_val:]
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    best_state, best_val, bad_epochs, best_epoch = None, float("inf"), 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(x_tr.shape[0], generator=generator)
        for start in range(0, x_tr.shape[0], BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_fn(model(x_tr[idx], w_tr[idx]), y_tr[idx])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val = float(loss_fn(model(x_va, w_va), y_va))
        if val < best_val - 1e-6:
            best_val, bad_epochs, best_epoch = val, 0, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {"best_valid_mse": best_val, "best_epoch": best_epoch}


def evaluate_arm(model: nn.Module, windows: np.ndarray, w_now: np.ndarray, targets: np.ndarray) -> dict:
    x = torch.tensor(_flatten(windows), dtype=torch.float32)
    w = torch.tensor(w_now, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        pred = model(x, w).numpy()
    err = pred - targets
    sse = float(np.sum(err ** 2))
    n_elements = int(err.size)
    return {"rmse": float(np.sqrt(sse / n_elements)), "sse": sse, "n_elements": n_elements,
            "mae": float(np.mean(np.abs(err))), "predictions": pred}


def setting_indices(train_w: np.ndarray, test_w: np.ndarray, setting: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if setting == "S-shift":
        return np.where(train_w <= SHIFT_TRAIN_MAX_W)[0], np.where(test_w > SHIFT_TEST_MIN_W)[0]
    if setting == "S-small":
        rng = np.random.default_rng(seed + 4242)
        n = int(SMALL_TRAIN_FRACTION * len(train_w))
        return np.sort(rng.choice(len(train_w), size=n, replace=False)), np.arange(len(test_w))
    if setting == "S-iid":
        return np.arange(len(train_w)), np.arange(len(test_w))
    raise ValueError(setting)


def run_single(seed: int, sign: float, horizon: int, setting: str, arm: str,
               prediction_store: dict | None = None) -> dict:
    prediction_store = {} if prediction_store is None else prediction_store
    data = build_dataset(seed, sign, horizon)
    tr_idx, te_idx = setting_indices(data.train_w, data.test_w, setting, seed)
    results = {}
    for name in ("uncon_add", "wcs_neg", "wcs_pos") if arm == "all" else (arm,):
        model = build_arm(name, horizon, seed)
        info = train_arm(model, data.train_windows[tr_idx], data.train_w[tr_idx], data.train_y[tr_idx], seed)
        metrics = evaluate_arm(model, data.test_windows[te_idx], data.test_w[te_idx], data.test_y[te_idx])
        predictions = metrics.pop("predictions")
        results[name] = {**metrics, **info, "parameters": count_parameters(model)}
        prediction_store[f"{setting}|{name}"] = predictions
    return {"seed": seed, "sign": sign, "horizon": horizon, "setting": setting,
            "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)), "arms": results,
            "test_end_indices": data.test_end_indices[te_idx].tolist()}


def monotonicity_probe(model: nn.Module, seed: int = 0, grid: int = 25) -> dict:
    """Numerically verify strict decrease and saturation of the candidate w.r.t. wind."""
    rng = np.random.default_rng(seed)
    x = torch.tensor(rng.normal(0, 1, size=(1, LOOKBACK * 2)), dtype=torch.float32)
    w_grid = torch.tensor(np.linspace(0.0, 12.0, grid), dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        y = np.array([model(x, w).numpy().ravel()[0] for w in w_grid])
    diffs = np.diff(y)
    return {"strictly_decreasing": bool(np.all(diffs < 0)),
            "differences": diffs.tolist(),
            "differences_abs_decreasing": bool(np.all(np.abs(diffs)[1:] <= np.abs(diffs)[:-1] + 1e-9))}


def _config_task(task: tuple) -> dict:
    """Run one (seed, dgp, horizon, setting) configuration; write its predictions; return the record."""
    seed, sign_label, horizon, setting, output_dir = task
    sign = +1.0 if sign_label == "M" else -1.0
    store: dict = {}
    record = run_single(seed, sign, horizon, setting, "all", prediction_store=store)
    record["dgp"] = sign_label
    np.savez_compressed(Path(output_dir) / f"pred_{seed}_{sign_label}_H{horizon}_{setting}.npz", **store)
    return record


def main() -> None:
    import argparse
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tasks = [(seed, dgp, horizon, setting, str(output))
             for seed in range(args.seed_start, args.seed_start + args.seeds)
             for dgp in ("M", "A")
             for horizon in HORIZONS
             for setting in ("S-shift", "S-small", "S-iid")]
    with mp.get_context("spawn").Pool(processes=args.workers) as pool:
        rows = pool.map(_config_task, tasks, chunksize=1)
    rows.sort(key=lambda r: (r["seed"], r["dgp"], r["horizon"], r["setting"]))
    (output / "stage0_runs.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    probe = monotonicity_probe(build_arm("wcs_neg", 1, 0))
    (output / "monotonicity_probe.json").write_text(json.dumps(probe, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"runs": len(rows), "workers": args.workers,
                      "monotonicity": {k: v for k, v in probe.items() if k != "differences"}}))


if __name__ == "__main__":
    main()
