"""Centre-only traditional baselines with train-only fitting.

The materialization boundary extracts the centre station immediately, yielding
``x=[N,1,L]`` and ``y=[N,1,H]``.  Consequently no estimator can accidentally
read a neighbour channel.  Validation is used only to select the preregistered
Ridge alpha; test is never read during fitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge


TRADITIONAL_ARMS = (
    "trad_persistence",
    "trad_daily_naive",
    "trad_climatology",
    "trad_ar",
    "trad_ridge",
)
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)


def dataset_arrays(dataset) -> tuple[np.ndarray, np.ndarray]:
    """Materialize centre-only windows as ``x=[N,1,L]``, ``y=[N,1,H]``."""
    indices = np.asarray(dataset.sample_indices, dtype=np.int64)
    center_idx = int(dataset.center_idx)
    x = np.stack(
        [
            dataset.values[start:start + dataset.history, center_idx][None, :]
            for start in indices
        ]
    ).astype(np.float32, copy=False)
    y = np.stack(
        [
            dataset.values[
                start + dataset.history:start + dataset.history + dataset.horizon,
                center_idx,
            ][None, :]
            for start in indices
        ]
    ).astype(np.float32, copy=False)
    if x.shape[1] != 1:
        raise AssertionError("traditional baseline input must have one channel")
    return x, y


@dataclass
class TraditionalBaseline:
    arm: str
    horizon: int
    input_channels: int = 1
    fit_sample_indices: np.ndarray | None = None
    selected_alpha: float | None = None
    estimator: object | None = None
    climatology_by_hour: np.ndarray | None = None

    def predict(self, dataset, metadata: dict) -> np.ndarray:
        x, _ = dataset_arrays(dataset)
        if self.input_channels != 1 or x.shape[1] != 1:
            raise AssertionError("single-station prediction received multiple channels")
        if self.arm == "trad_persistence":
            latest = x[:, 0, -1]
            return np.repeat(latest[:, None, None], self.horizon, axis=2)
        if self.arm == "trad_daily_naive":
            offsets = x.shape[2] - 24 + np.arange(self.horizon)
            if offsets.min() < 0 or offsets.max() >= x.shape[2]:
                raise ValueError("daily_naive requires L>=24 and H<=24")
            return x[:, 0, offsets][:, None, :]
        if self.arm == "trad_climatology":
            hours = _target_hours(dataset, metadata)
            values = self.climatology_by_hour[hours]
            return values[:, None, :].astype(np.float32)
        if self.arm == "trad_ar":
            p = min(x.shape[2], 24)
            values = np.asarray(self.estimator.predict(x[:, 0, -p:]), dtype=np.float32)
            return values.reshape(len(x), self.horizon)[:, None, :]
        if self.arm == "trad_ridge":
            values = np.asarray(self.estimator.predict(x[:, 0, :]), dtype=np.float32)
            return values.reshape(len(x), self.horizon)[:, None, :]
        raise ValueError(f"unknown traditional baseline arm: {self.arm}")


def _target_hours(dataset, metadata: dict) -> np.ndarray:
    start_time = pd.Timestamp(metadata["start_time"])
    starts = np.asarray(dataset.sample_indices, dtype=np.int64)
    target_rows = starts[:, None] + dataset.history + np.arange(dataset.horizon)[None, :]
    return (int(start_time.hour) + target_rows) % 24


def fit_traditional_baseline(
    arm: str,
    datasets: dict,
    metadata: dict,
) -> TraditionalBaseline:
    """Fit a centre-only arm under the no-test-fitting protocol."""
    if arm not in TRADITIONAL_ARMS:
        raise ValueError(f"unknown traditional baseline arm: {arm}")
    train = datasets["train"]
    model = TraditionalBaseline(arm=arm, horizon=int(train.horizon))
    if model.input_channels != 1:
        raise AssertionError("single-station traditional arm must have one channel")
    if arm in {"trad_persistence", "trad_daily_naive"}:
        return model

    x_train, y_train_3d = dataset_arrays(train)
    y_train = y_train_3d[:, 0, :]
    model.fit_sample_indices = np.asarray(train.sample_indices, dtype=np.int64).copy()

    if arm == "trad_climatology":
        hours = _target_hours(train, metadata)
        means = np.full(24, float(y_train.mean()), dtype=np.float32)
        for hour in range(24):
            mask = hours == hour
            if mask.any():
                means[hour] = float(y_train[mask].mean())
        model.climatology_by_hour = means
        return model

    if arm == "trad_ar":
        p = min(x_train.shape[2], 24)
        model.estimator = LinearRegression().fit(x_train[:, 0, -p:], y_train)
        return model

    # Ridge candidates and the final estimator are all fit on train only.
    x_valid, y_valid_3d = dataset_arrays(datasets["valid"])
    y_valid = y_valid_3d[:, 0, :]
    best_alpha = None
    best_loss = float("inf")
    for alpha in RIDGE_ALPHAS:
        candidate = Ridge(alpha=alpha).fit(x_train[:, 0, :], y_train)
        prediction = np.asarray(candidate.predict(x_valid[:, 0, :])).reshape(y_valid.shape)
        valid_loss = float(np.mean((prediction - y_valid) ** 2))
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_alpha = alpha
    model.selected_alpha = float(best_alpha)
    model.estimator = Ridge(alpha=best_alpha).fit(x_train[:, 0, :], y_train)
    return model
