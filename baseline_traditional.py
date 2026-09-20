"""Traditional and statistical baselines with train-only fitting."""

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
    "trad_spatial_linear",
)
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)


def dataset_arrays(dataset) -> tuple[np.ndarray, np.ndarray]:
    """Materialize protocol windows as x=[N,S,L], y=[N,1,H]."""
    indices = np.asarray(dataset.sample_indices, dtype=np.int64)
    x = np.stack(
        [dataset.values[start:start + dataset.history].T for start in indices]
    ).astype(np.float32, copy=False)
    y = np.stack(
        [
            dataset.values[
                start + dataset.history:start + dataset.history + dataset.horizon,
                dataset.center_idx,
            ][None, :]
            for start in indices
        ]
    ).astype(np.float32, copy=False)
    return x, y


@dataclass
class TraditionalBaseline:
    arm: str
    horizon: int
    center_idx: int
    fit_sample_indices: np.ndarray | None = None
    selected_alpha: float | None = None
    estimator: object | None = None
    climatology_by_hour: np.ndarray | None = None

    def predict(self, dataset, metadata: dict) -> np.ndarray:
        x, _ = dataset_arrays(dataset)
        if self.arm == "trad_persistence":
            latest = x[:, self.center_idx, -1]
            return np.repeat(latest[:, None, None], self.horizon, axis=2)
        if self.arm == "trad_daily_naive":
            offsets = x.shape[2] - 24 + np.arange(self.horizon)
            if offsets.min() < 0 or offsets.max() >= x.shape[2]:
                raise ValueError("daily_naive requires L>=24 and H<=24")
            return x[:, self.center_idx, offsets][:, None, :]
        if self.arm == "trad_climatology":
            hours = _target_hours(dataset, metadata)
            values = self.climatology_by_hour[hours]
            return values[:, None, :].astype(np.float32)
        if self.arm == "trad_ar":
            p = min(x.shape[2], 24)
            features = x[:, self.center_idx, -p:]
            values = np.asarray(self.estimator.predict(features), dtype=np.float32)
            return values.reshape(len(x), self.horizon)[:, None, :]
        if self.arm == "trad_ridge":
            features = x[:, self.center_idx, :]
            values = np.asarray(self.estimator.predict(features), dtype=np.float32)
            return values.reshape(len(x), self.horizon)[:, None, :]
        if self.arm == "trad_spatial_linear":
            features = x.reshape(len(x), -1)
            values = np.asarray(self.estimator.predict(features), dtype=np.float32)
            return values.reshape(len(x), self.horizon)[:, None, :]
        raise ValueError(f"Unknown traditional baseline arm: {self.arm}")


def _target_hours(dataset, metadata: dict) -> np.ndarray:
    start_time = pd.Timestamp(metadata["start_time"])
    starts = np.asarray(dataset.sample_indices, dtype=np.int64)
    target_rows = starts[:, None] + dataset.history + np.arange(dataset.horizon)[None, :]
    start_hour = int(start_time.hour)
    return (start_hour + target_rows) % 24


def _ridge_features(arm: str, x: np.ndarray, center_idx: int) -> np.ndarray:
    if arm == "trad_ridge":
        return x[:, center_idx, :]
    if arm == "trad_spatial_linear":
        return x.reshape(len(x), -1)
    raise ValueError(arm)


def fit_traditional_baseline(
    arm: str,
    datasets: dict,
    metadata: dict,
) -> TraditionalBaseline:
    """Fit an A-layer arm; learned state is derived from train only.

    Validation is read solely to choose Ridge alpha.  The chosen estimator is
    still fit on train alone and test is never touched here.
    """
    if arm not in TRADITIONAL_ARMS:
        raise ValueError(f"Unknown traditional baseline arm: {arm}")
    train = datasets["train"]
    model = TraditionalBaseline(
        arm=arm,
        horizon=int(train.horizon),
        center_idx=int(metadata["center_station_idx"]),
    )
    if arm in {"trad_persistence", "trad_daily_naive"}:
        return model

    x_train, y_train_3d = dataset_arrays(train)
    y_train = y_train_3d[:, 0, :]
    model.fit_sample_indices = np.asarray(train.sample_indices, dtype=np.int64).copy()

    if arm == "trad_climatology":
        hours = _target_hours(train, metadata)
        overall = float(y_train.mean())
        means = np.full(24, overall, dtype=np.float32)
        for hour in range(24):
            mask = hours == hour
            if mask.any():
                means[hour] = float(y_train[mask].mean())
        model.climatology_by_hour = means
        return model

    if arm == "trad_ar":
        p = min(x_train.shape[2], 24)
        features = x_train[:, model.center_idx, -p:]
        model.estimator = LinearRegression().fit(features, y_train)
        return model

    train_features = _ridge_features(arm, x_train, model.center_idx)
    x_valid, y_valid_3d = dataset_arrays(datasets["valid"])
    valid_features = _ridge_features(arm, x_valid, model.center_idx)
    y_valid = y_valid_3d[:, 0, :]
    best_alpha = None
    best_loss = float("inf")
    for alpha in RIDGE_ALPHAS:
        candidate = Ridge(alpha=alpha).fit(train_features, y_train)
        prediction = candidate.predict(valid_features)
        valid_loss = float(np.mean((prediction - y_valid) ** 2))
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_alpha = alpha
    model.selected_alpha = float(best_alpha)
    model.estimator = Ridge(alpha=best_alpha).fit(train_features, y_train)
    return model
