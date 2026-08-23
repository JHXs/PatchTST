"""Leakage-safe data preparation for the preregistered L0 ridge screen.

This module intentionally knows nothing about ST-PatchTST or MC-CVLRA.  It reads
all timestamps to establish the official 70/10/20 boundaries, but it reads
feature/target columns only for the train+validation selection interval.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TARGET_COLUMN = "PM25_Concentration"
SOURCE_AUXILIARY_COLUMNS = (
    "NO2_Concentration",
    "CO_Concentration",
    "O3_Concentration",
    "SO2_Concentration",
    "temperature",
    "pressure",
    "humidity",
)
AUXILIARY_COLUMNS = (
    "NO2",
    "CO",
    "O3",
    "SO2",
    "temperature",
    "pressure_hPa_fixed",
    "humidity_clean",
)
ALLOWED_SOURCE_COLUMNS = ("time", TARGET_COLUMN, *SOURCE_AUXILIARY_COLUMNS)
PROHIBITED_COLUMNS = (
    "PM10_Concentration",
    "wind_speed",
    "weather",
    "wind_direction",
)
LAGS = (0, 1, 2, 3, 6, 12)
ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
MODEL_FEATURES = ("R_self", "R_recent", "R_lag")


def stable_hash(value: Any) -> str:
    """Return a stable SHA-256 hash for JSON-compatible configuration/state."""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class L0TaskConfig:
    history: int
    horizon: int
    center_station_id: int = 1013
    train_fraction: float = 0.7
    validation_fraction: float = 0.1
    max_ffill_hours: int = 6
    pressure_conversion_threshold: float = 900.0
    pressure_conversion_factor: float = 1.33322
    pressure_min_hpa: float = 900.0
    pressure_max_hpa: float = 1100.0
    temperature_min_c: float = -50.0
    temperature_max_c: float = 60.0
    auxiliary_missing_rate_limit: float = 0.10
    inner_fit_fraction: float = 0.85
    lags: tuple[int, ...] = LAGS
    alphas: tuple[float, ...] = ALPHAS

    @property
    def embargo_origins(self) -> int:
        return self.horizon - 1

    @property
    def task_name(self) -> str:
        return f"{self.history}h_{self.horizon}h"

    @property
    def config_hash(self) -> str:
        return stable_hash(asdict(self))


@dataclass(frozen=True)
class OfficialBoundaries:
    total_timestamp_rows: int
    train_row_stop: int
    selection_row_stop: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        return {key: str(value) if isinstance(value, pd.Timestamp) else value for key, value in result.items()}


@dataclass
class SelectionData:
    frame: pd.DataFrame
    boundaries: OfficialBoundaries
    processing_ledger: list[dict[str, Any]]
    data_quality_rows: list[dict[str, Any]]
    access_metadata: dict[str, Any]


@dataclass
class WindowData:
    config: L0TaskConfig
    row_times: np.ndarray
    origin_rows: np.ndarray
    origin_times: np.ndarray
    target_times: np.ndarray
    targets: np.ndarray
    features: dict[str, np.ndarray]
    block_ids: np.ndarray
    nominal_blocks: dict[int, np.ndarray]
    block_embargo_rows: dict[int, np.ndarray]
    candidate_origin_count: int
    embargoed_origin_count: int
    residual_missing_window_count: int


@dataclass
class FoldSplit:
    fold: int
    outer_train_indices: np.ndarray
    outer_validation_indices: np.ndarray
    inner_fit_indices: np.ndarray
    inner_validation_indices: np.ndarray
    outer_embargo_origin_rows: np.ndarray
    inner_embargo_indices: np.ndarray


@dataclass
class L0Splits:
    folds: list[FoldSplit]
    boundary_checks: list[dict[str, Any]]
    target_time_rows: list[dict[str, Any]]
    embargo_rows: list[dict[str, Any]]


def _timestamp_strings(values: np.ndarray | pd.Index) -> list[str]:
    return [pd.Timestamp(value).isoformat() for value in values]


def establish_official_boundaries(
    timestamps: pd.Series | pd.DatetimeIndex,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.1,
) -> OfficialBoundaries:
    """Build row boundaries from timestamps before any forecast windows exist."""
    parsed = pd.DatetimeIndex(pd.to_datetime(timestamps, errors="raise"))
    if len(parsed) < 3:
        raise ValueError("At least three timestamp rows are required")
    if parsed.has_duplicates:
        raise ValueError("Duplicate timestamps are not allowed")
    if not parsed.is_monotonic_increasing:
        raise ValueError("Timestamps must be strictly increasing")
    deltas = np.diff(parsed.asi8)
    expected_hour_ns = pd.Timedelta(hours=1).value
    if not np.all(deltas == expected_hour_ns):
        raise ValueError("Timestamps must be hourly and contiguous")

    total_rows = len(parsed)
    # Avoid the 0.7 + 0.1 == 0.7999... truncation trap. The protocol fixes
    # exact 70/10/20 row boundaries, so calculate tenths with integer arithmetic.
    if not (
        math.isclose(train_fraction, 0.7, rel_tol=0.0, abs_tol=1e-12)
        and math.isclose(validation_fraction, 0.1, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError("Official L0 fractions are frozen at 70/10/20")
    train_row_stop = total_rows * 7 // 10
    selection_row_stop = total_rows * 8 // 10
    if not 0 < train_row_stop < selection_row_stop < total_rows:
        raise ValueError("The 70/10/20 boundaries would create an empty split")
    return OfficialBoundaries(
        total_timestamp_rows=total_rows,
        train_row_stop=train_row_stop,
        selection_row_stop=selection_row_stop,
        train_start=parsed[0],
        train_end=parsed[train_row_stop - 1],
        validation_start=parsed[train_row_stop],
        validation_end=parsed[selection_row_stop - 1],
        test_start=parsed[selection_row_stop],
    )


def convert_pressure_to_hpa(
    values: pd.Series | np.ndarray,
    threshold: float = 900.0,
    factor: float = 1.33322,
) -> pd.Series:
    """Apply the preregistered deterministic pressure unit conversion."""
    series = pd.to_numeric(pd.Series(values, copy=True), errors="coerce").astype(float)
    finite = np.isfinite(series.to_numpy())
    low = finite & (series.to_numpy() < threshold)
    converted = series.to_numpy(copy=True)
    converted[low] *= factor
    converted[~finite] = np.nan
    return pd.Series(converted, index=series.index, dtype=float)


def causal_ffill(series: pd.Series, limit: int = 6) -> pd.Series:
    """Forward-fill from past observations only, for at most ``limit`` rows."""
    if limit < 0:
        raise ValueError("Forward-fill limit cannot be negative")
    if limit == 0:
        return series.copy()
    return series.ffill(limit=limit)


def _valid_numeric(source: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(source, errors="coerce").astype(float)
    return numeric.where(np.isfinite(numeric))


def _clean_split(
    raw_split: pd.DataFrame,
    split_name: str,
    config: L0TaskConfig,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    cleaned = pd.DataFrame(index=raw_split.index)
    cleaned[TARGET_COLUMN] = _valid_numeric(raw_split[TARGET_COLUMN]).where(
        lambda values: values > 0
    )

    marked: dict[str, pd.Series] = {}
    for source, output in zip(SOURCE_AUXILIARY_COLUMNS[:4], AUXILIARY_COLUMNS[:4]):
        values = _valid_numeric(raw_split[source])
        marked[output] = values.where(values > 0)

    temperature = _valid_numeric(raw_split["temperature"])
    marked["temperature"] = temperature.where(
        temperature.between(config.temperature_min_c, config.temperature_max_c)
    )

    pressure = convert_pressure_to_hpa(
        raw_split["pressure"],
        threshold=config.pressure_conversion_threshold,
        factor=config.pressure_conversion_factor,
    )
    marked["pressure_hPa_fixed"] = pressure.where(
        pressure.between(config.pressure_min_hpa, config.pressure_max_hpa)
    )

    humidity = _valid_numeric(raw_split["humidity"])
    marked["humidity_clean"] = humidity.where((humidity > 0) & (humidity <= 100))

    ledger_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    fit_start = raw_split.index.min().isoformat()
    fit_end = raw_split.index.max().isoformat()
    for output in AUXILIARY_COLUMNS:
        before = marked[output]
        after = causal_ffill(before, limit=config.max_ffill_hours)
        cleaned[output] = after
        invalid_before = int(before.isna().sum())
        residual = int(after.isna().sum())
        rule_config = {
            "object_type": "fixed_missing_rule_and_causal_ffill",
            "variable": output,
            "split": split_name,
            "max_ffill_hours": config.max_ffill_hours,
            "task_config_hash": config.config_hash,
        }
        ledger_rows.append(
            {
                "object_id": f"cleaning:{split_name}:{output}",
                "object_type": "deterministic_processing_no_fit",
                "task": config.task_name,
                "fold": "",
                "model": "",
                "horizon_step": "",
                "fit_start": fit_start,
                "fit_end": fit_end,
                "row_count": len(raw_split),
                "config_hash": stable_hash(rule_config),
                "state_hash": stable_hash(
                    {
                        "invalid_before": invalid_before,
                        "filled": invalid_before - residual,
                        "residual": residual,
                    }
                ),
                "details": json.dumps(rule_config, ensure_ascii=False, sort_keys=True),
            }
        )
        quality_rows.append(
            {
                "task": config.task_name,
                "split": split_name,
                "variable": output,
                "rows": len(raw_split),
                "invalid_before_ffill": invalid_before,
                "filled_by_causal_ffill": invalid_before - residual,
                "residual_missing": residual,
                "residual_missing_rate": residual / len(raw_split),
                "max_ffill_hours": config.max_ffill_hours,
            }
        )

    target_missing = int(cleaned[TARGET_COLUMN].isna().sum())
    quality_rows.append(
        {
            "task": config.task_name,
            "split": split_name,
            "variable": TARGET_COLUMN,
            "rows": len(raw_split),
            "invalid_before_ffill": target_missing,
            "filled_by_causal_ffill": 0,
            "residual_missing": target_missing,
            "residual_missing_rate": target_missing / len(raw_split),
            "max_ffill_hours": 0,
        }
    )
    return cleaned, ledger_rows, quality_rows


def clean_selection_frame(
    raw_selection: pd.DataFrame,
    boundaries: OfficialBoundaries,
    config: L0TaskConfig,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    """Clean train and validation independently so filling cannot cross the split."""
    if len(raw_selection) != boundaries.selection_row_stop:
        raise ValueError("Selection feature rows do not match the timestamp boundary")
    if tuple(raw_selection.columns) != (TARGET_COLUMN, *SOURCE_AUXILIARY_COLUMNS):
        raise ValueError("Unexpected or prohibited source columns reached cleaning")

    train_raw = raw_selection.iloc[: boundaries.train_row_stop]
    validation_raw = raw_selection.iloc[boundaries.train_row_stop :]
    train_clean, train_ledger, train_quality = _clean_split(
        train_raw, "train", config
    )
    validation_clean, validation_ledger, validation_quality = _clean_split(
        validation_raw, "validation", config
    )
    cleaned = pd.concat([train_clean, validation_clean], axis=0)

    for row in train_quality:
        if (
            row["variable"] in AUXILIARY_COLUMNS
            and row["residual_missing_rate"]
            > config.auxiliary_missing_rate_limit
        ):
            raise ValueError(
                f"Training missing-rate gate failed for {row['variable']}: "
                f"{row['residual_missing_rate']:.6f}"
            )
    return cleaned, train_ledger + validation_ledger, train_quality + validation_quality


def load_selection_data(csv_path: str | Path, config: L0TaskConfig) -> SelectionData:
    """Load timestamps globally, then load allowed feature columns only before test."""
    path = Path(csv_path)
    timestamp_frame = pd.read_csv(path, usecols=["time"])
    boundaries = establish_official_boundaries(
        timestamp_frame["time"],
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
    )
    timestamp_index = pd.DatetimeIndex(pd.to_datetime(timestamp_frame["time"]))

    feature_frame = pd.read_csv(
        path,
        usecols=list(ALLOWED_SOURCE_COLUMNS),
        nrows=boundaries.selection_row_stop,
    )
    if any(column in feature_frame.columns for column in PROHIBITED_COLUMNS):
        raise AssertionError("A prohibited input column was loaded")
    feature_times = pd.DatetimeIndex(pd.to_datetime(feature_frame.pop("time")))
    expected_times = timestamp_index[: boundaries.selection_row_stop]
    if not feature_times.equals(expected_times):
        raise ValueError("Feature timestamp prefix differs from the boundary timestamps")
    feature_frame.index = feature_times
    feature_frame = feature_frame.loc[:, [TARGET_COLUMN, *SOURCE_AUXILIARY_COLUMNS]]

    cleaned, processing_ledger, quality_rows = clean_selection_frame(
        feature_frame, boundaries, config
    )
    access_config = {
        "source": str(path),
        "timestamp_columns_read": ["time"],
        "feature_columns_read": list(ALLOWED_SOURCE_COLUMNS),
        "feature_row_stop_exclusive": boundaries.selection_row_stop,
        "test_feature_rows_loaded": 0,
        "test_samples_constructed": 0,
        "task_config_hash": config.config_hash,
    }
    processing_ledger.insert(
        0,
        {
            "object_id": "official_split_boundaries",
            "object_type": "timestamp_boundary_no_fit",
            "task": config.task_name,
            "fold": "",
            "model": "",
            "horizon_step": "",
            "fit_start": boundaries.train_start.isoformat(),
            "fit_end": boundaries.validation_end.isoformat(),
            "row_count": boundaries.selection_row_stop,
            "config_hash": stable_hash(access_config),
            "state_hash": stable_hash(boundaries.to_dict()),
            "details": json.dumps(access_config, ensure_ascii=False, sort_keys=True),
        },
    )
    access_metadata = {
        **access_config,
        "total_timestamp_rows_read": len(timestamp_index),
        "selection_feature_rows_loaded": len(feature_frame),
        "selection_data_hash": stable_hash(
            {
                "times": _timestamp_strings(feature_times),
                "values": feature_frame.to_numpy(dtype=float).tolist(),
            }
        ),
        "formal_test_policy": (
            "Only timestamps establish the boundary; no test feature, target, window, "
            "prediction, cache, or metric is constructed."
        ),
    }
    return SelectionData(
        frame=cleaned,
        boundaries=boundaries,
        processing_ledger=processing_ledger,
        data_quality_rows=quality_rows,
        access_metadata=access_metadata,
    )


def _feature_columns_for_lags(prefix: str, variables: tuple[str, ...]) -> list[str]:
    names = []
    for variable in variables:
        for lag in LAGS:
            names.extend(
                [
                    f"{prefix}:{variable}:lag{lag}:value",
                    f"{prefix}:{variable}:lag{lag}:difference",
                ]
            )
    return names


def feature_names() -> dict[str, list[str]]:
    self_names = _feature_columns_for_lags("self", ("PM25",))
    recent_aux = []
    for variable in AUXILIARY_COLUMNS:
        recent_aux.extend(
            [f"aux:{variable}:lag0:value", f"aux:{variable}:lag0:difference"]
        )
    lag_aux = _feature_columns_for_lags("aux", AUXILIARY_COLUMNS)
    return {
        "R_self": self_names,
        "R_recent": self_names + recent_aux,
        "R_lag": self_names + lag_aux,
    }


def build_window_data(frame: pd.DataFrame, config: L0TaskConfig) -> WindowData:
    """Build common complete-case L0 samples without ever defining test samples."""
    expected_columns = (TARGET_COLUMN, *AUXILIARY_COLUMNS)
    if tuple(frame.columns) != expected_columns:
        raise ValueError("Clean selection frame has an unexpected feature schema")
    if tuple(config.lags) != LAGS:
        raise ValueError("The L0 lag set is frozen and cannot be changed")

    num_rows = len(frame)
    nominal_origins = np.arange(
        config.history,
        num_rows - config.horizon + 1,
        dtype=np.int64,
    )
    if len(nominal_origins) < 8:
        raise ValueError("Selection interval is too short for Q1-Q8")
    nominal_blocks = {
        block_id: values.astype(np.int64, copy=False)
        for block_id, values in enumerate(np.array_split(nominal_origins, 8), start=1)
    }

    keep = np.ones(num_rows, dtype=bool)
    block_embargo_rows: dict[int, np.ndarray] = {}
    for left_block in range(1, 8):
        embargo = config.embargo_origins
        removed = nominal_blocks[left_block][-embargo:] if embargo else np.empty(0, dtype=np.int64)
        block_embargo_rows[left_block] = removed
        keep[removed] = False
    embargoed_origins = nominal_origins[keep[nominal_origins]]

    lag_array = np.asarray(config.lags, dtype=np.int64)
    current_indices = embargoed_origins[:, None] - 1 - lag_array[None, :]
    previous_indices = current_indices - 1
    if previous_indices.min() < 0:
        raise AssertionError("A lag feature attempted to read before the selection data")

    pm25 = frame[TARGET_COLUMN].to_numpy(dtype=np.float64)
    self_values = pm25[current_indices]
    self_differences = self_values - pm25[previous_indices]
    self_features = np.stack([self_values, self_differences], axis=-1).reshape(
        len(embargoed_origins), -1
    )

    auxiliary = frame.loc[:, AUXILIARY_COLUMNS].to_numpy(dtype=np.float64)
    auxiliary_values = auxiliary[current_indices].transpose(0, 2, 1)
    auxiliary_differences = (
        auxiliary[current_indices] - auxiliary[previous_indices]
    ).transpose(0, 2, 1)
    auxiliary_tokens = np.stack(
        [auxiliary_values, auxiliary_differences], axis=-1
    )
    lag_features = auxiliary_tokens.reshape(len(embargoed_origins), -1)
    recent_features = auxiliary_tokens[:, :, 0, :].reshape(
        len(embargoed_origins), -1
    )

    target_indices = embargoed_origins[:, None] + np.arange(config.horizon)[None, :]
    targets = pm25[target_indices]
    all_features = np.concatenate([self_features, lag_features], axis=1)
    finite = np.isfinite(all_features).all(axis=1) & np.isfinite(targets).all(axis=1)

    retained_origins = embargoed_origins[finite]
    retained_target_indices = target_indices[finite]
    retained_blocks = np.empty(len(retained_origins), dtype=np.int64)
    origin_to_block = {
        int(origin): block_id
        for block_id, values in nominal_blocks.items()
        for origin in values
    }
    for index, origin in enumerate(retained_origins):
        retained_blocks[index] = origin_to_block[int(origin)]

    names = feature_names()
    features = {
        "R_self": self_features[finite],
        "R_recent": np.concatenate(
            [self_features[finite], recent_features[finite]], axis=1
        ),
        "R_lag": np.concatenate(
            [self_features[finite], lag_features[finite]], axis=1
        ),
    }
    for model, values in features.items():
        if values.shape[1] != len(names[model]):
            raise AssertionError(f"Feature shape mismatch for {model}")

    timestamps = frame.index.to_numpy(dtype="datetime64[ns]")
    origin_times = timestamps[retained_origins]
    target_times = timestamps[retained_target_indices]
    if target_times.size and target_times.max() > timestamps[-1]:
        raise AssertionError("A target escaped the train+validation selection interval")
    return WindowData(
        config=config,
        row_times=timestamps,
        origin_rows=retained_origins,
        origin_times=origin_times,
        target_times=target_times,
        targets=targets[finite],
        features=features,
        block_ids=retained_blocks,
        nominal_blocks=nominal_blocks,
        block_embargo_rows=block_embargo_rows,
        candidate_origin_count=len(nominal_origins),
        embargoed_origin_count=len(nominal_origins) - len(embargoed_origins),
        residual_missing_window_count=int((~finite).sum()),
    )


def _target_time_set(window: WindowData, indices: np.ndarray) -> set[np.datetime64]:
    return set(window.target_times[indices].reshape(-1))


def _record_boundary(
    window: WindowData,
    boundary_type: str,
    boundary_id: str,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    embargo_count: int,
    boundary_checks: list[dict[str, Any]],
    target_time_rows: list[dict[str, Any]],
) -> None:
    left_set = _target_time_set(window, left_indices)
    right_set = _target_time_set(window, right_indices)
    intersection = left_set & right_set
    if intersection:
        raise AssertionError(
            f"Target-time leakage at {boundary_type}:{boundary_id}: {len(intersection)}"
        )
    boundary_checks.append(
        {
            "task": window.config.task_name,
            "boundary_type": boundary_type,
            "boundary_id": boundary_id,
            "embargo_origin_count": embargo_count,
            "left_sample_count": len(left_indices),
            "right_sample_count": len(right_indices),
            "left_target_time_count": len(left_set),
            "right_target_time_count": len(right_set),
            "target_time_intersection_count": 0,
            "assertion_passed": True,
        }
    )
    for side, values in (("left", left_set), ("right", right_set)):
        for value in sorted(values):
            target_time_rows.append(
                {
                    "task": window.config.task_name,
                    "boundary_type": boundary_type,
                    "boundary_id": boundary_id,
                    "side": side,
                    "target_time": pd.Timestamp(value).isoformat(),
                }
            )


def build_l0_splits(window: WindowData) -> L0Splits:
    """Create all Q, outer, and inner splits with target-time assertions."""
    boundary_checks: list[dict[str, Any]] = []
    target_time_rows: list[dict[str, Any]] = []
    embargo_rows: list[dict[str, Any]] = []
    embargo = window.config.embargo_origins

    for left_block in range(1, 8):
        left_indices = np.flatnonzero(window.block_ids == left_block)
        right_indices = np.flatnonzero(window.block_ids == left_block + 1)
        _record_boundary(
            window,
            "block",
            f"Q{left_block}_to_Q{left_block + 1}",
            left_indices,
            right_indices,
            embargo,
            boundary_checks,
            target_time_rows,
        )
        for origin in window.block_embargo_rows[left_block]:
            embargo_rows.append(
                {
                    "task": window.config.task_name,
                    "boundary_type": "block",
                    "boundary_id": f"Q{left_block}_to_Q{left_block + 1}",
                    "origin_row": int(origin),
                    "origin_time": pd.Timestamp(window.row_times[origin]).isoformat(),
                }
            )

    folds: list[FoldSplit] = []
    for fold in range(1, 6):
        last_train_block = fold + 2
        validation_block = fold + 3
        outer_train = np.flatnonzero(window.block_ids <= last_train_block)
        outer_validation = np.flatnonzero(window.block_ids == validation_block)
        order = np.argsort(window.origin_rows[outer_train])
        outer_train = outer_train[order]
        outer_validation = outer_validation[np.argsort(window.origin_rows[outer_validation])]
        outer_embargo_rows = window.block_embargo_rows[last_train_block]

        _record_boundary(
            window,
            "outer",
            f"fold_{fold}",
            outer_train,
            outer_validation,
            embargo,
            boundary_checks,
            target_time_rows,
        )
        for origin in outer_embargo_rows:
            embargo_rows.append(
                {
                    "task": window.config.task_name,
                    "boundary_type": "outer",
                    "boundary_id": f"fold_{fold}",
                    "origin_row": int(origin),
                    "origin_time": pd.Timestamp(window.row_times[origin]).isoformat(),
                }
            )

        nominal_inner_fit_count = int(
            np.floor(window.config.inner_fit_fraction * len(outer_train))
        )
        if nominal_inner_fit_count <= embargo or nominal_inner_fit_count >= len(outer_train):
            raise ValueError(f"Fold {fold} is too short for the inner split")
        inner_embargo_indices = outer_train[
            nominal_inner_fit_count - embargo : nominal_inner_fit_count
        ] if embargo else np.empty(0, dtype=np.int64)
        inner_fit = outer_train[: nominal_inner_fit_count - embargo]
        inner_validation = outer_train[nominal_inner_fit_count:]
        _record_boundary(
            window,
            "inner",
            f"fold_{fold}",
            inner_fit,
            inner_validation,
            embargo,
            boundary_checks,
            target_time_rows,
        )
        for index in inner_embargo_indices:
            embargo_rows.append(
                {
                    "task": window.config.task_name,
                    "boundary_type": "inner",
                    "boundary_id": f"fold_{fold}",
                    "origin_row": int(window.origin_rows[index]),
                    "origin_time": pd.Timestamp(window.origin_times[index]).isoformat(),
                }
            )
        folds.append(
            FoldSplit(
                fold=fold,
                outer_train_indices=outer_train,
                outer_validation_indices=outer_validation,
                inner_fit_indices=inner_fit,
                inner_validation_indices=inner_validation,
                outer_embargo_origin_rows=outer_embargo_rows,
                inner_embargo_indices=inner_embargo_indices,
            )
        )

    return L0Splits(
        folds=folds,
        boundary_checks=boundary_checks,
        target_time_rows=target_time_rows,
        embargo_rows=embargo_rows,
    )
