"""Leakage-safe train/validation data pipeline for MC-CVLRA L1.

Only the complete timestamp column is read beyond the official 80% selection
boundary.  PM2.5 station features and the signed P=7 center-station variables are
loaded with ``nrows=selection_row_stop``.  No test feature, target, sample,
prediction, cache, or metric is represented by this module.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from cross_variable_lag_l0_data import (
    ALLOWED_SOURCE_COLUMNS,
    AUXILIARY_COLUMNS,
    LAGS,
    PROHIBITED_COLUMNS,
    SOURCE_AUXILIARY_COLUMNS,
    TARGET_COLUMN,
    L0TaskConfig,
    OfficialBoundaries,
    causal_ffill,
    clean_selection_frame,
    establish_official_boundaries,
    stable_hash,
)


BEIJING_STATION_POOL = tuple(range(1001, 1037))
CENTER_STATION_ID = 1013
CORRELATION_THRESHOLD = 0.85


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class L1TaskConfig:
    history: int
    horizon: int
    center_station_id: int = CENTER_STATION_ID
    station_pool: tuple[int, ...] = BEIJING_STATION_POOL
    correlation_threshold: float = CORRELATION_THRESHOLD
    train_fraction: float = 0.7
    validation_fraction: float = 0.1
    max_ffill_hours: int = 6
    auxiliary_missing_rate_limit: float = 0.10
    lags: tuple[int, ...] = LAGS

    def __post_init__(self) -> None:
        if (self.history, self.horizon) not in {(24, 1), (168, 6)}:
            raise ValueError("L1 tasks are frozen at 24->1 and 168->6")
        if self.center_station_id != CENTER_STATION_ID:
            raise ValueError("L1 center station is frozen at Beijing 1013")
        if tuple(self.station_pool) != BEIJING_STATION_POOL:
            raise ValueError("Beijing station pool must be exactly 1001-1036")
        if 1036 not in self.station_pool:
            raise ValueError("Beijing station pool must include 1036")
        if tuple(self.lags) != LAGS:
            raise ValueError("L1 lags are frozen at 0,1,2,3,6,12")
        if self.correlation_threshold != CORRELATION_THRESHOLD:
            raise ValueError("Station Pearson threshold is frozen at >0.85")

    @property
    def task_name(self) -> str:
        return f"{self.history}h_{self.horizon}h"

    @property
    def config_hash(self) -> str:
        return stable_hash(asdict(self))

    @property
    def l0_cleaning_config(self) -> L0TaskConfig:
        return L0TaskConfig(
            history=self.history,
            horizon=self.horizon,
            center_station_id=self.center_station_id,
            train_fraction=self.train_fraction,
            validation_fraction=self.validation_fraction,
            max_ffill_hours=self.max_ffill_hours,
            auxiliary_missing_rate_limit=self.auxiliary_missing_rate_limit,
        )


@dataclass
class L1SelectionData:
    config: L1TaskConfig
    boundaries: OfficialBoundaries
    timestamps: pd.DatetimeIndex
    center_clean: pd.DataFrame
    station_values: pd.DataFrame
    selected_station_ids: list[int]
    station_selection_rows: list[dict[str, Any]]
    processing_ledger: list[dict[str, Any]]
    data_quality_rows: list[dict[str, Any]]
    access_metadata: dict[str, Any]


@dataclass
class L1WindowArrays:
    config: L1TaskConfig
    split: str
    x_st: np.ndarray
    auxiliary_history: np.ndarray
    auxiliary_tokens: np.ndarray
    self_tokens: np.ndarray
    context: np.ndarray
    targets_scaled: np.ndarray
    targets_physical: np.ndarray
    origin_rows: np.ndarray
    origin_times: np.ndarray
    target_rows: np.ndarray
    target_times: np.ndarray
    high_pollution: np.ndarray
    rapid_rise: np.ndarray
    decline: np.ndarray
    ordinary: np.ndarray
    candidate_samples: int
    residual_missing_samples: int

    def subset(self, limit: int | None) -> "L1WindowArrays":
        if limit is None or limit >= len(self.origin_rows):
            return self
        if limit < 1:
            raise ValueError("sample limit must be positive")
        indices = np.arange(limit, dtype=np.int64)
        return L1WindowArrays(
            config=self.config,
            split=self.split,
            x_st=self.x_st[indices],
            auxiliary_history=self.auxiliary_history[indices],
            auxiliary_tokens=self.auxiliary_tokens[indices],
            self_tokens=self.self_tokens[indices],
            context=self.context[indices],
            targets_scaled=self.targets_scaled[indices],
            targets_physical=self.targets_physical[indices],
            origin_rows=self.origin_rows[indices],
            origin_times=self.origin_times[indices],
            target_rows=self.target_rows[indices],
            target_times=self.target_times[indices],
            high_pollution=self.high_pollution[indices],
            rapid_rise=self.rapid_rise[indices],
            decline=self.decline[indices],
            ordinary=self.ordinary[indices],
            candidate_samples=self.candidate_samples,
            residual_missing_samples=self.residual_missing_samples,
        )


@dataclass
class L1PreparedData:
    config: L1TaskConfig
    train: L1WindowArrays
    validation: L1WindowArrays
    metadata: dict[str, Any]
    processing_ledger: list[dict[str, Any]]
    station_selection_rows: list[dict[str, Any]]
    data_quality_rows: list[dict[str, Any]]
    target_time_rows: list[dict[str, Any]]
    boundary_checks: list[dict[str, Any]]


class L1TensorDataset(Dataset):
    """Tensor view used by both base and adapter training loops."""

    def __init__(self, arrays: L1WindowArrays) -> None:
        self.arrays = arrays

    def __len__(self) -> int:
        return len(self.arrays.origin_rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return (
            torch.from_numpy(self.arrays.x_st[index]),
            torch.from_numpy(self.arrays.auxiliary_tokens[index]),
            torch.from_numpy(self.arrays.self_tokens[index]),
            torch.from_numpy(self.arrays.context[index]),
            torch.from_numpy(self.arrays.targets_scaled[index]),
            torch.tensor(index, dtype=torch.long),
        )


def _finite_positive(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    return numeric.where(np.isfinite(numeric) & (numeric > 0))


def _selection_frame_from_center(
    path: Path,
    timestamps: pd.DatetimeIndex,
    boundaries: OfficialBoundaries,
    config: L1TaskConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    raw = pd.read_csv(
        path,
        usecols=list(ALLOWED_SOURCE_COLUMNS),
        nrows=boundaries.selection_row_stop,
    )
    if set(PROHIBITED_COLUMNS) & set(raw.columns):
        raise AssertionError("A prohibited center-station column was loaded")
    feature_times = pd.DatetimeIndex(pd.to_datetime(raw.pop("time"), errors="raise"))
    if not feature_times.equals(timestamps[: boundaries.selection_row_stop]):
        raise ValueError("Center feature timestamps differ from official prefix")
    raw.index = feature_times
    raw = raw.loc[:, [TARGET_COLUMN, *SOURCE_AUXILIARY_COLUMNS]]
    cleaned, ledger, quality = clean_selection_frame(
        raw,
        boundaries,
        config.l0_cleaning_config,
    )
    return raw, cleaned, ledger, quality


def select_training_stations(
    station_frame: pd.DataFrame,
    train_row_stop: int,
    center_station_id: int = CENTER_STATION_ID,
    correlation_threshold: float = CORRELATION_THRESHOLD,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Select stations from raw training rows only; validation is never accepted."""
    if train_row_stop <= 1 or train_row_stop > len(station_frame):
        raise ValueError("Invalid raw training range for station selection")
    if center_station_id not in station_frame.columns:
        raise ValueError("Center station is absent from the fixed pool")
    if list(station_frame.columns) != list(BEIJING_STATION_POOL):
        raise ValueError("Station selection frame must be the ordered 1001-1036 pool")

    raw_train = station_frame.iloc[:train_row_stop]
    center = raw_train[center_station_id]
    rows: list[dict[str, Any]] = []
    selected_neighbors: list[int] = []
    for station_id in BEIJING_STATION_POOL:
        values = raw_train[station_id]
        valid = values.notna()
        complete_count = int(valid.sum())
        completeness = complete_count / len(raw_train)
        pair_valid = valid & center.notna()
        pair_count = int(pair_valid.sum())
        correlation = (
            float(values[pair_valid].corr(center[pair_valid]))
            if pair_count >= 2
            else float("nan")
        )
        selected = (
            station_id != center_station_id
            and np.isfinite(correlation)
            and correlation > correlation_threshold
        )
        if selected and completeness != 1.0:
            raise ValueError(
                f"Selected station {station_id} is not complete in raw training rows"
            )
        if selected:
            selected_neighbors.append(station_id)
        rows.append(
            {
                "station_id": station_id,
                "is_center": station_id == center_station_id,
                "candidate_neighbor": station_id != center_station_id,
                "train_start_row": 0,
                "train_stop_row_exclusive": train_row_stop,
                "raw_train_rows": len(raw_train),
                "raw_valid_rows": complete_count,
                "raw_completeness": completeness,
                "pairwise_rows": pair_count,
                "pearson_to_center": correlation,
                "threshold": correlation_threshold,
                "selected_neighbor": selected,
            }
        )
    selected_station_ids = sorted([center_station_id, *selected_neighbors])
    if len(selected_neighbors) < 5:
        raise ValueError("Training-only station screen produced fewer than five neighbors")
    return selected_station_ids, rows


def _clean_station_pm25_by_split(
    raw_station_values: pd.DataFrame,
    boundaries: OfficialBoundaries,
    max_ffill_hours: int,
    task_name: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    """Causally clean station inputs without carrying fill state across splits."""
    parts = []
    ledger_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    for split_name, start, stop in (
        ("train", 0, boundaries.train_row_stop),
        ("validation", boundaries.train_row_stop, boundaries.selection_row_stop),
    ):
        split = raw_station_values.iloc[start:stop]
        cleaned = pd.DataFrame(index=split.index)
        for station_id in raw_station_values.columns:
            before = split[station_id]
            after = causal_ffill(before, limit=max_ffill_hours)
            cleaned[station_id] = after
            invalid_before = int(before.isna().sum())
            residual = int(after.isna().sum())
            rule = {
                "variable": "PM25_Concentration",
                "station_id": int(station_id),
                "split": split_name,
                "validity": "finite_and_strictly_positive",
                "max_causal_ffill_hours": max_ffill_hours,
                "bfill": False,
            }
            ledger_rows.append(
                {
                    "object_id": f"station_pm25_cleaning:{split_name}:{station_id}",
                    "object_type": "deterministic_processing_no_fit",
                    "task": task_name,
                    "fold": "",
                    "model": "F00/F10",
                    "horizon_step": "",
                    "fit_start": split.index.min().isoformat(),
                    "fit_end": split.index.max().isoformat(),
                    "row_count": len(split),
                    "config_hash": stable_hash(rule),
                    "state_hash": stable_hash(
                        {
                            "invalid_before": invalid_before,
                            "filled": invalid_before - residual,
                            "residual": residual,
                        }
                    ),
                    "details": json.dumps(rule, ensure_ascii=False, sort_keys=True),
                }
            )
            quality_rows.append(
                {
                    "task": task_name,
                    "split": split_name,
                    "variable": f"station_{station_id}_PM25",
                    "rows": len(split),
                    "invalid_before_ffill": invalid_before,
                    "filled_by_causal_ffill": invalid_before - residual,
                    "residual_missing": residual,
                    "residual_missing_rate": residual / len(split),
                    "max_ffill_hours": max_ffill_hours,
                }
            )
        parts.append(cleaned)
    return pd.concat(parts, axis=0), ledger_rows, quality_rows


def load_l1_selection_data(
    data_dir: str | Path,
    config: L1TaskConfig,
) -> L1SelectionData:
    """Load timestamp metadata globally and all features only through validation."""
    root = Path(data_dir)
    center_path = root / f"df_station_{config.center_station_id}.csv"
    timestamp_frame = pd.read_csv(center_path, usecols=["time"])
    boundaries = establish_official_boundaries(
        timestamp_frame["time"],
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
    )
    timestamps = pd.DatetimeIndex(
        pd.to_datetime(timestamp_frame["time"], errors="raise")
    )
    center_raw, center_clean, p7_ledger, quality_rows = _selection_frame_from_center(
        center_path, timestamps, boundaries, config
    )

    station_columns: dict[int, pd.Series] = {
        config.center_station_id: _finite_positive(center_raw[TARGET_COLUMN])
    }
    station_feature_read_calls = 1  # Center PM2.5 arrived in the signed P=7 read.
    for station_id in config.station_pool:
        if station_id == config.center_station_id:
            continue
        path = root / f"df_station_{station_id}.csv"
        raw = pd.read_csv(
            path,
            usecols=["time", TARGET_COLUMN],
            nrows=boundaries.selection_row_stop,
        )
        station_feature_read_calls += 1
        station_times = pd.DatetimeIndex(
            pd.to_datetime(raw.pop("time"), errors="raise")
        )
        if not station_times.equals(timestamps[: boundaries.selection_row_stop]):
            raise ValueError(f"Station {station_id} timestamps differ from center prefix")
        values = _finite_positive(raw[TARGET_COLUMN])
        values.index = station_times
        station_columns[station_id] = values

    raw_station_frame = pd.DataFrame(station_columns).loc[:, config.station_pool]
    selected_ids, station_rows = select_training_stations(
        raw_station_frame,
        boundaries.train_row_stop,
        center_station_id=config.center_station_id,
        correlation_threshold=config.correlation_threshold,
    )
    station_values, station_cleaning_ledger, station_quality_rows = _clean_station_pm25_by_split(
        raw_station_frame.loc[:, selected_ids],
        boundaries,
        config.max_ffill_hours,
        config.task_name,
    )
    p7_ledger.extend(station_cleaning_ledger)
    quality_rows.extend(station_quality_rows)

    selector_config = {
        "station_pool": list(config.station_pool),
        "center_station_id": config.center_station_id,
        "correlation_rule": "raw_train_pairwise_Pearson_strictly_greater_than_0.85",
        "selected_station_ids": selected_ids,
        "validation_rows_accepted_by_selector": 0,
        "test_rows_accepted_by_selector": 0,
        "task_config_hash": config.config_hash,
    }
    selector_state = {
        str(row["station_id"]): {
            "completeness": row["raw_completeness"],
            "correlation": row["pearson_to_center"],
            "selected": row["selected_neighbor"],
        }
        for row in station_rows
    }
    p7_ledger.append(
        {
            "object_id": "training_only_station_selector",
            "object_type": "fitted_selector",
            "task": config.task_name,
            "fold": "",
            "model": "F00/F10",
            "horizon_step": "",
            "fit_start": boundaries.train_start.isoformat(),
            "fit_end": boundaries.train_end.isoformat(),
            "row_count": boundaries.train_row_stop,
            "config_hash": stable_hash(selector_config),
            "state_hash": stable_hash(selector_state),
            "details": json.dumps(selector_config, ensure_ascii=False, sort_keys=True),
        }
    )

    access_metadata = {
        "scope": "L1_selection_train_validation_only_no_test_features",
        "total_timestamp_rows_read": len(timestamps),
        "selection_feature_row_stop_exclusive": boundaries.selection_row_stop,
        "center_allowed_feature_rows_loaded": len(center_raw),
        "station_feature_read_calls": station_feature_read_calls,
        "station_feature_rows_loaded_per_call": boundaries.selection_row_stop,
        "station_pool": list(config.station_pool),
        "station_pool_includes_1036": 1036 in config.station_pool,
        "center_excluded_from_neighbor_pool": config.center_station_id not in [
            value for value in config.station_pool if value != config.center_station_id
        ],
        "selected_station_ids": selected_ids,
        "selected_neighbor_ids": [
            value for value in selected_ids if value != config.center_station_id
        ],
        "test_feature_rows_loaded": 0,
        "test_target_values_loaded": 0,
        "test_samples_constructed": 0,
        "test_predictions_constructed": 0,
        "test_caches_constructed": 0,
        "test_metrics_constructed": 0,
        "bfill_calls": 0,
        "task_config_hash": config.config_hash,
        "center_selection_raw_hash": array_sha256(center_raw.to_numpy(dtype=np.float64)),
        "selected_station_raw_hash": array_sha256(
            raw_station_frame.loc[:, selected_ids].to_numpy(dtype=np.float64)
        ),
    }
    return L1SelectionData(
        config=config,
        boundaries=boundaries,
        timestamps=timestamps,
        center_clean=center_clean,
        station_values=station_values,
        selected_station_ids=selected_ids,
        station_selection_rows=station_rows,
        processing_ledger=p7_ledger,
        data_quality_rows=quality_rows,
        access_metadata=access_metadata,
    )


def _fit_standardizer(
    values: np.ndarray,
    object_id: str,
    columns: list[str],
    selection: L1SelectionData,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if values.ndim != 2 or len(values) != selection.boundaries.train_row_stop:
        raise ValueError("Standardizer must receive exactly the raw training rows")
    mean = np.nanmean(values, axis=0)
    scale = np.nanstd(values, axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError(f"Non-finite training standardizer state for {object_id}")
    state = {"columns": columns, "mean": mean.tolist(), "scale": scale.tolist()}
    config = {
        "fit_split": "train",
        "columns": columns,
        "ddof": 0,
        "task_config_hash": selection.config.config_hash,
    }
    ledger = {
        "object_id": object_id,
        "object_type": "training_standardizer",
        "task": selection.config.task_name,
        "fold": "",
        "model": "all",
        "horizon_step": "",
        "fit_start": selection.boundaries.train_start.isoformat(),
        "fit_end": selection.boundaries.train_end.isoformat(),
        "row_count": len(values),
        "config_hash": stable_hash(config),
        "state_hash": stable_hash(state),
        "details": json.dumps(config, ensure_ascii=False, sort_keys=True),
    }
    return mean, scale, ledger


def _scenario_thresholds(
    center_physical: np.ndarray,
    selection: L1SelectionData,
) -> tuple[dict[str, float], dict[str, Any]]:
    train = center_physical[: selection.boundaries.train_row_stop]
    changes = np.diff(train)
    thresholds = {
        "high_pollution_q90": float(np.quantile(train, 0.90)),
        "rapid_rise_change_q90": float(np.quantile(changes, 0.90)),
        "decline_change_q10": float(np.quantile(changes, 0.10)),
    }
    config = {
        "fit_split": "train",
        "level_quantile": 0.90,
        "rise_change_quantile": 0.90,
        "decline_change_quantile": 0.10,
        "task_config_hash": selection.config.config_hash,
    }
    ledger = {
        "object_id": "training_scenario_thresholds",
        "object_type": "training_quantile_thresholds",
        "task": selection.config.task_name,
        "fold": "",
        "model": "all",
        "horizon_step": "",
        "fit_start": selection.boundaries.train_start.isoformat(),
        "fit_end": selection.boundaries.train_end.isoformat(),
        "row_count": len(train),
        "config_hash": stable_hash(config),
        "state_hash": stable_hash(thresholds),
        "details": json.dumps(config, ensure_ascii=False, sort_keys=True),
    }
    return thresholds, ledger


def _build_split_windows(
    selection: L1SelectionData,
    split: str,
    station_normalized: np.ndarray,
    auxiliary_normalized: np.ndarray,
    center_mean: float,
    center_scale: float,
    scenario_thresholds: dict[str, float],
) -> L1WindowArrays:
    config = selection.config
    boundaries = selection.boundaries
    if split == "train":
        origins = np.arange(
            config.history,
            boundaries.train_row_stop - config.horizon + 1,
            dtype=np.int64,
        )
    elif split == "validation":
        origins = np.arange(
            boundaries.train_row_stop,
            boundaries.selection_row_stop - config.horizon + 1,
            dtype=np.int64,
        )
    else:
        raise ValueError("Only train and validation windows may be constructed in L1")
    if len(origins) == 0:
        raise ValueError(f"No {split} origins remain")

    history_offsets = np.arange(config.history, dtype=np.int64)
    history_rows = origins[:, None] - config.history + history_offsets[None, :]
    x_st = station_normalized[history_rows].transpose(0, 2, 1)
    center_index = selection.selected_station_ids.index(config.center_station_id)
    center_history = x_st[:, center_index]

    lag_array = np.asarray(config.lags, dtype=np.int64)
    current_rows = origins[:, None] - 1 - lag_array[None, :]
    previous_rows = current_rows - 1
    if previous_rows.min() < 0 or current_rows.max() >= origins.max():
        raise AssertionError("A lag token escaped the strict historical range")
    auxiliary_values = auxiliary_normalized[current_rows].transpose(0, 2, 1)
    auxiliary_differences = (
        auxiliary_normalized[current_rows] - auxiliary_normalized[previous_rows]
    ).transpose(0, 2, 1)
    auxiliary_tokens = np.stack(
        [auxiliary_values, auxiliary_differences], axis=-1
    )
    auxiliary_history = auxiliary_normalized[history_rows].transpose(0, 2, 1)

    center_series = station_normalized[:, center_index]
    self_values = center_series[current_rows]
    self_differences = self_values - center_series[previous_rows]
    self_tokens = np.stack([self_values, self_differences], axis=-1)[:, None, :, :]
    context = np.stack(
        [
            center_history[:, -1],
            center_history.mean(axis=1),
            center_history.std(axis=1),
            (center_history[:, -1] - center_history[:, 0])
            / max(config.history - 1, 1),
        ],
        axis=1,
    )

    target_rows = origins[:, None] + np.arange(config.horizon, dtype=np.int64)[None, :]
    center_physical = selection.center_clean[TARGET_COLUMN].to_numpy(dtype=np.float64)
    targets_physical = center_physical[target_rows]
    targets_scaled = ((targets_physical - center_mean) / center_scale)[:, None, :]
    target_changes = targets_physical - center_physical[target_rows - 1]
    high_pollution = targets_physical >= scenario_thresholds["high_pollution_q90"]
    rapid_rise = target_changes >= scenario_thresholds["rapid_rise_change_q90"]
    decline = target_changes <= scenario_thresholds["decline_change_q10"]
    ordinary = ~(high_pollution | rapid_rise | decline)

    finite = (
        np.isfinite(x_st).all(axis=(1, 2))
        & np.isfinite(auxiliary_history).all(axis=(1, 2))
        & np.isfinite(auxiliary_tokens).all(axis=(1, 2, 3))
        & np.isfinite(self_tokens).all(axis=(1, 2, 3))
        & np.isfinite(context).all(axis=1)
        & np.isfinite(targets_scaled).all(axis=(1, 2))
    )
    timestamps = selection.timestamps.to_numpy(dtype="datetime64[ns]")
    retained_origins = origins[finite]
    retained_targets = target_rows[finite]
    return L1WindowArrays(
        config=config,
        split=split,
        x_st=x_st[finite].astype(np.float32),
        auxiliary_history=auxiliary_history[finite].astype(np.float32),
        auxiliary_tokens=auxiliary_tokens[finite].astype(np.float32),
        self_tokens=self_tokens[finite].astype(np.float32),
        context=context[finite].astype(np.float32),
        targets_scaled=targets_scaled[finite].astype(np.float32),
        targets_physical=targets_physical[finite].astype(np.float32),
        origin_rows=retained_origins,
        origin_times=timestamps[retained_origins],
        target_rows=retained_targets,
        target_times=timestamps[retained_targets],
        high_pollution=high_pollution[finite],
        rapid_rise=rapid_rise[finite],
        decline=decline[finite],
        ordinary=ordinary[finite],
        candidate_samples=len(origins),
        residual_missing_samples=int((~finite).sum()),
    )


def _target_time_audit(
    selection: L1SelectionData,
    train: L1WindowArrays,
    validation: L1WindowArrays,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_set = set(train.target_times.reshape(-1))
    validation_set = set(validation.target_times.reshape(-1))
    # Timestamp-only reserved set. No test origin, feature, target value, or sample exists.
    reserved_test_set = set(
        selection.timestamps[
            selection.boundaries.selection_row_stop :
        ].to_numpy(dtype="datetime64[ns]")
    )
    sets = {
        "train": train_set,
        "validation": validation_set,
        "reserved_test_timestamps_only": reserved_test_set,
    }
    rows = [
        {
            "split": split,
            "target_time": pd.Timestamp(value).isoformat(),
            "timestamp_only_no_target_value": split == "reserved_test_timestamps_only",
        }
        for split, values in sets.items()
        for value in sorted(values)
    ]
    checks = []
    pairs = (
        ("train", "validation"),
        ("train", "reserved_test_timestamps_only"),
        ("validation", "reserved_test_timestamps_only"),
    )
    for left, right in pairs:
        intersection = sets[left] & sets[right]
        if intersection:
            raise AssertionError(f"Target-time leakage between {left} and {right}")
        checks.append(
            {
                "boundary_type": "official_split",
                "left_split": left,
                "right_split": right,
                "left_target_time_count": len(sets[left]),
                "right_target_time_count": len(sets[right]),
                "target_time_intersection_count": 0,
                "assertion_passed": True,
            }
        )
    return rows, checks


def prepare_l1_data(
    data_dir: str | Path,
    config: L1TaskConfig,
    train_sample_limit: int | None = None,
    validation_sample_limit: int | None = None,
) -> L1PreparedData:
    selection = load_l1_selection_data(data_dir, config)
    train_stop = selection.boundaries.train_row_stop
    station_raw = selection.station_values.to_numpy(dtype=np.float64)
    station_mean, station_scale, station_ledger = _fit_standardizer(
        station_raw[:train_stop],
        "training_station_pm25_standardizer",
        [str(value) for value in selection.selected_station_ids],
        selection,
    )
    station_normalized = (station_raw - station_mean) / station_scale

    auxiliary_raw = selection.center_clean.loc[:, AUXILIARY_COLUMNS].to_numpy(
        dtype=np.float64
    )
    auxiliary_mean, auxiliary_scale, auxiliary_ledger = _fit_standardizer(
        auxiliary_raw[:train_stop],
        "training_p7_standardizer",
        list(AUXILIARY_COLUMNS),
        selection,
    )
    auxiliary_normalized = (auxiliary_raw - auxiliary_mean) / auxiliary_scale
    center_index = selection.selected_station_ids.index(config.center_station_id)
    center_mean = float(station_mean[center_index])
    center_scale = float(station_scale[center_index])
    center_physical = selection.center_clean[TARGET_COLUMN].to_numpy(dtype=np.float64)
    scenario_thresholds, scenario_ledger = _scenario_thresholds(
        center_physical, selection
    )

    full_train = _build_split_windows(
        selection,
        "train",
        station_normalized,
        auxiliary_normalized,
        center_mean,
        center_scale,
        scenario_thresholds,
    )
    full_validation = _build_split_windows(
        selection,
        "validation",
        station_normalized,
        auxiliary_normalized,
        center_mean,
        center_scale,
        scenario_thresholds,
    )
    target_rows, boundary_checks = _target_time_audit(
        selection, full_train, full_validation
    )
    train = full_train.subset(train_sample_limit)
    validation = full_validation.subset(validation_sample_limit)
    metadata = {
        "task": config.task_name,
        "task_config_hash": config.config_hash,
        "official_boundaries": selection.boundaries.to_dict(),
        "access_metadata": selection.access_metadata,
        "selected_station_ids": selection.selected_station_ids,
        "selected_neighbor_ids": [
            value
            for value in selection.selected_station_ids
            if value != config.center_station_id
        ],
        "center_station_idx": center_index,
        "center_mean": center_mean,
        "center_scale": center_scale,
        "station_mean": station_mean.tolist(),
        "station_scale": station_scale.tolist(),
        "auxiliary_variable_order": list(AUXILIARY_COLUMNS),
        "auxiliary_mean": auxiliary_mean.tolist(),
        "auxiliary_scale": auxiliary_scale.tolist(),
        "lags": list(config.lags),
        "scenario_thresholds": scenario_thresholds,
        "full_train_samples": len(full_train.origin_rows),
        "full_validation_samples": len(full_validation.origin_rows),
        "materialized_train_samples": len(train.origin_rows),
        "materialized_validation_samples": len(validation.origin_rows),
        "smoke_sample_limit_applied": (
            train_sample_limit is not None or validation_sample_limit is not None
        ),
        "train_residual_missing_samples": full_train.residual_missing_samples,
        "validation_residual_missing_samples": full_validation.residual_missing_samples,
        "train_latest_feature_before_origin": bool(
            np.all(full_train.origin_rows[:, None] - 1 >= 0)
        ),
        "all_target_time_sets_disjoint": all(
            row["target_time_intersection_count"] == 0 for row in boundary_checks
        ),
        "test_feature_rows_loaded": 0,
        "test_target_values_loaded": 0,
        "test_samples_constructed": 0,
        "test_predictions_constructed": 0,
        "test_metrics_constructed": 0,
    }
    ledger = [
        *selection.processing_ledger,
        station_ledger,
        auxiliary_ledger,
        scenario_ledger,
    ]
    return L1PreparedData(
        config=config,
        train=train,
        validation=validation,
        metadata=metadata,
        processing_ledger=ledger,
        station_selection_rows=selection.station_selection_rows,
        data_quality_rows=selection.data_quality_rows,
        target_time_rows=target_rows,
        boundary_checks=boundary_checks,
    )
