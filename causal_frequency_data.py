"""Leak-safe data preparation for the causal frequency research branch.

This module is intentionally independent from ``data_preparation.py`` and the
legacy ST ablation data helpers.  It never writes shared ``tsai/data`` assets.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


HASH_ENCODING = "sorted_unique_unix_ns_int64_le_v1"
WINDOW_HASH_ENCODING = "ordered_target_windows_unix_ns_int64_le_v1"


@dataclass(frozen=True)
class CausalFrequencyDataConfig:
    """Minimum configuration required by the leak-safe data pipeline."""

    data_dir: str = "tsai/data/stations_data"
    center_station_id: int = 1013
    station_start: int = 1001
    station_end: int = 1035
    correlation_threshold: float = 0.85
    history: int = 168
    horizon: int = 6
    train_fraction: float = 0.7
    valid_fraction: float = 0.1


class CausalForecastWindowDataset(Dataset):
    """Windows indexed by forecast origin over normalized ``[time, station]`` data."""

    def __init__(
        self,
        values: np.ndarray,
        target_origins: np.ndarray,
        history: int,
        horizon: int,
        center_idx: int,
        time_index: pd.DatetimeIndex,
    ) -> None:
        values = np.asarray(values, dtype=np.float32)
        target_origins = np.asarray(target_origins, dtype=np.int64)
        if values.ndim != 2:
            raise ValueError(f"values 必须为 [time, station]，实际形状为 {values.shape}")
        if target_origins.ndim != 1:
            raise ValueError("target_origins 必须是一维数组")
        if len(time_index) != len(values):
            raise ValueError("time_index 长度必须与 values 时间维一致")
        if len(target_origins) and (
            target_origins.min() < history
            or target_origins.max() + horizon > len(values)
        ):
            raise ValueError("预测起点无法构造完整历史或目标窗口")
        if not np.isfinite(values).all():
            raise ValueError("数据窗口包含 NaN 或 Inf")

        self.values = values
        self.target_origins = target_origins
        self.history = int(history)
        self.horizon = int(horizon)
        self.center_idx = int(center_idx)
        self.time_index = pd.DatetimeIndex(time_index)

    def __len__(self) -> int:
        return len(self.target_origins)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        target_start = int(self.target_origins[item])
        history_start = target_start - self.history
        # x: [station, history], y: [1, horizon]
        x = self.values[history_start:target_start].T
        y = self.values[
            target_start:target_start + self.horizon, self.center_idx
        ][None, :]
        return torch.from_numpy(x), torch.from_numpy(y)

    def target_timestamp_matrix_ns(self) -> np.ndarray:
        """Return ordered target timestamps as an ``[sample, horizon]`` int64 matrix."""

        offsets = np.arange(self.horizon, dtype=np.int64)
        positions = self.target_origins[:, None] + offsets[None, :]
        return self.time_index.asi8[positions]


def _json_float(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _sha256_int64(values: np.ndarray) -> str:
    payload = np.asarray(values, dtype="<i8").tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def _normalize_aligned_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Sort, deduplicate and expand a station frame to a complete hourly grid."""

    if not isinstance(frame.index, pd.DatetimeIndex):
        try:
            frame = frame.copy()
            frame.index = pd.to_datetime(frame.index, errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError("站点数据索引必须可解析为时间") from error
    if frame.empty:
        raise ValueError("站点数据为空")
    if frame.columns.duplicated().any():
        raise ValueError("站点列存在重复 ID")

    normalized = frame.copy()
    normalized.columns = [int(column) for column in normalized.columns]
    normalized = normalized.sort_index(kind="stable")
    duplicate_rows = int(normalized.index.duplicated(keep="first").sum())
    normalized = normalized.loc[~normalized.index.duplicated(keep="first")]
    normalized = normalized.reindex(sorted(normalized.columns), axis=1)
    normalized = normalized.apply(pd.to_numeric, errors="coerce")

    full_index = pd.date_range(
        normalized.index.min(), normalized.index.max(), freq="h"
    )
    inserted_hour_rows = int(len(full_index) - len(normalized))
    normalized = normalized.reindex(full_index)
    normalized.index.name = "time"
    metadata = {
        "deduplicated_time_rows": duplicate_rows,
        "inserted_missing_hour_rows": inserted_hour_rows,
        "aligned_rows_before_trim": len(normalized),
    }
    return normalized, metadata


def causal_forward_fill_and_trim(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Forward-fill from the past and remove the common incomplete prefix."""

    normalized, alignment_metadata = _normalize_aligned_frame(frame)
    missing_before = {
        str(station_id): int(value)
        for station_id, value in normalized.isna().sum().items()
    }
    filled = normalized.ffill()
    complete_rows = filled.notna().all(axis=1).to_numpy()
    if not complete_rows.any():
        raise ValueError("因果前向填充后没有所有站点同时可用的时间行")
    first_complete_position = int(np.flatnonzero(complete_rows)[0])
    trimmed = filled.iloc[first_complete_position:].copy()
    if trimmed.isna().any().any():
        raise ValueError("公共前缀裁剪后仍包含缺失值")

    metadata = {
        **alignment_metadata,
        "missing_values_before_ffill": missing_before,
        "leading_rows_trimmed": first_complete_position,
        "usable_rows": len(trimmed),
        "fill_method": "causal_forward_fill_only",
        "duplicate_time_policy": "keep_first_after_stable_time_sort",
        "time_grid": "complete_hourly_grid",
    }
    return trimmed, metadata


def read_aligned_station_pm25(config) -> tuple[pd.DataFrame, dict]:
    """Read candidate PM2.5 series, recording unusable non-center stations."""

    series_by_station: dict[int, pd.Series] = {}
    excluded: dict[str, str] = {}
    duplicate_rows: dict[str, int] = {}
    for station_id in range(config.station_start, config.station_end + 1):
        path = Path(config.data_dir) / f"df_station_{station_id}.csv"
        if not path.is_file():
            excluded[str(station_id)] = "missing_file"
            continue
        try:
            station_frame = pd.read_csv(
                path, usecols=["time", "PM25_Concentration"]
            )
            timestamps = pd.to_datetime(station_frame["time"], errors="coerce")
            if timestamps.isna().any():
                raise ValueError("invalid_timestamp")
            values = pd.to_numeric(
                station_frame["PM25_Concentration"], errors="coerce"
            )
            station_series = pd.Series(
                values.to_numpy(), index=pd.DatetimeIndex(timestamps), name=station_id
            ).sort_index(kind="stable")
            duplicates = int(station_series.index.duplicated(keep="first").sum())
            station_series = station_series.loc[
                ~station_series.index.duplicated(keep="first")
            ]
            if station_series.empty:
                raise ValueError("empty_series")
        except (OSError, KeyError, TypeError, ValueError) as error:
            excluded[str(station_id)] = f"unreadable_or_invalid:{error}"
            continue
        series_by_station[station_id] = station_series
        duplicate_rows[str(station_id)] = duplicates

    if config.center_station_id not in series_by_station:
        reason = excluded.get(str(config.center_station_id), "not_in_candidate_range")
        raise ValueError(
            f"中心站 {config.center_station_id} 不可用，原因: {reason}"
        )
    if len(series_by_station) < 2:
        raise ValueError("至少需要中心站和一个候选邻站")

    aligned = pd.concat(series_by_station.values(), axis=1, join="outer")
    aligned, fill_metadata = causal_forward_fill_and_trim(aligned)
    metadata = {
        **fill_metadata,
        "available_station_ids": [int(value) for value in aligned.columns],
        "excluded_stations": excluded,
        "source_duplicate_rows": duplicate_rows,
    }
    return aligned, metadata


def split_target_origins(
    row_count: int,
    history: int,
    horizon: int,
    train_boundary: int,
    valid_boundary: int,
) -> dict[str, np.ndarray]:
    """Build split origins so target timestamps never cross split boundaries."""

    if history < 1 or horizon < 1:
        raise ValueError("history 和 horizon 必须为正整数")
    if not history < train_boundary < valid_boundary < row_count:
        raise ValueError("时间边界不足以形成 train/valid/test")

    origins = {
        "train": np.arange(
            history, train_boundary - horizon + 1, dtype=np.int64
        ),
        "valid": np.arange(
            train_boundary, valid_boundary - horizon + 1, dtype=np.int64
        ),
        "test": np.arange(
            valid_boundary, row_count - horizon + 1, dtype=np.int64
        ),
    }
    empty = [name for name, values in origins.items() if len(values) == 0]
    if empty:
        raise ValueError(f"以下划分没有可用窗口: {empty}")
    return origins


def _target_timestamp_metadata(
    dataset: CausalForecastWindowDataset,
) -> tuple[dict, set[int]]:
    matrix = dataset.target_timestamp_matrix_ns()
    unique_ns = np.unique(matrix.reshape(-1))
    metadata = {
        "sample_windows": int(matrix.shape[0]),
        "unique_timestamp_count": int(len(unique_ns)),
        "min": str(pd.Timestamp(unique_ns.min())),
        "max": str(pd.Timestamp(unique_ns.max())),
        "unique_timestamp_sha256": _sha256_int64(unique_ns),
        "unique_hash_encoding": HASH_ENCODING,
        "ordered_window_sha256": _sha256_int64(matrix),
        "ordered_window_hash_encoding": WINDOW_HASH_ENCODING,
    }
    return metadata, set(int(value) for value in unique_ns)


def prepare_causal_datasets(
    config,
    aligned_frame: pd.DataFrame | None = None,
) -> tuple[dict[str, CausalForecastWindowDataset], dict]:
    """Prepare leak-safe train/valid/test windows and complete audit metadata."""

    if not 0 < config.train_fraction < 1:
        raise ValueError("train_fraction 必须位于 (0, 1)")
    if not 0 < config.valid_fraction < 1 - config.train_fraction:
        raise ValueError("valid_fraction 必须为正且 train+valid 小于 1")

    if aligned_frame is None:
        frame, source_metadata = read_aligned_station_pm25(config)
    else:
        frame, fill_metadata = causal_forward_fill_and_trim(aligned_frame)
        source_metadata = {
            **fill_metadata,
            "available_station_ids": [int(value) for value in frame.columns],
            "excluded_stations": {},
            "source_duplicate_rows": {},
            "source": "provided_aligned_frame",
        }

    row_count = len(frame)
    # Decimal avoids 0.7 + 0.1 becoming 0.799999... and moving a boundary
    # one row earlier.  Fractions originate from CLI/config decimal strings.
    train_fraction = Decimal(str(config.train_fraction))
    valid_fraction = Decimal(str(config.valid_fraction))
    train_boundary = int(train_fraction * row_count)
    valid_boundary = int((train_fraction + valid_fraction) * row_count)
    origins = split_target_origins(
        row_count,
        config.history,
        config.horizon,
        train_boundary,
        valid_boundary,
    )

    if config.center_station_id not in frame.columns:
        raise ValueError(f"中心站 {config.center_station_id} 不在对齐数据中")
    train_frame = frame.iloc[:train_boundary]
    correlations = train_frame.corr()[config.center_station_id]
    selected_ids = [
        int(station_id)
        for station_id in frame.columns
        if np.isfinite(correlations.loc[station_id])
        and correlations.loc[station_id] > config.correlation_threshold
    ]
    if config.center_station_id not in selected_ids:
        selected_ids.append(int(config.center_station_id))
    selected_ids = sorted(selected_ids)
    selected = frame[selected_ids]

    raw_values = selected.to_numpy(dtype=np.float32)
    train_values = raw_values[:train_boundary]
    means = train_values.mean(axis=0, dtype=np.float64).astype(np.float32)
    raw_stds = train_values.std(axis=0, ddof=0, dtype=np.float64).astype(np.float32)
    scales = np.where(raw_stds < 1e-6, 1.0, raw_stds).astype(np.float32)
    normalized = ((raw_values - means) / scales).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("标准化后出现 NaN 或 Inf")

    center_idx = selected_ids.index(config.center_station_id)
    datasets = {
        split: CausalForecastWindowDataset(
            normalized,
            split_origins,
            config.history,
            config.horizon,
            center_idx,
            frame.index,
        )
        for split, split_origins in origins.items()
    }

    target_metadata = {}
    target_sets = {}
    for split, dataset in datasets.items():
        target_metadata[split], target_sets[split] = _target_timestamp_metadata(dataset)
    intersections = {
        "train_valid": len(target_sets["train"] & target_sets["valid"]),
        "train_test": len(target_sets["train"] & target_sets["test"]),
        "valid_test": len(target_sets["valid"] & target_sets["test"]),
    }
    if any(intersections.values()):
        raise AssertionError(f"目标时间戳跨划分重叠: {intersections}")

    train_target_positions = np.unique(
        origins["train"][:, None]
        + np.arange(config.horizon, dtype=np.int64)[None, :]
    )
    train_target_physical = raw_values[train_target_positions, center_idx]
    peak_q90 = float(np.quantile(train_target_physical, 0.9))

    candidate_correlations = {
        str(station_id): _json_float(correlations.loc[station_id])
        for station_id in frame.columns
    }
    metadata = {
        "protocol_version": "FREQ-PROTOCOL-v1",
        "rows": row_count,
        "start_time": str(frame.index.min()),
        "end_time": str(frame.index.max()),
        "time_index_sha256": _sha256_int64(frame.index.asi8),
        "time_index_hash_encoding": HASH_ENCODING,
        "boundary_basis": "usable_rows_after_causal_ffill_and_common_prefix_trim",
        "train_boundary_row": train_boundary,
        "valid_boundary_row": valid_boundary,
        "train_boundary_time": str(frame.index[train_boundary]),
        "valid_boundary_time": str(frame.index[valid_boundary]),
        "split_assignment": {
            "train": "history <= t and t + horizon <= train_boundary",
            "valid": "train_boundary <= t and t + horizon <= valid_boundary",
            "test": "valid_boundary <= t and t + horizon <= rows",
        },
        "split_sizes": {split: len(dataset) for split, dataset in datasets.items()},
        "target_timestamps": target_metadata,
        "target_timestamp_intersections": intersections,
        "target_timestamp_intersections_empty": True,
        "candidate_station_ids": [int(value) for value in frame.columns],
        "station_ids": selected_ids,
        "station_selection_period": f"rows[0:{train_boundary})",
        "station_selection_rule": (
            f"train_only_pearson_strictly_greater_than_{config.correlation_threshold}"
        ),
        "candidate_train_correlations": candidate_correlations,
        "station_correlations": {
            str(station_id): candidate_correlations[str(station_id)]
            for station_id in selected_ids
        },
        "center_station_idx": center_idx,
        "center_mean": float(means[center_idx]),
        "center_std": float(scales[center_idx]),
        "station_means": [float(value) for value in means],
        "station_raw_stds_ddof0": [float(value) for value in raw_stds],
        "station_stds": [float(value) for value in scales],
        "standardization_period": f"rows[0:{train_boundary})",
        "standardization_ddof": 0,
        "near_zero_std_scale": 1.0,
        "peak_q90_ugm3": peak_q90,
        "peak_q90_scaled": float(
            (peak_q90 - means[center_idx]) / scales[center_idx]
        ),
        "peak_q90_definition": (
            "90th percentile of center PM2.5 at sorted unique train target timestamps"
        ),
        "source_preprocessing": source_metadata,
    }
    return datasets, metadata
