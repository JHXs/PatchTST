import os
from csv import reader
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def infer_num_metrics(y):
    y = np.asarray(y)
    if y.ndim <= 2:
        return 1
    return int(y.shape[1])


def load_target_names(expected_count, params_path="tsai/data/model_params.npz"):
    fallback_names = [f"metric_{idx}" for idx in range(expected_count)]
    if not os.path.exists(params_path):
        inferred_names = infer_target_names_from_raw_data(expected_count)
        return inferred_names if inferred_names is not None else fallback_names

    try:
        with np.load(params_path, allow_pickle=True) as params:
            for key in ("target_names", "center_vars", "y_vars"):
                if key not in params.files:
                    continue
                target_names = _normalize_target_names(
                    params[key],
                    center_station_id=_extract_scalar(params, "center_station_id"),
                )
                if len(target_names) == expected_count:
                    return target_names
            inferred_names = infer_target_names_from_raw_data(
                expected_count,
                center_station_id=_extract_scalar(params, "center_station_id"),
            )
            if inferred_names is not None:
                return inferred_names
    except Exception as exc:
        print(f"警告: 读取目标变量名失败，将使用默认名称。原因: {exc}")

    return fallback_names


def build_split_evaluation(y_true, y_pred, split_name, target_names):
    y_true = _ensure_3d(y_true)
    y_pred = _ensure_3d(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"{split_name} 的真实值与预测值形状不一致: {y_true.shape} vs {y_pred.shape}")

    metric_count = y_true.shape[1]
    metric_names = list(target_names)
    if len(metric_names) != metric_count:
        metric_names = [f"metric_{idx}" for idx in range(metric_count)]

    summary_row = {"split": split_name, **compute_metrics(y_true, y_pred)}
    per_metric_rows = []
    for metric_idx, metric_name in enumerate(metric_names):
        per_metric_rows.append(
            {
                "split": split_name,
                "metric_name": metric_name,
                **compute_metrics(y_true[:, metric_idx, :], y_pred[:, metric_idx, :]),
            }
        )

    return summary_row, per_metric_rows


def build_results_dataframes(summary_rows, per_metric_rows):
    summary_df = pd.DataFrame(summary_rows).set_index("split")
    per_metric_df = pd.DataFrame(per_metric_rows)
    return summary_df, per_metric_df


def print_evaluation_results(summary_df, per_metric_df):
    print("\n总体评估指标:")
    print(summary_df.to_string())

    for split_name in summary_df.index:
        split_metrics = per_metric_df[per_metric_df["split"] == split_name].reset_index(drop=True)
        print(f"\n{split_name} 单指标评估结果:")
        print(split_metrics.to_string(index=False))


def compute_metrics(y_true, y_pred):
    y_true_flat = np.asarray(y_true).reshape(-1)
    y_pred_flat = np.asarray(y_pred).reshape(-1)
    mse_value = mean_squared_error(y_true_flat, y_pred_flat)
    return {
        "mse": mse_value,
        "rmse": float(np.sqrt(mse_value)),
        "mae": mean_absolute_error(y_true_flat, y_pred_flat),
    }


def _ensure_3d(array):
    array = np.asarray(array)
    if array.ndim == 1:
        return array[:, None, None]
    if array.ndim == 2:
        return array[:, None, :]
    if array.ndim == 3:
        return array
    raise ValueError(f"只支持 1D/2D/3D 评估输入，当前维度为 {array.ndim}")


def infer_target_names_from_raw_data(expected_count, center_station_id=None):
    candidate_paths = []
    if center_station_id is not None:
        for base_dir in _candidate_station_dirs():
            candidate_paths.append(base_dir / f"df_station_{center_station_id}.csv")

    for base_dir in _candidate_station_dirs():
        candidate_paths.extend(sorted(base_dir.glob("df_station_*.csv")))

    seen_paths = set()
    for candidate_path in candidate_paths:
        if candidate_path in seen_paths or not candidate_path.exists():
            continue
        seen_paths.add(candidate_path)
        header = _read_csv_header(candidate_path)
        feature_names = [name for name in header if name not in {"station_id", "time"}]
        if len(feature_names) == expected_count:
            return feature_names
    return None


def _candidate_station_dirs():
    return [
        Path("tsai/data/stations_data_Guangzhou"),
        Path("tsai/data/stations_data"),
    ]


def _read_csv_header(csv_path):
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        header = next(reader(handle), [])
    return header


def _extract_scalar(params, key):
    if key not in params.files:
        return None
    value = params[key]
    if np.isscalar(value):
        return value.item() if hasattr(value, "item") else value
    if getattr(value, "shape", ()) == ():
        return value.item()
    return None


def _normalize_target_names(values, center_station_id=None):
    normalized_names = []
    for value in np.atleast_1d(values).tolist():
        if isinstance(value, bytes):
            decoded_value = value.decode("utf-8")
        else:
            decoded_value = str(value)
        if center_station_id is not None:
            suffix = f"_{center_station_id}"
            if decoded_value.endswith(suffix):
                decoded_value = decoded_value[: -len(suffix)]
        normalized_names.append(decoded_value)
    return normalized_names
