"""Diagnose whether selected neighbors lead center-station PM2.5 on training data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from run_st_patchtst_ablation import ExperimentConfig, load_station_data


OUTPUT_DIR = Path("experiments/results/st_patchtst_ablation/lag_diagnostics")
LAGS = (0, 1, 2, 3, 6, 12)
NUM_FOLDS = 5
FOLD_SIZE = 600
RIDGE = 1e-6


def fit_linear(train_x, train_y, valid_x):
    """Fit a standardized ridge linear model and return physical-scale predictions."""
    x_mean = train_x.mean(axis=0)
    x_std = train_x.std(axis=0)
    x_std[x_std < 1e-8] = 1.0
    y_mean = train_y.mean()
    y_std = train_y.std()
    train_scaled = (train_x - x_mean) / x_std
    valid_scaled = (valid_x - x_mean) / x_std
    target_scaled = (train_y - y_mean) / y_std
    design = np.column_stack([np.ones(len(train_scaled)), train_scaled])
    penalty = np.eye(design.shape[1]) * RIDGE
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ target_scaled,
    )
    valid_design = np.column_stack([np.ones(len(valid_scaled)), valid_scaled])
    return (valid_design @ coefficients) * y_std + y_mean


def correlation_rows(train_frame, center_station_id):
    center = train_frame[center_station_id].to_numpy(dtype=np.float64)
    center_difference = np.diff(center)
    rows = []
    for station_id in train_frame.columns:
        if station_id == center_station_id:
            continue
        neighbor = train_frame[station_id].to_numpy(dtype=np.float64)
        neighbor_difference = np.diff(neighbor)
        for lag in LAGS:
            if lag == 0:
                raw_center, raw_neighbor = center, neighbor
                diff_center, diff_neighbor = center_difference, neighbor_difference
            else:
                raw_center, raw_neighbor = center[lag:], neighbor[:-lag]
                diff_center = center_difference[lag:]
                diff_neighbor = neighbor_difference[:-lag]
            rows.append(
                {
                    "station_id": int(station_id),
                    "lag_hours": lag,
                    "raw_pearson_r": np.corrcoef(raw_center, raw_neighbor)[0, 1],
                    "difference_pearson_r": np.corrcoef(
                        diff_center, diff_neighbor
                    )[0, 1],
                }
            )
    return pd.DataFrame(rows)


def expanding_cv_rows(train_frame, config):
    values = train_frame.to_numpy(dtype=np.float64)
    station_ids = [int(value) for value in train_frame.columns]
    center_idx = station_ids.index(config.center_station_id)
    center = values[:, center_idx]
    target_times = np.arange(config.history, len(train_frame))
    ar_features = np.stack(
        [center[target_time - config.history:target_time] for target_time in target_times]
    )
    targets = center[target_times]
    first_valid = len(target_times) - NUM_FOLDS * FOLD_SIZE
    if first_valid <= config.history:
        raise ValueError("训练时间段不足以构造指定的扩展窗口交叉验证")

    baseline_fold_rmse = {}
    for fold in range(NUM_FOLDS):
        valid_start = first_valid + fold * FOLD_SIZE
        valid_stop = valid_start + FOLD_SIZE
        prediction = fit_linear(
            ar_features[:valid_start],
            targets[:valid_start],
            ar_features[valid_start:valid_stop],
        )
        baseline_fold_rmse[fold] = float(
            np.sqrt(np.mean((prediction - targets[valid_start:valid_stop]) ** 2))
        )

    rows = []
    for station_idx, station_id in enumerate(station_ids):
        if station_id == config.center_station_id:
            continue
        neighbor = values[:, station_idx]
        for lag in LAGS:
            neighbor_latest = neighbor[target_times - 1 - lag, None]
            augmented_features = np.concatenate([ar_features, neighbor_latest], axis=1)
            for fold in range(NUM_FOLDS):
                valid_start = first_valid + fold * FOLD_SIZE
                valid_stop = valid_start + FOLD_SIZE
                prediction = fit_linear(
                    augmented_features[:valid_start],
                    targets[:valid_start],
                    augmented_features[valid_start:valid_stop],
                )
                rmse = float(
                    np.sqrt(
                        np.mean((prediction - targets[valid_start:valid_stop]) ** 2)
                    )
                )
                baseline_rmse = baseline_fold_rmse[fold]
                rows.append(
                    {
                        "station_id": station_id,
                        "lag_hours": lag,
                        "fold": fold + 1,
                        "train_samples": valid_start,
                        "valid_samples": FOLD_SIZE,
                        "baseline_rmse_ugm3": baseline_rmse,
                        "augmented_rmse_ugm3": rmse,
                        "rmse_improvement_percent": (
                            baseline_rmse / rmse - 1
                        )
                        * 100,
                    }
                )
    return pd.DataFrame(rows)


def main():
    config = ExperimentConfig(history=24, horizon=1)
    frame, _ = load_station_data(config)
    num_samples = len(frame) - config.history - config.horizon + 1
    train_end = int(num_samples * config.train_fraction)
    train_row_stop = train_end + config.history + config.horizon - 1
    train_frame = frame.iloc[:train_row_stop]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    correlations = correlation_rows(train_frame, config.center_station_id)
    correlations.to_csv(OUTPUT_DIR / "lag_correlations.csv", index=False)
    correlation_summary = (
        correlations.groupby("lag_hours", sort=False)
        .agg(
            raw_pearson_r_mean=("raw_pearson_r", "mean"),
            raw_pearson_r_std=("raw_pearson_r", "std"),
            difference_pearson_r_mean=("difference_pearson_r", "mean"),
            difference_pearson_r_std=("difference_pearson_r", "std"),
        )
        .reset_index()
    )
    correlation_summary.to_csv(
        OUTPUT_DIR / "lag_correlation_summary.csv", index=False
    )

    cv = expanding_cv_rows(train_frame, config)
    cv.to_csv(OUTPUT_DIR / "lag_cv_metrics.csv", index=False)
    station_summary = (
        cv.groupby(["station_id", "lag_hours"], sort=False)
        .agg(
            baseline_rmse_ugm3_mean=("baseline_rmse_ugm3", "mean"),
            augmented_rmse_ugm3_mean=("augmented_rmse_ugm3", "mean"),
            rmse_improvement_percent_mean=("rmse_improvement_percent", "mean"),
            positive_fold_count=(
                "rmse_improvement_percent",
                lambda values: int((values > 0).sum()),
            ),
        )
        .reset_index()
    )
    station_summary.to_csv(OUTPUT_DIR / "lag_station_summary.csv", index=False)
    lag_summary = (
        station_summary.groupby("lag_hours", sort=False)
        .agg(
            baseline_rmse_ugm3_mean=("baseline_rmse_ugm3_mean", "mean"),
            augmented_rmse_ugm3_mean=("augmented_rmse_ugm3_mean", "mean"),
            rmse_improvement_percent_mean=("rmse_improvement_percent_mean", "mean"),
            positive_station_count=(
                "rmse_improvement_percent_mean",
                lambda values: int((values > 0).sum()),
            ),
        )
        .reset_index()
    )
    lag_summary.to_csv(OUTPUT_DIR / "lag_cv_summary.csv", index=False)

    lag_zero = station_summary[station_summary["lag_hours"] == 0].set_index(
        "station_id"
    )
    lag_zero_correlations = correlations[
        correlations["lag_hours"] == 0
    ].set_index("station_id")
    lag_zero_best_station_count = 0
    for _, group in station_summary.groupby("station_id"):
        lag_zero_gain = group.loc[
            group["lag_hours"] == 0, "rmse_improvement_percent_mean"
        ].iloc[0]
        lag_zero_best_station_count += int(
            lag_zero_gain >= group["rmse_improvement_percent_mean"].max()
        )
    correlation_gain_spearman = lag_zero_correlations["raw_pearson_r"].corr(
        lag_zero["rmse_improvement_percent_mean"], method="spearman"
    )

    metadata = pd.DataFrame(
        [
            {
                "center_station_id": config.center_station_id,
                "history_hours": config.history,
                "horizon_hours": config.horizon,
                "training_rows": len(train_frame),
                "start_time": train_frame.index.min(),
                "end_time": train_frame.index.max(),
                "num_folds": NUM_FOLDS,
                "fold_size": FOLD_SIZE,
                "candidate_lags": ",".join(map(str, LAGS)),
                "neighbor_feature": "latest observed neighbor PM2.5 at t-1-lag",
                "lag_zero_positive_station_count": int(
                    (lag_zero["rmse_improvement_percent_mean"] > 0).sum()
                ),
                "lag_zero_best_station_count": lag_zero_best_station_count,
                "lag_zero_raw_correlation_gain_spearman": correlation_gain_spearman,
            }
        ]
    )
    metadata.to_csv(OUTPUT_DIR / "lag_diagnostic_metadata.csv", index=False)

    print(correlation_summary.to_string(index=False))
    print(lag_summary.to_string(index=False))
    print(
        station_summary[station_summary["lag_hours"] == 0]
        .sort_values("rmse_improvement_percent_mean", ascending=False)
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
