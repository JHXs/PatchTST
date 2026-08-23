"""Run the preregistered selection-interval L0 forward-chaining ridge screen.

The output is evidence from train+validation selection time only.  It is neither
formal-test performance nor MC-CVLRA/ST-PatchTST neural performance.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cross_variable_lag_l0_data import (
    ALPHAS,
    AUXILIARY_COLUMNS,
    LAGS,
    MODEL_FEATURES,
    L0TaskConfig,
    SelectionData,
    WindowData,
    build_l0_splits,
    build_window_data,
    feature_names,
    load_selection_data,
    stable_hash,
)


SCOPE_LABEL = "selection_interval_L0_not_test_not_neural"
DEFAULT_OUTPUT_DIR = Path("experiments/results/cross_variable_lag_adapter/l0")
DEFAULT_DATA_PATH = Path("tsai/data/stations_data/df_station_1013.csv")


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale

    @property
    def state_hash(self) -> str:
        return stable_hash(
            {"mean": self.mean.tolist(), "scale": self.scale.tolist()}
        )


@dataclass
class DirectRidge:
    coefficients: np.ndarray
    x_mean: np.ndarray
    y_mean: np.ndarray
    alpha: float

    def predict(self, values: np.ndarray) -> np.ndarray:
        return (values - self.x_mean) @ self.coefficients + self.y_mean

    def horizon_state_hash(self, horizon_index: int) -> str:
        return stable_hash(
            {
                "alpha": self.alpha,
                "coefficient": self.coefficients[:, horizon_index].tolist(),
                "x_mean": self.x_mean.tolist(),
                "intercept_mean": float(self.y_mean[horizon_index]),
            }
        )


@dataclass
class TaskArtifacts:
    config: L0TaskConfig
    metadata: dict[str, Any]
    frames: dict[str, pd.DataFrame]


def fit_standardizer(train_x: np.ndarray) -> Standardizer:
    """Fit a population standardizer on exactly the supplied training slice."""
    if train_x.ndim != 2 or len(train_x) == 0:
        raise ValueError("Standardizer requires a non-empty 2D training matrix")
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0, ddof=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return Standardizer(mean=mean, scale=scale)


def fit_direct_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    alpha: float,
) -> DirectRidge:
    """Fit independent direct ridge coefficients for every horizon.

    The solve uses a matrix right-hand side for efficiency, but each output column
    has its own coefficient vector and intercept, exactly matching separate fits.
    """
    if alpha <= 0:
        raise ValueError("Ridge alpha must be positive")
    if train_x.ndim != 2 or train_y.ndim != 2 or len(train_x) != len(train_y):
        raise ValueError("Ridge inputs must be aligned 2D matrices")
    x_mean = train_x.mean(axis=0)
    y_mean = train_y.mean(axis=0)
    centered_x = train_x - x_mean
    centered_y = train_y - y_mean
    gram = centered_x.T @ centered_x
    penalty = np.eye(gram.shape[0], dtype=np.float64) * float(alpha)
    coefficients = np.linalg.solve(
        gram + penalty,
        centered_x.T @ centered_y,
    )
    return DirectRidge(
        coefficients=coefficients,
        x_mean=x_mean,
        y_mean=y_mean,
        alpha=float(alpha),
    )


def regression_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if actual.shape != prediction.shape or actual.size == 0:
        raise ValueError("Metric arrays must be non-empty and have identical shapes")
    errors = prediction - actual
    denominator = np.abs(actual) + np.abs(prediction)
    smape_terms = np.divide(
        2.0 * np.abs(errors),
        denominator,
        out=np.zeros_like(errors),
        where=denominator > 0,
    )
    return {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(np.abs(errors))),
        "smape_percent": float(np.mean(smape_terms) * 100.0),
    }


def select_alpha(
    scores: dict[float, float],
    tie_tolerance: float = 1e-12,
) -> float:
    """Select minimum RMSE_all; numerical ties choose the larger alpha."""
    if not scores:
        raise ValueError("At least one alpha score is required")
    minimum = min(scores.values())
    tied = [
        alpha
        for alpha, score in scores.items()
        if math.isclose(score, minimum, rel_tol=0.0, abs_tol=tie_tolerance)
    ]
    return float(max(tied))


def relative_improvement_percent(baseline: float, candidate: float) -> float:
    if baseline <= 0:
        raise ValueError("Baseline RMSE must be positive")
    return (baseline - candidate) / baseline * 100.0


def compute_gate(
    self_rmse: list[float] | np.ndarray,
    recent_rmse: list[float] | np.ndarray,
    lag_rmse: list[float] | np.ndarray,
    self_threshold_percent: float = 0.5,
    recent_threshold_percent: float = 0.2,
    required_positive_folds: int = 4,
) -> dict[str, Any]:
    """Compute the sole preregistered 168→6 L0 hard gate."""
    self_values = np.asarray(self_rmse, dtype=np.float64)
    recent_values = np.asarray(recent_rmse, dtype=np.float64)
    lag_values = np.asarray(lag_rmse, dtype=np.float64)
    if not (
        self_values.shape == recent_values.shape == lag_values.shape == (5,)
    ):
        raise ValueError("The L0 hard gate requires exactly five paired outer folds")
    self_mean = float(self_values.mean())
    recent_mean = float(recent_values.mean())
    lag_mean = float(lag_values.mean())
    vs_self = relative_improvement_percent(self_mean, lag_mean)
    vs_recent = relative_improvement_percent(recent_mean, lag_mean)
    self_positive = int(np.sum(lag_values < self_values))
    recent_positive = int(np.sum(lag_values < recent_values))
    checks = {
        "lag_vs_self_mean_improvement_at_least_0_5_percent": (
            vs_self >= self_threshold_percent
        ),
        "lag_vs_recent_mean_improvement_at_least_0_2_percent": (
            vs_recent >= recent_threshold_percent
        ),
        "lag_vs_self_positive_folds_at_least_4_of_5": (
            self_positive >= required_positive_folds
        ),
        "lag_vs_recent_positive_folds_at_least_4_of_5": (
            recent_positive >= required_positive_folds
        ),
    }
    passed = bool(all(checks.values()))
    return {
        "scope": SCOPE_LABEL,
        "primary_task": "168h_6h",
        "thresholds": {
            "lag_vs_self_mean_improvement_percent": self_threshold_percent,
            "lag_vs_recent_mean_improvement_percent": recent_threshold_percent,
            "required_positive_folds": required_positive_folds,
        },
        "five_fold_mean_rmse": {
            "R_self": self_mean,
            "R_recent": recent_mean,
            "R_lag": lag_mean,
        },
        "observed": {
            "lag_vs_self_mean_improvement_percent": vs_self,
            "lag_vs_recent_mean_improvement_percent": vs_recent,
            "lag_vs_self_positive_folds": self_positive,
            "lag_vs_recent_positive_folds": recent_positive,
        },
        "checks": checks,
        "passed": passed,
        "status": "PASS" if passed else "STOP",
        "next_step": (
            "PASS：仅允许进入冻结的 L1 神经实现；L0 本身不证明 MC-CVLRA 有效。"
            if passed
            else "STOP：归档该方向，不实现 MC-CVLRA 神经模块。"
        ),
    }


def _fit_range(window: WindowData, indices: np.ndarray) -> tuple[str, str, int]:
    times = window.origin_times[indices]
    return (
        pd.Timestamp(times.min()).isoformat(),
        pd.Timestamp(times.max()).isoformat(),
        len(indices),
    )


def _ledger_row(
    *,
    object_id: str,
    object_type: str,
    config: L0TaskConfig,
    window: WindowData,
    indices: np.ndarray,
    fold: int | str = "",
    model: str = "",
    horizon_step: int | str = "",
    object_config: dict[str, Any],
    state_hash: str,
) -> dict[str, Any]:
    fit_start, fit_end, row_count = _fit_range(window, indices)
    return {
        "object_id": object_id,
        "object_type": object_type,
        "task": config.task_name,
        "fold": fold,
        "model": model,
        "horizon_step": horizon_step,
        "fit_start": fit_start,
        "fit_end": fit_end,
        "row_count": row_count,
        "config_hash": stable_hash(object_config),
        "state_hash": state_hash,
        "details": json.dumps(object_config, ensure_ascii=False, sort_keys=True),
    }


def _append_metric_rows(
    *,
    config: L0TaskConfig,
    fold: int,
    model: str,
    alpha: float,
    train_samples: int,
    validation_samples: int,
    actual: np.ndarray,
    prediction: np.ndarray,
    horizon_rows: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
) -> None:
    for horizon_index in range(config.horizon):
        values = regression_metrics(
            actual[:, horizon_index], prediction[:, horizon_index]
        )
        horizon_rows.append(
            {
                "scope": SCOPE_LABEL,
                "task": config.task_name,
                "fold": fold,
                "model": model,
                "horizon_step": horizon_index + 1,
                "selected_alpha": alpha,
                "train_samples": train_samples,
                "validation_samples": validation_samples,
                **values,
            }
        )
    values = regression_metrics(actual.reshape(-1), prediction.reshape(-1))
    all_rows.append(
        {
            "scope": SCOPE_LABEL,
            "task": config.task_name,
            "fold": fold,
            "model": model,
            "selected_alpha": alpha,
            "train_samples": train_samples,
            "validation_samples": validation_samples,
            "target_values": int(actual.size),
            "rmse_all": values["rmse"],
            "mae_all": values["mae"],
            "smape_all_percent": values["smape_percent"],
        }
    )


def _window_manifest(window: WindowData) -> pd.DataFrame:
    rows = []
    for index, origin in enumerate(window.origin_rows):
        rows.append(
            {
                "scope": SCOPE_LABEL,
                "task": window.config.task_name,
                "sample_index": index,
                "origin_row": int(origin),
                "origin_time": pd.Timestamp(window.origin_times[index]).isoformat(),
                "target_start": pd.Timestamp(window.target_times[index, 0]).isoformat(),
                "target_end": pd.Timestamp(window.target_times[index, -1]).isoformat(),
                "block": f"Q{int(window.block_ids[index])}",
                "all_required_values_finite": True,
                "formal_test_sample": False,
            }
        )
    return pd.DataFrame(rows)


def run_task(csv_path: Path, config: L0TaskConfig) -> TaskArtifacts:
    selection: SelectionData = load_selection_data(csv_path, config)
    window = build_window_data(selection.frame, config)
    splits = build_l0_splits(window)
    names = feature_names()

    fit_ledger_rows: list[dict[str, Any]] = []
    alpha_rows: list[dict[str, Any]] = []
    alpha_horizon_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    fold_horizon_rows: list[dict[str, Any]] = []
    fold_all_rows: list[dict[str, Any]] = []

    for split in splits.folds:
        for model in MODEL_FEATURES:
            values = window.features[model]
            inner_train_x = values[split.inner_fit_indices]
            inner_validation_x = values[split.inner_validation_indices]
            inner_train_y = window.targets[split.inner_fit_indices]
            inner_validation_y = window.targets[split.inner_validation_indices]

            inner_scaler = fit_standardizer(inner_train_x)
            scaler_config = {
                "task_config_hash": config.config_hash,
                "fold": split.fold,
                "model": model,
                "stage": "inner",
                "feature_count": inner_train_x.shape[1],
                "feature_names": names[model],
                "ddof": 0,
            }
            fit_ledger_rows.append(
                _ledger_row(
                    object_id=f"inner_scaler:fold{split.fold}:{model}",
                    object_type="standardizer",
                    config=config,
                    window=window,
                    indices=split.inner_fit_indices,
                    fold=split.fold,
                    model=model,
                    object_config=scaler_config,
                    state_hash=inner_scaler.state_hash,
                )
            )
            inner_train_scaled = inner_scaler.transform(inner_train_x)
            inner_validation_scaled = inner_scaler.transform(inner_validation_x)

            scores: dict[float, float] = {}
            score_row_indices: dict[float, int] = {}
            for alpha in config.alphas:
                ridge = fit_direct_ridge(inner_train_scaled, inner_train_y, alpha)
                prediction = ridge.predict(inner_validation_scaled)
                metrics_all = regression_metrics(
                    inner_validation_y.reshape(-1), prediction.reshape(-1)
                )
                scores[alpha] = metrics_all["rmse"]
                score_row_indices[alpha] = len(alpha_rows)
                alpha_rows.append(
                    {
                        "scope": SCOPE_LABEL,
                        "task": config.task_name,
                        "fold": split.fold,
                        "model": model,
                        "alpha": alpha,
                        "inner_fit_samples": len(split.inner_fit_indices),
                        "inner_validation_samples": len(split.inner_validation_indices),
                        "rmse_all": metrics_all["rmse"],
                        "mae_all": metrics_all["mae"],
                        "smape_all_percent": metrics_all["smape_percent"],
                        "selected": False,
                    }
                )
                for horizon_index in range(config.horizon):
                    horizon_metrics = regression_metrics(
                        inner_validation_y[:, horizon_index],
                        prediction[:, horizon_index],
                    )
                    alpha_horizon_rows.append(
                        {
                            "scope": SCOPE_LABEL,
                            "task": config.task_name,
                            "fold": split.fold,
                            "model": model,
                            "alpha": alpha,
                            "horizon_step": horizon_index + 1,
                            **horizon_metrics,
                        }
                    )
                    ridge_config = {
                        "task_config_hash": config.config_hash,
                        "fold": split.fold,
                        "model": model,
                        "stage": "inner",
                        "alpha": alpha,
                        "horizon_step": horizon_index + 1,
                        "direct_target": True,
                    }
                    fit_ledger_rows.append(
                        _ledger_row(
                            object_id=(
                                f"inner_ridge:fold{split.fold}:{model}:"
                                f"alpha{alpha:g}:h{horizon_index + 1}"
                            ),
                            object_type="ridge",
                            config=config,
                            window=window,
                            indices=split.inner_fit_indices,
                            fold=split.fold,
                            model=model,
                            horizon_step=horizon_index + 1,
                            object_config=ridge_config,
                            state_hash=ridge.horizon_state_hash(horizon_index),
                        )
                    )

            selected_alpha = select_alpha(scores)
            alpha_rows[score_row_indices[selected_alpha]]["selected"] = True
            selection_indices = np.concatenate(
                [split.inner_fit_indices, split.inner_validation_indices]
            )
            selector_config = {
                "task_config_hash": config.config_hash,
                "fold": split.fold,
                "model": model,
                "alphas": list(config.alphas),
                "metric": "RMSE_all",
                "shared_across_horizons": True,
                "tie_rule": "absolute_difference_le_1e-12_then_larger_alpha",
                "inner_fit_rows": len(split.inner_fit_indices),
                "inner_validation_rows": len(split.inner_validation_indices),
            }
            fit_ledger_rows.append(
                _ledger_row(
                    object_id=f"alpha_selector:fold{split.fold}:{model}",
                    object_type="hyperparameter_selector",
                    config=config,
                    window=window,
                    indices=selection_indices,
                    fold=split.fold,
                    model=model,
                    object_config=selector_config,
                    state_hash=stable_hash(
                        {"scores": scores, "selected_alpha": selected_alpha}
                    ),
                )
            )

            outer_train_x = values[split.outer_train_indices]
            outer_validation_x = values[split.outer_validation_indices]
            outer_train_y = window.targets[split.outer_train_indices]
            outer_validation_y = window.targets[split.outer_validation_indices]
            outer_scaler = fit_standardizer(outer_train_x)
            outer_scaler_config = {
                "task_config_hash": config.config_hash,
                "fold": split.fold,
                "model": model,
                "stage": "outer_refit",
                "feature_count": outer_train_x.shape[1],
                "feature_names": names[model],
                "ddof": 0,
            }
            fit_ledger_rows.append(
                _ledger_row(
                    object_id=f"outer_scaler:fold{split.fold}:{model}",
                    object_type="standardizer",
                    config=config,
                    window=window,
                    indices=split.outer_train_indices,
                    fold=split.fold,
                    model=model,
                    object_config=outer_scaler_config,
                    state_hash=outer_scaler.state_hash,
                )
            )
            outer_train_scaled = outer_scaler.transform(outer_train_x)
            outer_validation_scaled = outer_scaler.transform(outer_validation_x)
            outer_ridge = fit_direct_ridge(
                outer_train_scaled, outer_train_y, selected_alpha
            )
            outer_prediction = outer_ridge.predict(outer_validation_scaled)
            for horizon_index in range(config.horizon):
                ridge_config = {
                    "task_config_hash": config.config_hash,
                    "fold": split.fold,
                    "model": model,
                    "stage": "outer_refit",
                    "alpha": selected_alpha,
                    "horizon_step": horizon_index + 1,
                    "direct_target": True,
                }
                fit_ledger_rows.append(
                    _ledger_row(
                        object_id=(
                            f"outer_ridge:fold{split.fold}:{model}:"
                            f"alpha{selected_alpha:g}:h{horizon_index + 1}"
                        ),
                        object_type="ridge",
                        config=config,
                        window=window,
                        indices=split.outer_train_indices,
                        fold=split.fold,
                        model=model,
                        horizon_step=horizon_index + 1,
                        object_config=ridge_config,
                        state_hash=outer_ridge.horizon_state_hash(horizon_index),
                    )
                )

            for local_index, sample_index in enumerate(
                split.outer_validation_indices
            ):
                for horizon_index in range(config.horizon):
                    prediction_rows.append(
                        {
                            "scope": SCOPE_LABEL,
                            "task": config.task_name,
                            "fold": split.fold,
                            "model": model,
                            "selected_alpha": selected_alpha,
                            "origin_time": pd.Timestamp(
                                window.origin_times[sample_index]
                            ).isoformat(),
                            "target_time": pd.Timestamp(
                                window.target_times[sample_index, horizon_index]
                            ).isoformat(),
                            "horizon_step": horizon_index + 1,
                            "actual": float(
                                outer_validation_y[local_index, horizon_index]
                            ),
                            "prediction": float(
                                outer_prediction[local_index, horizon_index]
                            ),
                            "formal_test_sample": False,
                        }
                    )
            _append_metric_rows(
                config=config,
                fold=split.fold,
                model=model,
                alpha=selected_alpha,
                train_samples=len(split.outer_train_indices),
                validation_samples=len(split.outer_validation_indices),
                actual=outer_validation_y,
                prediction=outer_prediction,
                horizon_rows=fold_horizon_rows,
                all_rows=fold_all_rows,
            )

    fold_all = pd.DataFrame(fold_all_rows)
    paired_rows = []
    for fold in range(1, 6):
        fold_values = fold_all[fold_all["fold"] == fold].set_index("model")
        lag_rmse = float(fold_values.loc["R_lag", "rmse_all"])
        for baseline in ("R_self", "R_recent"):
            baseline_rmse = float(fold_values.loc[baseline, "rmse_all"])
            absolute = baseline_rmse - lag_rmse
            paired_rows.append(
                {
                    "scope": SCOPE_LABEL,
                    "task": config.task_name,
                    "fold": fold,
                    "baseline": baseline,
                    "candidate": "R_lag",
                    "baseline_rmse_all": baseline_rmse,
                    "candidate_rmse_all": lag_rmse,
                    "absolute_rmse_improvement": absolute,
                    "relative_rmse_improvement_percent": (
                        relative_improvement_percent(baseline_rmse, lag_rmse)
                    ),
                    "candidate_better": bool(absolute > 0),
                }
            )
    paired = pd.DataFrame(paired_rows)

    summary_rows = []
    for model in MODEL_FEATURES:
        values = fold_all.loc[fold_all["model"] == model, "rmse_all"]
        summary_rows.append(
            {
                "scope": SCOPE_LABEL,
                "task": config.task_name,
                "row_type": "model",
                "name": model,
                "baseline": "",
                "candidate": "",
                "five_fold_mean_rmse_all": float(values.mean()),
                "five_fold_sample_std_rmse_all": float(values.std(ddof=1)),
                "mean_absolute_improvement": np.nan,
                "relative_improvement_of_mean_percent": np.nan,
                "mean_fold_relative_improvement_percent": np.nan,
                "positive_fold_count": np.nan,
            }
        )
    for baseline in ("R_self", "R_recent"):
        rows = paired[paired["baseline"] == baseline]
        baseline_mean = float(rows["baseline_rmse_all"].mean())
        candidate_mean = float(rows["candidate_rmse_all"].mean())
        summary_rows.append(
            {
                "scope": SCOPE_LABEL,
                "task": config.task_name,
                "row_type": "comparison",
                "name": f"R_lag_vs_{baseline}",
                "baseline": baseline,
                "candidate": "R_lag",
                "five_fold_mean_rmse_all": candidate_mean,
                "five_fold_sample_std_rmse_all": float(
                    rows["candidate_rmse_all"].std(ddof=1)
                ),
                "mean_absolute_improvement": baseline_mean - candidate_mean,
                "relative_improvement_of_mean_percent": (
                    relative_improvement_percent(baseline_mean, candidate_mean)
                ),
                "mean_fold_relative_improvement_percent": float(
                    rows["relative_rmse_improvement_percent"].mean()
                ),
                "positive_fold_count": int(rows["candidate_better"].sum()),
            }
        )

    metadata = {
        "scope": SCOPE_LABEL,
        "warning_cn": "本结果仅为train+validation选择区间L0，不是正式test或神经模型性能。",
        "task_config": asdict(config),
        "task_config_hash": config.config_hash,
        "official_boundaries": selection.boundaries.to_dict(),
        "access_metadata": selection.access_metadata,
        "candidate_origins_before_embargo": window.candidate_origin_count,
        "block_embargoed_origins": window.embargoed_origin_count,
        "residual_missing_windows_deleted": window.residual_missing_window_count,
        "retained_common_samples": len(window.origin_rows),
        "feature_shapes": {
            model: list(values.shape) for model, values in window.features.items()
        },
        "feature_names": names,
        "lags": list(LAGS),
        "alphas": list(ALPHAS),
        "test_feature_rows_loaded": 0,
        "test_samples_constructed": 0,
        "test_predictions_constructed": 0,
        "test_metrics_constructed": 0,
    }
    frames = {
        "processing_ledger": pd.DataFrame(selection.processing_ledger),
        "fit_ledger": pd.DataFrame(fit_ledger_rows),
        "data_quality": pd.DataFrame(selection.data_quality_rows),
        "window_manifest": _window_manifest(window),
        "boundary_checks": pd.DataFrame(splits.boundary_checks),
        "target_time_sets": pd.DataFrame(splits.target_time_rows),
        "embargoed_origins": pd.DataFrame(splits.embargo_rows),
        "inner_alpha_scores": pd.DataFrame(alpha_rows),
        "inner_alpha_horizon_metrics": pd.DataFrame(alpha_horizon_rows),
        "outer_predictions": pd.DataFrame(prediction_rows),
        "fold_horizon_metrics": pd.DataFrame(fold_horizon_rows),
        "fold_metrics_all": fold_all,
        "paired_differences": paired,
        "task_summary": pd.DataFrame(summary_rows),
    }
    return TaskArtifacts(config=config, metadata=metadata, frames=frames)


def _parse_tasks(value: str) -> list[L0TaskConfig]:
    tasks = []
    for item in value.split(","):
        history_text, horizon_text = item.strip().split(":", maxsplit=1)
        tasks.append(L0TaskConfig(history=int(history_text), horizon=int(horizon_text)))
    if [(task.history, task.horizon) for task in tasks] != [(168, 6), (24, 1)]:
        raise ValueError("The frozen L0 run must be exactly 168:6,24:1 in that order")
    return tasks


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _render_report(
    gate: dict[str, Any],
    task_summary: pd.DataFrame,
    metadata: dict[str, dict[str, Any]],
) -> str:
    lines = [
        "# 北京 1013 跨变量滞后 L0 选择区间报告",
        "",
        "> 本报告只汇总 train+validation 选择区间的 L0 forward-chaining ridge。"
        "它不是正式 test 结果，也不是 MC-CVLRA、ST-PatchTST 或其他神经模型性能。",
        "",
        "## 冻结协议与无泄漏边界",
        "",
        "- 主任务为 168→6，24→1 仅作支持性报告；变量、顺序、lag、alpha、折、阈值和清洗均在结果前冻结。",
        "- 全 CSV 只读取时间戳以建立 70/10/20 边界；特征和目标只加载前 80% train+validation 行。",
        "- test 特征行、窗口、预测、缓存和指标构造数均为 0。",
        "- 三模型使用相同完整窗口；全部 Q、outer 和 inner 目标时间集合空交集断言通过。",
        "",
        "## 五折总体 RMSE",
        "",
        "| 任务 | R_self | R_recent | R_lag | lag vs self | lag vs recent | 同向折数（self/recent） |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task in ("168h_6h", "24h_1h"):
        table = task_summary[task_summary["task"] == task]
        models = table[table["row_type"] == "model"].set_index("name")
        comparisons = table[table["row_type"] == "comparison"].set_index("baseline")
        lines.append(
            "| {task} | {self_rmse:.6f} | {recent_rmse:.6f} | {lag_rmse:.6f} | "
            "{self_gain:.6f}% | {recent_gain:.6f}% | {self_count}/5, {recent_count}/5 |".format(
                task=task.replace("h_", "→"),
                self_rmse=models.loc["R_self", "five_fold_mean_rmse_all"],
                recent_rmse=models.loc["R_recent", "five_fold_mean_rmse_all"],
                lag_rmse=models.loc["R_lag", "five_fold_mean_rmse_all"],
                self_gain=comparisons.loc[
                    "R_self", "relative_improvement_of_mean_percent"
                ],
                recent_gain=comparisons.loc[
                    "R_recent", "relative_improvement_of_mean_percent"
                ],
                self_count=int(comparisons.loc["R_self", "positive_fold_count"]),
                recent_count=int(
                    comparisons.loc["R_recent", "positive_fold_count"]
                ),
            )
        )
    lines.extend(
        [
            "",
            "## 唯一阶段门结论",
            "",
            f"**{gate['status']}**。{gate['next_step']}",
            "",
            "168→6 的四项硬门：",
            "",
        ]
    )
    for name, passed in gate["checks"].items():
        lines.append(f"- `{name}`：{'通过' if passed else '失败'}")
    lines.extend(
        [
            "",
            "## 样本与审计",
            "",
        ]
    )
    for task in ("168h_6h", "24h_1h"):
        task_metadata = metadata[task]
        lines.append(
            f"- {task}：共同保留样本 {task_metadata['retained_common_samples']}；"
            f"block embargo 删除 {task_metadata['block_embargoed_origins']}；"
            f"残余缺失窗口删除 {task_metadata['residual_missing_windows_deleted']}。"
        )
    lines.extend(
        [
            "",
            "## 解释边界与风险",
            "",
            "- 结果只说明固定七变量与固定离散 lag 在北京 1013 历史回放选择区间中的线性预测关联。",
            "- L0 PASS 也不证明神经适配器有效、变量具有因果作用、真实在线无延迟或跨城市泛化。",
            "- 仓库缺少精确数据血缘、真实报送/入库日志和原始缺失掩码；pressure 单位与零值缺失均为预签字工程假设。",
            "- 24→1 不参与阶段门，不能挽救 168→6 的 STOP。",
            "",
        ]
    )
    return "\n".join(lines)


def write_artifacts(
    artifacts: list[TaskArtifacts],
    output_dir: Path,
    command: list[str],
    data_path: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    combined: dict[str, list[pd.DataFrame]] = {}
    metadata = {}
    for artifact in artifacts:
        metadata[artifact.config.task_name] = artifact.metadata
        for name, frame in artifact.frames.items():
            combined.setdefault(name, []).append(frame)
    combined_frames = {
        name: pd.concat(frames, ignore_index=True) for name, frames in combined.items()
    }
    for name, frame in combined_frames.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)

    primary = combined_frames["fold_metrics_all"]
    primary = primary[primary["task"] == "168h_6h"]
    values = {
        model: primary[primary["model"] == model]
        .sort_values("fold")["rmse_all"]
        .to_numpy()
        for model in MODEL_FEATURES
    }
    gate = compute_gate(values["R_self"], values["R_recent"], values["R_lag"])
    gate["selection_interval_only"] = True
    gate["formal_test_used"] = False
    gate["neural_model_used"] = False
    gate["test_access_counts"] = {
        task: {
            key: metadata[task][key]
            for key in (
                "test_feature_rows_loaded",
                "test_samples_constructed",
                "test_predictions_constructed",
                "test_metrics_constructed",
            )
        }
        for task in metadata
    }
    gate["gate_config_hash"] = stable_hash(
        {"thresholds": gate["thresholds"], "primary_task": gate["primary_task"]}
    )

    run_config = {
        "scope": SCOPE_LABEL,
        "command": command,
        "data_path": str(data_path),
        "tasks": [asdict(artifact.config) for artifact in artifacts],
        "task_config_hashes": {
            artifact.config.task_name: artifact.config.config_hash
            for artifact in artifacts
        },
        "frozen_variable_order": list(AUXILIARY_COLUMNS),
        "frozen_lags": list(LAGS),
        "frozen_alphas": list(ALPHAS),
        "formal_test_used": False,
        "neural_model_used": False,
    }
    _write_json(output_dir / "config.json", run_config)
    _write_json(output_dir / "task_metadata.json", metadata)
    _write_json(output_dir / "gate_status.json", gate)
    report = _render_report(gate, combined_frames["task_summary"], metadata)
    (output_dir / "L0选择区间报告.md").write_text(report, encoding="utf-8")
    return gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--tasks", default="168:6,24:1")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = _parse_tasks(args.tasks)
    artifacts = []
    for config in tasks:
        print(f"Running {config.task_name} selection-interval L0...")
        artifacts.append(run_task(args.data_path, config))
    gate = write_artifacts(artifacts, args.output_dir, sys.argv, args.data_path)
    print(
        json.dumps(
            {
                "scope": SCOPE_LABEL,
                "status": gate["status"],
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
