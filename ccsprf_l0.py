"""Pure-synthetic CCSPRF L0 runner primitives.

The only learned backbone here is tsai's PatchTST. This module does not load
project station data, checkpoints, or historical experiment results.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tsai.models.PatchTST import PatchTST


FORMAL_SEEDS = tuple(range(20260939, 20260969))
FORMAL_BOOTSTRAP_SEED = 20260969
FORMAL_BOOTSTRAP_REPLICATES = 10_000
TAU_L0 = 50.0
ARMS = ("BASE", "EMB", "CCSPRF-global", "CCSPRF")
PRIOR_MODES = {"BASE": "none", "EMB": "embedding_bias", "CCSPRF-global": "global", "CCSPRF": "calendar"}
CONDITIONS = ("calendar_on", "calendar_off")
TASKS = ((24, 1), (168, 6))
SCHEMA_VERSION = "ccsprf-l0-v2"
FORMAL_NUM_POINTS = 8736
TRAINING_RNG_OFFSET = 1_000_003
HOUR_NS = 3_600_000_000_000
DAY_NS = 86_400_000_000_000

HOUR_EFFECT_RAW = np.asarray(
    [-1.10, -1.25, -1.30, -1.15, -0.80, -0.20, 0.55, 1.00, 1.25, 1.10, 0.80, 0.45,
     0.15, -0.05, -0.20, -0.05, 0.25, 0.70, 1.10, 0.95, 0.45, -0.10, -0.60, -0.90],
    dtype=np.float64,
)
WEEKDAY_EFFECT_RAW = np.asarray([0.95, 0.40, -0.25, -0.65, -0.45, 0.15, 0.85], dtype=np.float64)
HOUR_EFFECT = HOUR_EFFECT_RAW - HOUR_EFFECT_RAW.mean()
WEEKDAY_EFFECT = WEEKDAY_EFFECT_RAW - WEEKDAY_EFFECT_RAW.mean()


@dataclass(frozen=True)
class SyntheticConfig:
    history: int = 24
    horizon: int = 1
    seed: int = 20260939
    calendar_on: bool = True
    num_points: int = FORMAL_NUM_POINTS
    mu: float = 10.0
    rho: float = 0.60
    noise_std: float = 0.80


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 16
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 128
    dropout: float = 0.2
    attn_dropout: float = 0.0
    patch_len: int = 4
    stride: int = 2
    padding_patch: bool = True
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 40
    patience: int = 8
    batch_size: int = 256
    embedding_dim: int = 4


@dataclass(frozen=True)
class PreparedExperiment:
    condition: str
    task: str
    seed: int
    raw_values: np.ndarray
    timestamps_ns: np.ndarray
    normalized_values: np.ndarray
    center_mean: float
    center_std: float
    split_indices: dict[str, np.ndarray]
    train_target_indices: np.ndarray
    metadata: dict


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def training_rng_seed(seed: int) -> int:
    """Fair post-construction RNG state shared by all four arms."""

    return int(seed) + TRAINING_RNG_OFFSET


def calendar_key(timestamp_ns: int | np.integer) -> tuple[int, int]:
    value = int(timestamp_ns)
    return int((value // HOUR_NS) % 24), int((value // DAY_NS + 3) % 7)


def calendar_key_matrix(timestamps_ns: np.ndarray) -> np.ndarray:
    values = np.asarray(timestamps_ns, dtype=np.int64)
    flat = np.asarray([calendar_key(value) for value in values.reshape(-1)], dtype=np.int64)
    return flat.reshape(values.shape + (2,))


def _dgp_metadata(config: SyntheticConfig, hour_effect: np.ndarray, weekday_effect: np.ndarray) -> dict:
    return {
        "formula": "y_t=mu+a_hour+b_weekday+rho*(y_prev-mu-a_prev-b_prev)+epsilon",
        "mu": float(config.mu), "rho": float(config.rho), "noise_std": float(config.noise_std),
        "initial_dynamic_residual": 0.0, "rng_constructor": "numpy.random.default_rng(seed)",
        "rng_bit_generator": "PCG64", "rng_normal_calls": 1, "rng_normal_shape": int(config.num_points),
        "rng_calls_after_normal": 0, "hour_effect_raw": HOUR_EFFECT_RAW.tolist(),
        "weekday_effect_raw": WEEKDAY_EFFECT_RAW.tolist(), "hour_effect_centered": hour_effect.tolist(),
        "weekday_effect_centered": weekday_effect.tolist(), "calendar_on": bool(config.calendar_on),
        "timestamps_origin": "1970-01-01T00:00:00Z", "timestamp_dtype": "numpy.int64 nanoseconds",
        "timestamp_step_ns": HOUR_NS, "split_rule": "train_cut=int(num_points*0.70), valid_cut=int(num_points*0.80)",
    }


def generate_synthetic_series(config: SyntheticConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    if config.num_points <= config.history + config.horizon + 2:
        raise ValueError("num_points 太小，无法构造窗口")
    if abs(config.rho) >= 1:
        raise ValueError("rho 必须处于平稳范围 (-1, 1)")
    timestamps = np.arange(config.num_points, dtype=np.int64) * HOUR_NS
    hour_effect = HOUR_EFFECT.copy() if config.calendar_on else np.zeros(24, dtype=np.float64)
    weekday_effect = WEEKDAY_EFFECT.copy() if config.calendar_on else np.zeros(7, dtype=np.float64)
    rng = np.random.default_rng(config.seed)
    epsilon = rng.normal(0.0, config.noise_std, size=config.num_points)
    values = np.empty(config.num_points, dtype=np.float64)
    previous_residual = 0.0
    for index, timestamp in enumerate(timestamps):
        hour, weekday = calendar_key(timestamp)
        conditional_mean = config.mu + hour_effect[hour] + weekday_effect[weekday]
        values[index] = conditional_mean + config.rho * previous_residual + epsilon[index]
        previous_residual = values[index] - conditional_mean
    return values, timestamps, _dgp_metadata(config, hour_effect, weekday_effect)


def _window_indices(num_points: int, history: int, horizon: int) -> dict[str, np.ndarray]:
    train_cut, valid_cut = int(num_points * 0.70), int(num_points * 0.80)
    last_start = num_points - history - horizon
    result = {
        "train": np.arange(0, train_cut - history - horizon + 1, dtype=np.int64),
        "valid": np.arange(train_cut - history, valid_cut - history - horizon + 1, dtype=np.int64),
        "test": np.arange(valid_cut - history, last_start + 1, dtype=np.int64),
    }
    if any(len(value) == 0 for value in result.values()):
        raise ValueError("某个 split 没有窗口")
    return result


def _unique_train_target_indices(split_indices: dict[str, np.ndarray], history: int, horizon: int) -> np.ndarray:
    targets = {int(start + history + offset) for start in split_indices["train"] for offset in range(horizon)}
    return np.asarray(sorted(targets), dtype=np.int64)


def prepare_experiment(config: SyntheticConfig) -> PreparedExperiment:
    raw, timestamps, dgp = generate_synthetic_series(config)
    splits = _window_indices(config.num_points, config.history, config.horizon)
    train_target_indices = _unique_train_target_indices(splits, config.history, config.horizon)
    center_mean, center_std = float(raw[train_target_indices].mean()), float(raw[train_target_indices].std())
    if center_std < 1e-6:
        center_std = 1.0
    normalized = ((raw - center_mean) / center_std).astype(np.float32)
    task, condition = f"{config.history}h_{config.horizon}h", "calendar_on" if config.calendar_on else "calendar_off"
    metadata = {
        "schema_version": SCHEMA_VERSION, "condition": condition, "task": task, "seed": int(config.seed),
        "history": int(config.history), "horizon": int(config.horizon), "num_points": int(config.num_points),
        "train_fraction": 0.70, "valid_fraction": 0.10, "test_fraction": 0.20,
        "train_cut": int(config.num_points * 0.70), "valid_cut": int(config.num_points * 0.80),
        "center_mean": center_mean, "center_std": center_std, "train_target_count": int(len(train_target_indices)),
        "split_sizes": {key: int(len(value)) for key, value in splits.items()},
        "split_start_indices": {key: value.tolist() for key, value in splits.items()},
        "train_target_indices": train_target_indices.tolist(), "dgp": dgp,
    }
    return PreparedExperiment(condition, task, config.seed, raw, timestamps, normalized,
                              center_mean, center_std, splits, train_target_indices, metadata)


@dataclass(frozen=True)
class CalendarPriorTable:
    tau: float
    mu0: float
    total_count: int
    groups: dict[tuple[int, int], tuple[int, float]]

    def full_prior(self, key: tuple[int, int]) -> float:
        count, total = self.groups.get(key, (0, 0.0))
        if count == 0:
            return float(self.mu0)
        weight = count / (count + self.tau)
        return float(weight * total / count + (1.0 - weight) * self.mu0)

    def loo_prior(self, timestamp_ns: int, value: float) -> float:
        if self.total_count <= 1:
            raise RuntimeError("ENGINEERING_FAILURE: N=1 禁止含自身 fallback")
        mu0_loo = (self.total_count * self.mu0 - float(value)) / (self.total_count - 1)
        count, total = self.groups.get(calendar_key(timestamp_ns), (0, 0.0))
        remaining = count - 1
        if remaining <= 0:
            return float(mu0_loo)
        remaining_mean = (total - float(value)) / remaining
        weight = remaining / (remaining + self.tau)
        return float(weight * remaining_mean + (1.0 - weight) * mu0_loo)


@dataclass(frozen=True)
class GlobalPriorTable:
    mu0: float
    total_count: int

    def full_prior(self, key: tuple[int, int] | None = None) -> float:
        return float(self.mu0)

    def loo_prior(self, timestamp_ns: int, value: float) -> float:
        if self.total_count <= 1:
            raise RuntimeError("ENGINEERING_FAILURE: N=1 禁止含自身 fallback")
        return float((self.total_count * self.mu0 - float(value)) / (self.total_count - 1))


def fit_calendar_prior(unique_timestamps_ns: Iterable[int], unique_values_scaled: Iterable[float], tau: float = TAU_L0) -> CalendarPriorTable:
    timestamps = np.asarray(list(unique_timestamps_ns), dtype=np.int64)
    values = np.asarray(list(unique_values_scaled), dtype=np.float64)
    if timestamps.ndim != 1 or values.ndim != 1 or len(timestamps) != len(values):
        raise ValueError("先验输入必须是一维且长度一致")
    if len(timestamps) == 0:
        raise ValueError("先验要求非空且每个物理目标时间戳唯一")
    if len(np.unique(timestamps)) != len(timestamps):
        raise ValueError("先验禁止重复物理目标时间戳")
    if tau <= 0:
        raise ValueError("tau 必须大于0")
    groups: dict[tuple[int, int], tuple[int, float]] = {}
    for timestamp, value in zip(timestamps, values):
        key = calendar_key(timestamp)
        count, total = groups.get(key, (0, 0.0))
        groups[key] = count + 1, total + float(value)
    return CalendarPriorTable(float(tau), float(values.mean()), len(values), groups)


def fit_global_prior(unique_timestamps_ns: Iterable[int], unique_values_scaled: Iterable[float], tau: float = TAU_L0) -> GlobalPriorTable:
    del tau
    timestamps = np.asarray(list(unique_timestamps_ns), dtype=np.int64)
    values = np.asarray(list(unique_values_scaled), dtype=np.float64)
    if len(values) == 0 or len(np.unique(timestamps)) != len(timestamps):
        raise ValueError("全局先验要求非空且每个物理目标时间戳唯一")
    return GlobalPriorTable(float(values.mean()), len(values))


def target_time_matrix(prepared: PreparedExperiment, split: str) -> np.ndarray:
    history, horizon = prepared.metadata["history"], prepared.metadata["horizon"]
    return np.asarray([prepared.timestamps_ns[start + history : start + history + horizon] for start in prepared.split_indices[split]], dtype=np.int64)


def values_matrix(prepared: PreparedExperiment, split: str) -> np.ndarray:
    history, horizon = prepared.metadata["history"], prepared.metadata["horizon"]
    return np.asarray([prepared.normalized_values[start + history : start + history + horizon] for start in prepared.split_indices[split]], dtype=np.float32)


def prior_values_for_targets(table, target_times_ns: np.ndarray, values_scaled: np.ndarray | None = None, loo: bool = False) -> np.ndarray:
    times = np.asarray(target_times_ns, dtype=np.int64)
    if loo and values_scaled is None:
        raise ValueError("LOO 需要目标值")
    values = None if values_scaled is None else np.asarray(values_scaled)
    result = np.empty(times.shape, dtype=np.float32)
    for index in np.ndindex(times.shape):
        result[index] = table.loo_prior(int(times[index]), float(values[index])) if loo else table.full_prior(calendar_key(int(times[index])))
    return result


def build_prior_arrays(prepared: PreparedExperiment, tau: float = TAU_L0) -> dict:
    train_times = prepared.timestamps_ns[prepared.train_target_indices]
    train_values = prepared.normalized_values[prepared.train_target_indices]
    calendar_table, global_table = fit_calendar_prior(train_times, train_values, tau), fit_global_prior(train_times, train_values, tau)
    arrays = {}
    for split in ("train", "valid", "test"):
        times, values = target_time_matrix(prepared, split), values_matrix(prepared, split)
        arrays[split] = {"times_ns": times, "values_scaled": values,
                         "calendar": prior_values_for_targets(calendar_table, times, values, split == "train"),
                         "global": prior_values_for_targets(global_table, times, values, split == "train")}
    return {"calendar_table": calendar_table, "global_table": global_table, "arrays": arrays}


class WindowDataset(Dataset):
    def __init__(self, prepared: PreparedExperiment, split: str, prior: np.ndarray):
        self.x = np.asarray([prepared.normalized_values[start : start + prepared.metadata["history"]] for start in prepared.split_indices[split]], dtype=np.float32)
        self.y, self.times, self.prior = values_matrix(prepared, split), target_time_matrix(prepared, split), np.asarray(prior, dtype=np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return torch.from_numpy(self.x[index]), torch.from_numpy(self.y[index]), torch.from_numpy(self.times[index]), torch.from_numpy(self.prior[index])


class PatchTSTWrapper(nn.Module):
    """Direct tsai PatchTST wrapper: [B,L] -> [B,1,L] -> [B,H]."""

    def __init__(self, seq_len: int, horizon: int, config: ModelConfig):
        super().__init__()
        self.model = PatchTST(c_in=1, c_out=1, seq_len=seq_len, pred_dim=horizon,
                              n_layers=config.n_layers, n_heads=config.n_heads, d_model=config.d_model,
                              d_ff=config.d_ff, dropout=config.dropout, attn_dropout=config.attn_dropout,
                              patch_len=config.patch_len, stride=config.stride, padding_patch=config.padding_patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"PatchTST 输入必须为 [B,L]，收到 {tuple(x.shape)}")
        output = self.model(x.unsqueeze(1))
        if output.ndim != 3 or output.shape[1] != 1:
            raise RuntimeError(f"ENGINEERING_FAILURE: tsai PatchTST 输出形状异常 {tuple(output.shape)}")
        return output[:, 0, :]


class ForecastArm(nn.Module):
    def __init__(self, seq_len: int, horizon: int, config: ModelConfig, arm: str):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"未知实验臂: {arm}")
        self.arm = arm
        self.backbone = PatchTSTWrapper(seq_len, horizon, config)
        if arm == "EMB":
            self.hour_embedding, self.weekday_embedding = nn.Embedding(24, config.embedding_dim), nn.Embedding(7, config.embedding_dim)

    def calendar_bias(self, target_times_ns: torch.Tensor) -> torch.Tensor:
        if self.arm != "EMB":
            return torch.zeros(target_times_ns.shape, dtype=torch.float32, device=target_times_ns.device)
        keys = calendar_key_matrix(target_times_ns.detach().cpu().numpy())
        hours = torch.as_tensor(keys[..., 0], dtype=torch.long, device=target_times_ns.device)
        weekdays = torch.as_tensor(keys[..., 1], dtype=torch.long, device=target_times_ns.device)
        return torch.cat((self.hour_embedding(hours), self.weekday_embedding(weekdays)), dim=-1).mean(dim=-1)

    def forward(self, x: torch.Tensor, target_times_ns: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.backbone(x)
        bias = self.calendar_bias(target_times_ns)
        return output + bias, bias


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def state_dict_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(value.dtype).encode()); digest.update(json.dumps(list(value.shape)).encode()); digest.update(value.tobytes())
    return digest.hexdigest()


def expected_backbone_hashes(seed: int, history: int, horizon: int, config: ModelConfig) -> dict[str, str]:
    result = {}
    for arm in ARMS:
        set_seed(seed)
        result[arm] = state_dict_hash(ForecastArm(history, horizon, config, arm).backbone)
    return result


def make_loader(dataset: WindowDataset, config: ModelConfig, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(dataset, batch_size=min(config.batch_size, len(dataset)), shuffle=shuffle, generator=torch.Generator().manual_seed(seed), num_workers=0)


def _predict_batches(model: ForecastArm, loader: DataLoader, device: torch.device, arm: str):
    model.eval(); predictions, targets, priors, residuals, times = [], [], [], [], []
    with torch.no_grad():
        for x, y, target_times, prior in loader:
            x, target_times = x.to(device), target_times.to(device)
            output, embedding_bias = model(x, target_times)
            if arm == "EMB":
                used_prior, residual = embedding_bias, output - embedding_bias
            elif arm in ("CCSPRF-global", "CCSPRF"):
                used_prior, residual = prior.to(device), output
            else:
                used_prior, residual = torch.zeros_like(output), output
            predictions.append((used_prior + residual).cpu().numpy()); targets.append(y.numpy()); priors.append(used_prior.cpu().numpy()); residuals.append(residual.cpu().numpy()); times.append(target_times.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(targets), np.concatenate(priors), np.concatenate(residuals), np.concatenate(times)


def regression_metrics(target_scaled: np.ndarray, prediction_scaled: np.ndarray, center_mean: float, center_std: float) -> dict:
    error = np.asarray(prediction_scaled) - np.asarray(target_scaled); physical_error = error * center_std
    target_raw = np.asarray(target_scaled) * center_std + center_mean; prediction_raw = np.asarray(prediction_scaled) * center_std + center_mean
    denominator = np.abs(target_raw) + np.abs(prediction_raw)
    return {"mse_scaled": float(np.mean(error ** 2)), "rmse_scaled": float(np.sqrt(np.mean(error ** 2))), "mae_scaled": float(np.mean(np.abs(error))),
            "rmse_raw": float(np.sqrt(np.mean(physical_error ** 2))), "mae_raw": float(np.mean(np.abs(physical_error))),
            "smape_percent": float(200 * np.mean(np.abs(physical_error) / np.maximum(denominator, 1e-6))),
            "per_step_rmse_raw": np.sqrt(np.mean(physical_error ** 2, axis=0)).astype(float).tolist()}


def paired_bootstrap_median(values: Iterable[float], seed: int = FORMAL_BOOTSTRAP_SEED, replicates: int = FORMAL_BOOTSTRAP_REPLICATES) -> tuple[float, float, float]:
    values = np.asarray(list(values), dtype=np.float64)
    if values.size == 0:
        raise ValueError("bootstrap 至少需要一个种子")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(replicates, values.size))
    medians = np.median(values[draws], axis=1)
    return float(np.median(values)), float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def atomic_json(path: Path, value: dict) -> None:
    _atomic_bytes(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode())


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def artifact_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_prediction_artifact(path: Path, arrays: dict[str, np.ndarray], metadata: dict) -> dict:
    if path.exists() or path.with_suffix(".json").exists():
        raise RuntimeError(f"ENGINEERING_FAILURE: 拒绝覆盖已有 artifact: {path}")
    atomic_npz(path, arrays)
    manifest = {"schema_version": SCHEMA_VERSION, "sha256": artifact_sha256(path), "arrays": {key: list(value.shape) for key, value in arrays.items()}, **metadata}
    atomic_json(path.with_suffix(".json"), manifest)
    return manifest


def verify_prediction_artifact(path: str | Path) -> dict:
    path = Path(path); manifest_path = path.with_suffix(".json")
    if not path.is_file() or not manifest_path.is_file(): raise RuntimeError(f"ENGINEERING_FAILURE: artifact/manifest 缺失: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if artifact_sha256(path) != manifest.get("sha256"): raise RuntimeError(f"ENGINEERING_FAILURE: artifact hash 不匹配: {path}")
    arm = manifest.get("arm")
    if manifest.get("schema_version") != SCHEMA_VERSION or arm not in ARMS: raise RuntimeError(f"ENGINEERING_FAILURE: artifact sidecar schema/arm 无效: {path}")
    if manifest.get("prior_mode") != PRIOR_MODES[arm] or manifest.get("tau_applied") != (arm == "CCSPRF") or manifest.get("tau") != (TAU_L0 if arm == "CCSPRF" else None): raise RuntimeError(f"ENGINEERING_FAILURE: artifact tau/prior 语义不一致: {path}")
    if int(manifest.get("training_rng_seed", -1)) != int(manifest.get("seed", -2)) + TRAINING_RNG_OFFSET: raise RuntimeError(f"ENGINEERING_FAILURE: artifact training_rng_seed 不一致: {path}")
    with np.load(path, allow_pickle=False) as loaded:
        base_required = {"prediction_scaled", "target_scaled", "prior_scaled", "residual_scaled", "prior_only_scaled", "target_times_ns", "prediction_raw", "target_raw", "prior_only_raw"}
        if manifest.get("arm") == "EMB":
            required = base_required | {"emb_hour_weight", "emb_weekday_weight"}
        elif manifest.get("arm") in ARMS:
            required = base_required
        else:
            raise RuntimeError(f"ENGINEERING_FAILURE: artifact arm 无效: {path}")
        if set(loaded.files) != required: raise RuntimeError(f"ENGINEERING_FAILURE: artifact 字段不精确: {path}")
        arrays = {key: loaded[key] for key in loaded.files}
    declared_shapes = manifest.get("arrays")
    actual_shapes = {key: list(value.shape) for key, value in arrays.items()}
    if declared_shapes != actual_shapes: raise RuntimeError(f"ENGINEERING_FAILURE: artifact arrays shape 声明不一致: {path}")
    if int(manifest.get("test_count", -1)) != int(arrays["prediction_scaled"].shape[0]): raise RuntimeError(f"ENGINEERING_FAILURE: artifact test_count 不一致: {path}")
    increment = 124 if arm == "EMB" else 0
    try:
        if int(manifest["parameter_count"]) - int(manifest["backbone_parameter_count"]) != increment or int(manifest["parameter_increment"]) != increment:
            raise ValueError
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"ENGINEERING_FAILURE: artifact parameter metadata 不一致: {path}") from error
    if not np.isfinite(np.concatenate([arrays[key].ravel() for key in ("prediction_scaled", "target_scaled", "prior_scaled", "residual_scaled")])).all(): raise RuntimeError(f"ENGINEERING_FAILURE: artifact 含非有限值: {path}")
    if not np.allclose(arrays["prediction_scaled"], arrays["prior_scaled"] + arrays["residual_scaled"], rtol=0, atol=1e-6): raise RuntimeError(f"ENGINEERING_FAILURE: prediction != prior + residual: {path}")
    mean, std = float(manifest["center_mean"]), float(manifest["center_std"])
    for scaled, raw, label in (("prediction_scaled", "prediction_raw", "prediction"), ("target_scaled", "target_raw", "target"), ("prior_only_scaled", "prior_only_raw", "prior-only")):
        if not np.allclose(arrays[raw], arrays[scaled] * std + mean, rtol=0, atol=1e-5): raise RuntimeError(f"ENGINEERING_FAILURE: {label} 反标准化失败: {path}")
    return {"manifest": manifest, "arrays": arrays}


def train_one_arm(prepared: PreparedExperiment, priors: dict, arm: str, seed: int, config: ModelConfig, device: torch.device | str = "cpu", output_dir: Path | None = None) -> dict:
    set_seed(seed); device = torch.device(device)
    model = ForecastArm(prepared.metadata["history"], prepared.metadata["horizon"], config, arm).to(device)
    initial_hash = state_dict_hash(model.backbone)
    # EMB construction consumes RNG for its 124 parameters. Reset only after
    # construction so loader shuffling and all dropout streams are identical.
    train_seed = training_rng_seed(seed)
    set_seed(train_seed)
    def prior_for(split):
        return np.zeros_like(priors["arrays"][split]["values_scaled"]) if arm in ("BASE", "EMB") else priors["arrays"][split]["global" if arm == "CCSPRF-global" else "calendar"]
    train_loader = make_loader(WindowDataset(prepared, "train", prior_for("train")), config, True, train_seed)
    valid_loader = make_loader(WindowDataset(prepared, "valid", prior_for("valid")), config, False, train_seed)
    test_loader = make_loader(WindowDataset(prepared, "test", prior_for("test")), config, False, train_seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3); loss_fn = nn.MSELoss()
    best_loss, best_epoch, no_improvement, best_state = math.inf, 0, 0, None; history_rows = []; started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train(); total, count = 0.0, 0
        for x, y, target_times, prior in train_loader:
            x, y, target_times, prior = x.to(device), y.to(device), target_times.to(device), prior.to(device); optimizer.zero_grad(set_to_none=True)
            output, _ = model(x, target_times); loss = loss_fn(output, y - prior) if arm in ("CCSPRF-global", "CCSPRF") else loss_fn(output, y)
            if not torch.isfinite(loss): raise FloatingPointError(f"{arm} seed={seed} 非有限损失")
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); total += float(loss.item()) * len(x); count += len(x)
        valid_prediction, valid_target, _, _, _ = _predict_batches(model, valid_loader, device, arm); valid_loss = float(np.mean((valid_prediction - valid_target) ** 2)); scheduler.step(valid_loss)
        history_rows.append({"epoch": epoch, "train_loss": total / max(count, 1), "valid_loss": valid_loss, "learning_rate": optimizer.param_groups[0]["lr"]})
        if valid_loss < best_loss - 1e-7: best_loss, best_epoch, no_improvement, best_state = valid_loss, epoch, 0, copy.deepcopy(model.state_dict())
        else: no_improvement += 1
        if no_improvement >= config.patience: break
    if best_state is None: raise RuntimeError(f"ENGINEERING_FAILURE: {arm} 没有最佳状态")
    training_seconds = time.perf_counter() - started; model.load_state_dict(best_state)
    prediction, target, prior, residual, times = _predict_batches(model, test_loader, device, arm); metrics = regression_metrics(target, prediction, prepared.center_mean, prepared.center_std); prior_only_metrics = regression_metrics(target, prior, prepared.center_mean, prepared.center_std)
    backbone_parameters = parameter_count(model.backbone)
    tau_applied = arm == "CCSPRF"
    row = {"schema_version": SCHEMA_VERSION, "condition": prepared.condition, "task": prepared.task, "seed": int(seed), "arm": arm, "training_rng_seed": train_seed, "best_epoch": int(best_epoch), "best_valid_loss": float(best_loss), "training_seconds": float(training_seconds), "parameter_count": parameter_count(model), "backbone_parameter_count": backbone_parameters, "parameter_increment": parameter_count(model) - backbone_parameters, "initial_backbone_hash": initial_hash, "prior_mode": PRIOR_MODES[arm], "tau": TAU_L0 if tau_applied else None, "tau_applied": tau_applied, "prior_only_rmse_raw": prior_only_metrics["rmse_raw"], "prior_only_mae_raw": prior_only_metrics["mae_raw"], "prior_only_per_step_rmse_raw": prior_only_metrics["per_step_rmse_raw"], **metrics}
    if output_dir is not None:
        artifact_path = output_dir / "predictions" / prepared.condition / prepared.task / f"seed{seed}" / f"{arm}.npz"
        artifact_relative = str(artifact_path.relative_to(output_dir))
        metadata = {key: value for key, value in row.items() if not isinstance(value, list)}; metadata.update({"center_mean": prepared.center_mean, "center_std": prepared.center_std, "test_count": int(len(target)), "prediction_artifact": artifact_relative, "per_step_rmse_raw": metrics["per_step_rmse_raw"], "prior_only_per_step_rmse_raw": prior_only_metrics["per_step_rmse_raw"]})
        artifact_arrays = {"prediction_scaled": prediction.astype(np.float32), "target_scaled": target.astype(np.float32), "prior_scaled": prior.astype(np.float32), "residual_scaled": residual.astype(np.float32), "prior_only_scaled": prior.astype(np.float32), "target_times_ns": times.astype(np.int64), "prediction_raw": (prediction * prepared.center_std + prepared.center_mean).astype(np.float32), "target_raw": (target * prepared.center_std + prepared.center_mean).astype(np.float32), "prior_only_raw": (prior * prepared.center_std + prepared.center_mean).astype(np.float32)}
        if arm == "EMB":
            artifact_arrays.update({"emb_hour_weight": model.hour_embedding.weight.detach().cpu().numpy().astype(np.float32), "emb_weekday_weight": model.weekday_embedding.weight.detach().cpu().numpy().astype(np.float32)})
        write_prediction_artifact(artifact_path, artifact_arrays, metadata)
        row["prediction_artifact"] = artifact_relative; log_path = output_dir / "training_logs" / prepared.condition / prepared.task / f"seed{seed}_{arm}.csv"; log_path.parent.mkdir(parents=True, exist_ok=True); atomic_csv(log_path, history_rows)
    return row


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]); fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".csv.tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def expected_keys(seeds: Iterable[int], tasks: Iterable[tuple[int, int]], conditions: Iterable[str], arms: Iterable[str]) -> list[str]:
    return [f"{condition}|{history}h_{horizon}h|{seed}|{arm}" for seed in sorted(seeds) for history, horizon in tasks for condition in conditions for arm in arms]


def repository_root() -> Path:
    return Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=Path(__file__).resolve().parent, check=True, capture_output=True, text=True).stdout.strip())


def source_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository_root(), check=True, capture_output=True, text=True).stdout.strip()


def canonical_manifest(seeds, tasks, conditions, arms, config: ModelConfig, num_points: int, mode: str) -> dict:
    keys = expected_keys(seeds, tasks, conditions, arms); canonical = {"schema_version": SCHEMA_VERSION, "mode": mode, "tau": TAU_L0, "training_rng_seed_rule": "seed+1000003", "seeds": list(seeds), "tasks": [f"{h}h_{p}h" for h, p in tasks], "conditions": list(conditions), "arms": list(arms), "model_config": asdict(config), "dgp_config": {"num_points": num_points, "mu": 10.0, "rho": 0.60, "noise_std": 0.80, "timestamps_origin": "1970-01-01T00:00:00Z", "timestamp_step_ns": HOUR_NS, "split_rule": "int(num_points*0.70), int(num_points*0.80)", "rng": "default_rng(seed)/PCG64/one normal call"}, "frozen_hour_effect_raw": HOUR_EFFECT_RAW.tolist(), "frozen_weekday_effect_raw": WEEKDAY_EFFECT_RAW.tolist()}
    source = source_commit()
    payload = {"source_commit": source, "canonical": canonical, "expected_keys": keys}
    return {"schema_version": SCHEMA_VERSION, "source_commit": source, "canonical": canonical, "expected_keys": keys, "expected_keys_sha256": hashlib.sha256(json.dumps(keys, separators=(",", ":")).encode()).hexdigest(), "canonical_payload_sha256": hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def check_clean_tree() -> None:
    if subprocess.run(["git", "status", "--porcelain"], cwd=repository_root(), check=True, capture_output=True, text=True).stdout.strip(): raise RuntimeError("ENGINEERING_FAILURE: formal 初跑要求 clean tree")


def check_source_tree(output_dir: Path) -> None:
    """For resume, permit only dirty paths contained by the result directory."""

    repo = repository_root()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=repo, check=True, capture_output=True, text=True).stdout
    violations = source_tree_violations(repo, output_dir, status)
    if violations:
        raise RuntimeError(f"ENGINEERING_FAILURE: resume 检测到 output 之外源码改动: {violations[0]}")


def source_tree_violations(repo: Path, output_dir: Path, porcelain_status: str) -> list[str]:
    """Return status records outside output_dir; kept pure for resume testing."""

    repo, allowed = Path(repo).resolve(), Path(output_dir).resolve()
    violations = []
    for line in porcelain_status.splitlines():
        if len(line) < 4:
            continue
        record = line[3:]
        # Git quotes unusual paths in porcelain output.  The experiment paths
        # are ASCII, but rejecting an unparseable quoted record is safer than
        # accidentally treating a source change as output-local.
        if record.startswith('"') or " -> " in record:
            violations.append(line)
            continue
        candidate = (repo / record).resolve()
        if candidate != allowed and allowed not in candidate.parents:
            violations.append(line)
    return violations


def ensure_new_output_dir(path: Path) -> None:
    if path.exists(): raise FileExistsError(f"ENGINEERING_FAILURE: 拒绝覆盖已有输出目录: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _key_parts(key: str) -> tuple[str, str, int, str]:
    condition, task, seed, arm = key.split("|"); return condition, task, int(seed), arm


def _artifact_for_key(output_dir: Path, key: str) -> Path:
    condition, task, seed, arm = _key_parts(key); return output_dir / "predictions" / condition / task / f"seed{seed}" / f"{arm}.npz"


def _write_dataset(output_dir: Path, prepared: PreparedExperiment) -> None:
    base = output_dir / "datasets" / prepared.condition / prepared.task; path = base / f"seed{prepared.seed}.npz"
    metadata_path = path.with_suffix(".json")
    if path.exists() != metadata_path.exists():
        raise RuntimeError(f"ENGINEERING_FAILURE: dataset 原子文件不完整: {path}")
    if path.exists():
        with np.load(path, allow_pickle=False) as loaded:
            if not np.array_equal(loaded["raw_values"], prepared.raw_values) or not np.array_equal(loaded["normalized_values"], prepared.normalized_values) or not np.array_equal(loaded["timestamps_ns"], prepared.timestamps_ns):
                raise RuntimeError(f"ENGINEERING_FAILURE: dataset 内容不一致，拒绝覆盖: {path}")
        if _load_json(metadata_path) != prepared.metadata:
            raise RuntimeError(f"ENGINEERING_FAILURE: dataset metadata 不一致，拒绝覆盖: {metadata_path}")
        return
    atomic_npz(path, {"raw_values": prepared.raw_values, "normalized_values": prepared.normalized_values, "timestamps_ns": prepared.timestamps_ns}); atomic_json(base / f"seed{prepared.seed}.json", prepared.metadata)


def _prior_audit_payload(prepared: PreparedExperiment, priors: dict, hashes: dict[str, str]) -> dict:
    groups = []; table = priors["calendar_table"]
    for hour in range(24):
        for weekday in range(7):
            count, total = table.groups.get((hour, weekday), (0, 0.0)); groups.append({"hour": hour, "weekday": weekday, "n_unique_timestamps": int(count), "sum_scaled": float(total), "full_prior_scaled": float(table.full_prior((hour, weekday)))})
    return {"schema_version": SCHEMA_VERSION, "condition": prepared.condition, "task": prepared.task, "seed": prepared.seed, "calendar_prior_tau": TAU_L0, "tau_applied": {arm: arm == "CCSPRF" for arm in ARMS}, "center_mean": prepared.center_mean, "center_std": prepared.center_std, "calendar_groups": groups, "global_mu0_scaled": priors["global_table"].mu0, "loo_checksums": {"CCSPRF": hashlib.sha256(np.asarray(priors["arrays"]["train"]["calendar"], dtype=np.float32).tobytes()).hexdigest(), "CCSPRF-global": hashlib.sha256(np.asarray(priors["arrays"]["train"]["global"], dtype=np.float32).tobytes()).hexdigest()}, "initial_backbone_hashes": hashes, "arms": {arm: {"prior_mode": PRIOR_MODES[arm], "tau": TAU_L0 if arm == "CCSPRF" else None, "tau_applied": arm == "CCSPRF", "parameter_increment": 124 if arm == "EMB" else 0} for arm in ARMS}}


def _save_prior_audit(output_dir: Path, prepared: PreparedExperiment, priors: dict, hashes: dict[str, str]) -> None:
    path = output_dir / "prior_tables" / prepared.condition / prepared.task / f"seed{prepared.seed}.json"
    value = _prior_audit_payload(prepared, priors, hashes)
    if path.exists():
        if _load_json(path) != value:
            raise RuntimeError(f"ENGINEERING_FAILURE: prior audit 不一致，拒绝覆盖: {path}")
        return
    atomic_json(path, value)


def _read_rows(path: Path) -> list[dict]:
    if not path.exists(): return []
    with path.open(newline="", encoding="utf-8") as handle: return list(csv.DictReader(handle))


def _write_metric_rows(output_dir: Path, rows: list[dict]) -> None:
    serializable = []
    for row in rows:
        value = dict(row)
        for key, item in list(value.items()):
            if isinstance(item, list): value[key] = json.dumps(item, separators=(",", ":"))
        serializable.append(value)
    atomic_csv(output_dir / "raw_metrics.csv", serializable)


def _verify_resume_key(output_dir: Path, key: str, raw_rows: list[dict], config: ModelConfig, num_points: int) -> None:
    artifact = _artifact_for_key(output_dir, key); verified = verify_prediction_artifact(artifact); condition, task, seed, arm = _key_parts(key); manifest = verified["manifest"]
    if (manifest.get("condition"), manifest.get("task"), int(manifest.get("seed")), manifest.get("arm")) != (condition, task, seed, arm): raise RuntimeError(f"ENGINEERING_FAILURE: resume artifact identity 不一致: {artifact}")
    matching_rows = [row for row in raw_rows if row.get("condition") == condition and row.get("task") == task and int(row.get("seed", -1)) == seed and row.get("arm") == arm]
    if len(matching_rows) != 1: raise RuntimeError(f"ENGINEERING_FAILURE: resume 已完成 key 的 raw_metrics 行非唯一: {key}")
    history, horizon = (int(value[:-1]) for value in task.split("_")); prepared = prepare_experiment(SyntheticConfig(history=history, horizon=horizon, seed=seed, calendar_on=condition == "calendar_on", num_points=num_points)); priors = build_prior_arrays(prepared); hashes = expected_backbone_hashes(seed, history, horizon, config)
    dataset_base = output_dir / "datasets" / condition / task; dataset_path, dataset_metadata_path = dataset_base / f"seed{seed}.npz", dataset_base / f"seed{seed}.json"; audit_path = output_dir / "prior_tables" / condition / task / f"seed{seed}.json"
    if not dataset_path.is_file() or not dataset_metadata_path.is_file() or not audit_path.is_file(): raise RuntimeError(f"ENGINEERING_FAILURE: resume completed key 缺少 dataset/prior: {key}")
    with np.load(dataset_path, allow_pickle=False) as loaded:
        if not np.array_equal(loaded["raw_values"], prepared.raw_values) or not np.array_equal(loaded["normalized_values"], prepared.normalized_values) or not np.array_equal(loaded["timestamps_ns"], prepared.timestamps_ns): raise RuntimeError(f"ENGINEERING_FAILURE: resume dataset 不一致: {key}")
    if _load_json(dataset_metadata_path) != prepared.metadata or _load_json(audit_path) != _prior_audit_payload(prepared, priors, hashes): raise RuntimeError(f"ENGINEERING_FAILURE: resume dataset/prior audit 不一致: {key}")
    arrays = verified["arrays"]; expected_times = target_time_matrix(prepared, "test"); expected_target = values_matrix(prepared, "test")
    if not np.array_equal(arrays["target_times_ns"], expected_times) or not np.allclose(arrays["target_scaled"], expected_target, rtol=0, atol=1e-6): raise RuntimeError(f"ENGINEERING_FAILURE: resume target 与 dataset 不一致: {key}")
    if manifest.get("prediction_artifact") != str(artifact.relative_to(output_dir)) or manifest.get("center_mean") != prepared.center_mean or manifest.get("center_std") != prepared.center_std or int(manifest.get("test_count", -1)) != len(expected_target): raise RuntimeError(f"ENGINEERING_FAILURE: resume artifact sidecar metadata 不一致: {key}")
    if arm == "CCSPRF": expected_prior = priors["arrays"]["test"]["calendar"]
    elif arm == "CCSPRF-global": expected_prior = priors["arrays"]["test"]["global"]
    elif arm == "BASE": expected_prior = np.zeros_like(expected_target)
    else: expected_prior = None
    if expected_prior is not None and not np.allclose(arrays["prior_scaled"], expected_prior, rtol=0, atol=1e-6): raise RuntimeError(f"ENGINEERING_FAILURE: resume frozen prior 不一致: {key}")
    if manifest.get("initial_backbone_hash") != hashes[arm]: raise RuntimeError(f"ENGINEERING_FAILURE: resume backbone hash 不一致: {key}")
    if int(manifest.get("training_rng_seed", -1)) != training_rng_seed(seed): raise RuntimeError(f"ENGINEERING_FAILURE: resume training_rng_seed 不一致: {key}")
    if manifest.get("prior_mode") != PRIOR_MODES[arm] or manifest.get("tau_applied") != (arm == "CCSPRF") or manifest.get("tau") != (TAU_L0 if arm == "CCSPRF" else None): raise RuntimeError(f"ENGINEERING_FAILURE: resume tau/prior 语义不一致: {key}")
    if arm == "EMB" and (arrays["emb_hour_weight"].shape != (24, config.embedding_dim) or arrays["emb_weekday_weight"].shape != (7, config.embedding_dim) or not np.isfinite(arrays["emb_hour_weight"]).all() or not np.isfinite(arrays["emb_weekday_weight"]).all()): raise RuntimeError(f"ENGINEERING_FAILURE: resume EMB 权重不完整: {key}")
    computed = regression_metrics(arrays["target_scaled"], arrays["prediction_scaled"], prepared.center_mean, prepared.center_std)
    prior_metrics = regression_metrics(arrays["target_scaled"], arrays["prior_only_scaled"], prepared.center_mean, prepared.center_std)
    expected_metrics = {**computed, "prior_only_rmse_raw": prior_metrics["rmse_raw"], "prior_only_mae_raw": prior_metrics["mae_raw"], "prior_only_per_step_rmse_raw": prior_metrics["per_step_rmse_raw"]}
    row = matching_rows[0]
    for name in ("schema_version", "condition", "task", "seed", "arm", "training_rng_seed", "best_epoch", "best_valid_loss", "training_seconds", "parameter_count", "backbone_parameter_count", "parameter_increment", "initial_backbone_hash", "prior_mode", "tau", "tau_applied", "prediction_artifact"):
        expected_value, actual_value = manifest.get(name), row.get(name)
        if name == "tau":
            actual_value = None if actual_value in ("", "null", "None") else float(actual_value)
        elif name == "tau_applied":
            actual_value = actual_value.strip().lower() == "true"
        elif name in {"seed", "training_rng_seed", "best_epoch", "parameter_count", "backbone_parameter_count", "parameter_increment"}:
            actual_value = int(actual_value)
        elif name in {"best_valid_loss", "training_seconds"}:
            actual_value = float(actual_value)
        if expected_value is None:
            matches = actual_value is None
        elif isinstance(expected_value, (int, float)) and not isinstance(expected_value, bool):
            matches = np.isclose(float(actual_value), float(expected_value), rtol=0, atol=1e-6)
        else:
            matches = actual_value == expected_value
        if not matches: raise RuntimeError(f"ENGINEERING_FAILURE: resume raw/sidecar {name} 不一致: {key}")
    for name, expected_value in expected_metrics.items():
        actual_value = json.loads(row[name]) if name.endswith("per_step_rmse_raw") else float(row[name])
        if isinstance(expected_value, list):
            if not np.allclose(actual_value, expected_value, rtol=0, atol=1e-6): raise RuntimeError(f"ENGINEERING_FAILURE: resume raw_metrics {name} 不一致: {key}")
        elif not np.isclose(actual_value, expected_value, rtol=0, atol=1e-6):
            raise RuntimeError(f"ENGINEERING_FAILURE: resume raw_metrics {name} 不一致: {key}")
        manifest_value = manifest.get(name)
        if isinstance(expected_value, list):
            if manifest_value is None or not np.allclose(manifest_value, expected_value, rtol=0, atol=1e-6): raise RuntimeError(f"ENGINEERING_FAILURE: resume artifact manifest {name} 不一致: {key}")
        elif manifest_value is None or not np.isclose(float(manifest_value), expected_value, rtol=0, atol=1e-6):
            raise RuntimeError(f"ENGINEERING_FAILURE: resume artifact manifest {name} 不一致: {key}")
    log_path = output_dir / "training_logs" / condition / task / f"seed{seed}_{arm}.csv"
    if not log_path.is_file(): raise RuntimeError(f"ENGINEERING_FAILURE: resume completed key 缺少 training log: {key}")
    with log_path.open(newline="", encoding="utf-8") as handle:
        log_rows = list(csv.DictReader(handle))
    if not log_rows or not {"epoch", "train_loss", "valid_loss", "learning_rate"}.issubset(log_rows[0]): raise RuntimeError(f"ENGINEERING_FAILURE: resume training log 不完整: {key}")


def run_experiment(output_dir: Path, seeds: Iterable[int], tasks: Iterable[tuple[int, int]], conditions: Iterable[str], arms: Iterable[str], config: ModelConfig, num_points: int, device: torch.device | str, formal: bool = False, resume: bool = False) -> list[dict]:
    output_dir = Path(output_dir); seeds, tasks, conditions, arms = list(seeds), list(tasks), list(conditions), list(arms)
    if formal and not resume: check_clean_tree(); ensure_new_output_dir(output_dir)
    elif not resume: ensure_new_output_dir(output_dir)
    elif not output_dir.is_dir(): raise RuntimeError("ENGINEERING_FAILURE: --resume 要求已有输出目录")
    if formal and resume: check_source_tree(output_dir)
    expected_manifest = canonical_manifest(seeds, tasks, conditions, arms, config, num_points, "formal" if formal else "fixture"); manifest_path, progress_path = output_dir / "canonical_manifest.json", output_dir / "progress.json"
    if resume:
        if not manifest_path.is_file() or not progress_path.is_file(): raise RuntimeError("ENGINEERING_FAILURE: resume 缺少 canonical manifest/progress")
        if _load_json(manifest_path) != expected_manifest: raise RuntimeError("ENGINEERING_FAILURE: resume source/config/DGP/key manifest 不一致")
        progress = _load_json(progress_path); completed_list = progress.get("completed_keys", [])
        if progress.get("expected_keys") != expected_manifest["expected_keys"] or progress.get("expected_keys_sha256") != expected_manifest["expected_keys_sha256"] or progress.get("canonical_payload_sha256") != expected_manifest["canonical_payload_sha256"] or completed_list != expected_manifest["expected_keys"][:len(completed_list)] or len(completed_list) != len(set(completed_list)): raise RuntimeError("ENGINEERING_FAILURE: progress key 集合损坏")
        completed = set(completed_list)
    else:
        atomic_json(manifest_path, expected_manifest); atomic_json(progress_path, {"schema_version": SCHEMA_VERSION, "expected_keys": expected_manifest["expected_keys"], "expected_keys_sha256": expected_manifest["expected_keys_sha256"], "canonical_payload_sha256": expected_manifest["canonical_payload_sha256"], "completed_keys": []}); completed = set()
    raw_rows = _read_rows(output_dir / "raw_metrics.csv"); expected_set = set(expected_manifest["expected_keys"])
    raw_keys = []
    for row in raw_rows:
        try:
            raw_key = f"{row['condition']}|{row['task']}|{int(row['seed'])}|{row['arm']}"
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("ENGINEERING_FAILURE: raw_metrics identity 无法解析") from error
        raw_keys.append(raw_key)
    if any(item not in expected_set for item in raw_keys) or len(raw_keys) != len(set(raw_keys)):
        raise RuntimeError("ENGINEERING_FAILURE: raw_metrics 存在额外或重复 key")
    if resume:
        prefix = expected_manifest["expected_keys"][:len(completed)]
        allowed_raw_keys = set(prefix)
        if len(completed) < len(expected_manifest["expected_keys"]):
            allowed_raw_keys.add(expected_manifest["expected_keys"][len(completed)])
        if set(raw_keys) not in (set(prefix), allowed_raw_keys):
            raise RuntimeError("ENGINEERING_FAILURE: resume raw_metrics 不符合 progress 前缀或紧邻 reconciliation key")
    predictions_dir = output_dir / "predictions"
    for path in predictions_dir.rglob("*") if predictions_dir.exists() else []:
        if not path.is_file():
            continue
        relative = path.relative_to(predictions_dir)
        if len(relative.parts) != 4: raise RuntimeError(f"ENGINEERING_FAILURE: 非法 prediction 路径: {path}")
        if path.suffix not in {".npz", ".json"} or not path.name.removesuffix(path.suffix): raise RuntimeError(f"ENGINEERING_FAILURE: 非法 prediction 文件: {path}")
        condition, task, seed_dir, arm_file = relative.parts
        if not seed_dir.startswith("seed") or not arm_file.endswith(path.suffix): raise RuntimeError(f"ENGINEERING_FAILURE: 非法 prediction 命名: {path}")
        try:
            seed = int(seed_dir.removeprefix("seed"))
        except ValueError as error:
            raise RuntimeError(f"ENGINEERING_FAILURE: 非法 prediction seed: {path}") from error
        key = f"{condition}|{task}|{seed}|{arm_file.removesuffix(path.suffix)}"
        if key not in expected_set: raise RuntimeError(f"ENGINEERING_FAILURE: 发现额外 prediction key: {key}")
    reconciled_key = expected_manifest["expected_keys"][len(completed)] if resume and len(completed) < len(expected_manifest["expected_keys"]) and set(raw_keys) == set(expected_manifest["expected_keys"][:len(completed) + 1]) else None
    if resume:
        for pending_key in expected_manifest["expected_keys"][len(completed):]:
            pending_condition, pending_task, pending_seed, pending_arm = _key_parts(pending_key)
            pending_artifact = _artifact_for_key(output_dir, pending_key)
            pending_log = output_dir / "training_logs" / pending_condition / pending_task / f"seed{pending_seed}_{pending_arm}.csv"
            pending_state = (pending_artifact.exists(), pending_artifact.with_suffix(".json").exists(), pending_log.exists(), pending_key in raw_keys)
            if pending_key == reconciled_key:
                if pending_state != (True, True, True, True): raise RuntimeError(f"ENGINEERING_FAILURE: reconciliation key 文件不完整: {pending_key}")
            elif any(pending_state):
                raise RuntimeError(f"ENGINEERING_FAILURE: 未完成 key 存在 artifact/sidecar/log/raw 残留，拒绝覆盖: {pending_key}")
    for key in expected_manifest["expected_keys"]:
        artifact = _artifact_for_key(output_dir, key)
        if key in completed: _verify_resume_key(output_dir, key, raw_rows, config, num_points); continue
        log_path = output_dir / "training_logs" / key.split("|")[0] / key.split("|")[1] / f"seed{key.split('|')[2]}_{key.split('|')[3]}.csv"
        raw_present = key in raw_keys
        file_state = (artifact.exists(), artifact.with_suffix(".json").exists(), log_path.exists(), raw_present)
        if key == reconciled_key:
            if file_state != (True, True, True, True): raise RuntimeError(f"ENGINEERING_FAILURE: reconciliation key 文件不完整: {key}")
            _verify_resume_key(output_dir, key, raw_rows, config, num_points)
            completed.add(key)
            atomic_json(progress_path, {"schema_version": SCHEMA_VERSION, "expected_keys": expected_manifest["expected_keys"], "expected_keys_sha256": expected_manifest["expected_keys_sha256"], "canonical_payload_sha256": expected_manifest["canonical_payload_sha256"], "completed_keys": [item for item in expected_manifest["expected_keys"] if item in completed]})
            continue
        if any(file_state): raise RuntimeError(f"ENGINEERING_FAILURE: 未完成 key 存在 artifact/sidecar/log/raw 残留，拒绝覆盖: {key}")
        condition, task_name, seed, arm = _key_parts(key); history, horizon = (int(value[:-1]) for value in task_name.split("_")); prepared = prepare_experiment(SyntheticConfig(history=history, horizon=horizon, seed=seed, calendar_on=condition == "calendar_on", num_points=num_points)); priors = build_prior_arrays(prepared); _write_dataset(output_dir, prepared); _save_prior_audit(output_dir, prepared, priors, expected_backbone_hashes(seed, history, horizon, config)); row = train_one_arm(prepared, priors, arm, seed, config, device, output_dir); raw_rows.append(row); _write_metric_rows(output_dir, raw_rows); completed.add(key); atomic_json(progress_path, {"schema_version": SCHEMA_VERSION, "expected_keys": expected_manifest["expected_keys"], "expected_keys_sha256": expected_manifest["expected_keys_sha256"], "canonical_payload_sha256": expected_manifest["canonical_payload_sha256"], "completed_keys": [item for item in expected_manifest["expected_keys"] if item in completed]}); print(f"[{condition} {task_name} seed={seed} {arm}] RMSE={row['rmse_raw']:.6f} training={row['training_seconds']:.2f}s")
    return raw_rows
