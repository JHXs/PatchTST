"""Independent CCSPRF L0 audit and gate recomputation.

This file intentionally does not import ccsprf_l0, the runner, its metrics,
its prior fitter, or its artifact verifier.  The duplicated code below is the
independent audit path required by the frozen execution addendum.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
from tsai.models.PatchTST import PatchTST


SCHEMA = "ccsprf-l0-v2"
SEEDS = tuple(range(20260939, 20260969))
BOOTSTRAP_SEED, BOOTSTRAP_B = 20260969, 10_000
TAU = 50.0
ARMS = ("BASE", "EMB", "CCSPRF-global", "CCSPRF")
CONDITIONS = ("calendar_on", "calendar_off")
TASKS = ((24, 1), (168, 6))
TRAINING_RNG_OFFSET = 1_000_003
HOUR_NS, DAY_NS = 3_600_000_000_000, 86_400_000_000_000
HOUR_RAW = [-1.10, -1.25, -1.30, -1.15, -0.80, -0.20, 0.55, 1.00, 1.25, 1.10, 0.80, 0.45, 0.15, -0.05, -0.20, -0.05, 0.25, 0.70, 1.10, 0.95, 0.45, -0.10, -0.60, -0.90]
WEEKDAY_RAW = [0.95, 0.40, -0.25, -0.65, -0.45, 0.15, 0.85]
HOUR = np.asarray(HOUR_RAW) - np.mean(HOUR_RAW)
WEEKDAY = np.asarray(WEEKDAY_RAW) - np.mean(WEEKDAY_RAW)


def fail(message: str) -> None:
    raise RuntimeError(f"ENGINEERING_FAILURE: {message}")


def key(timestamp: int) -> tuple[int, int]:
    return int((int(timestamp) // HOUR_NS) % 24), int((int(timestamp) // DAY_NS + 3) % 7)


def expected_keys(seeds, tasks, conditions, arms):
    return [f"{condition}|{h}h_{p}h|{seed}|{arm}" for seed in sorted(seeds) for h, p in tasks for condition in conditions for arm in arms]


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_payload_sha256(manifest: dict) -> str:
    """Recompute the runner's canonical payload digest independently."""

    payload = {
        "source_commit": manifest.get("source_commit"),
        "canonical": manifest.get("canonical"),
        "expected_keys": manifest.get("expected_keys"),
    }
    return sha_bytes(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def validate_progress(manifest: dict, progress: dict, formal: bool) -> None:
    """Validate exact ordered completion without consulting runner code."""

    keys = manifest.get("expected_keys")
    if not isinstance(keys, list) or progress.get("expected_keys") != keys:
        fail("progress expected_keys 不完整或顺序不一致")
    if progress.get("expected_keys_sha256") != manifest.get("expected_keys_sha256"):
        fail("progress expected_keys_sha256 不一致")
    if progress.get("canonical_payload_sha256") != manifest.get("canonical_payload_sha256"):
        fail("progress canonical_payload_sha256 不一致")
    completed = progress.get("completed_keys")
    if not isinstance(completed, list) or len(completed) != len(set(completed)):
        fail("progress completed_keys 重复或类型错误")
    if completed != keys[:len(completed)]:
        fail("progress completed_keys 不是 canonical 前缀")
    if formal and completed != keys:
        fail("formal progress 未完整按序覆盖 480 key")
    if not formal and completed != keys:
        fail("fixture progress 未完整按序覆盖 manifest key")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def array_sha(value: np.ndarray) -> str:
    return sha_bytes(np.asarray(value, dtype=np.float32).tobytes())


def dgp(seed: int, calendar_on: bool, num_points: int):
    timestamps = np.arange(num_points, dtype=np.int64) * HOUR_NS
    hour, weekday = (HOUR.copy(), WEEKDAY.copy()) if calendar_on else (np.zeros(24), np.zeros(7))
    rng = np.random.default_rng(seed)
    epsilon = rng.normal(0.0, 0.80, size=num_points)
    values = np.empty(num_points, dtype=np.float64)
    previous_residual = 0.0
    for index, timestamp in enumerate(timestamps):
        h, w = key(timestamp)
        conditional_mean = 10.0 + hour[h] + weekday[w]
        values[index] = conditional_mean + 0.60 * previous_residual + epsilon[index]
        previous_residual = values[index] - conditional_mean
    metadata = {
        "formula": "y_t=mu+a_hour+b_weekday+rho*(y_prev-mu-a_prev-b_prev)+epsilon",
        "mu": 10.0, "rho": 0.60, "noise_std": 0.80, "initial_dynamic_residual": 0.0,
        "rng_constructor": "numpy.random.default_rng(seed)", "rng_bit_generator": "PCG64",
        "rng_normal_calls": 1, "rng_normal_shape": num_points, "rng_calls_after_normal": 0,
        "hour_effect_raw": HOUR_RAW, "weekday_effect_raw": WEEKDAY_RAW,
        "hour_effect_centered": hour.tolist(), "weekday_effect_centered": weekday.tolist(),
        "calendar_on": calendar_on, "timestamps_origin": "1970-01-01T00:00:00Z",
        "timestamp_dtype": "numpy.int64 nanoseconds", "timestamp_step_ns": HOUR_NS,
        "split_rule": "train_cut=int(num_points*0.70), valid_cut=int(num_points*0.80)",
    }
    return values, timestamps, metadata


def split_starts(num_points: int, history: int, horizon: int):
    train_cut, valid_cut = int(num_points * 0.70), int(num_points * 0.80)
    last = num_points - history - horizon
    return {
        "train": np.arange(0, train_cut - history - horizon + 1, dtype=np.int64),
        "valid": np.arange(train_cut - history, valid_cut - history - horizon + 1, dtype=np.int64),
        "test": np.arange(valid_cut - history, last + 1, dtype=np.int64),
    }


def values_for(starts, values, history, horizon):
    return np.asarray([values[start + history:start + history + horizon] for start in starts], dtype=np.float32)


def times_for(starts, timestamps, history, horizon):
    return np.asarray([timestamps[start + history:start + history + horizon] for start in starts], dtype=np.int64)


def independent_prior(train_times, train_values):
    if len(train_times) <= 1 or len(np.unique(train_times)) != len(train_times):
        fail("训练唯一目标时间戳 N<=1 或存在重复")
    values64 = np.asarray(train_values, dtype=np.float64)
    mu0 = float(np.mean(values64)); groups = {}
    for timestamp, value in zip(train_times, values64):
        k = key(timestamp); count, total = groups.get(k, (0, 0.0)); groups[k] = count + 1, total + float(value)
    full = {}
    loo = np.empty(len(train_times), dtype=np.float32)
    for h in range(24):
        for w in range(7):
            count, total = groups.get((h, w), (0, 0.0)); weight = count / (count + TAU) if count else 0.0
            full[(h, w)] = float(weight * total / count + (1 - weight) * mu0) if count else mu0
    for index, (timestamp, value) in enumerate(zip(train_times, values64)):
        n, total = groups[key(timestamp)]; n_minus = n - 1
        mu0_minus = (len(train_times) * mu0 - float(value)) / (len(train_times) - 1)
        if n_minus == 0:
            loo[index] = mu0_minus
        else:
            mean_minus = (total - float(value)) / n_minus; weight = n_minus / (n_minus + TAU)
            loo[index] = weight * mean_minus + (1 - weight) * mu0_minus
    return mu0, groups, full, loo


def global_loo(train_times, train_values):
    if len(train_times) <= 1 or len(np.unique(train_times)) != len(train_times): fail("global LOO N<=1 或重复")
    values64 = np.asarray(train_values, dtype=np.float64)
    mu0 = float(np.mean(values64)); return mu0, np.asarray([(len(train_times) * mu0 - value) / (len(train_times) - 1) for value in values64], dtype=np.float32)


def full_prior_matrix(times, full, mu0):
    return np.asarray([[full.get(key(value), mu0) for value in row] for row in times], dtype=np.float32)


def metrics(target, prediction, mean, std):
    error = (np.asarray(prediction) - np.asarray(target)) * std
    raw_target, raw_prediction = target * std + mean, prediction * std + mean
    denominator = np.abs(raw_target) + np.abs(raw_prediction)
    return {"mse_scaled": float(np.mean((prediction - target) ** 2)), "rmse_scaled": float(np.sqrt(np.mean((prediction - target) ** 2))), "mae_scaled": float(np.mean(np.abs(prediction - target))), "rmse_raw": float(np.sqrt(np.mean(error ** 2))), "mae_raw": float(np.mean(np.abs(error))), "smape_percent": float(200 * np.mean(np.abs(error) / np.maximum(denominator, 1e-6))), "per_step_rmse_raw": np.sqrt(np.mean(error ** 2, axis=0)).tolist()}


def independent_backbone_parameter_count(history: int, horizon: int, model_config: dict) -> int:
    model = PatchTST(c_in=1, c_out=1, seq_len=history, pred_dim=horizon,
                     n_layers=int(model_config["n_layers"]), n_heads=int(model_config["n_heads"]),
                     d_model=int(model_config["d_model"]), d_ff=int(model_config["d_ff"]),
                     dropout=float(model_config["dropout"]), attn_dropout=float(model_config["attn_dropout"]),
                     patch_len=int(model_config["patch_len"]), stride=int(model_config["stride"]),
                     padding_patch=bool(model_config["padding_patch"]))
    return sum(parameter.numel() for parameter in model.parameters())


def verify_dataset(output_dir: Path, condition: str, task: str, seed: int):
    base = output_dir / "datasets" / condition / task; dataset_path, metadata_path = base / f"seed{seed}.npz", base / f"seed{seed}.json"; audit_path = output_dir / "prior_tables" / condition / task / f"seed{seed}.json"
    if not dataset_path.is_file() or not metadata_path.is_file() or not audit_path.is_file(): fail(f"dataset/metadata/prior 缺失: {condition} {task} {seed}")
    metadata, audit = json.loads(metadata_path.read_text(encoding="utf-8")), json.loads(audit_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA or audit.get("schema_version") != SCHEMA: fail("dataset schema 不匹配")
    h, horizon, n = int(metadata["history"]), int(metadata["horizon"]), int(metadata["num_points"])
    with np.load(dataset_path, allow_pickle=False) as loaded:
        raw, normalized, timestamps = loaded["raw_values"], loaded["normalized_values"], loaded["timestamps_ns"]
    expected_raw, expected_times, expected_dgp = dgp(seed, condition == "calendar_on", n)
    if not np.allclose(raw, expected_raw.astype(np.float32), rtol=0, atol=1e-6): fail(f"DGP raw 不一致: {condition} {task} {seed}")
    if not np.array_equal(timestamps, expected_times) or timestamps.dtype != np.int64: fail("时间戳不一致")
    if metadata["dgp"] != expected_dgp: fail(f"DGP metadata 不一致: {condition} {task} {seed}")
    starts = split_starts(n, h, horizon)
    if metadata["split_start_indices"] != {k: v.tolist() for k, v in starts.items()}: fail("split 起点不一致")
    train_indices = np.asarray(sorted({int(start + h + offset) for start in starts["train"] for offset in range(horizon)}), dtype=np.int64)
    if metadata["train_target_indices"] != train_indices.tolist(): fail("训练唯一目标集合不一致")
    mean, std = float(raw[train_indices].mean()), float(raw[train_indices].std())
    if std < 1e-6: std = 1.0
    _assert_close(normalized, ((raw - mean) / std).astype(np.float32), "标准化")
    if not np.isclose(float(metadata["center_mean"]), mean) or not np.isclose(float(metadata["center_std"]), std): fail("scale metadata 不一致")
    train_times, train_values = timestamps[train_indices], normalized[train_indices]
    mu0, groups, full, cal_loo = independent_prior(train_times, train_values); global_mu0, global_loo_values = global_loo(train_times, train_values)
    expected_groups = []
    for hour in range(24):
        for weekday in range(7):
            count, total = groups.get((hour, weekday), (0, 0.0)); expected_groups.append({"hour": hour, "weekday": weekday, "n_unique_timestamps": int(count), "sum_scaled": float(total), "full_prior_scaled": float(full[(hour, weekday)]) if count else float(mu0)})
    expected_tau_applied = {arm: arm == "CCSPRF" for arm in ARMS}
    if audit["calendar_groups"] != expected_groups or not np.isclose(float(audit["global_mu0_scaled"]), global_mu0) or audit.get("calendar_prior_tau") != TAU or audit.get("tau_applied") != expected_tau_applied: fail("逐 c/global 先验表或 tau 语义不一致")
    expected_audit_arms = {arm: {"prior_mode": "none" if arm == "BASE" else "embedding_bias" if arm == "EMB" else "global" if arm == "CCSPRF-global" else "calendar", "tau": TAU if arm == "CCSPRF" else None, "tau_applied": arm == "CCSPRF", "parameter_increment": 124 if arm == "EMB" else 0} for arm in ARMS}
    if audit.get("arms") != expected_audit_arms: fail("audit arm metadata 或 tau 语义不一致")
    train_times_matrix, train_values_matrix = times_for(starts["train"], timestamps, h, horizon), values_for(starts["train"], normalized, h, horizon)
    expected_cal_loo = np.asarray([[cal_loo[np.where(train_times == value)[0][0]] for value in row] for row in train_times_matrix], dtype=np.float32)
    expected_global_loo = np.asarray([[(len(train_times) * global_mu0 - float(value)) / (len(train_times) - 1) for value in row] for row in train_values_matrix], dtype=np.float32)
    if audit.get("loo_checksums", {}).get("CCSPRF") != array_sha(expected_cal_loo) or audit.get("loo_checksums", {}).get("CCSPRF-global") != array_sha(expected_global_loo): fail("LOO checksum 不一致")
    test_times = times_for(starts["test"], timestamps, h, horizon); test_values = values_for(starts["test"], normalized, h, horizon); test_raw = np.asarray([raw[start + h:start + h + horizon] for start in starts["test"]], dtype=np.float64)
    return {"metadata": metadata, "audit": audit, "raw": raw, "normalized": normalized, "timestamps": timestamps, "mean": mean, "std": std, "full": full, "global_mu0": global_mu0, "test_times": test_times, "test_values": test_values, "test_raw": test_raw, "test_calendar": full_prior_matrix(test_times, full, mu0), "test_global": np.full(test_times.shape, global_mu0, dtype=np.float32)}


def _assert_close(actual, expected, label, atol=1e-6):
    if not np.allclose(actual, expected, rtol=0, atol=atol): fail(f"{label} 不一致")


def verify_artifact(path: Path, record: dict, condition: str, task: str, seed: int, arm: str, artifact_relative: str, model_config: dict):
    manifest_path = path.with_suffix(".json")
    if not manifest_path.is_file(): fail(f"manifest 缺失: {path}")
    manifest, digest = json.loads(manifest_path.read_text(encoding="utf-8")), hashlib.sha256(path.read_bytes()).hexdigest()
    if manifest.get("sha256") != digest: fail(f"artifact hash 不一致: {path}")
    if manifest.get("schema_version") != SCHEMA: fail(f"artifact schema 不一致: {path}")
    if (manifest.get("condition"), manifest.get("task"), int(manifest.get("seed", -1)), manifest.get("arm")) != (condition, task, seed, arm): fail(f"artifact identity 不一致: {path}")
    if manifest.get("prediction_artifact") != artifact_relative: fail(f"artifact path sidecar 不一致: {path}")
    if int(manifest.get("training_rng_seed", -1)) != seed + TRAINING_RNG_OFFSET: fail("training_rng_seed 不符合公平 RNG 规则")
    expected_prior_mode = "none" if arm == "BASE" else "embedding_bias" if arm == "EMB" else "global" if arm == "CCSPRF-global" else "calendar"
    expected_tau = TAU if arm == "CCSPRF" else None
    if manifest.get("prior_mode") != expected_prior_mode or manifest.get("tau") != expected_tau or manifest.get("tau_applied") != (arm == "CCSPRF"): fail("artifact tau/prior metadata 不一致")
    if not np.isfinite(float(manifest.get("center_mean", np.nan))) or not np.isfinite(float(manifest.get("center_std", np.nan))) or float(manifest.get("center_std", 0.0)) <= 0: fail("artifact center_mean/std 不合法")
    with np.load(path, allow_pickle=False) as loaded: arrays = {key: loaded[key] for key in loaded.files}
    base_required = {"prediction_scaled", "target_scaled", "prior_scaled", "residual_scaled", "prior_only_scaled", "target_times_ns", "prediction_raw", "target_raw", "prior_only_raw"}
    required = base_required | {"emb_hour_weight", "emb_weekday_weight"} if arm == "EMB" else base_required
    if set(arrays) != required: fail(f"artifact 字段不精确: {path}")
    if manifest.get("arrays") != {name: list(value.shape) for name, value in arrays.items()}: fail(f"artifact arrays shape 声明不一致: {path}")
    h, horizon = int(record["metadata"]["history"]), int(record["metadata"]["horizon"]); expected_shape = (len(record["test_times"]), horizon)
    for name in ("prediction_scaled", "target_scaled", "prior_scaled", "residual_scaled", "prior_only_scaled", "prediction_raw", "target_raw", "prior_only_raw"):
        if arrays[name].shape != expected_shape: fail(f"shape 错误 {name}: {path}")
        if not np.isfinite(arrays[name]).all(): fail(f"非有限值 {name}: {path}")
    if arrays["target_times_ns"].shape != expected_shape or arrays["target_times_ns"].dtype != np.int64: fail(f"时间戳 shape/dtype 错误: {path}")
    if not np.isclose(float(manifest.get("center_mean")), record["mean"], rtol=0, atol=1e-12) or not np.isclose(float(manifest.get("center_std")), record["std"], rtol=0, atol=1e-12): fail("artifact center_mean/std 不一致")
    if int(manifest.get("test_count", -1)) != expected_shape[0]: fail("artifact test_count 不一致")
    _assert_close(arrays["target_times_ns"], record["test_times"], "artifact target_times", 0)
    _assert_close(arrays["target_scaled"], record["test_values"], "artifact target_scaled", 1e-6)
    _assert_close(arrays["target_raw"], record["test_raw"], "artifact target_raw vs raw dataset", 1e-5)
    _assert_close(arrays["target_raw"], record["test_values"] * record["std"] + record["mean"], "artifact target_raw inverse", 1e-5)
    expected_prior = np.zeros(expected_shape, dtype=np.float32) if arm == "BASE" else record["test_global"] if arm == "CCSPRF-global" else record["test_calendar"] if arm == "CCSPRF" else None
    if arm == "EMB":
        if arrays["emb_hour_weight"].shape != (24, 4) or arrays["emb_weekday_weight"].shape != (7, 4): fail("EMB 权重 shape 不一致")
        if not np.isfinite(arrays["emb_hour_weight"]).all() or not np.isfinite(arrays["emb_weekday_weight"]).all(): fail("EMB 权重含非有限值")
        hours = np.asarray([[key(value)[0] for value in row] for row in record["test_times"]], dtype=np.int64)
        weekdays = np.asarray([[key(value)[1] for value in row] for row in record["test_times"]], dtype=np.int64)
        expected_prior = np.concatenate((arrays["emb_hour_weight"][hours], arrays["emb_weekday_weight"][weekdays]), axis=-1).mean(axis=-1)
    if expected_prior is not None: _assert_close(arrays["prior_scaled"], expected_prior, "冻结测试先验")
    _assert_close(arrays["prior_only_scaled"], arrays["prior_scaled"], "prior-only")
    _assert_close(arrays["prediction_scaled"], arrays["prior_scaled"] + arrays["residual_scaled"], "prediction reconstruction")
    for scaled, raw_name in (("prediction_scaled", "prediction_raw"), ("target_scaled", "target_raw"), ("prior_only_scaled", "prior_only_raw")): _assert_close(arrays[raw_name], arrays[scaled] * record["std"] + record["mean"], f"{raw_name} inverse")
    computed = metrics(arrays["target_scaled"], arrays["prediction_scaled"], record["mean"], record["std"]); prior_metrics = metrics(arrays["target_scaled"], arrays["prior_only_scaled"], record["mean"], record["std"])
    if manifest.get("initial_backbone_hash") != record["audit"]["initial_backbone_hashes"].get(arm): fail("initial backbone hash 审计不一致")
    expected_increment = 124 if arm == "EMB" else 0
    expected_backbone_count = independent_backbone_parameter_count(h, horizon, model_config)
    if int(manifest.get("backbone_parameter_count", -1)) != expected_backbone_count or int(manifest.get("parameter_count", -1)) != expected_backbone_count + expected_increment or manifest.get("parameter_increment") != expected_increment: fail("参数 metadata/增量不一致")
    expected_manifest_values = {**computed, "prior_only_rmse_raw": prior_metrics["rmse_raw"], "prior_only_mae_raw": prior_metrics["mae_raw"], "prior_only_per_step_rmse_raw": prior_metrics["per_step_rmse_raw"]}
    for name, value in expected_manifest_values.items():
        if name == "per_step_rmse_raw": actual = manifest.get(name)
        else: actual = manifest.get(name)
        if isinstance(value, list):
            if not np.allclose(np.asarray(actual), value, rtol=0, atol=1e-6): fail(f"manifest {name} 不一致")
        elif actual is None or not np.isclose(float(actual), float(value), rtol=0, atol=1e-6): fail(f"manifest {name} 不一致")
    return manifest, arrays, computed, prior_metrics


def bootstrap(values):
    values = np.asarray(values, dtype=np.float64); rng = np.random.default_rng(BOOTSTRAP_SEED); draws = rng.integers(0, len(values), size=(BOOTSTRAP_B, len(values))); medians = np.median(values[draws], axis=1); return float(np.median(values)), float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def gates(rows_by_key):
    result = {}
    for h, horizon in TASKS:
        task = f"{h}h_{horizon}h"; get = lambda condition, arm: {seed: rows_by_key[(condition, task, seed, arm)] for seed in SEEDS}
        base = get("calendar_on", "BASE")
        emb = get("calendar_on", "EMB")
        global_ = get("calendar_on", "CCSPRF-global")
        cc = get("calendar_on", "CCSPRF")
        off_base = get("calendar_off", "BASE")
        off_cc = get("calendar_off", "CCSPRF")
        main = [1 - cc[s]["rmse_raw"] / base[s]["rmse_raw"] for s in SEEDS]; centering = [1 - global_[s]["rmse_raw"] / base[s]["rmse_raw"] for s in SEEDS]; calendar = [1 - cc[s]["rmse_raw"] / global_[s]["rmse_raw"] for s in SEEDS]; emb_delta = [1 - cc[s]["rmse_raw"] / emb[s]["rmse_raw"] for s in SEEDS]; residual = [1 - cc[s]["rmse_raw"] / cc[s]["prior_only_rmse_raw"] for s in SEEDS]; off = [1 - off_cc[s]["rmse_raw"] / off_base[s]["rmse_raw"] for s in SEEDS]
        late = None
        if task == "168h_6h": late = [1 - np.sqrt(np.mean(np.asarray(cc[s]["per_step_rmse_raw"])[3:6] ** 2)) / np.sqrt(np.mean(np.asarray(base[s]["per_step_rmse_raw"])[3:6] ** 2)) for s in SEEDS]
        intervals = {name: bootstrap(values) for name, values in (("calendar", calendar), ("emb", emb_delta), ("off", off), ("late", late or [])) if values}
        checks = {"main_effect": np.median(main) >= .05 and all(value > 0 for value in main), "calendar_increment": np.median(calendar) >= .02 and intervals["calendar"][1] > 0, "calendar_off_null": abs(np.median(off)) <= .02 and intervals["off"][1] <= 0 <= intervals["off"][2], "delta_emb": np.median(emb_delta) >= 0 and intervals["emb"][1] > -.02, "prior_only_residual": np.median(residual) >= .05 and all(value > 0 for value in residual)}
        if late is not None: checks["late_4_6"] = np.median(late) >= 0 and intervals["late"][1] > -.02
        result[task] = {"delta_centering": bootstrap(centering), "main_effect": bootstrap(main), "delta_calendar": intervals["calendar"], "delta_emb": intervals["emb"], "delta_residual": bootstrap(residual), "calendar_off": intervals["off"], "delta_late": intervals.get("late"), "gates": checks, "pass": all(checks.values())}
    return {"schema_version": SCHEMA, "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": BOOTSTRAP_B, "tasks": result, "pass": all(value["pass"] for value in result.values())}


def summarize(output_dir: Path):
    output_dir = Path(output_dir)
    manifest_path = output_dir / "canonical_manifest.json"
    if not manifest_path.is_file(): fail("canonical_manifest.json 缺失")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")); canonical = manifest.get("canonical", {}); mode = canonical.get("mode")
    keys = manifest.get("expected_keys", []); key_hash = sha_bytes(json.dumps(keys, separators=(",", ":")).encode())
    if manifest.get("expected_keys_sha256") != key_hash or len(keys) != len(set(keys)): fail("manifest key hash/重复错误")
    if not manifest.get("source_commit") or manifest.get("canonical_payload_sha256") != canonical_payload_sha256(manifest): fail("canonical payload sha256 不一致")
    progress_path = output_dir / "progress.json"
    if not progress_path.is_file(): fail("progress.json 缺失")
    validate_progress(manifest, json.loads(progress_path.read_text(encoding="utf-8")), mode == "formal")
    if canonical.get("tau") != TAU or canonical.get("frozen_hour_effect_raw") != HOUR_RAW or canonical.get("frozen_weekday_effect_raw") != WEEKDAY_RAW: fail("canonical 冻结常量不一致")
    expected_model = {"d_model": 16, "n_heads": 4, "n_layers": 3, "d_ff": 128, "dropout": 0.2, "attn_dropout": 0.0, "patch_len": 4, "stride": 2, "padding_patch": True, "learning_rate": 1e-3, "weight_decay": 1e-4, "epochs": 40, "patience": 8, "batch_size": 256, "embedding_dim": 4}
    if mode == "formal" and canonical.get("model_config") != expected_model: fail("formal ModelConfig 不符合冻结配置")
    dgp_config = canonical.get("dgp_config", {})
    if mode == "formal" and (dgp_config.get("num_points") != 8736 or dgp_config.get("mu") != 10.0 or dgp_config.get("rho") != 0.60 or dgp_config.get("noise_std") != 0.80 or dgp_config.get("timestamps_origin") != "1970-01-01T00:00:00Z" or dgp_config.get("timestamp_step_ns") != 3_600_000_000_000 or dgp_config.get("split_rule") != "int(num_points*0.70), int(num_points*0.80)" or dgp_config.get("rng") != "default_rng(seed)/PCG64/one normal call"): fail("formal DGP canonical 配置不符合冻结协议")
    if mode == "formal":
        expected = expected_keys(SEEDS, TASKS, CONDITIONS, ARMS)
        if keys != expected: fail("formal manifest 未精确枚举 480 key")
    else:
        expected = keys
    artifact_files = sorted((output_dir / "predictions").rglob("*.npz"))
    actual_keys = []
    for path in artifact_files:
        relative = path.relative_to(output_dir / "predictions")
        if len(relative.parts) != 4: fail(f"非法 prediction 路径: {path}")
        condition, task, seed_dir, arm_file = relative.parts
        if not seed_dir.startswith("seed") or not arm_file.endswith(".npz"):
            fail(f"非法 prediction 命名: {path}")
        try:
            seed = int(seed_dir.removeprefix("seed"))
        except ValueError:
            fail(f"非法 prediction seed: {path}")
        arm = arm_file.removesuffix(".npz")
        actual_keys.append(f"{condition}|{task}|{seed}|{arm}")
    all_prediction_files = [path for path in (output_dir / "predictions").rglob("*") if path.is_file()]
    allowed = {path.with_suffix(".json") for path in artifact_files} | set(artifact_files)
    if set(all_prediction_files) != allowed: fail("prediction 目录存在缺失 manifest 或额外文件")
    if sorted(actual_keys) != sorted(expected): fail("实际 artifact key 与 canonical manifest 不一致")
    rows, cache = [], {}
    for path, key_value in zip(artifact_files, actual_keys):
        condition, task, seed_text, arm = key_value.split("|"); seed = int(seed_text); cache_key = (condition, task, seed)
        if cache_key not in cache: cache[cache_key] = verify_dataset(output_dir, condition, task, seed)
        manifest_item, arrays, computed, prior_metrics = verify_artifact(path, cache[cache_key], condition, task, seed, arm, str(path.relative_to(output_dir)), canonical["model_config"])
        rows.append({"schema_version": SCHEMA, "condition": condition, "task": task, "seed": seed, "arm": arm, "training_rng_seed": int(manifest_item["training_rng_seed"]), "best_epoch": int(manifest_item["best_epoch"]), "best_valid_loss": float(manifest_item["best_valid_loss"]), "training_seconds": float(manifest_item["training_seconds"]), "parameter_count": int(manifest_item["parameter_count"]), "backbone_parameter_count": int(manifest_item["backbone_parameter_count"]), "parameter_increment": int(manifest_item["parameter_increment"]), "initial_backbone_hash": manifest_item["initial_backbone_hash"], "prior_mode": manifest_item["prior_mode"], "tau": manifest_item.get("tau"), "tau_applied": bool(manifest_item["tau_applied"]), "prior_only_rmse_raw": prior_metrics["rmse_raw"], "prior_only_mae_raw": prior_metrics["mae_raw"], "prior_only_per_step_rmse_raw": prior_metrics["per_step_rmse_raw"], **computed, "prediction_artifact": str(path.relative_to(output_dir))})
    raw_path = output_dir / "raw_metrics.csv"
    if raw_path.is_file():
        with raw_path.open(newline="", encoding="utf-8") as handle:
            raw_rows = list(csv.DictReader(handle))
    else:
        raw_rows = []
    if len(raw_rows) != len(rows): fail("raw_metrics 行数与 artifact 不一致")
    def value(row, name):
        if name in ("per_step_rmse_raw", "prior_only_per_step_rmse_raw"): return json.loads(row[name])
        if name == "tau": return None if row[name] in ("", "null", "None") else float(row[name])
        if name == "tau_applied": return row[name].strip().lower() == "true"
        return row[name]
    for expected_row in rows:
        matches = [row for row in raw_rows if row.get("condition") == expected_row["condition"] and row.get("task") == expected_row["task"] and int(row.get("seed", -1)) == expected_row["seed"] and row.get("arm") == expected_row["arm"]]
        if len(matches) != 1: fail("raw_metrics 缺失/重复 identity")
        for name, expected_value in expected_row.items():
            actual = value(matches[0], name)
            if isinstance(expected_value, list):
                if not np.allclose(actual, expected_value, rtol=0, atol=1e-6): fail(f"raw_metrics {name} 不一致")
            elif isinstance(expected_value, bool):
                if actual != expected_value: fail(f"raw_metrics {name} 不一致")
            elif expected_value is None:
                if actual is not None: fail(f"raw_metrics {name} 不一致")
            elif isinstance(expected_value, (int, float)):
                if not np.isclose(float(actual), float(expected_value), rtol=0, atol=1e-6): fail(f"raw_metrics {name} 不一致")
            elif actual != expected_value: fail(f"raw_metrics {name} 不一致")
    for cache_key, record in cache.items():
        hashes = record["audit"]["initial_backbone_hashes"]
        if set(hashes) != set(ARMS) or len(set(hashes.values())) != 1: fail(f"四臂 initial backbone hash 不一致: {cache_key}")
    formal = mode == "formal"
    gate_result = gates({(row["condition"], row["task"], row["seed"], row["arm"]): row for row in rows}) if formal else {"schema_version": SCHEMA, "status": "NOT_APPLICABLE_FIXTURE", "pass": None}
    atomic_json(output_dir / "gate_summary.json", gate_result)
    with (output_dir / "recomputed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    status = "PASS" if formal and gate_result["pass"] else "STOP" if formal else "NOT_APPLICABLE_FIXTURE"
    atomic_json(output_dir / "strict_release_status.json", {"schema_version": SCHEMA, "mode": mode, "status": status, "engineering_verification": "PASS", "gate_pass": gate_result.get("pass"), "gate_summary": "gate_summary.json", "derived_summary": "recomputed_metrics.csv"})
    return gate_result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, required=True); args = parser.parse_args(); print(json.dumps(summarize(args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
