"""Independent integrity summarizer for the Round 11 synthetic fixture.

The runner's factor arrays, conditions, method predictions and metrics are not
trusted here.  Only the saved neural base outputs are inputs; all labels,
plans, pools, operators and raw prediction formulas are rebuilt from the
recorded seed/configuration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import cf_cv_smearing_fixture as fx


def _same(actual: Any, expected: Any, *, name: str, equal_nan: bool = False) -> None:
    try:
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-10, atol=1e-12,
                                  equal_nan=equal_nan)
    except AssertionError as exc:
        raise ValueError(f"independent reconstruction mismatch: {name}") from exc


def _pair(output: Path, seed: int, task: str, expected_keys: list[str]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    stem = f"task_{seed}_{fx.safe_task_name(task)}"
    json_path, npz_path = output / f"{stem}.json", output / f"{stem}.npz"
    if not json_path.exists() or not npz_path.exists():
        raise ValueError(f"missing task pair: {stem}")
    info = json.loads(json_path.read_text(encoding="utf-8"))
    if info.get("schema") != "round11-cf-cv-task/v2":
        raise ValueError(f"task {stem} schema mismatch")
    if info.get("unit_key") != f"seed={seed}/task={task}":
        raise ValueError(f"task {stem} unit key mismatch")
    if info.get("fit_keys") != expected_keys:
        raise ValueError(f"task {stem} does not contain its exact seven fit keys")
    if info.get("npz_sha256") != fx.sha256_file(npz_path):
        raise ValueError(f"task {stem} NPZ hash mismatch")
    with np.load(npz_path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key].copy() for key in loaded.files}
    required = {
        "y_valid", "y_test", "valid_timestamps", "test_timestamps", "m_query_rows",
        "m_prediction_rows", "m_rows", "m_leads", "m_timestamps", "m_blocks",
        "m_condition_raw", "m_r", "m_u", "s_m_indices", "s_leads", "s_timestamps",
        "s_blocks", "s_c_raw", "s_r", "s_u", "s_b", "s_v0", "s_vtilde",
        "s_oof_rem", "s_oof_cv", "s_oof_rhs", "s_oof_lambda", "valid_log_m", "test_log_m",
        "valid_condition_raw", "test_condition_raw", "valid_factor_removal", "test_factor_removal",
        "valid_factor_cv", "test_factor_cv", "lambda_by_lead",
    }
    required |= {f"valid_{method}" for method in fx.METHODS}
    required |= {f"test_{method}" for method in fx.METHODS}
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"task {stem} NPZ missing keys: {sorted(missing)}")
    return info, arrays


def validate_mode_metadata(metadata: dict[str, Any]) -> None:
    """Reject semantic labels that could make this fixture look like formal L0."""
    expected = {"formal_l0": False, "formal": False, "non_formal": True}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"fixture metadata has invalid semantic flag {key}={metadata.get(key)!r}")
    smoke = metadata.get("smoke")
    if not isinstance(smoke, bool):
        raise ValueError("fixture metadata must explicitly declare smoke")
    if metadata.get("registered_fixture") != (not smoke):
        raise ValueError("registered_fixture and smoke mode flags disagree")
    if not isinstance(metadata.get("rho_diagnostic_stop"), bool):
        raise ValueError("fixture metadata must explicitly declare rho diagnostic status")
    if smoke:
        expected_mode = {"status": "SMOKE_COMPLETE", "rho_gate_applied": False, "rho_stop": False}
    else:
        if metadata.get("status") not in {"COMPLETE", "STOP"}:
            raise ValueError("registered fixture has an invalid status")
        expected_mode = {"rho_gate_applied": True}
    for key, value in expected_mode.items():
        if metadata.get(key) != value:
            raise ValueError(f"fixture metadata has invalid mode field {key}={metadata.get(key)!r}")


def _assert_fit_records(info: dict[str, Any], task: fx.WindowTask, plan: fx.OOFPlan,
                        raw_target: np.ndarray, log_target: np.ndarray) -> None:
    train = np.asarray(task.split_indices["train"], dtype=np.int64)
    purge = task.history + task.horizon + 1
    full_fit, full_valid, full_excluded = fx.split_train_inner(train, task.target_timestamps, purge)
    if info.get("full_inner_excluded") != full_excluded.tolist():
        raise ValueError("full train-inner exclusion mismatch")
    records = info.get("fit_records")
    if not isinstance(records, list) or len(records) != 7:
        raise ValueError("sidecar must contain seven fit records")
    expected: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    expected[f"seed={info['seed']}/task={task.name}/stage=raw_full"] = (raw_target, full_fit, full_valid)
    expected[f"seed={info['seed']}/task={task.name}/stage=log_full"] = (log_target, full_fit, full_valid)
    for block in plan.blocks:
        expected[f"seed={info['seed']}/task={task.name}/stage=log_oof/block={block.block}"] = (
            log_target, block.inner_fit_indices, block.inner_valid_indices)
    for record in records:
        key = record.get("key")
        if key not in expected:
            raise ValueError(f"unexpected fit record {key}")
        target, fit_indices, valid_indices = expected[key]
        if record.get("fit_indices") != np.asarray(fit_indices, dtype=np.int64).tolist():
            raise ValueError(f"fit indices mismatch for {key}")
        if record.get("valid_indices") != np.asarray(valid_indices, dtype=np.int64).tolist():
            raise ValueError(f"inner validation indices mismatch for {key}")
        expected_input = fx.fit_input_scaler(task, fit_indices)
        expected_target = fx.fit_target_scaler(target, fit_indices)
        _same(record.get("input_scaler", {}).get("mean"), expected_input["mean"], name=f"{key} input mean")
        _same(record.get("input_scaler", {}).get("scale"), expected_input["scale"], name=f"{key} input scale")
        _same(record.get("target_scaler", {}).get("mean"), expected_target["mean"], name=f"{key} target mean")
        _same(record.get("target_scaler", {}).get("scale"), expected_target["scale"], name=f"{key} target scale")
    if {record.get("key") for record in records} != set(expected):
        raise ValueError("fit record set is incomplete")


def _evaluate(query_c: np.ndarray, fit_c: np.ndarray, fit_r: np.ndarray, fit_b: np.ndarray,
              horizon: int, allowed: list[np.ndarray], name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, dict[str, Any]]]:
    n_query = query_c.shape[0] // horizon
    removal = np.full((n_query, horizon), np.nan, dtype=np.float64)
    candidate = np.full_like(removal, np.nan)
    rhs = np.full_like(removal, np.nan)
    diagnostics: dict[int, dict[str, Any]] = {}
    for lead in range(horizon):
        q_ids = np.arange(lead, n_query * horizon, horizon, dtype=np.int64)
        fit_ids = np.asarray(allowed[lead], dtype=np.int64)
        prefix = fx.compute_prefix_control_variate(fit_r[fit_ids], fit_b[fit_ids])
        H = fx.build_s_prefix_operator(
            query_c[q_ids], fit_c[fit_ids], np.zeros(q_ids.size, dtype=np.int64),
            np.zeros(fit_ids.size, dtype=np.int64),
            [np.arange(fit_ids.size, dtype=np.int64)] * q_ids.size, name=f"{name}_h{lead}")
        h_u = H.apply(prefix["u"])
        h_v = H.apply(prefix["vtilde"])
        removal[:, lead] = h_u
        candidate[:, lead] = H.apply(prefix["u"] - prefix["lambda"] * prefix["vtilde"])
        rhs[:, lead] = -prefix["lambda"] * h_v
        np.testing.assert_allclose(candidate[:, lead] - removal[:, lead], rhs[:, lead], rtol=1e-10, atol=1e-12)
        diagnostics[lead] = {
            "shape": list(H.shape), "lambda": prefix["lambda"], "var_v": prefix["var_v"],
            "cov_uv": prefix["cov_uv"], "variance_ratio": prefix["variance_ratio"],
            "mean_v0": prefix["mean_v0"],
            "rho": float(np.linalg.norm(h_v) / (np.linalg.norm(h_u) + 1e-12)),
        }
    return removal, candidate, rhs, diagnostics


def build_method_predictions(log_mean: np.ndarray, raw_base: np.ndarray, global_factor: np.ndarray,
                              hpc: np.ndarray, removal_factor: np.ndarray,
                              cv_factor: np.ndarray) -> dict[str, np.ndarray]:
    """Apply the frozen raw/log formulas; raw neural output is never boundary-clipped."""
    log_mean = np.asarray(log_mean, dtype=np.float64)
    return {
        "raw_mse_base": np.asarray(raw_base, dtype=np.float64),
        "naive_log": fx.apply_raw_boundary(log_mean, np.ones_like(log_mean)),
        "global_smearing": fx.apply_raw_boundary(log_mean, np.broadcast_to(global_factor, log_mean.shape)),
        "hpc": fx.apply_raw_boundary(log_mean, np.broadcast_to(hpc, log_mean.shape)),
        "strongest_existing_conditional_smearing": fx.apply_raw_boundary(log_mean, removal_factor),
        "cf_cv_cs_removal": fx.apply_raw_boundary(log_mean, removal_factor),
        "cf_cv_cs": fx.apply_raw_boundary(log_mean, cv_factor),
    }


def write_verification(output: Path, summary: dict[str, Any], runner_manifest_hash: str) -> None:
    """Write deterministic verification artifacts without touching runner manifest."""
    verification = output / "verification"
    verification.mkdir(parents=True, exist_ok=True)
    summary_path = verification / "summary.json"
    fx.atomic_json(summary_path, summary)
    verification_manifest = {
        "schema": "round11-cf-cv-verification-manifest/v1",
        "source_runner_manifest_sha256": runner_manifest_hash,
        "artifacts": {"summary.json": fx.sha256_file(summary_path)},
    }
    fx.atomic_json(verification / "manifest.json", verification_manifest)


def _rebuild_task(sequence: fx.SequenceDGP, task: fx.WindowTask, info: dict[str, Any],
                  arrays: dict[str, np.ndarray], warmup_fraction: float) -> dict[str, Any]:
    task = fx.with_splits(task)
    plan = fx.build_oof_plan(task, warmup_fraction=warmup_fraction)
    raw_target = task.y_raw.astype(np.float64)
    log_target = task.z_target[:, None, :].astype(np.float64)
    _assert_fit_records(info, task, plan, raw_target, log_target)
    for name, index_values in (("train_indices", plan.train_indices), ("warmup_indices", plan.warmup_indices)):
        if info.get(name) != np.asarray(index_values, dtype=np.int64).tolist():
            raise ValueError(f"{task.name} {name} mismatch")
    if info.get("purge") != task.history + task.horizon + 1:
        raise ValueError("purge mismatch")
    if len(info.get("blocks", [])) != 5:
        raise ValueError("OOF block count is not five")
    for saved, rebuilt in zip(info["blocks"], plan.blocks):
        for key in ("block", "query_indices", "fit_indices", "inner_fit_indices", "inner_valid_indices", "excluded_indices", "query_min_timestamp", "fit_max_timestamp", "purge", "exclusion_reason"):
            expected = getattr(rebuilt, key)
            if isinstance(expected, np.ndarray):
                expected = expected.tolist()
            if saved.get(key) != expected:
                raise ValueError(f"OOF block {rebuilt.block} field {key} mismatch")

    oof_rows = np.concatenate([block.query_indices for block in plan.blocks])
    m_prediction_rows = np.asarray(arrays["m_prediction_rows"], dtype=np.float64)
    if m_prediction_rows.shape != (oof_rows.size, task.horizon) or not np.isfinite(m_prediction_rows).all():
        raise ValueError("saved OOF neural base output has the wrong shape or non-finite values")
    m_c_row, m_leads_row, m_ts_row = fx.build_conditions(task, m_prediction_rows, row_indices=oof_rows)
    m_block_row = np.repeat(np.concatenate([np.full(block.query_indices.size, block.block) for block in plan.blocks]), task.horizon)
    order = np.lexsort((m_leads_row, m_ts_row))
    m_rows = np.repeat(oof_rows, task.horizon)[order]
    m_leads = m_leads_row[order]
    m_ts = m_ts_row[order]
    m_blocks = m_block_row[order]
    m_c_raw = m_c_row[order]
    m_pred = m_prediction_rows.ravel()[order]
    m_r = log_target[oof_rows].ravel()[order] - m_pred
    m_u = np.exp(m_r)
    for key, expected in (("m_query_rows", oof_rows), ("m_rows", m_rows), ("m_leads", m_leads),
                          ("m_timestamps", m_ts), ("m_blocks", m_blocks), ("m_condition_raw", m_c_raw),
                          ("m_r", m_r), ("m_u", m_u)):
        _same(arrays[key], expected, name=key)

    purge = task.history + task.horizon + 1
    s_m_indices, b_support = fx.select_s_pool(m_ts, m_leads, purge)
    _same(arrays["s_m_indices"], s_m_indices, name="S selector")
    if info.get("s_m_indices") != s_m_indices.tolist():
        raise ValueError("sidecar S selector mismatch")
    if info.get("b_support_indices") != [ids.tolist() for ids in b_support]:
        raise ValueError("sidecar B support mismatch")
    B = fx.build_prefix_scaled_operator(m_c_raw[s_m_indices], m_c_raw, b_support, name="B")
    if list(B.shape) != info.get("B_shape") or B.shape != (s_m_indices.size, m_r.size):
        raise ValueError("B is not the expected rectangular operator")
    s_c_raw, s_leads, s_ts = m_c_raw[s_m_indices], m_leads[s_m_indices], m_ts[s_m_indices]
    s_blocks, s_r, s_u = m_blocks[s_m_indices], m_r[s_m_indices], m_u[s_m_indices]
    s_b = B.apply(m_r)
    s_v0 = s_r - s_b
    s_vtilde = np.full_like(s_v0, np.nan)
    lambdas = np.empty(task.horizon, dtype=np.float64)
    var_v, cov_uv, var_ratio, mean_v0 = [], [], [], []
    for lead in range(task.horizon):
        ids = np.flatnonzero(s_leads == lead)
        prefix = fx.compute_prefix_control_variate(s_r[ids], s_b[ids])
        s_vtilde[ids] = prefix["vtilde"]
        lambdas[lead] = prefix["lambda"]
        var_v.append(prefix["var_v"]); cov_uv.append(prefix["cov_uv"])
        var_ratio.append(prefix["variance_ratio"]); mean_v0.append(prefix["mean_v0"])
        _same(s_vtilde[ids].sum(), 0.0, name=f"lead {lead} vtilde centering")
    if info.get("m_count") != int(m_r.size) or info.get("s_count") != int(s_r.size):
        raise ValueError("M/S sidecar counts mismatch")
    if info.get("bandwidth") != fx.SMOOTHER_BANDWIDTH or info.get("smoother_max_support") != fx.SMOOTHER_MAX_SUPPORT:
        raise ValueError("smoother freeze metadata mismatch")
    for name, expected in (("lambda_by_lead", lambdas), ("var_v_by_lead", var_v),
                           ("cov_u_v_by_lead", cov_uv), ("variance_ratio_by_lead", var_ratio),
                           ("mean_v0_by_lead", mean_v0)):
        _same(info.get(name), expected, name=f"sidecar {name}")
    for key, expected in (("s_leads", s_leads), ("s_timestamps", s_ts), ("s_blocks", s_blocks),
                          ("s_c_raw", s_c_raw), ("s_r", s_r), ("s_u", s_u), ("s_b", s_b),
                          ("s_v0", s_v0), ("s_vtilde", s_vtilde), ("lambda_by_lead", lambdas)):
        _same(arrays[key], expected, name=key)

    valid = np.asarray(task.split_indices["valid"], dtype=np.int64)
    test = np.asarray(task.split_indices["test"], dtype=np.int64)
    valid_log = np.asarray(arrays["valid_log_m"], dtype=np.float64)
    test_log = np.asarray(arrays["test_log_m"], dtype=np.float64)
    valid_c, _, _ = fx.build_conditions(task, valid_log, row_indices=valid)
    test_c, _, _ = fx.build_conditions(task, test_log, row_indices=test)
    _same(arrays["valid_condition_raw"], valid_c, name="valid raw C")
    _same(arrays["test_condition_raw"], test_c, name="test raw C")
    by_lead = [np.flatnonzero(s_leads == lead) for lead in range(task.horizon)]
    valid_rem, valid_cv, valid_rhs, valid_diag = _evaluate(valid_c, s_c_raw, s_r, s_b, task.horizon, by_lead, "H_valid")
    test_rem, test_cv, test_rhs, test_diag = _evaluate(test_c, s_c_raw, s_r, s_b, task.horizon, by_lead, "H_test")
    _same(arrays["valid_factor_removal"], valid_rem, name="valid removal factor")
    _same(arrays["test_factor_removal"], test_rem, name="test removal factor")
    _same(arrays["valid_factor_cv"], valid_cv, name="valid CV factor")
    _same(arrays["test_factor_cv"], test_cv, name="test CV factor")
    for split, diagnostics in (("valid", valid_diag), ("test", test_diag)):
        expected_dims = [diagnostics[lead]["shape"] for lead in range(task.horizon)]
        if info["H_dimensions"][split] != expected_dims:
            raise ValueError(f"{split} H dimensions mismatch")
        saved_diagnostics = info.get("H_diagnostics", {}).get(split, {})
        for lead in range(task.horizon):
            if str(lead) not in saved_diagnostics:
                raise ValueError(f"{split} H diagnostic is missing lead {lead}")
            for field in ("shape", "lambda", "var_v", "cov_uv", "variance_ratio", "mean_v0", "rho"):
                expected = diagnostics[lead][field]
                actual = saved_diagnostics[str(lead)].get(field)
                if field == "shape":
                    if actual != expected:
                        raise ValueError(f"{split} H diagnostic {lead} {field} mismatch")
                else:
                    _same(actual, expected, name=f"{split} H diagnostic {lead} {field}")

    # Rebuild the second cold-start layer; excluded blocks must remain NaN.
    s_oof_rem = np.full(s_r.size, np.nan); s_oof_cv = np.full(s_r.size, np.nan)
    s_oof_rhs = np.full(s_r.size, np.nan); s_oof_lambda = np.full(s_r.size, np.nan)
    excluded: list[int] = []
    for block in plan.blocks:
        query = np.flatnonzero(s_blocks == block.block)
        allowed = [np.flatnonzero((s_leads == lead) & (s_blocks < block.block) &
                                  (s_ts <= block.query_min_timestamp - purge)) for lead in range(task.horizon)]
        if query.size == 0 or any(ids.size < 2 for ids in allowed):
            excluded.append(block.block)
            continue
        for lead in range(task.horizon):
            q_ids, fit_ids = query[s_leads[query] == lead], allowed[lead]
            prefix = fx.compute_prefix_control_variate(s_r[fit_ids], s_b[fit_ids])
            H = fx.build_s_prefix_operator(s_c_raw[q_ids], s_c_raw[fit_ids],
                                            np.zeros(q_ids.size, dtype=np.int64), np.zeros(fit_ids.size, dtype=np.int64),
                                            [np.arange(fit_ids.size, dtype=np.int64)] * q_ids.size,
                                            name=f"H_s_oof_{block.block}_h{lead}")
            s_oof_rem[q_ids] = H.apply(prefix["u"])
            s_oof_cv[q_ids] = H.apply(prefix["u"] - prefix["lambda"] * prefix["vtilde"])
            s_oof_rhs[q_ids] = -prefix["lambda"] * H.apply(prefix["vtilde"])
            s_oof_lambda[q_ids] = prefix["lambda"]
            np.testing.assert_allclose(s_oof_cv[q_ids] - s_oof_rem[q_ids], s_oof_rhs[q_ids], rtol=1e-10, atol=1e-12)
    if info.get("S_oof_excluded_blocks") != excluded:
        raise ValueError("S-OOF exclusion mismatch")
    for key, expected in (("s_oof_rem", s_oof_rem), ("s_oof_cv", s_oof_cv), ("s_oof_rhs", s_oof_rhs),
                          ("s_oof_lambda", s_oof_lambda)):
        _same(arrays[key], expected, name=key, equal_nan=True)

    global_factor = np.asarray([np.mean(s_u[by_lead[lead]]) for lead in range(task.horizon)])
    hpc = np.asarray([fx.hpc_factor(s_r[by_lead[lead]]) for lead in range(task.horizon)])
    valid_predictions = build_method_predictions(valid_log, arrays["valid_raw_mse_base"], global_factor, hpc,
                                                  valid_rem, valid_cv)
    test_predictions = build_method_predictions(test_log, arrays["test_raw_mse_base"], global_factor, hpc,
                                                 test_rem, test_cv)
    for split, predictions in (("valid", valid_predictions), ("test", test_predictions)):
        truth = raw_target[valid if split == "valid" else test, 0, :]
        for method in fx.METHODS:
            _same(arrays[f"{split}_{method}"], predictions[method], name=f"{split} {method}")
        if not np.array_equal(predictions["strongest_existing_conditional_smearing"], predictions["cf_cv_cs_removal"]):
            raise ValueError("strongest conditional and removal differ")
    if not np.array_equal(arrays["y_valid"], raw_target[valid, 0, :]) or not np.array_equal(arrays["y_test"], raw_target[test, 0, :]):
        raise ValueError("saved labels do not match DGP reconstruction")
    _same(arrays["valid_timestamps"], task.target_timestamps[valid], name="valid timestamps")
    _same(arrays["test_timestamps"], task.target_timestamps[test], name="test timestamps")
    metrics = {
        split: {method: fx.metric_rows(raw_target[valid if split == "valid" else test, 0, :], predictions[method])
                for method in fx.METHODS}
        for split, predictions in (("valid", valid_predictions), ("test", test_predictions))
    }
    return {"task": task.name, "seed": int(info["seed"]), "m_count": int(m_r.size), "s_count": int(s_r.size),
            "metrics": metrics, "rho_valid_by_lead": [valid_diag[i]["rho"] for i in range(task.horizon)],
            "rho_test_by_lead": [test_diag[i]["rho"] for i in range(task.horizon)]}


def summarize(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    fx.validate_manifest(output)
    metadata_path = output / "metadata.json"
    if not metadata_path.exists():
        raise ValueError("missing fixture metadata")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_mode_metadata(metadata)
    strict_path = output / "strict_fixture_status.json"
    if not strict_path.exists():
        raise ValueError("missing strict fixture status")
    strict_status = json.loads(strict_path.read_text(encoding="utf-8"))
    validate_mode_metadata(strict_status)
    seeds = tuple(int(seed) for seed in metadata.get("seeds", []))
    tasks = tuple(metadata.get("tasks", []))
    if tasks != tuple(fx.TASKS) or not seeds:
        raise ValueError("metadata task/seed contract is incomplete")
    expected_keys = fx.expected_fit_keys(seeds)
    if metadata.get("expected_fit_keys") != sorted(expected_keys) or metadata.get("fit_keys") != sorted(expected_keys):
        raise ValueError("metadata fit-key contract is not independently complete")
    if metadata.get("fit_count") != len(expected_keys) or len(expected_keys) != 7 * len(seeds) * len(tasks):
        raise ValueError("metadata fit count does not match seed x task x seven")
    if metadata.get("historical_seeds") != list(fx.HISTORICAL_SEEDS) or metadata.get("patchtst_config") != fx.PATCHTST_CONFIG:
        raise ValueError("metadata historical-seed or PatchTST configuration binding mismatch")
    if not metadata["smoke"]:
        registered_rules = {"seeds": list(fx.FORMAL_SEEDS), "sequence_n": 4096, "burn_in": 512,
                            "warmup_fraction": 0.20, "max_epochs": 40, "patience": 8,
                            "batch_size": None, "dgp_variant": "heteroskedastic"}
        actual_rules = {"seeds": list(seeds), "sequence_n": metadata.get("sequence_n"),
                        "burn_in": metadata.get("burn_in"), "warmup_fraction": metadata.get("warmup_fraction"),
                        "max_epochs": metadata.get("resolved_rules", {}).get("max_epochs"),
                        "patience": metadata.get("resolved_rules", {}).get("patience"),
                        "batch_size": metadata.get("resolved_rules", {}).get("batch_size"),
                        "dgp_variant": metadata.get("dgp_variant")}
        if actual_rules != registered_rules:
            raise ValueError("registered fixture violates its frozen execution contract")
    sequence_results = []
    for seed in seeds:
        sequence = fx.generate_sequence(seed, n=int(metadata["sequence_n"]), burn_in=int(metadata["burn_in"]), variant=metadata["dgp_variant"])
        tasks_by_name = {name: fx.with_splits(fx.build_windows(sequence, *fx.TASKS[name])) for name in fx.TASKS}
        fx.assert_cross_task_labels(sequence, tasks_by_name.values())
        for task_name in tasks:
            expected_task_keys = [key for key in expected_keys if key.startswith(f"seed={seed}/task={task_name}/")]
            if len(expected_task_keys) != 7 or sum("stage=log_oof" in key for key in expected_task_keys) != 5:
                raise ValueError("per-task seven-fit contract failed")
            info, arrays = _pair(output, seed, task_name, expected_task_keys)
            sequence_results.append(_rebuild_task(sequence, tasks_by_name[task_name], info, arrays,
                                                  float(metadata["warmup_fraction"])))
    rho_diagnostic_stop = any(
        fx.rho_gate_status(result["rho_valid_by_lead"])["stop"] or
        fx.rho_gate_status(result["rho_test_by_lead"])["stop"]
        for result in sequence_results
    )
    rho_gate_applied = not metadata["smoke"]
    rho_stop = rho_diagnostic_stop if rho_gate_applied else False
    expected_status = "SMOKE_COMPLETE" if metadata["smoke"] else ("STOP" if rho_stop else "COMPLETE")
    if metadata.get("rho_diagnostic_stop") is not rho_diagnostic_stop or metadata.get("rho_stop") is not rho_stop or metadata.get("status") != expected_status:
        raise ValueError("metadata rho/status evidence does not match independent reconstruction")
    if strict_status.get("rho_diagnostic_stop") is not rho_diagnostic_stop or strict_status.get("rho_gate_applied") != rho_gate_applied or strict_status.get("rho_stop") is not rho_stop or strict_status.get("status") != expected_status:
        raise ValueError("strict fixture status does not match independent reconstruction")
    if not strict_status.get("engineering_complete"):
        raise ValueError("strict fixture status is not engineering-complete")
    runner_manifest_hash = fx.sha256_file(output / "manifest.json")
    summary = {"schema": "round11-cf-cv-summary/v3", "source_runner_manifest_sha256": runner_manifest_hash,
               "registered_fixture": not bool(metadata["smoke"]), "formal_l0": False, "formal": False,
               "non_formal": True, "smoke": bool(metadata["smoke"]),
               "engineering_complete": True, "rho_diagnostic_stop": rho_diagnostic_stop,
               "rho_gate_applied": rho_gate_applied, "rho_stop": rho_stop, "status": expected_status,
               "historical_seeds": list(metadata["historical_seeds"]),
               "fit_count": len(expected_keys), "fit_keys": sorted(expected_keys), "verified": True,
               "tasks": sequence_results}
    write_verification(output, summary, runner_manifest_hash)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = summarize(args.output_dir)
    print(json.dumps({"output_dir": args.output_dir, "verified": result["verified"],
                      "fit_count": result["fit_count"], "formal_l0": result["formal_l0"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
