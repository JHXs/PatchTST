"""Registered Round 11 synthetic fixture runner.

The registered 42-fit mode is intentionally not a formal-L0 performance
command; ``--smoke`` is the only reduced execution path.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import cf_cv_smearing_fixture as fx


def _fit_model(task: fx.WindowTask, X_raw: np.ndarray, target: np.ndarray, fit_idx: np.ndarray,
               valid_idx: np.ndarray, *, seed: int, max_epochs: int, patience: int,
               device: str, batch_size: int | None) -> tuple[fx.PatchTSTAdapter, dict[str, Any], dict[str, Any]]:
    input_scaler = fx.fit_input_scaler(task, fit_idx)
    target_scaler = fx.fit_target_scaler(target, fit_idx)
    X = ((X_raw - input_scaler["mean"]) / input_scaler["scale"]).astype(np.float32)
    y = fx.transform_target(target, target_scaler)
    adapter = fx.PatchTSTAdapter(task.history, task.horizon, seed, device=device,
                                 max_epochs=max_epochs, patience=patience, batch_size=batch_size)
    adapter.fit(X, y, fit_idx, valid_idx)
    return adapter, input_scaler, target_scaler


def _append_fit(fits: list[dict[str, Any]], ledger: fx.FitLedger, seed: int, task: str,
                stage: str, block: int | None, adapter: fx.PatchTSTAdapter,
                input_scaler: Mapping[str, Any], target_scaler: Mapping[str, Any],
                fit_indices: np.ndarray, valid_indices: np.ndarray) -> None:
    key = ledger.add(seed, task, stage, block)
    fits.append({"key": key, **adapter.fit_info, "input_scaler": dict(input_scaler),
                 "target_scaler": dict(target_scaler),
                 "fit_indices": np.asarray(fit_indices, dtype=np.int64).tolist(),
                 "valid_indices": np.asarray(valid_indices, dtype=np.int64).tolist()})


def _prefix_sets_by_lead(lead_values: np.ndarray, timestamps: np.ndarray, *, horizon: int,
                         purge: int, limit: int | None = None) -> list[np.ndarray]:
    result = []
    for lead in range(horizon):
        ids = np.flatnonzero(lead_values == lead)
        if limit is not None:
            ids = ids[timestamps[ids] <= limit - purge]
        result.append(ids)
    return result


def _evaluate_s_factors(query_c_raw: np.ndarray, query_leads: np.ndarray,
                        fit_c_raw: np.ndarray, fit_leads: np.ndarray,
                        fit_r: np.ndarray, fit_b: np.ndarray, *, horizon: int,
                        allowed_by_lead: list[np.ndarray], name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, dict[str, Any]]]:
    n_query = query_c_raw.shape[0] // horizon
    rem = np.full((n_query, horizon), np.nan, dtype=np.float64)
    cv = np.full_like(rem, np.nan)
    rhs = np.full_like(rem, np.nan)
    diagnostics: dict[int, dict[str, Any]] = {}
    for lead in range(horizon):
        q_ids = np.arange(lead, n_query * horizon, horizon, dtype=np.int64)
        f_ids = np.asarray(allowed_by_lead[lead], dtype=np.int64)
        if f_ids.size < 2:
            raise ValueError(f"STOP: {name} lead {lead} has fewer than two legal S fit rows")
        prefix = fx.compute_prefix_control_variate(fit_r[f_ids], fit_b[f_ids])
        H = fx.build_s_prefix_operator(query_c_raw[q_ids], fit_c_raw[f_ids],
                                       np.zeros(q_ids.size, dtype=np.int64),
                                       np.zeros(f_ids.size, dtype=np.int64),
                                       [np.arange(f_ids.size, dtype=np.int64)] * q_ids.size,
                                       name=f"{name}_h{lead}")
        factor_rem = H.apply(prefix["u"])
        factor_cv = H.apply(prefix["u"] - prefix["lambda"] * prefix["vtilde"])
        identity_rhs = -prefix["lambda"] * H.apply(prefix["vtilde"])
        np.testing.assert_allclose(factor_cv - factor_rem, identity_rhs, rtol=1e-10, atol=1e-12)
        rem[:, lead] = factor_rem
        cv[:, lead] = factor_cv
        rhs[:, lead] = identity_rhs
        diagnostics[lead] = {"shape": list(H.shape), "lambda": prefix["lambda"],
                             "var_v": prefix["var_v"], "cov_uv": prefix["cov_uv"],
                             "variance_ratio": prefix["variance_ratio"],
                             "mean_v0": prefix["mean_v0"],
                             "rho": float(np.linalg.norm(H.apply(prefix["vtilde"])) /
                                           (np.linalg.norm(H.apply(prefix["u"])) + 1e-12))}
    return rem, cv, rhs, diagnostics


def _task_fit_keys(seed: int, task: str) -> list[str]:
    return ([f"seed={seed}/task={task}/stage=raw_full"] +
            [f"seed={seed}/task={task}/stage=log_oof/block={i}" for i in range(5)] +
            [f"seed={seed}/task={task}/stage=log_full"])


def run_one_task(sequence: fx.SequenceDGP, task: fx.WindowTask, *, seed: int, output_dir: Path,
                 ledger: fx.FitLedger, max_epochs: int, patience: int, device: str,
                 batch_size: int | None, warmup_fraction: float) -> tuple[dict[str, Any], Path, Path]:
    task_started = time.monotonic()
    task = fx.with_splits(task)
    plan = fx.build_oof_plan(task, warmup_fraction=warmup_fraction)
    train = np.asarray(task.split_indices["train"], dtype=np.int64)
    valid = np.asarray(task.split_indices["valid"], dtype=np.int64)
    test = np.asarray(task.split_indices["test"], dtype=np.int64)
    purge = task.history + task.horizon + 1
    X_raw = task.X_raw.astype(np.float64)
    raw_target = task.y_raw.astype(np.float64)
    log_target = task.z_target[:, None, :].astype(np.float64)
    full_inner_fit, full_inner_valid, full_excluded = fx.split_train_inner(train, task.target_timestamps, purge)
    fits: list[dict[str, Any]] = []

    raw_adapter, raw_in_scaler, raw_y_scaler = _fit_model(task, X_raw, raw_target, full_inner_fit, full_inner_valid,
                                                            seed=seed, max_epochs=max_epochs, patience=patience,
                                                            device=device, batch_size=batch_size)
    _append_fit(fits, ledger, seed, task.name, "raw_full", None, raw_adapter, raw_in_scaler, raw_y_scaler,
                full_inner_fit, full_inner_valid)
    raw_valid = fx.inverse_target(raw_adapter.predict(((X_raw - raw_in_scaler["mean"]) / raw_in_scaler["scale"]).astype(np.float32))[valid], raw_y_scaler)[:, 0, :]
    raw_test = fx.inverse_target(raw_adapter.predict(((X_raw - raw_in_scaler["mean"]) / raw_in_scaler["scale"]).astype(np.float32))[test], raw_y_scaler)[:, 0, :]

    oof_rows = np.concatenate([block.query_indices for block in plan.blocks])
    m_prediction_rows = np.full((len(oof_rows), task.horizon), np.nan, dtype=np.float64)
    for block in plan.blocks:
        if block.inner_fit_indices.size == 0:
            raise RuntimeError(f"STOP: OOF block {block.block} has no legal train-inner fit")
        adapter, in_scaler, y_scaler = _fit_model(task, X_raw, log_target, block.inner_fit_indices,
                                                   block.inner_valid_indices, seed=seed, max_epochs=max_epochs,
                                                   patience=patience, device=device, batch_size=batch_size)
        _append_fit(fits, ledger, seed, task.name, "log_oof", block.block, adapter, in_scaler, y_scaler,
                    block.inner_fit_indices, block.inner_valid_indices)
        query_pos = np.concatenate([np.flatnonzero(oof_rows == row) for row in block.query_indices])
        pred = adapter.predict(((X_raw - in_scaler["mean"]) / in_scaler["scale"]).astype(np.float32)[block.query_indices])
        m_prediction_rows[query_pos] = fx.inverse_target(pred, y_scaler)[:, 0, :]
    if not np.isfinite(m_prediction_rows).all():
        raise RuntimeError("STOP: log OOF predictions do not cover all five query blocks")

    log_adapter, log_in_scaler, log_y_scaler = _fit_model(task, X_raw, log_target, full_inner_fit, full_inner_valid,
                                                           seed=seed, max_epochs=max_epochs, patience=patience,
                                                           device=device, batch_size=batch_size)
    _append_fit(fits, ledger, seed, task.name, "log_full", None, log_adapter, log_in_scaler, log_y_scaler,
                full_inner_fit, full_inner_valid)
    X_log_full = ((X_raw - log_in_scaler["mean"]) / log_in_scaler["scale"]).astype(np.float32)
    log_valid = fx.inverse_target(log_adapter.predict(X_log_full[valid]), log_y_scaler)[:, 0, :]
    log_test = fx.inverse_target(log_adapter.predict(X_log_full[test]), log_y_scaler)[:, 0, :]

    # M is sorted by (absolute target timestamp, lead).  It is the only pool used by B.
    m_c_row_raw, m_leads_row, m_ts_row = fx.build_conditions(task, m_prediction_rows, row_indices=oof_rows)
    m_block_row = np.repeat(np.concatenate([np.full(len(block.query_indices), block.block) for block in plan.blocks]), task.horizon)
    sort = np.lexsort((m_leads_row, m_ts_row))
    m_rows = np.repeat(oof_rows, task.horizon)[sort]
    m_leads = m_leads_row[sort]
    m_ts = m_ts_row[sort]
    m_blocks = m_block_row[sort]
    m_c_raw = m_c_row_raw[sort]
    m_pred = m_prediction_rows.ravel()[sort]
    m_r = log_target[oof_rows].ravel()[sort] - m_pred
    m_u = np.exp(m_r)
    s_m_indices, b_support = fx.select_s_pool(m_ts, m_leads, purge)
    if s_m_indices.size == 0:
        raise RuntimeError("STOP: no legal S rows")
    B = fx.build_prefix_scaled_operator(m_c_raw[s_m_indices], m_c_raw, b_support, name="B")
    s_c_raw, s_leads, s_ts = m_c_raw[s_m_indices], m_leads[s_m_indices], m_ts[s_m_indices]
    s_blocks, s_r, s_u = m_blocks[s_m_indices], m_r[s_m_indices], m_u[s_m_indices]
    s_b = B.apply(m_r)
    s_v0 = s_r - s_b
    s_vtilde = np.full_like(s_v0, np.nan)
    final_lambda = np.empty(task.horizon, dtype=np.float64)
    var_v, cov_uv, var_ratio, mean_v0 = [], [], [], []
    for lead in range(task.horizon):
        ids = np.flatnonzero(s_leads == lead)
        prefix = fx.compute_prefix_control_variate(s_r[ids], s_b[ids])
        s_vtilde[ids] = prefix["vtilde"]
        final_lambda[lead] = prefix["lambda"]
        var_v.append(prefix["var_v"]); cov_uv.append(prefix["cov_uv"])
        var_ratio.append(prefix["variance_ratio"]); mean_v0.append(prefix["mean_v0"])

    valid_c_raw, valid_leads, _ = fx.build_conditions(task, log_valid, row_indices=valid)
    test_c_raw, test_leads, _ = fx.build_conditions(task, log_test, row_indices=test)
    all_s_by_lead = [np.flatnonzero(s_leads == lead) for lead in range(task.horizon)]
    valid_rem, valid_cv, valid_rhs, valid_diag = _evaluate_s_factors(valid_c_raw, valid_leads, s_c_raw, s_leads,
                                                                       s_r, s_b, horizon=task.horizon,
                                                                       allowed_by_lead=all_s_by_lead, name="H_valid")
    test_rem, test_cv, test_rhs, test_diag = _evaluate_s_factors(test_c_raw, test_leads, s_c_raw, s_leads,
                                                                  s_r, s_b, horizon=task.horizon,
                                                                  allowed_by_lead=all_s_by_lead, name="H_test")

    # S-OOF diagnostics have their own earlier-S prefix and its own centered v/lambda.
    s_oof_rem = np.full(s_r.size, np.nan); s_oof_cv = np.full(s_r.size, np.nan)
    s_oof_rhs = np.full(s_r.size, np.nan); s_oof_lambda = np.full(s_r.size, np.nan)
    s_oof_excluded: list[int] = []
    for block in plan.blocks:
        q = np.flatnonzero(s_blocks == block.block)
        if q.size == 0:
            s_oof_excluded.append(block.block); continue
        allowed = [np.flatnonzero((s_leads == lead) & (s_blocks < block.block) &
                                  (s_ts <= block.query_min_timestamp - purge)) for lead in range(task.horizon)]
        if any(ids.size < 2 for ids in allowed):
            s_oof_excluded.append(block.block); continue
        for lead in range(task.horizon):
            q_ids = q[s_leads[q] == lead]
            f_ids = allowed[lead]
            prefix = fx.compute_prefix_control_variate(s_r[f_ids], s_b[f_ids])
            H = fx.build_s_prefix_operator(s_c_raw[q_ids], s_c_raw[f_ids], np.zeros(q_ids.size, dtype=np.int64),
                                            np.zeros(f_ids.size, dtype=np.int64),
                                            [np.arange(f_ids.size, dtype=np.int64)] * q_ids.size,
                                            name=f"H_s_oof_{block.block}_h{lead}")
            s_oof_rem[q_ids] = H.apply(prefix["u"])
            s_oof_cv[q_ids] = H.apply(prefix["u"] - prefix["lambda"] * prefix["vtilde"])
            s_oof_rhs[q_ids] = -prefix["lambda"] * H.apply(prefix["vtilde"])
            s_oof_lambda[q_ids] = prefix["lambda"]
            np.testing.assert_allclose(s_oof_cv[q_ids] - s_oof_rem[q_ids], s_oof_rhs[q_ids], rtol=1e-10, atol=1e-12)

    global_factor = np.asarray([np.mean(s_u[all_s_by_lead[lead]]) for lead in range(task.horizon)])
    hpc = np.asarray([fx.hpc_factor(s_r[all_s_by_lead[lead]]) for lead in range(task.horizon)])
    valid_predictions = {
        "raw_mse_base": raw_valid, "naive_log": fx.apply_raw_boundary(log_valid, np.ones_like(log_valid)),
        "global_smearing": fx.apply_raw_boundary(log_valid, np.broadcast_to(global_factor, log_valid.shape)),
        "hpc": fx.apply_raw_boundary(log_valid, np.broadcast_to(hpc, log_valid.shape)),
        "strongest_existing_conditional_smearing": fx.apply_raw_boundary(log_valid, valid_rem),
        "cf_cv_cs_removal": fx.apply_raw_boundary(log_valid, valid_rem),
        "cf_cv_cs": fx.apply_raw_boundary(log_valid, valid_cv),
    }
    test_predictions = {
        "raw_mse_base": raw_test, "naive_log": fx.apply_raw_boundary(log_test, np.ones_like(log_test)),
        "global_smearing": fx.apply_raw_boundary(log_test, np.broadcast_to(global_factor, log_test.shape)),
        "hpc": fx.apply_raw_boundary(log_test, np.broadcast_to(hpc, log_test.shape)),
        "strongest_existing_conditional_smearing": fx.apply_raw_boundary(log_test, test_rem),
        "cf_cv_cs_removal": fx.apply_raw_boundary(log_test, test_rem),
        "cf_cv_cs": fx.apply_raw_boundary(log_test, test_cv),
    }
    npz_path = output_dir / f"task_{seed}_{fx.safe_task_name(task.name)}.npz"
    json_path = output_dir / f"task_{seed}_{fx.safe_task_name(task.name)}.json"
    arrays: dict[str, np.ndarray] = {
        "y_valid": raw_target[valid, 0, :], "y_test": raw_target[test, 0, :],
        "valid_timestamps": task.target_timestamps[valid], "test_timestamps": task.target_timestamps[test],
        "m_rows": m_rows, "m_leads": m_leads, "m_timestamps": m_ts, "m_blocks": m_blocks,
        "m_condition_raw": m_c_raw, "m_prediction_rows": m_prediction_rows,
        "m_query_rows": oof_rows, "m_r": m_r, "m_u": m_u,
        "s_m_indices": s_m_indices, "s_leads": s_leads, "s_timestamps": s_ts, "s_blocks": s_blocks,
        "s_c_raw": s_c_raw, "s_r": s_r, "s_u": s_u, "s_b": s_b, "s_v0": s_v0, "s_vtilde": s_vtilde,
        "s_oof_rem": s_oof_rem, "s_oof_cv": s_oof_cv, "s_oof_rhs": s_oof_rhs, "s_oof_lambda": s_oof_lambda,
        "valid_log_m": log_valid, "test_log_m": log_test,
        "valid_condition_raw": valid_c_raw, "test_condition_raw": test_c_raw,
        "valid_factor_removal": valid_rem, "test_factor_removal": test_rem,
        "valid_factor_cv": valid_cv, "test_factor_cv": test_cv,
        "lambda_by_lead": final_lambda,
    }
    for method in fx.METHODS:
        arrays[f"valid_{method}"] = valid_predictions[method]
        arrays[f"test_{method}"] = test_predictions[method]
    fx.atomic_npz(npz_path, **arrays)
    info = fx.task_metadata(task, plan)
    info.update({
        "schema": "round11-cf-cv-task/v2", "unit_key": f"seed={seed}/task={task.name}", "seed": seed,
        "dgp_variant": sequence.variant, "fit_keys": [fit["key"] for fit in fits], "fit_records": fits,
        "full_inner_excluded": full_excluded.tolist(), "m_count": int(m_r.size), "s_count": int(s_r.size),
        "s_m_indices": s_m_indices.tolist(), "b_support_indices": [ids.tolist() for ids in b_support],
        "B_shape": list(B.shape), "H_dimensions": {"valid": [valid_diag[i]["shape"] for i in range(task.horizon)],
                                                       "test": [test_diag[i]["shape"] for i in range(task.horizon)]},
        "H_diagnostics": {"valid": valid_diag, "test": test_diag},
        "S_oof_excluded_blocks": s_oof_excluded, "bandwidth": fx.SMOOTHER_BANDWIDTH,
        "smoother_max_support": fx.SMOOTHER_MAX_SUPPORT,
        "lambda_by_lead": final_lambda.tolist(), "var_v_by_lead": var_v, "cov_u_v_by_lead": cov_uv,
        "variance_ratio_by_lead": var_ratio, "mean_v0_by_lead": mean_v0,
        "rho_valid_by_lead": [valid_diag[i]["rho"] for i in range(task.horizon)],
        "rho_test_by_lead": [test_diag[i]["rho"] for i in range(task.horizon)],
        "rho_gate": {"valid": fx.rho_gate_status([valid_diag[i]["rho"] for i in range(task.horizon)]),
                      "test": fx.rho_gate_status([test_diag[i]["rho"] for i in range(task.horizon)])},
        "metrics_run_only": {split: {method: fx.metric_rows(truth, pred[method]) for method in fx.METHODS}
                              for split, truth, pred in (("valid", raw_target[valid, 0, :], valid_predictions),
                                                         ("test", raw_target[test, 0, :], test_predictions))},
    })
    info["npz_sha256"] = fx.sha256_file(npz_path)
    info["elapsed_seconds"] = float(time.monotonic() - task_started)
    return info, json_path, npz_path


def validate_task_pair(json_path: Path, npz_path: Path, *, expected_unit: str, expected_fit_keys: list[str]) -> dict[str, Any]:
    if not json_path.exists() or not npz_path.exists():
        raise ValueError("task pair is incomplete")
    info = json.loads(json_path.read_text(encoding="utf-8"))
    if info.get("schema") != "round11-cf-cv-task/v2" or info.get("unit_key") != expected_unit:
        raise ValueError("task sidecar schema/unit key mismatch")
    if info.get("fit_keys") != expected_fit_keys:
        raise ValueError("task sidecar fit keys mismatch")
    if not isinstance(info.get("elapsed_seconds"), (int, float)) or info["elapsed_seconds"] < 0.0:
        raise ValueError("task sidecar elapsed_seconds is missing or invalid")
    if info.get("npz_sha256") != fx.sha256_file(npz_path):
        raise ValueError("task NPZ hash mismatch; refusing resume")
    with np.load(npz_path, allow_pickle=False) as arrays:
        required = {"y_valid", "y_test", "m_r", "s_r", "valid_log_m", "test_log_m"}
        if not required.issubset(arrays.files):
            raise ValueError("task NPZ schema is incomplete")
    return info


def run_fixture(output_dir: str | Path = "experiments/results/round11_cf_cv_fixture", *,
                seeds: tuple[int, ...] = fx.FORMAL_SEEDS, smoke: bool = False,
                max_epochs: int | None = None, patience: int | None = None,
                device: str = "cpu", batch_size: int | None = None,
                dgp_variant: str = "heteroskedastic", fail_after: int | None = None,
                sequence_n: int | None = None, burn_in: int | None = None) -> dict[str, Any]:
    started = time.monotonic()
    output = Path(output_dir)
    if dgp_variant != "heteroskedastic":
        raise ValueError("runner training only permits the heteroskedastic DGP; negative controls are unit-test only")
    if smoke:
        if not output.name.endswith("_smoke") or len(seeds) != 1:
            raise ValueError("smoke requires one seed and an output directory ending in _smoke")
    else:
        if sequence_n not in (None, 4096) or burn_in not in (None, 512):
            raise ValueError("registered fixture requires N=4096 and burn-in=512")
    seeds = fx.validate_seed_set(seeds, formal=not smoke)
    max_epochs = int(max_epochs if max_epochs is not None else (1 if smoke else 40))
    patience = int(patience if patience is not None else (1 if smoke else 8))
    sequence_n = int(sequence_n if sequence_n is not None else (2048 if smoke else 4096))
    burn_in = int(burn_in if burn_in is not None else (256 if smoke else 512))
    warmup_fraction = 0.35 if smoke else 0.20
    if not smoke and (max_epochs != 40 or patience != 8 or batch_size is not None):
        raise ValueError("registered fixture fixes max_epochs=40, patience=8, and batch_size=None")
    mode = {"registered_fixture": not smoke, "formal_l0": False, "formal": False,
            "non_formal": True, "smoke": bool(smoke)}
    resolved_rules = {"max_epochs": max_epochs, "patience": patience, "batch_size": batch_size,
                      "device": device, "sequence_n": sequence_n, "burn_in": burn_in,
                      "warmup_fraction": warmup_fraction, "dgp_variant": dgp_variant}
    config = {**mode, "seeds": list(seeds), "dgp_variant": dgp_variant,
              "sequence_n": sequence_n, "burn_in": burn_in, "warmup_fraction": warmup_fraction,
              "max_epochs": max_epochs, "patience": patience, "device": device,
              "batch_size": batch_size, "patchtst_config": fx.PATCHTST_CONFIG,
              "resolved_rules": resolved_rules}
    manifest_path = output / "manifest.json"
    previous_wall_time = 0.0
    if manifest_path.exists():
        manifest = fx.validate_manifest(output)
        if any(manifest.get("config", {}).get(key) != value for key, value in config.items()):
            raise ValueError("resume configuration differs from manifest")
        preflight = manifest.get("preflight")
        if not isinstance(preflight, dict) or not preflight.get("commit") or not preflight.get("branch"):
            raise ValueError("manifest is missing its preflight binding")
        current = fx.git_binding(Path.cwd(), exclude_path=output)
        if current["commit"] != preflight["commit"] or current["branch"] != preflight["branch"]:
            raise RuntimeError("resume source commit/branch differs from preflight binding")
        if not current["clean_tree"]:
            raise RuntimeError("resume source has dirty paths outside the registered output directory")
        previous_metadata_path = output / "metadata.json"
        if previous_metadata_path.exists():
            previous_metadata = json.loads(previous_metadata_path.read_text(encoding="utf-8"))
            previous_wall_time = float(previous_metadata.get("wall_time", 0.0))
    else:
        if output.exists():
            raise ValueError("registered fixture first run requires an output directory that does not exist")
        current = fx.git_binding(Path.cwd(), exclude_path=output)
        if not current["clean_tree"]:
            raise RuntimeError("registered fixture requires a clean source tree outside its output directory")
        if not smoke and "round11" not in current["branch"]:
            raise RuntimeError("registered fixture requires a round11 branch")
        output.mkdir(parents=True, exist_ok=False)
        manifest = {"schema": "round11-cf-cv-fixture/v3", "config": config,
                    "preflight": {"commit": current["commit"], "branch": current["branch"],
                                  "clean_tree": current["clean_tree"]}, "artifacts": {}}
        fx.atomic_json(manifest_path, manifest)
    progress_path = output / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {"completed": []}
    ledger = fx.FitLedger()
    infos: list[dict[str, Any]] = []
    sequences = {seed: fx.generate_sequence(seed, n=sequence_n, burn_in=burn_in, variant=dgp_variant) for seed in seeds}
    for seed in seeds:
        tasks = [fx.with_splits(fx.build_windows(sequences[seed], *fx.TASKS[name])) for name in fx.TASKS]
        fx.assert_cross_task_labels(sequences[seed], tasks)
        for task in tasks:
            unit = f"seed={seed}/task={task.name}"
            jp = output / f"task_{seed}_{fx.safe_task_name(task.name)}.json"
            npz = output / f"task_{seed}_{fx.safe_task_name(task.name)}.npz"
            if jp.exists() or npz.exists():
                info = validate_task_pair(jp, npz, expected_unit=unit, expected_fit_keys=_task_fit_keys(seed, task.name))
                infos.append(info)
                manifest["artifacts"][jp.name] = fx.sha256_file(jp)
                manifest["artifacts"][npz.name] = fx.sha256_file(npz)
                if unit not in progress.get("completed", []):
                    progress.setdefault("completed", []).append(unit)
                continue
            info, jp, npz = run_one_task(sequences[seed], task, seed=seed, output_dir=output, ledger=ledger,
                                         max_epochs=max_epochs, patience=patience, device=device,
                                         batch_size=batch_size, warmup_fraction=warmup_fraction)
            if fail_after == 1 and not progress.get("completed"):
                # Real crash window: NPZ exists, sidecar is complete, manifest/progress are not updated.
                info["crash_window_test_marker"] = True
                fx.atomic_json(jp, info)
                raise RuntimeError("injected crash after sidecar/NPZ commit")
            fx.atomic_json(jp, info)
            validate_task_pair(jp, npz, expected_unit=unit, expected_fit_keys=_task_fit_keys(seed, task.name))
            infos.append(info)
            manifest["artifacts"][jp.name] = fx.sha256_file(jp)
            manifest["artifacts"][npz.name] = fx.sha256_file(npz)
            progress.setdefault("completed", []).append(unit)
            fx.atomic_json(manifest_path, manifest); fx.atomic_json(progress_path, progress)
    expected_keys = fx.expected_fit_keys(seeds)
    all_keys = sorted(key for info in infos for key in info["fit_keys"])
    if all_keys != sorted(expected_keys):
        raise AssertionError("fit key contract is incomplete")
    progress["fit_count"] = len(all_keys); progress["completed"] = sorted(set(progress.get("completed", [])))
    fx.atomic_json(progress_path, progress)
    rho_diagnostic_stop = any(info.get("rho_gate", {}).get(split, {}).get("stop", False)
                              for info in infos for split in ("valid", "test"))
    rho_gate_applied = not smoke
    rho_stop = rho_diagnostic_stop if rho_gate_applied else False
    if smoke:
        status = "SMOKE_COMPLETE"
    else:
        status = "STOP" if rho_stop else "COMPLETE"
    final_binding = fx.git_binding(Path.cwd(), exclude_path=output)
    invocation_wall_time = float(time.monotonic() - started)
    task_elapsed_sum = float(sum(float(info.get("elapsed_seconds", 0.0)) for info in infos))
    metadata = {"schema": "round11-cf-cv-fixture/v3", **mode,
                "status": status, "engineering_complete": True,
                "rho_diagnostic_stop": rho_diagnostic_stop, "rho_gate_applied": rho_gate_applied,
                "rho_stop": rho_stop,
                "dgp_variant": dgp_variant, "seeds": list(seeds),
                "historical_seeds": list(fx.HISTORICAL_SEEDS), "tasks": list(fx.TASKS),
                "sequence_n": sequence_n, "burn_in": burn_in, "warmup_fraction": warmup_fraction,
                "fit_count": len(all_keys), "fit_keys": all_keys, "expected_fit_keys": sorted(expected_keys),
                "patchtst_config": fx.PATCHTST_CONFIG, "preflight": manifest["preflight"],
                "resolved_rules": resolved_rules, "binding": final_binding,
                "task_elapsed_seconds_sum": task_elapsed_sum,
                "invocation_wall_time_seconds": invocation_wall_time,
                "wall_time": previous_wall_time + invocation_wall_time,
                "ru_maxrss": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                "valid_test_not_used_for_selection": True, "old_h2_kappa_interface": "rejected"}
    fx.atomic_json(output / "metadata.json", metadata)
    strict_status = {"schema": "round11-cf-cv-strict-status/v1", **mode,
                     "engineering_complete": True, "rho_diagnostic_stop": rho_diagnostic_stop,
                     "rho_gate_applied": rho_gate_applied, "rho_stop": rho_stop,
                     "status": status, "fit_count": len(all_keys)}
    fx.atomic_json(output / "strict_fixture_status.json", strict_status)
    manifest["artifacts"]["metadata.json"] = fx.sha256_file(output / "metadata.json")
    manifest["artifacts"]["progress.json"] = fx.sha256_file(progress_path)
    manifest["artifacts"]["strict_fixture_status.json"] = fx.sha256_file(output / "strict_fixture_status.json")
    fx.atomic_json(manifest_path, manifest)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="experiments/results/round11_cf_cv_fixture")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seeds", default="4001,4002,4003")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dgp-variant", choices=("heteroskedastic",), default="heteroskedastic")
    parser.add_argument("--smoke-n", type=int, default=None)
    parser.add_argument("--smoke-burn-in", type=int, default=None)
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    result = run_fixture(args.output_dir, seeds=seeds, smoke=args.smoke, max_epochs=args.max_epochs,
                         patience=args.patience, device=args.device, batch_size=args.batch_size,
                         dgp_variant=args.dgp_variant, sequence_n=args.smoke_n, burn_in=args.smoke_burn_in)
    print(json.dumps({"output_dir": args.output_dir, "status": result["status"],
                      "formal_l0": result["formal_l0"], "fit_count": result["fit_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
