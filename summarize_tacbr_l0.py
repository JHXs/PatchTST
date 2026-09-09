"""Independent TACBR L0 certificate, gate, and prediction reconstruction."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from tacbr_l0 import (
    C_GRID, TF_METHODS, DGPConfig, _array_sha256, config_to_json, json_dump,
    make_synthetic, non_tf_certificate, prepare_data, tf_certificate,
)


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _reconstruct_and_verify(prepared, method: str, rows: list[dict], bundle_path: Path) -> tuple[list[dict], bool]:
    bundle = np.load(bundle_path, allow_pickle=False)
    required = {"c", "beta", "intercept", "nuisance", "val_prediction", "test_prediction"}
    if not required.issubset(bundle.files):
        raise ValueError(f"bundle missing arrays: {sorted(required - set(bundle.files))}")
    results = []
    ok = True
    for index, row in enumerate(sorted(rows, key=lambda item: item["c"])):
        if row.get("status") != "optimal":
            ok = False
        c = float(bundle["c"][index])
        if abs(c - float(row["c"])) > 1e-14:
            raise ValueError("candidate c order mismatch")
        beta = np.asarray(bundle["beta"][index], dtype=np.float64)
        intercept = float(bundle["intercept"][index])
        x_val = prepared.x_target[prepared.data.validation_slice]
        x_test = prepared.x_target[prepared.data.test_slice]
        val_rebuilt = x_val @ beta + intercept
        test_rebuilt = x_test @ beta + intercept
        val_stored = np.asarray(bundle["val_prediction"][index], dtype=np.float64)
        test_stored = np.asarray(bundle["test_prediction"][index], dtype=np.float64)
        reconstruction_error = max(float(np.max(np.abs(val_rebuilt - val_stored))),
                                   float(np.max(np.abs(test_rebuilt - test_stored))))
        hash_ok = _array_sha256(val_stored, test_stored) == row["prediction_sha256"]
        row_out = dict(row)
        row_out["reconstruction_max_abs"] = reconstruction_error
        val_mse_rebuilt = float(np.mean((val_rebuilt - prepared.y_target[prepared.data.validation_slice]) ** 2))
        test_mse_rebuilt = float(np.mean((test_rebuilt - prepared.y_target[prepared.data.test_slice]) ** 2))
        mse_ok = (abs(val_mse_rebuilt - float(row["validation_mse"])) <= 1e-8 and
                  abs(test_mse_rebuilt - float(row["test_mse"])) <= 1e-8)
        row_out["independent_validation_mse"] = val_mse_rebuilt
        row_out["independent_test_mse"] = test_mse_rebuilt
        if reconstruction_error > 1e-8 or not hash_ok or not mse_ok:
            row_out["status"] = "PREDICTION_RECONSTRUCTION_FAILURE"
            ok = False
        nuisance = np.asarray(bundle["nuisance"][index], dtype=np.float64)
        coefficient_arrays = [beta, np.asarray([intercept]), nuisance]
        if method in TF_METHODS:
            for name in ("tf_task_betas", "tf_task_intercepts", "tf_delta", "tf_delta_intercept"):
                if name not in bundle.files:
                    raise ValueError(f"TF bundle missing {name}")
            coefficient_arrays.extend([np.asarray(bundle["tf_task_betas"][index]),
                                        np.asarray(bundle["tf_task_intercepts"][index]),
                                        np.asarray(bundle["tf_delta"][index]),
                                        np.asarray([float(bundle["tf_delta_intercept"][index])])])
            cert_objective, cert_kkt = tf_certificate(
                prepared, method, c, np.asarray(bundle["tf_task_betas"][index]),
                np.asarray(bundle["tf_task_intercepts"][index]), np.asarray(bundle["tf_delta"][index]),
                float(bundle["tf_delta_intercept"][index]), beta, intercept,
            )
        else:
            cert_objective, cert_kkt = non_tf_certificate(prepared, method, beta, intercept, nuisance, c)
        if _array_sha256(*coefficient_arrays) != row.get("coefficient_sha256"):
            row_out["status"] = "COEFFICIENT_RECONSTRUCTION_FAILURE"
            ok = False
        expected_data_hash = _array_sha256(prepared.data.x_target, prepared.data.y_target,
                                           prepared.data.x_sources, prepared.data.y_sources)
        if expected_data_hash != row.get("data_hash"):
            row_out["status"] = "DATA_GENERATION_HASH_FAILURE"
            ok = False
        objective_rel = abs(cert_objective - float(row["objective"])) / max(1.0, abs(cert_objective))
        row_out["independent_objective"] = cert_objective
        row_out["independent_kkt_inf"] = cert_kkt
        row_out["objective_relative_difference"] = objective_rel
        if not np.isfinite(cert_objective) or cert_kkt > 1e-6 or objective_rel > 1e-8:
            row_out["status"] = "ENGINEERING_FAILURE"
            ok = False
        results.append(row_out)
    return results, ok


def _select(rows: list[dict]) -> dict | None:
    if not rows or any(r.get("status") not in {"optimal"} for r in rows):
        return None
    return min(rows, key=lambda r: (float(r["validation_mse"]), -float(r["c"])))


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else float("nan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    root = args.input_dir
    metadata = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    config = DGPConfig(**metadata["config"])
    rows = _load_jsonl(root / "candidate_results.jsonl")
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["seed"], row["group"], row["label_count"], row["method"])].append(row)
    verified_rows: list[dict] = []
    selected: list[dict] = []
    verification_failures = []
    for (seed, group, label_count, method), method_rows in sorted(grouped.items()):
        data = make_synthetic(seed, group, label_count, config)
        prepared = prepare_data(data, label_count)
        bundle = root / method_rows[0]["bundle"]
        expected_c = np.asarray(metadata["c_grid"], dtype=np.float64)
        actual_c = np.sort(np.asarray([row["c"] for row in method_rows], dtype=np.float64))
        if actual_c.shape != expected_c.shape or not np.allclose(actual_c, np.sort(expected_c), rtol=0, atol=1e-14):
            verification_failures.append((seed, group, label_count, method))
            for row in method_rows:
                row = dict(row)
                row["status"] = "MISSING_OR_DUPLICATE_C_FAILURE"
                verified_rows.append(row)
            continue
        checked, ok = _reconstruct_and_verify(prepared, method, method_rows, bundle)
        verified_rows.extend(checked)
        if not ok:
            verification_failures.append((seed, group, label_count, method))
        winner = _select(checked)
        if winner is not None:
            selected.append(winner)
    selected_path = args.output or (root / "selected_results.jsonl")
    selected_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")

    by_key = {(r["seed"], r["group"], r["label_count"], r["method"]): r for r in selected}
    summaries = {}
    for group in ("V", "F"):
        for label_count in metadata["labels"]:
            methods = {m: [r for (seed, g, n, meth), r in by_key.items() if (g, n, meth) == (group, label_count, m)] for m in metadata["methods"]}
            target_mse = {r["seed"]: r["test_mse"] for r in methods.get("target_only", [])}
            matched_bias = {r["seed"]: r["bias_mse"] for r in methods.get("matched_global", [])}
            metrics = {}
            for method, method_results in methods.items():
                gains = [1.0 - r["test_mse"] / target_mse[r["seed"]] for r in method_results if r["seed"] in target_mse]
                metrics[method] = {
                    "n": len(method_results), "median_test_mse": _median([r["test_mse"] for r in method_results]),
                    "median_gain_vs_target": _median(gains),
                }
            tacbr_rows = methods.get("tacbr", [])
            bias_gains = [1.0 - r["bias_mse"] / matched_bias[r["seed"]] for r in tacbr_rows
                          if r["seed"] in matched_bias and matched_bias[r["seed"]] > 1e-12]
            tacbr_gains = metrics.get("tacbr", {}).get("median_gain_vs_target", float("nan"))
            oracle_gain = metrics.get("tacbr_oracle", {}).get("median_gain_vs_target", float("nan"))
            comparisons = []
            for seed in target_mse:
                if all(seed in {r["seed"] for r in methods.get(m, [])} for m in
                       ("tacbr", "global_full", "matched_global", "matched_global_station_intercept")):
                    vals = {m: next(r["test_mse"] for r in methods[m] if r["seed"] == seed)
                            for m in ("tacbr", "target_only", "global_full", "matched_global", "matched_global_station_intercept")}
                    comparisons.append(all(vals["tacbr"] < vals[m] for m in vals if m != "tacbr"))
            tf_medians = [metrics[m]["median_test_mse"] for m in TF_METHODS if metrics.get(m, {}).get("n")]
            summaries[f"{group}_{label_count}"] = {
                "metrics": metrics,
                "mechanism_bias_gain_median": _median(bias_gains),
                "oracle_gain_median": oracle_gain,
                "tacbr_gain_median": tacbr_gains,
                "tacbr_beats_four_controls_count": int(sum(comparisons)),
                "transfusion_median_test_mse": tf_medians,
                "gate": {
                    "engineering_ok": not verification_failures,
                    "less_labeled_V_mechanism": group == "V" and label_count == 96 and
                        _median(bias_gains) >= 0.5 and np.isfinite(oracle_gain) and oracle_gain > 0 and
                        tacbr_gains >= 0.5 * oracle_gain,
                    "less_labeled_V_competition": group == "V" and label_count == 96 and
                        sum(comparisons) >= 24 and all(metrics.get(m, {}).get("n", 0) == 30 for m in TF_METHODS) and
                        all(metrics.get(m, {}).get("median_test_mse", np.inf) >= metrics.get("tacbr", {}).get("median_test_mse", -np.inf) for m in TF_METHODS),
                    "full_label_V_positive": group == "V" and label_count == 768 and tacbr_gains > 0,
                    "full_label_V_competition": group == "V" and label_count == 768 and all(
                        metrics.get(m, {}).get("median_test_mse", np.inf) >= metrics.get("tacbr", {}).get("median_test_mse", -np.inf)
                        for m in ("global_full", "matched_global", "matched_global_station_intercept", *TF_METHODS)
                    ),
                },
            }
    output = {
        "synthetic": True, "metadata": metadata, "verified_candidate_count": len(verified_rows),
        "selected_count": len(selected), "verification_failures": [list(x) for x in verification_failures],
        "summaries": summaries,
    }
    json_dump(root / "summary.json", output)
    print(json.dumps({"selected_count": len(selected), "verification_failures": len(verification_failures)}, ensure_ascii=False))
    return 1 if verification_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
