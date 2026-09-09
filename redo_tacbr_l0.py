"""Redo the OSQP-stalled formal configs with the SCS-fallback fix.

Reads an already-written formal output directory, identifies every
(seed, group, label_count, method) config that contains at least one
ENGINEERING_FAILURE row, refits only those configs with the fixed solver
path, and rewrites their candidate rows and bundles in place.  Every refit
config must end up fully optimal with KKT <= 1e-6, otherwise the script
fails loudly.  A ``redo_log.json`` records exactly which configs and c values
were refit and why.  This is an engineering fix to solver convergence, not a
change to the DGP, thresholds, methods, or gates.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from tacbr_l0 import (
    C_GRID, DGPConfig, _array_sha256, fit_grid, fit_to_record, json_dump,
    make_synthetic, method_bias_mse, prepare_data, save_fit_bundle,
)


def main(argv: list[str] | None = None) -> int:
    root = Path(argv[1] if argv and len(argv) > 1 else "experiments/results/tacbr_l0/l0_formal")
    metadata = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    config = DGPConfig(**metadata["config"])
    rows = [json.loads(line) for line in (root / "candidate_results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    failed_keys = {}
    for row in rows:
        if row.get("status") == "ENGINEERING_FAILURE":
            failed_keys[(row["seed"], row["group"], row["label_count"], row["method"])] = None
    if not failed_keys:
        print("no ENGINEERING_FAILURE configs to redo")
        return 0

    grouped: dict[tuple, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[(row["seed"], row["group"], row["label_count"], row["method"])].append(index)

    redo_log = []
    for key in sorted(failed_keys):
        seed, group, label_count, method = key
        indices = grouped[key]
        if len(indices) != len(C_GRID):
            raise SystemExit(f"config {key} has {len(indices)} rows, expected {len(C_GRID)}")
        data = make_synthetic(seed, group, label_count, config)
        prepared = prepare_data(data, label_count)
        fits = fit_grid(prepared, method)
        bad = [f for f in fits if f.status != "optimal" or f.kkt_inf > 1e-6 or not _isfinite_objective(f)]
        if bad:
            raise SystemExit(f"refit still failed for {key}: {[(f.c, f.status, f.kkt_inf) for f in bad]}")
        bundle_name = f"seed_{seed}_{group}_{label_count}_{method}.npz"
        save_fit_bundle(root / "models" / bundle_name, fits, prepared)
        data_hash = _array_sha256(data.x_target, data.y_target, data.x_sources, data.y_sources)
        for index, fit in zip(indices, fits):
            record = fit_to_record(prepared, fit)
            record.update({"synthetic": True, "data_hash": data_hash, "bundle": f"models/{bundle_name}",
                           "bias_mse": method_bias_mse(prepared, fit, method)})
            if (fit.status != "optimal" or not _isfinite_objective(fit)
                    or fit.reported_objective is None
                    or abs(fit.objective - fit.reported_objective) / max(1.0, abs(fit.objective)) > 1e-8
                    or fit.kkt_inf > 1e-6):
                record["status"] = "ENGINEERING_FAILURE"
            rows[index] = record
        redo_log.append({
            "seed": seed, "group": group, "label_count": label_count, "method": method,
            "reason": "OSQP max_iter exhausted at formal scale; refit with SCS fallback",
            "c": [float(f.c) for f in fits],
            "status_after": [f.status for f in fits],
            "kkt_after": [float(f.kkt_inf) for f in fits],
        })
        print(f"redid {key}: {len(fits)} c all optimal", flush=True)

    (root / "candidate_results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    json_dump(root / "redo_log.json", {"redone_configs": len(redo_log), "entries": redo_log})
    print(f"redid {len(redo_log)} configs; redo_log.json written")
    return 0


def _isfinite_objective(fit) -> bool:
    import numpy as np
    return bool(np.isfinite(fit.objective))


if __name__ == "__main__":
    sys.exit(main(sys.argv))
