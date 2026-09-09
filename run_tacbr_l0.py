"""Formal TACBR L0 runner.

The default command is intentionally expensive and is guarded by a clean-tree
check.  ``--fixture`` is the only mode intended for development on a dirty
worktree and never uses a formal seed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from tacbr_l0 import (
    C_GRID, METHODS, SEEDS, DGPConfig, _array_sha256, config_to_json,
    fit_grid, fit_to_record, json_dump, load_environment_versions, make_synthetic,
    method_bias_mse, prepare_data, save_fit_bundle,
)


def _clean_tree() -> list[str]:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        check=True, capture_output=True, text=True,
    )
    return [line for line in completed.stdout.splitlines() if line.strip()]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixture", action="store_true", help="small synthetic fixture; seed 8702 only")
    parser.add_argument("--seeds", default=None, help="comma-separated formal seeds")
    parser.add_argument("--groups", default="V,F")
    parser.add_argument("--labels", default="96,768")
    parser.add_argument("--methods", default=",".join(METHODS))
    return parser


def _parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.fixture:
        config = DGPConfig(p=16, n_train=48, n_validation=12, n_test=16, target_label_stride=2)
        seeds, groups, labels = [8702], ["V", "F"], [24, 48]
    else:
        dirty = _clean_tree()
        if dirty:
            print("REFUSED: formal TACBR L0 requires a clean tree:")
            print("\n".join(dirty))
            return 2
        config = DGPConfig()
        seeds = _parse_ints(args.seeds) if args.seeds else list(SEEDS)
        groups, labels = args.groups.split(","), _parse_ints(args.labels)
        if tuple(seeds) != SEEDS:
            raise SystemExit("formal run requires exactly seeds 20260909..20260938")
        if groups != ["V", "F"] or labels != [96, 768]:
            raise SystemExit("formal run requires groups V,F and labels 96,768")
    methods = args.methods.split(",")
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise SystemExit(f"unknown methods: {unknown}")
    if not args.fixture and tuple(methods) != METHODS:
        raise SystemExit("formal run requires all nine frozen methods")
    if args.output_dir.exists():
        raise SystemExit(f"REFUSED: output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "models").mkdir()
    metadata = {
        "schema": "tacbr_l0/1",
        "synthetic": True,
        "fixture": args.fixture,
        "config": config_to_json(config),
        "seeds": seeds, "groups": groups, "labels": labels, "methods": methods,
        "c_grid": C_GRID.tolist(), "environment": load_environment_versions(),
        "dependencies": {
            "cvxpy": {"version": "1.7.2", "license": "Apache-2.0"},
            "osqp": {"version": "1.0.4", "license": "Apache-2.0"},
            "numpy": {"license": "BSD-3-Clause"},
            "scipy": {"license": "BSD-3-Clause"},
            "scikit-learn": {"license": "BSD-3-Clause"},
        },
        "source_data_access": {"station_csv": 0, "checkpoint": 0, "historical_metrics": 0},
    }
    json_dump(args.output_dir / "manifest.json", metadata)
    records_path = args.output_dir / "candidate_results.jsonl"
    with records_path.open("w", encoding="utf-8") as records:
        for seed in seeds:
            for group in groups:
                for label_count in labels:
                    data = make_synthetic(seed, group, label_count, config)
                    prepared = prepare_data(data, label_count)
                    data_hash = _array_sha256(data.x_target, data.y_target, data.x_sources, data.y_sources)
                    for method in methods:
                        print(f"fit seed={seed} group={group} labels={label_count} method={method}", flush=True)
                        fits = fit_grid(prepared, method)
                        bundle_name = f"seed_{seed}_{group}_{label_count}_{method}.npz"
                        save_fit_bundle(args.output_dir / "models" / bundle_name, fits, prepared)
                        for fit in fits:
                            record = fit_to_record(prepared, fit)
                            record.update({"synthetic": True, "data_hash": data_hash, "bundle": f"models/{bundle_name}",
                                           "bias_mse": method_bias_mse(prepared, fit, method)})
                            # A failed certificate invalidates the entire grid;
                            # no candidate is silently removed before selection.
                            if (fit.status != "optimal" or not np.isfinite(fit.objective) or
                                    fit.reported_objective is None or
                                    abs(fit.objective - fit.reported_objective) / max(1.0, abs(fit.objective)) > 1e-8 or
                                    fit.kkt_inf > 1e-6):
                                record["status"] = "ENGINEERING_FAILURE"
                            records.write(json.dumps(record, ensure_ascii=False) + "\n")
                            records.flush()
    print(f"wrote synthetic results to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
