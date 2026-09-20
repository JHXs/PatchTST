"""Run the frozen baseline-comparison matrix with resumable artifacts.

P1 smoke example (the only run authorized during implementation)::

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
        run_baseline_comparison.py --smoke --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import torch

import run_beijing_leakfree_coverage as beijing
import run_cross_city_generalization as guangzhou
import run_st_patchtst_ablation as legacy
from baseline_models import CAPACITY_TIERS, NEURAL_ARMS, build_baseline_model
from baseline_traditional import TRADITIONAL_ARMS
from baseline_training import evaluate_traditional_baseline, train_baseline


TASK_GRID = tuple(beijing.TASK_GRID)
HEADLINE_TASKS = ((24, 1), (168, 6))
BEIJING_GRID_SEEDS = (2047, 2048, 2049)
BEIJING_HEADLINE_SEEDS = (2047, 2048, 2049, 2050, 2051)
GUANGZHOU_SEEDS = (7001, 7002, 7003)
GUANGZHOU_ARMS = (
    "plain_patchtst_all",
    "plain_patchtst_top5",
    "multi_gru",
    "center_gru",
)
GUANGZHOU_TRADITIONAL_ARMS = (
    "trad_persistence",
    "trad_daily_naive",
    "trad_climatology",
    "trad_spatial_linear",
)
IMPLEMENTATION_FILES = (
    "baseline_models.py",
    "baseline_traditional.py",
    "baseline_training.py",
    "run_baseline_comparison.py",
    "summarize_baselines.py",
    "run_st_patchtst_ablation.py",
    "run_beijing_leakfree_coverage.py",
    "run_cross_city_generalization.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", choices=("beijing", "guangzhou"), default="beijing")
    parser.add_argument("--configs", default=None, help="Comma-separated LxH tasks")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-root", default="experiments/results/baselines")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_state() -> dict:
    hashes = {
        path: sha256_file(path)
        for path in IMPLEMENTATION_FILES
        if Path(path).is_file()
    }
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"commit": commit, "hashes": hashes, "git_status": status}


def parse_tasks(value: str | None, defaults: tuple[tuple[int, int], ...]):
    if value is None:
        return defaults
    tasks = []
    for item in value.split(","):
        history, horizon = item.lower().split("x")
        tasks.append((int(history), int(horizon)))
    return tuple(tasks)


def config_for(history: int, horizon: int, args: argparse.Namespace):
    config = beijing.config_for(history, horizon)
    if args.epochs is not None:
        config = replace(config, epochs=args.epochs)
    if args.patience is not None:
        config = replace(config, patience=args.patience)
    return replace(
        config,
        evaluation_split="test",
        initialize_from_degraded=False,
        freeze_backbone=False,
    )


def _prepare_guangzhou(center_station_id: int, config):
    frame, opened = guangzhou.load_center_candidate_frame(center_station_id)
    guangzhou_config = guangzhou.RunConfig(
        history=config.history,
        horizon=config.horizon,
        batch_size=config.batch_size,
        epochs=config.epochs,
        patience=config.patience,
    )
    source_datasets, metadata = guangzhou.prepare_shared_data(
        frame, center_station_id, guangzhou_config
    )
    datasets = {
        "train": source_datasets["fit"],
        "valid": source_datasets["val"],
        "test": source_datasets["confirm"],
    }
    for source_name, target_name in (("fit", "train"), ("val", "valid"), ("confirm", "test")):
        datasets[target_name].sample_indices = datasets[target_name].indices
    metadata = {
        **metadata,
        "center_station_id": center_station_id,
        "start_time": str(frame.index.min()),
        "opened_station_ids": list(opened),
        "split_sizes": {name: len(dataset) for name, dataset in datasets.items()},
        "station_correlations": metadata["candidate_correlations"],
    }
    return datasets, metadata


def prepare_data(city: str, config, center_station_id: int | None = None):
    if city == "beijing":
        return beijing.prepare_datasets_leakfree(config)
    if center_station_id is None:
        raise ValueError("Guangzhou requires a center station")
    return _prepare_guangzhou(center_station_id, config)


def expected_prediction_names(neural_arms, traditional_arms, capacities, seeds):
    names = {
        f"{arm}_{capacity}_seed{seed}.npz"
        for arm in neural_arms
        for capacity in capacities
        for seed in seeds
    }
    names.update(f"{arm}.npz" for arm in traditional_arms)
    return names


def is_complete(output_dir: Path, expected_names: set[str]) -> bool:
    raw_path = output_dir / "raw_metrics.csv"
    if not raw_path.is_file():
        return False
    actual = {path.name for path in (output_dir / "predictions").glob("*.npz")}
    return expected_names <= actual


def save_rows(rows: list[dict], output_dir: Path) -> None:
    pd.DataFrame(rows).to_csv(output_dir / "raw_metrics.csv", index=False)


def run_one_dataset(
    config,
    datasets: dict,
    metadata: dict,
    output_dir: Path,
    neural_arms: tuple[str, ...],
    traditional_arms: tuple[str, ...],
    capacities: tuple[str, ...],
    seeds: tuple[int, ...],
    device: torch.device,
    start_state: dict,
    smoke: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = expected_prediction_names(neural_arms, traditional_arms, capacities, seeds)
    if is_complete(output_dir, expected):
        print(f"[skip complete] {output_dir}")
        return
    raw_path = output_dir / "raw_metrics.csv"
    rows = pd.read_csv(raw_path).to_dict("records") if raw_path.is_file() else []
    completed = {
        Path(str(row["prediction_file"])).name
        for row in rows
        if row.get("prediction_file") and str(row.get("prediction_file")) != "nan"
    }

    for arm in traditional_arms:
        filename = f"{arm}.npz"
        if filename in completed and (output_dir / "predictions" / filename).is_file():
            continue
        rows.append(
            evaluate_traditional_baseline(config, arm, datasets, metadata, output_dir)
        )
        save_rows(rows, output_dir)

    for capacity in capacities:
        for arm in neural_arms:
            for seed in seeds:
                variant = f"{arm}_{capacity}"
                filename = f"{variant}_seed{seed}.npz"
                if filename in completed and (output_dir / "predictions" / filename).is_file():
                    continue
                _, registration = build_baseline_model(arm, capacity, config, metadata)

                def model_factory(arm=arm, capacity=capacity):
                    return build_baseline_model(arm, capacity, config, metadata)[0]

                registration_dict = asdict(registration)
                registration_dict["selected_channel_indices"] = json.dumps(
                    registration_dict["selected_channel_indices"]
                )
                registration_dict["hyperparameters"] = json.dumps(
                    registration_dict["hyperparameters"], sort_keys=True
                )
                rows.append(
                    train_baseline(
                        config,
                        model_factory,
                        datasets,
                        metadata,
                        seed,
                        device,
                        variant,
                        output_dir,
                        registration=registration_dict,
                    )
                )
                save_rows(rows, output_dir)

    metadata_path = output_dir / "dataset_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    experiment = {
        **config.__dict__,
        "seeds": list(seeds),
        "neural_arms": list(neural_arms),
        "traditional_arms": list(traditional_arms),
        "capacities": list(capacities),
        "smoke": smoke,
        "warning": "SMOKE TEST DATA - DO NOT USE AS FINAL EVIDENCE" if smoke else None,
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(experiment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    end_state = code_state()
    if start_state != end_state:
        raise RuntimeError("Code or git status changed while the matrix was running")
    (output_dir / "run_metadata.json").write_text(
        json.dumps({"code_state_before": start_state, "code_state_after": end_state}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    start_state = code_state()
    device = torch.device(args.device)
    if args.smoke:
        tasks = ((24, 1),)
        seeds = (2047,)
        capacities = CAPACITY_TIERS
        neural_arms = NEURAL_ARMS
        traditional_arms = TRADITIONAL_ARMS
        if args.epochs is None:
            args.epochs = 2
        if args.patience is None:
            args.patience = 2
    elif args.city == "beijing":
        tasks = parse_tasks(args.configs, TASK_GRID)
        seeds = tuple(int(value) for value in args.seeds.split(",")) if args.seeds else None
        neural_arms = NEURAL_ARMS
        traditional_arms = TRADITIONAL_ARMS
        capacities = ("default",)
    else:
        tasks = parse_tasks(args.configs, HEADLINE_TASKS)
        seeds = tuple(int(value) for value in args.seeds.split(",")) if args.seeds else GUANGZHOU_SEEDS
        neural_arms = GUANGZHOU_ARMS
        traditional_arms = GUANGZHOU_TRADITIONAL_ARMS
        capacities = ("default", "matched")

    output_root = Path(args.output_root) / args.city
    started = time.perf_counter()
    centers = (None,) if args.city == "beijing" else guangzhou.B1_STATIONS
    for history, horizon in tasks:
        config = config_for(history, horizon, args)
        task_seeds = seeds
        task_capacities = capacities
        if not args.smoke and args.city == "beijing":
            task_seeds = (
                BEIJING_HEADLINE_SEEDS if (history, horizon) in HEADLINE_TASKS
                else BEIJING_GRID_SEEDS
            ) if seeds is None else seeds
            task_capacities = (
                ("default", "matched")
                if (history, horizon) in HEADLINE_TASKS
                else ("default",)
            )
        for center in centers:
            datasets, metadata = prepare_data(args.city, config, center)
            output_dir = output_root / f"{history}h_{horizon}h"
            if center is not None:
                output_dir = output_dir / f"station_{center}"
            print(
                f"[{args.city} {history}→{horizon}] stations={len(metadata['station_ids'])}, "
                f"splits={metadata['split_sizes']}"
            )
            run_one_dataset(
                config,
                datasets,
                metadata,
                output_dir,
                tuple(neural_arms),
                tuple(traditional_arms),
                tuple(task_capacities),
                tuple(task_seeds),
                device,
                start_state,
                args.smoke,
            )
    print(f"BASELINE_MATRIX_DONE seconds={time.perf_counter() - started:.1f}")


if __name__ == "__main__":
    main()
