"""Ingest and, only when explicitly authorized, run single-station baselines.

The default path is non-training: ``--ingest-source`` validates reusable rows,
writes them below this experiment's result root, and prints the missing-run
plan.  No training starts unless ``--execute-missing`` is supplied explicitly.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn

import run_beijing_leakfree_coverage as beijing
import run_cross_city_generalization as guangzhou
import run_st_patchtst_ablation as legacy
from single_station_models import build_single_station_model
from single_station_traditional import (
    TRADITIONAL_ARMS,
    dataset_arrays,
    fit_traditional_baseline,
)


TASK_GRID = tuple(beijing.TASK_GRID)
HEADLINE_TASKS = ((24, 1), (168, 6))
BEIJING_GRID_SEEDS = (2047, 2048, 2049)
BEIJING_HEADLINE_SEEDS = (2047, 2048, 2049, 2050, 2051)
GUANGZHOU_SEEDS = (7001, 7002, 7003)
GUANGZHOU_STATIONS = tuple(int(value) for value in guangzhou.B1_STATIONS)
GRID_NEURAL_ARMS = ("center_gru", "center_lstm", "center_tcn")
HEADLINE_NEURAL_ARMS = (
    "center_mlp",
    "center_gru",
    "center_lstm",
    "center_tcn",
    "center_resnet",
    "center_tst",
)
INGESTABLE_ARMS = frozenset((*HEADLINE_NEURAL_ARMS, *TRADITIONAL_ARMS))
PROVENANCE = (
    "ingested from experiment/baseline-comparison-ablation "
    "(same arm definition, same config/seed/data)"
)
IMPLEMENTATION_FILES = (
    "single_station_models.py",
    "single_station_traditional.py",
    "run_single_station_baselines.py",
    "summarize_single_station.py",
    "test_single_station.py",
    "run_st_patchtst_ablation.py",
    "run_beijing_leakfree_coverage.py",
    "run_cross_city_generalization.py",
)


def _canonical_seed(value: object) -> str:
    if str(value) == "deterministic":
        return "deterministic"
    try:
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
    except (TypeError, ValueError):
        pass
    return str(value)


@dataclass(frozen=True, order=True)
class RunKey:
    city: str
    history: int
    horizon: int
    station_id: int | None
    arm: str
    seed: str
    capacity_tier: str

    def result_parts(self) -> tuple[str, ...]:
        parts = (self.city, f"{self.history}h_{self.horizon}h")
        if self.station_id is not None:
            parts += (f"station_{self.station_id}",)
        return parts


def expected_run_keys() -> tuple[RunKey, ...]:
    """Return the frozen reusable/run matrix from work order W1 §2."""
    keys: list[RunKey] = []
    for history, horizon in TASK_GRID:
        headline = (history, horizon) in HEADLINE_TASKS
        seeds = BEIJING_HEADLINE_SEEDS if headline else BEIJING_GRID_SEEDS
        neural_arms = HEADLINE_NEURAL_ARMS if headline else GRID_NEURAL_ARMS
        capacities = ("default", "matched") if headline else ("default",)
        for arm in neural_arms:
            for capacity in capacities:
                for seed in seeds:
                    keys.append(
                        RunKey("beijing", history, horizon, None, arm, str(seed), capacity)
                    )
        for arm in TRADITIONAL_ARMS:
            keys.append(
                RunKey(
                    "beijing",
                    history,
                    horizon,
                    None,
                    arm,
                    "deterministic",
                    "not_applicable",
                )
            )

    # Guangzhou W1 reuses the measured centre-GRU competitor at both capacity
    # tiers.  All five traditional centre-only arms remain required; AR/Ridge
    # are expected to appear in the printed gap plan because the old run did
    # not contain them.
    for history, horizon in HEADLINE_TASKS:
        for station_id in GUANGZHOU_STATIONS:
            for capacity in ("default", "matched"):
                for seed in GUANGZHOU_SEEDS:
                    keys.append(
                        RunKey(
                            "guangzhou",
                            history,
                            horizon,
                            station_id,
                            "center_gru",
                            str(seed),
                            capacity,
                        )
                    )
            for arm in TRADITIONAL_ARMS:
                keys.append(
                    RunKey(
                        "guangzhou",
                        history,
                        horizon,
                        station_id,
                        arm,
                        "deterministic",
                        "not_applicable",
                    )
                )
    return tuple(sorted(keys))


def parse_source_context(raw_path: Path, source_root: Path) -> tuple[str, int, int, int | None]:
    relative = raw_path.relative_to(source_root)
    if len(relative.parts) < 3:
        raise ValueError(f"unexpected source result path: {raw_path}")
    city = relative.parts[0]
    task_match = re.fullmatch(r"(\d+)h_(\d+)h", relative.parts[1])
    if city not in {"beijing", "guangzhou"} or task_match is None:
        raise ValueError(f"cannot infer city/config from {raw_path}")
    station_parts = [part for part in relative.parts if part.startswith("station_")]
    station_id = int(station_parts[0].removeprefix("station_")) if station_parts else None
    if city == "guangzhou" and station_id is None:
        raise ValueError(f"Guangzhou source lacks station directory: {raw_path}")
    if city == "beijing" and station_id is not None:
        raise ValueError(f"Beijing source unexpectedly has station directory: {raw_path}")
    return city, int(task_match.group(1)), int(task_match.group(2)), station_id


def key_from_source_row(
    row: pd.Series | dict,
    city: str,
    history: int,
    horizon: int,
    station_id: int | None,
) -> RunKey:
    arm = str(row.get("arm", row.get("variant", "")))
    capacity = str(row.get("capacity_tier", row.get("requested_capacity", "")))
    if capacity in {"", "nan", "None"} and arm in TRADITIONAL_ARMS:
        capacity = "not_applicable"
    return RunKey(
        city,
        int(history),
        int(horizon),
        station_id,
        arm,
        _canonical_seed(row.get("seed")),
        capacity,
    )


def validate_ingest_row(
    row: pd.Series | dict,
    expected: RunKey,
    source_context: tuple[str, int, int, int | None],
) -> None:
    """Reject information-set or identity drift before a row is ingested."""
    actual = key_from_source_row(row, *source_context)
    if actual != expected:
        raise ValueError(f"ingest identity mismatch: expected={expected}, actual={actual}")
    try:
        input_channels = int(float(row.get("input_channels")))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{actual} lacks a valid input_channels value") from error
    if input_channels != 1:
        raise ValueError(f"refusing non-single-station ingest row {actual}: input_channels={input_channels}")
    if str(row.get("evaluation_split", "test")) != "test":
        raise ValueError(f"refusing non-test result row {actual}")
    expected_variant = (
        actual.arm
        if actual.arm in TRADITIONAL_ARMS
        else f"{actual.arm}_{actual.capacity_tier}"
    )
    if str(row.get("variant")) != expected_variant:
        raise ValueError(
            f"variant mismatch for {actual}: expected {expected_variant}, got {row.get('variant')}"
        )


def _read_source_config(raw_path: Path) -> dict:
    config_path = raw_path.parent / "experiment_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    return json.loads(config_path.read_text(encoding="utf-8"))


def _verify_source_config(config: dict, context: tuple[str, int, int, int | None]) -> None:
    _, history, horizon, _ = context
    if int(config.get("history", -1)) != history or int(config.get("horizon", -1)) != horizon:
        raise ValueError(
            f"source experiment_config disagrees with path: config={config}, context={context}"
        )
    if bool(config.get("smoke", False)):
        raise ValueError(f"refusing smoke-test source for context={context}")


def _prediction_source(raw_path: Path, row: pd.Series) -> Path | None:
    status = str(row.get("status", "completed"))
    if status.startswith(("infeasible", "nonfinite")):
        return None
    value = str(row.get("prediction_file", ""))
    if value in {"", "nan", "None"}:
        raise ValueError(f"completed row has no prediction_file in {raw_path}")
    path = Path(value)
    if not path.is_absolute():
        path = raw_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def ingest_source(source_root: str | Path, output_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate and materialize the reusable single-station rows."""
    source_root = Path(source_root).resolve()
    output_root = Path(output_root)
    expected = set(expected_run_keys())
    selected: dict[RunKey, dict] = {}
    audit: list[dict] = []
    source_files = sorted(source_root.rglob("raw_metrics.csv"))
    if not source_files:
        raise FileNotFoundError(f"no raw_metrics.csv below {source_root}")

    for raw_path in source_files:
        context = parse_source_context(raw_path, source_root)
        _verify_source_config(_read_source_config(raw_path), context)
        raw = pd.read_csv(raw_path)
        for _, source_row in raw.iterrows():
            arm = str(source_row.get("arm", source_row.get("variant", "")))
            if arm not in INGESTABLE_ARMS:
                reason = "not_registered_single_station_arm"
                if arm.startswith("trad_") and int(float(source_row.get("input_channels", -1))) != 1:
                    reason = "excluded_non_single_station_input"
                audit.append(
                    {
                        "source_file": str(raw_path),
                        "arm": arm,
                        "reason": reason,
                        "input_channels": source_row.get("input_channels"),
                        "count": 1,
                    }
                )
                continue
            key = key_from_source_row(source_row, *context)
            if key not in expected:
                raise ValueError(f"source row is outside the frozen W1 matrix: {key}")
            validate_ingest_row(source_row, key, context)
            if key in selected:
                raise ValueError(f"duplicate ingest identity: {key}")
            prediction_path = _prediction_source(raw_path, source_row)
            row = source_row.to_dict()
            row.update(
                {
                    "city": key.city,
                    "history": key.history,
                    "horizon": key.horizon,
                    "station_id": key.station_id,
                    "capacity_tier": key.capacity_tier,
                    "input_channels": 1,
                    "provenance": PROVENANCE,
                    "source_raw_metrics": str(raw_path),
                    "source_prediction_file": str(prediction_path) if prediction_path else "",
                    # Absolute read-only reference avoids copying hundreds of NPZ files.
                    "prediction_file": str(prediction_path) if prediction_path else "",
                }
            )
            selected[key] = row

    by_directory: dict[tuple[str, ...], list[dict]] = {}
    for key, row in sorted(selected.items()):
        by_directory.setdefault(key.result_parts(), []).append(row)
    for parts, rows in by_directory.items():
        directory = output_root.joinpath(*parts)
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(directory / "raw_metrics.csv", index=False)
        (directory / "experiment_config.json").write_text(
            json.dumps(
                {
                    "city": parts[0],
                    "history": int(rows[0]["history"]),
                    "horizon": int(rows[0]["horizon"]),
                    "evaluation_split": "test",
                    "smoke": False,
                    "ingest_only": True,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    missing = sorted(expected - set(selected))
    historical = pd.DataFrame(selected.values())
    plan = build_missing_plan(missing, historical)
    output_root.mkdir(parents=True, exist_ok=True)
    plan.to_csv(output_root / "missing_plan.csv", index=False)
    audit_frame = pd.DataFrame(audit)
    audit_frame.to_csv(output_root / "ingest_audit.csv", index=False)
    summary = {
        "source_root": str(source_root),
        "provenance": PROVENANCE,
        "expected_rows": len(expected),
        "ingested_rows": len(selected),
        "missing_rows": len(missing),
        "input_channel_validation": "pass: all ingested rows input_channels == 1",
        "excluded_source_rows": len(audit_frame),
    }
    (output_root / "ingest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return pd.DataFrame(selected.values()), plan


def build_missing_plan(missing: list[RunKey], historical: pd.DataFrame) -> pd.DataFrame:
    records = []
    for key in missing:
        estimate = np.nan
        if not historical.empty and "training_seconds" in historical:
            same = historical[
                (historical["arm"] == key.arm)
                & (historical["capacity_tier"] == key.capacity_tier)
            ]
            values = pd.to_numeric(same["training_seconds"], errors="coerce")
            values = values[np.isfinite(values) & (values >= 0)]
            if len(values):
                estimate = float(values.median())
        if not np.isfinite(estimate):
            estimate = 5.0 if key.arm.startswith("trad_") else 600.0
        records.append({**asdict(key), "estimated_seconds": estimate})
    return pd.DataFrame(records)


def print_missing_plan(plan: pd.DataFrame) -> None:
    if plan.empty:
        print("补跑计划：0 个组合；预计 0.0 小时。")
        return
    grouped = (
        plan.groupby(["city", "arm", "capacity_tier"], dropna=False)
        .agg(combinations=("arm", "size"), estimated_seconds=("estimated_seconds", "sum"))
        .reset_index()
    )
    print("补跑计划（仅打印，未启动）：")
    print(grouped.to_string(index=False))
    print(
        f"缺失合计={len(plan)} 组合，预计={plan['estimated_seconds'].sum() / 3600:.3f} 小时"
    )


def _ensure_test_split(config) -> None:
    if config.evaluation_split != "test":
        raise ValueError("single-station protocol fixes evaluation_split='test'")


def train_neural_baseline(
    config,
    model: nn.Module | Callable[[], nn.Module],
    datasets: dict,
    metadata: dict,
    seed: int,
    device: torch.device,
    variant: str,
    output_dir: str | Path,
    registration: dict | None = None,
) -> dict:
    """Replica of ``legacy.train_one_run`` used by neural baseline arms."""
    _ensure_test_split(config)
    legacy.set_seed(seed)
    model_instance = model() if callable(model) else model
    model_instance = model_instance.to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model_instance.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    loss_fn = nn.MSELoss()
    train_loader = legacy.make_loader(datasets["train"], config, True, seed)
    valid_loader = legacy.make_loader(datasets["valid"], config, False, seed)
    evaluation_loader = legacy.make_loader(datasets[config.evaluation_split], config, False, seed)

    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{variant}_seed{seed}.pt"
    best_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history_rows = []
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model_instance.train()
        train_loss_sum = 0.0
        train_count = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model_instance(x)
            loss = loss_fn(prediction, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{variant} seed={seed} has non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model_instance.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(x)
            train_count += len(x)

        valid_prediction, valid_target, _ = legacy.predict(
            model_instance, valid_loader, device, metadata["center_station_idx"]
        )
        train_loss = train_loss_sum / train_count
        valid_loss = float(np.mean((valid_prediction - valid_target) ** 2))
        scheduler.step(valid_loss)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if valid_loss < best_loss - 1e-7:
            best_loss = valid_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model_instance.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.patience:
            break

    training_seconds = time.perf_counter() - started
    model_instance.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    prediction, target, inference_seconds = legacy.predict(
        model_instance,
        evaluation_loader,
        device,
        metadata["center_station_idx"],
    )
    metrics = legacy.regression_metrics(
        target, prediction, metadata["center_mean"], metadata["center_std"]
    )
    logs_dir = output_dir / "training_logs"
    predictions_dir = output_dir / "predictions"
    logs_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history_rows).to_csv(logs_dir / f"{variant}_seed{seed}.csv", index=False)
    prediction_path = predictions_dir / f"{variant}_seed{seed}.npz"
    np.savez_compressed(
        prediction_path,
        prediction_scaled=prediction,
        target_scaled=target,
        prediction_ugm3=prediction * metadata["center_std"] + metadata["center_mean"],
        target_ugm3=target * metadata["center_std"] + metadata["center_mean"],
    )
    return {
        "variant": variant,
        "seed": seed,
        "status": "completed",
        "best_epoch": best_epoch,
        "best_valid_loss": best_loss,
        "training_seconds": training_seconds,
        "test_inference_seconds": inference_seconds,
        "evaluation_split": config.evaluation_split,
        "parameter_count": sum(parameter.numel() for parameter in model_instance.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model_instance.parameters() if parameter.requires_grad
        ),
        "prediction_file": str(prediction_path.relative_to(output_dir)),
        **(registration or {}),
        **metrics,
    }


def evaluate_traditional_baseline(
    config,
    arm: str,
    datasets: dict,
    metadata: dict,
    output_dir: str | Path,
) -> dict:
    _ensure_test_split(config)
    output_dir = Path(output_dir)
    started = time.perf_counter()
    model = fit_traditional_baseline(arm, datasets, metadata)
    training_seconds = time.perf_counter() - started
    inference_started = time.perf_counter()
    prediction = model.predict(datasets[config.evaluation_split], metadata)
    inference_seconds = time.perf_counter() - inference_started
    _, target = dataset_arrays(datasets[config.evaluation_split])
    metrics = legacy.regression_metrics(
        target, prediction, metadata["center_mean"], metadata["center_std"]
    )
    predictions_dir = output_dir / "predictions"
    logs_dir = output_dir / "training_logs"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = predictions_dir / f"{arm}.npz"
    np.savez_compressed(
        prediction_path,
        prediction_scaled=prediction,
        target_scaled=target,
        prediction_ugm3=prediction * metadata["center_std"] + metadata["center_mean"],
        target_ugm3=target * metadata["center_std"] + metadata["center_mean"],
    )
    pd.DataFrame(
        [{"fit": "train_only", "selected_alpha": model.selected_alpha}]
    ).to_csv(logs_dir / f"{arm}.csv", index=False)
    return {
        "variant": arm,
        "arm": arm,
        "seed": "deterministic",
        "status": "completed",
        "best_epoch": 0,
        "best_valid_loss": np.nan,
        "training_seconds": training_seconds,
        "test_inference_seconds": inference_seconds,
        "evaluation_split": config.evaluation_split,
        "parameter_count": 0,
        "trainable_parameter_count": 0,
        "layer": "single_station_traditional",
        "requested_capacity": "not_applicable",
        "capacity_tier": "not_applicable",
        "capacity_status": "not_applicable",
        "input_channels": 1,
        "selected_alpha": model.selected_alpha,
        "prediction_file": str(prediction_path.relative_to(output_dir)),
        **metrics,
    }


def is_cuda_oom(error: BaseException, device: torch.device) -> bool:
    message = str(error).lower()
    return device.type == "cuda" and (
        isinstance(error, torch.OutOfMemoryError)
        or "out of memory" in message
        or "hip error out of memory" in message
    )


def row_identity(row: dict) -> tuple[str, str, str]:
    return (
        str(row.get("arm", row.get("variant", ""))),
        _canonical_seed(row.get("seed")),
        str(row.get("capacity_tier", row.get("requested_capacity", ""))),
    )


def save_rows(rows: list[dict], output_dir: Path) -> None:
    deduplicated = {row_identity(row): row for row in rows}
    pd.DataFrame(deduplicated.values()).to_csv(output_dir / "raw_metrics.csv", index=False)


def code_state() -> dict:
    hashes = {}
    for path_value in IMPLEMENTATION_FILES:
        path = Path(path_value)
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hashes[path_value] = digest
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"commit": commit, "hashes": hashes, "git_status": status}


def config_for(history: int, horizon: int):
    return replace(
        beijing.config_for(history, horizon),
        evaluation_split="test",
        initialize_from_degraded=False,
        freeze_backbone=False,
    )


def _prepare_guangzhou(center_station_id: int, config):
    frame, opened = guangzhou.load_center_candidate_frame(center_station_id)
    source, metadata = guangzhou.prepare_shared_data(
        frame,
        center_station_id,
        guangzhou.RunConfig(
            history=config.history,
            horizon=config.horizon,
            batch_size=config.batch_size,
            epochs=config.epochs,
            patience=config.patience,
        ),
    )
    datasets = {"train": source["fit"], "valid": source["val"], "test": source["confirm"]}
    for dataset in datasets.values():
        dataset.sample_indices = dataset.indices
    return datasets, {
        **metadata,
        "center_station_id": center_station_id,
        "start_time": str(frame.index.min()),
        "opened_station_ids": list(opened),
        "station_correlations": metadata["candidate_correlations"],
    }


def prepare_data(key: RunKey, config):
    if key.city == "beijing":
        return beijing.prepare_datasets_leakfree(config)
    return _prepare_guangzhou(int(key.station_id), config)


def _exception_row(key: RunKey, status: str, reason: str) -> dict:
    variant = key.arm if key.arm in TRADITIONAL_ARMS else f"{key.arm}_{key.capacity_tier}"
    return {
        "variant": variant,
        "arm": key.arm,
        "seed": key.seed,
        "status": status,
        "evaluation_split": "test",
        "capacity_tier": key.capacity_tier,
        "requested_capacity": key.capacity_tier,
        "input_channels": 1,
        "prediction_file": "",
        "rmse_ugm3": np.nan,
        "mae_ugm3": np.nan,
        "smape_percent": np.nan,
        "infeasible_reason": reason[:1000],
    }


def execute_missing(plan: pd.DataFrame, output_root: Path, device: torch.device) -> None:
    """Run only planned gaps; caller must explicitly opt in at the CLI."""
    if plan.empty:
        return
    start_state = code_state()
    group_columns = ["city", "history", "horizon", "station_id"]
    for _, group in plan.groupby(group_columns, dropna=False, sort=True):
        first = group.iloc[0]
        station_id = None if pd.isna(first["station_id"]) else int(first["station_id"])
        template = RunKey(
            str(first["city"]),
            int(first["history"]),
            int(first["horizon"]),
            station_id,
            str(first["arm"]),
            str(first["seed"]),
            str(first["capacity_tier"]),
        )
        config = config_for(template.history, template.horizon)
        datasets, metadata = prepare_data(template, config)
        output_dir = output_root.joinpath(*template.result_parts())
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_path = output_dir / "raw_metrics.csv"
        rows = pd.read_csv(raw_path).to_dict("records") if raw_path.is_file() else []
        completed = {row_identity(row) for row in rows}
        for _, plan_row in group.iterrows():
            key = RunKey(
                str(plan_row["city"]),
                int(plan_row["history"]),
                int(plan_row["horizon"]),
                station_id,
                str(plan_row["arm"]),
                _canonical_seed(plan_row["seed"]),
                str(plan_row["capacity_tier"]),
            )
            identity = (key.arm, key.seed, key.capacity_tier)
            if identity in completed:
                continue
            try:
                if key.arm in TRADITIONAL_ARMS:
                    row = evaluate_traditional_baseline(
                        config, key.arm, datasets, metadata, output_dir
                    )
                else:
                    seed = int(key.seed)

                    def factory(key=key):
                        return build_single_station_model(
                            key.arm, key.capacity_tier, config, metadata
                        )[0]

                    _, registration = build_single_station_model(
                        key.arm, key.capacity_tier, config, metadata
                    )
                    registration_dict = asdict(registration)
                    registration_dict["selected_channel_indices"] = json.dumps(
                        registration_dict["selected_channel_indices"]
                    )
                    registration_dict["hyperparameters"] = json.dumps(
                        registration_dict["hyperparameters"], sort_keys=True
                    )
                    variant = f"{key.arm}_{key.capacity_tier}"
                    try:
                        row = train_neural_baseline(
                            config,
                            factory,
                            datasets,
                            metadata,
                            seed,
                            device,
                            variant,
                            output_dir,
                            registration_dict,
                        )
                    except FloatingPointError:
                        retry = replace(config, learning_rate=config.learning_rate / 10.0)
                        try:
                            row = train_neural_baseline(
                                retry,
                                factory,
                                datasets,
                                metadata,
                                seed,
                                device,
                                variant,
                                output_dir,
                                registration_dict,
                            )
                            row["learning_rate_retry"] = True
                        except FloatingPointError as error:
                            row = _exception_row(key, "nonfinite", str(error))
                row.update(
                    {
                        "city": key.city,
                        "history": key.history,
                        "horizon": key.horizon,
                        "station_id": key.station_id,
                        "capacity_tier": key.capacity_tier,
                        "provenance": "measured by run_single_station_baselines.py",
                    }
                )
            except (RuntimeError, torch.OutOfMemoryError) as error:
                if not is_cuda_oom(error, device):
                    raise
                row = _exception_row(key, "infeasible_oom", str(error))
            rows.append(row)
            save_rows(rows, output_dir)
            completed.add(identity)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if code_state() != start_state:
        raise RuntimeError("code or tracked git state changed during the missing-run matrix")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ingest-source", required=True)
    parser.add_argument(
        "--output-root", default="experiments/results/single_station_baselines"
    )
    parser.add_argument("--execute-missing", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    ingested, plan = ingest_source(args.ingest_source, output_root)
    print(
        f"ingest_rows={len(ingested)} input_channels_check=PASS provenance={PROVENANCE}"
    )
    print_missing_plan(plan)
    if not args.execute_missing:
        print("正式矩阵未启动；如获批准，显式追加 --execute-missing。")
        print("SINGLE_STATION_INGEST_PLAN_DONE")
        return
    execute_missing(plan, output_root, torch.device(args.device))
    print("SINGLE_STATION_MISSING_RUNS_DONE")


if __name__ == "__main__":
    main()
