"""Run the frozen baseline-comparison matrix with resumable artifacts.

P1 smoke example (the only run authorized during implementation)::

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python \
        run_baseline_comparison.py --smoke --device cuda
"""

from __future__ import annotations

import argparse
import gc
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
# 预注册的覆盖限制（协议 §5 精简档，用户 2026-09-20 批准）：
# concat 把输入拉长 S 倍，实测单次训练约 10 分钟且 L>=72 显存溢出，
# 因此只在前两个 L=24 配置上运行（其余配置不记录、不参与 best-of-baselines）。
CONCAT_TASKS = ((24, 1), (24, 6))
# `patchtst_ci_all` 在 L=168 下需要 18 通道 × 83 patch = 1494 个 token，
# 注意力矩阵在 batch=256 下超出 8GB 显存；实测 168→1 三颗种子全部 OOM、168→3 两颗 OOM，
# 并触发驱动级 GPU Hang（HW Exception ... GPU Hang，可复现）。该臂仅用于语义登记
# （“只给通道独立模型多嗂通道 ≠ 空间融合”），非同信息集主对手，因此预注册其覆盖为 **L ≤ 72**：
# L=168 不运行、不参与 best-of-baselines，并在产物中显式登记该排除。
CI_ALL_EXCLUDED_HISTORY = 168
BEIJING_GRID_SEEDS = (2047, 2048, 2049)
BEIJING_HEADLINE_SEEDS = (2047, 2048, 2049, 2050, 2051)
GUANGZHOU_SEEDS = (7001, 7002, 7003)
GUANGZHOU_ARMS = (
    "plain_mix_patchtst_all",
    "plain_mix_patchtst_top5",
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
    "test_baselines.py",
    "docs/基线对比/00_协议.md",
    "docs/基线对比/01_实现工作单.md",
    "run_st_patchtst_ablation.py",
    "run_beijing_leakfree_coverage.py",
    "run_cross_city_generalization.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", choices=("beijing", "guangzhou"), default="beijing")
    parser.add_argument("--grid-arms", default=None,
                        help="逗号分隔：跑满全部配置的神经臂；缺省=全部臂（完整档）")
    parser.add_argument("--headline-arms", default=None,
                        help="逗号分隔：只在头条配置（24→1、168→6）跑的神经臂（精简档）")
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
    raw = pd.read_csv(raw_path)
    if "arm" not in raw or raw["arm"].isna().any():
        return False
    if len(deduplicate_rows(raw.to_dict("records"))) != len(raw):
        return False
    infeasible = set()
    if "status" in raw:
        recorded = raw[
            raw["status"].astype(str).str.startswith("infeasible")
            | raw["status"].astype(str).str.startswith("nonfinite")
        ]
        for _, row in recorded.iterrows():
            infeasible.add(f"{row['variant']}_seed{int(row['seed'])}.npz")
    return expected_names <= (actual | infeasible)


def is_cuda_oom(error: BaseException, device: torch.device) -> bool:
    message = str(error).lower()
    return device.type == "cuda" and (
        isinstance(error, torch.OutOfMemoryError)
        or "out of memory" in message
        or "hip error out of memory" in message
    )


def _normalized_seed(value) -> str:
    try:
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
    except (TypeError, ValueError):
        pass
    return str(value)


def row_identity(row: dict) -> tuple[str, str, str]:
    """Stable idempotency key for one arm/seed/capacity result row."""
    capacity = row.get("requested_capacity", row.get("capacity_tier", ""))
    if pd.isna(capacity):
        capacity = ""
    return str(row.get("variant", "")), _normalized_seed(row.get("seed")), str(capacity)


def deduplicate_rows(rows: list[dict]) -> list[dict]:
    """Keep the latest row for each identity while preserving first-seen order."""
    deduplicated: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        deduplicated[row_identity(row)] = row
    return list(deduplicated.values())


def save_rows(rows: list[dict], output_dir: Path) -> None:
    rows[:] = deduplicate_rows(rows)
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
    rows = deduplicate_rows(rows)
    for row in rows:
        if not row.get("arm") or str(row.get("arm")) == "nan":
            row["arm"] = row.get("variant")
    if rows:
        save_rows(rows, output_dir)
    completed = {
        Path(str(row["prediction_file"])).name
        for row in rows
        if row.get("prediction_file") and str(row.get("prediction_file")) != "nan"
    }
    recorded_infeasible = {
        f"{row['variant']}_seed{int(row['seed'])}.npz"
        for row in rows
        if str(row.get("status", "")).startswith("infeasible")
        or str(row.get("status", "")).startswith("nonfinite")
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
                if filename in recorded_infeasible or (
                    filename in completed
                    and (output_dir / "predictions" / filename).is_file()
                ):
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
                try:
                    row = train_baseline(
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
                except FloatingPointError as error:
                    # 训练发散（非有限损失）：**统一规则，不针对具体臂**——
                    # 先用 1/10 学习率重试一次；若仍发散则登记为 nonfinite 行并继续。
                    # 理由：某条基线在共享训练配方下发散不应中断整个矩阵，也不应被静默丢弃；
                    # 重试事实与是否仍失败都会写进产物。
                    retry_config = replace(
                        config, learning_rate=config.learning_rate / 10.0
                    )
                    print(
                        f"[{variant} seed={seed}] 非有限损失，改用 lr/10 重试一次"
                    )
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    try:
                        row = train_baseline(
                            retry_config,
                            model_factory,
                            datasets,
                            metadata,
                            seed,
                            device,
                            variant,
                            output_dir,
                            registration=registration_dict,
                        )
                        row["learning_rate_retry"] = True
                        row["note"] = "nonfinite loss at protocol lr; retried once with lr/10"
                    except FloatingPointError:
                        row = {
                            "variant": variant,
                            "seed": seed,
                            "status": "nonfinite",
                            "evaluation_split": config.evaluation_split,
                            "best_epoch": 0,
                            "best_valid_loss": float("nan"),
                            "training_seconds": float("nan"),
                            "test_inference_seconds": float("nan"),
                            "trainable_parameter_count": registration.parameter_count,
                            "prediction_file": "",
                            "rmse_ugm3": float("nan"),
                            "mae_ugm3": float("nan"),
                            "smape_percent": float("nan"),
                            "mse_scaled": float("nan"),
                            "rmse_scaled": float("nan"),
                            "mae_scaled": float("nan"),
                            "infeasible_reason": (
                                "non-finite training loss at protocol lr and at lr/10: "
                                f"{str(error)[:400]}"
                            ),
                            "device_name": torch.cuda.get_device_name(device),
                            "batch_size": config.batch_size,
                            "history": config.history,
                            "horizon": config.horizon,
                            **registration_dict,
                        }
                        print(
                            f"[{variant} seed={seed}] status=nonfinite（lr 与 lr/10 均发散）"
                        )
                except (RuntimeError, torch.OutOfMemoryError) as error:
                    # 任何臂的 CUDA OOM 都不应杀死整个矩阵：登记为不可行行后继续，
                    # 保证"不可行"在产物里显式可见，而不是静默跳过或中断整轮。
                    # 非 OOM 的真实错误仍然向上抛。
                    if not is_cuda_oom(error, device):
                        raise
                    gc.collect()
                    torch.cuda.empty_cache()
                    row = {
                        "variant": variant,
                        "seed": seed,
                        "status": "infeasible_oom",
                        "evaluation_split": config.evaluation_split,
                        "best_epoch": 0,
                        "best_valid_loss": float("nan"),
                        "training_seconds": float("nan"),
                        "test_inference_seconds": float("nan"),
                        "trainable_parameter_count": registration.parameter_count,
                        "prediction_file": "",
                        "rmse_ugm3": float("nan"),
                        "mae_ugm3": float("nan"),
                        "smape_percent": float("nan"),
                        "mse_scaled": float("nan"),
                        "rmse_scaled": float("nan"),
                        "mae_scaled": float("nan"),
                        "infeasible_reason": str(error).replace("\n", " ")[:1000],
                        "device_name": torch.cuda.get_device_name(device),
                        "batch_size": config.batch_size,
                        "history": config.history,
                        "horizon": config.horizon,
                        **registration_dict,
                    }
                    print(
                        f"[{variant} seed={seed}] status=infeasible_oom "
                        f"history={config.history} batch={config.batch_size}"
                    )
                rows.append(row)
                save_rows(rows, output_dir)
                # 跨运行显存回收：单臂峰值本身可接受（实测 ci_all@L=168 约 4.6 GB），
                # 但同一进程连续跑多个臂/种子会累积缓存块，导致后续运行假性 OOM。
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

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
        grid_arms = tuple(neural_arms)
        headline_arms = ()
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
        grid_arms = tuple(neural_arms)
        headline_arms = ()
    else:
        tasks = parse_tasks(args.configs, HEADLINE_TASKS)
        seeds = tuple(int(value) for value in args.seeds.split(",")) if args.seeds else GUANGZHOU_SEEDS
        neural_arms = GUANGZHOU_ARMS
        traditional_arms = GUANGZHOU_TRADITIONAL_ARMS
        capacities = ("default", "matched")
        grid_arms = tuple(neural_arms)
        headline_arms = ()

    # 精简档：--grid-arms 指定跑满网格的臂，--headline-arms 指定只在头条配置跑的臂。
    if args.grid_arms is not None or args.headline_arms is not None:
        grid_selection = {name for name in (args.grid_arms or "").split(",") if name}
        headline_selection = {name for name in (args.headline_arms or "").split(",") if name}
        unknown = (grid_selection | headline_selection) - set(neural_arms)
        if unknown:
            raise ValueError(f"--grid-arms/--headline-arms 含未知臂: {sorted(unknown)}")
        grid_arms = tuple(arm for arm in neural_arms if arm in grid_selection)
        headline_arms = tuple(arm for arm in neural_arms if arm in headline_selection)

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
            task_arms = tuple(
                arm for arm in neural_arms
                if arm in grid_arms
                or ((history, horizon) in HEADLINE_TASKS and arm in headline_arms)
            )
            task_arms = tuple(
                arm for arm in task_arms
                if arm != "concat_patchtst_all" or (history, horizon) in CONCAT_TASKS
            )
            task_arms = tuple(
                arm for arm in task_arms
                if not (arm == "patchtst_ci_all" and history >= CI_ALL_EXCLUDED_HISTORY)
            )
            print(
                f"[{args.city} {history}→{horizon}] 神经臂={len(task_arms)} "
                f"(网格 {sum(1 for a in task_arms if a in grid_arms)} / 仅头条 "
                f"{sum(1 for a in task_arms if a not in grid_arms)})"
            )
            run_one_dataset(
                config,
                datasets,
                metadata,
                output_dir,
                tuple(task_arms),
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
