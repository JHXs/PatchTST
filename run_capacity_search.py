"""Run the preregistered P1 spatial-branch capacity search.

The selected backbone for every group is read unchanged from the formal
backbone-upgrade selection artifact.  This runner reuses that experiment's
data, model-mounting, freezing, and training path; only the three registered
spatial-branch hyperparameters differ between candidates.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from run_backbone_upgrade import (
    _is_oom,
    config_for,
    prepare_data,
    train_one_upgrade,
)


BACKBONE_SELECTION = Path(
    "experiments/results/backbone_upgrade/formal/selection.csv"
)
OUTPUT_ROOT = Path("experiments/results/capacity_search")
HEADLINE_TASKS = ((24, 1), (168, 6))
HEADLINE_SEEDS = (2047, 2048, 2049, 2050, 2051)
CENTER_STATION_ID = 1013


@dataclass(frozen=True)
class CapacityCandidate:
    name: str
    neighbor_hidden_dim: int
    spatial_pool_bins: int
    forecast_alpha_init: float
    forecast_alpha_max: float = 0.5


CAPACITY_CANDIDATES = (
    CapacityCandidate("cap32_b4_a10", 32, 4, 0.1),
    CapacityCandidate("cap128_b4_a10", 128, 4, 0.1),
    CapacityCandidate("cap128_b8_a20", 128, 8, 0.2),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--backbone-selection", default=str(BACKBONE_SELECTION))
    parser.add_argument("--configs", default="24x1,168x6", help="逗号分隔 LxH")
    parser.add_argument("--seeds", default=",".join(map(str, HEADLINE_SEEDS)))
    return parser.parse_args()


def _as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1"})


def load_selected_backbone(
    selection_path: str | Path,
    history: int,
    horizon: int,
    seed: int,
    station_id: int = CENTER_STATION_ID,
) -> dict[str, Any]:
    """Read the one frozen P1 winner without re-running backbone selection."""
    selection = pd.read_csv(selection_path)
    required = {
        "city", "history", "horizon", "station_id", "seed", "selected",
        "variant", "arm", "capacity", "best_valid_loss", "checkpoint_path",
        "hyperparameters",
    }
    missing = required - set(selection.columns)
    if missing:
        raise RuntimeError(f"主干选择表缺少字段: {sorted(missing)}")
    rows = selection[
        (selection["city"] == "beijing")
        & (selection["history"] == history)
        & (selection["horizon"] == horizon)
        & (selection["station_id"] == station_id)
        & (selection["seed"] == seed)
        & _as_bool(selection["selected"])
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"既定主干必须恰有一个: L={history}, H={horizon}, seed={seed}, 实际={len(rows)}"
        )
    winner = rows.iloc[0].to_dict()
    checkpoint = Path(str(winner["checkpoint_path"]))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"既定主干检查点不存在: {checkpoint}")
    if not math.isfinite(float(winner["best_valid_loss"])):
        raise RuntimeError("既定主干的验证损失非有限")
    return winner


def apply_capacity_candidate(config, candidate: CapacityCandidate):
    """Return the backbone-upgrade recipe with only registered branch fields changed."""
    return replace(
        config,
        neighbor_hidden_dim=candidate.neighbor_hidden_dim,
        spatial_pool_bins=candidate.spatial_pool_bins,
        forecast_alpha_init=candidate.forecast_alpha_init,
        forecast_alpha_max=candidate.forecast_alpha_max,
    )


def select_capacity_by_validation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select one capacity using status and validation loss only."""
    copied = [{**row, "selected": False} for row in rows]
    eligible = [
        row for row in copied
        if row.get("status") == "completed"
        and math.isfinite(float(row.get("best_valid_loss", math.nan)))
    ]
    if not eligible:
        return copied
    winner = min(
        eligible,
        key=lambda row: (float(row["best_valid_loss"]), str(row["capacity_candidate"])),
    )
    winner_name = str(winner["capacity_candidate"])
    for row in copied:
        row["selected"] = (
            row.get("status") == "completed"
            and str(row.get("capacity_candidate")) == winner_name
        )
    return copied


def _identity(history: int, horizon: int, seed: int, candidate: str) -> str:
    return (
        f"beijing_{history}h_{horizon}h_station{CENTER_STATION_ID}_"
        f"seed{seed}_{candidate}"
    )


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return pd.read_csv(path).to_dict("records") if path.is_file() else []


def _save_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _upsert(rows: list[dict[str, Any]], new_row: dict[str, Any]) -> None:
    rows[:] = [row for row in rows if row.get("run_id") != new_row.get("run_id")]
    rows.append(new_row)


def run_spec(configs: str, seeds: str) -> tuple[tuple[int, int, int], ...]:
    tasks = tuple(
        tuple(int(part) for part in item.lower().split("x"))
        for item in configs.split(",")
    )
    parsed_seeds = tuple(int(item) for item in seeds.split(","))
    return tuple((history, horizon, seed) for history, horizon in tasks for seed in parsed_seeds)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    raw_path = output_root / "raw_metrics.csv"
    raw_rows = _read_rows(raw_path)
    device = torch.device(args.device)

    for history, horizon, seed in run_spec(args.configs, args.seeds):
        winner = load_selected_backbone(args.backbone_selection, history, horizon, seed)
        base_config = config_for(history, horizon, smoke=False)
        datasets, metadata = prepare_data("beijing", base_config, CENTER_STATION_ID)
        for candidate in CAPACITY_CANDIDATES:
            config = apply_capacity_candidate(base_config, candidate)
            identity = _identity(history, horizon, seed, candidate.name)
            prior = [row for row in raw_rows if row.get("run_id") == identity]
            if prior and str(prior[-1].get("status")) in {
                "completed", "infeasible_oom", "nonfinite"
            }:
                print(f"[跳过] {identity}: {prior[-1]['status']}")
                continue
            base = {
                "run_id": identity,
                "city": "beijing",
                "history": history,
                "horizon": horizon,
                "station_id": CENTER_STATION_ID,
                "seed": seed,
                "capacity_candidate": candidate.name,
                "neighbor_hidden_dim": candidate.neighbor_hidden_dim,
                "spatial_pool_bins": candidate.spatial_pool_bins,
                "forecast_alpha_init": candidate.forecast_alpha_init,
                "forecast_alpha_max": candidate.forecast_alpha_max,
            }
            try:
                result = train_one_upgrade(
                    config,
                    datasets,
                    metadata,
                    winner,
                    seed,
                    output_root,
                    identity,
                    device,
                )
                if not math.isfinite(float(result["best_valid_loss"])):
                    raise FloatingPointError(f"{identity} 出现非有限验证损失")
                result["learning_rate_retry"] = False
            except FloatingPointError as first_error:
                print(f"[{identity}] 非有限损失，按协议以 lr/10 重试一次")
                retry_config = replace(config, learning_rate=config.learning_rate / 10.0)
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                try:
                    result = train_one_upgrade(
                        retry_config,
                        datasets,
                        metadata,
                        winner,
                        seed,
                        output_root,
                        identity,
                        device,
                    )
                    if not math.isfinite(float(result["best_valid_loss"])):
                        raise FloatingPointError(f"{identity} 重试后验证损失仍非有限")
                    result["learning_rate_retry"] = True
                    result["retry_reason"] = str(first_error)
                except FloatingPointError as retry_error:
                    result = {
                        "status": "nonfinite",
                        "failure_reason": str(retry_error),
                        "learning_rate_retry": True,
                        "selected_variant": winner["variant"],
                        "source_checkpoint": winner["checkpoint_path"],
                    }
            except (RuntimeError, torch.OutOfMemoryError) as error:
                if not _is_oom(error, device):
                    raise
                result = {
                    "status": "infeasible_oom",
                    "failure_reason": str(error),
                    "selected_variant": winner["variant"],
                    "source_checkpoint": winner["checkpoint_path"],
                }
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            row = {**base, **result}
            _upsert(raw_rows, row)
            _save_rows(raw_rows, raw_path)
            metadata_dir = output_root / "metadata"
            metadata_dir.mkdir(exist_ok=True)
            (metadata_dir / f"{identity}.json").write_text(
                json.dumps(
                    {
                        "config": asdict(config),
                        "capacity_candidate": asdict(candidate),
                        "dataset": metadata,
                        "winner": winner,
                        "backbone_selection_source": str(args.backbone_selection),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            if result["status"] == "completed":
                reduction = 100 * (
                    result["backbone_rmse_ugm3"] - result["rmse_ugm3"]
                ) / result["backbone_rmse_ugm3"]
                print(
                    f"[完成] {identity}: RMSE={result['rmse_ugm3']:.4f}，"
                    f"C1={reduction:+.3f}%"
                )
            else:
                print(f"[登记] {identity}: {result['status']}")
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
