"""Independently rebuild Round 14 metrics and evaluate preregistered G1--G4."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from run_cross_city_generalization import (
    ARMS,
    B1_STATIONS,
    FORBIDDEN_STATIONS,
    FORMAL_SEEDS,
    TASKS,
    authorized_station_ids,
    build_frozen_run_payload,
    config_fingerprint,
    formal_config,
)


BASELINE, SPATIAL = ARMS


class ArtifactValidationError(RuntimeError):
    """An artifact is incomplete, non-finite, inconsistent, or tampered with."""


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.shape != prediction.shape or target.size == 0:
        raise ArtifactValidationError("真实值与预测值形状不一致或为空")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ArtifactValidationError("真实值或预测值包含非有限数")
    error = prediction - target
    denominator = np.maximum(np.abs(target) + np.abs(prediction), 1e-6)
    return {
        "sse": float(np.square(error).sum()),
        "element_count": int(error.size),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "mae": float(np.abs(error).mean()),
        "smape_percent": float(200 * np.mean(np.abs(error) / denominator)),
    }


def reduction_percent(base_rmse: float, spatial_rmse: float) -> float:
    if not np.isfinite(base_rmse) or base_rmse <= 0 or not np.isfinite(spatial_rmse):
        raise ArtifactValidationError("RMSE 非有限或基线 RMSE 非正")
    return float(100 * (base_rmse - spatial_rmse) / base_rmse)


def _rebuild_block_labels(confirm_timestamps_ns: np.ndarray) -> np.ndarray:
    timestamps = np.asarray(confirm_timestamps_ns, dtype=np.int64)
    if timestamps.ndim != 1 or len(timestamps) < 5:
        raise ArtifactValidationError("完整 C-confirm 时间轴为空或不足五个时间戳")
    if np.any(np.diff(timestamps) <= 0):
        raise ArtifactValidationError("完整 C-confirm 时间轴必须严格递增且唯一")
    quotient, remainder = divmod(len(timestamps), 5)
    sizes = [quotient + (1 if block < remainder else 0) for block in range(5)]
    return np.concatenate([
        np.full(size, block + 1, dtype=np.int16)
        for block, size in enumerate(sizes)
    ])


def _rebuild_scalar_support(metadata: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    history = int(metadata["config"]["history"])
    horizon = int(metadata["config"]["horizon"])
    confirm_start, confirm_stop = [int(value) for value in metadata["bounds"]["confirm"]]
    full_timestamps = np.asarray(metadata["confirm_timestamps_ns"], dtype=np.int64)
    if len(full_timestamps) != confirm_stop - confirm_start:
        raise ArtifactValidationError("完整 C-confirm 时间轴长度与 bounds 不一致")
    full_blocks = _rebuild_block_labels(full_timestamps)
    if not np.array_equal(full_blocks, np.asarray(metadata["confirm_block_labels"], dtype=np.int16)):
        raise ArtifactValidationError("metadata 完整 C-confirm 块标签与协议重建不一致")
    scalar_timestamps, scalar_leads, scalar_blocks = [], [], []
    for start in metadata["sample_indices"]["confirm"]:
        target_start = int(start) + history
        local_start = target_start - confirm_start
        local_stop = local_start + horizon
        if local_start < 0 or local_stop > len(full_timestamps):
            raise ArtifactValidationError("确认样本索引的目标越过 C-confirm 边界")
        scalar_timestamps.extend(full_timestamps[local_start:local_stop].tolist())
        scalar_leads.extend(range(1, horizon + 1))
        scalar_blocks.extend(full_blocks[local_start:local_stop].tolist())
    rebuilt = (
        np.asarray(scalar_timestamps, dtype=np.int64),
        np.asarray(scalar_leads, dtype=np.int16),
        np.asarray(scalar_blocks, dtype=np.int16),
    )
    saved = (
        np.asarray(metadata["confirm_scalar_timestamps_ns"], dtype=np.int64),
        np.asarray(metadata["confirm_scalar_leads"], dtype=np.int16),
        np.asarray(metadata["confirm_scalar_block_labels"], dtype=np.int16),
    )
    for name, rebuilt_values, saved_values in zip(
        ("target timestamps", "leads", "block labels"), rebuilt, saved
    ):
        if not np.array_equal(rebuilt_values, saved_values):
            raise ArtifactValidationError(f"metadata 共享标量 {name} 与独立重建不一致")
    if len(rebuilt[1]) == 0 or len(rebuilt[1]) % horizon != 0:
        raise ArtifactValidationError("metadata 共享标量 lead 数量非法")
    expected_leads = np.tile(np.arange(1, horizon + 1, dtype=np.int16), len(rebuilt[1]) // horizon)
    if not np.array_equal(rebuilt[1], expected_leads):
        raise ArtifactValidationError("每个预测窗口必须严格包含 lead 1..H")
    return rebuilt


def _load_artifact(path: Path, metadata: dict) -> dict[str, np.ndarray]:
    required = {
        "prediction_ugm3", "target_ugm3", "target_timestamp_ns", "lead",
        "block_label", "sse_by_block", "element_count_by_block",
    }
    try:
        with np.load(path, allow_pickle=False) as artifact:
            missing = required - set(artifact.files)
            if missing:
                raise ArtifactValidationError(f"{path} 缺少数组: {sorted(missing)}")
            arrays = {name: np.asarray(artifact[name]) for name in required}
    except ArtifactValidationError:
        raise
    except Exception as exc:
        raise ArtifactValidationError(f"无法读取 {path}: {exc}") from exc
    lengths = [len(arrays[name]) for name in (
        "prediction_ugm3", "target_ugm3", "target_timestamp_ns", "lead", "block_label"
    )]
    if len(set(lengths)) != 1 or lengths[0] == 0:
        raise ArtifactValidationError(f"{path} 逐标量数组长度不一致或为空")
    expected_timestamps, expected_leads, rebuilt_blocks = _rebuild_scalar_support(metadata)
    for name, actual, expected in (
        ("target_timestamp_ns", arrays["target_timestamp_ns"], expected_timestamps),
        ("lead", arrays["lead"], expected_leads),
        ("block_label", arrays["block_label"], rebuilt_blocks),
    ):
        if not np.array_equal(actual, expected):
            raise ArtifactValidationError(f"{path} 的 {name} 与 metadata/协议重建支持不一致")
    horizon = int(metadata["config"]["horizon"])
    if ((arrays["lead"] < 1) | (arrays["lead"] > horizon)).any():
        raise ArtifactValidationError(f"{path} 包含超出 [1,H] 的 lead")
    errors = arrays["prediction_ugm3"].astype(float) - arrays["target_ugm3"].astype(float)
    rebuilt_sse = np.asarray([
        np.square(errors[rebuilt_blocks == block]).sum() for block in range(1, 6)
    ])
    rebuilt_counts = np.asarray([
        (rebuilt_blocks == block).sum() for block in range(1, 6)
    ])
    if not np.allclose(rebuilt_sse, arrays["sse_by_block"], rtol=1e-10, atol=1e-8):
        raise ArtifactValidationError(f"{path} 保存 SSE 与逐预测复算不一致")
    if not np.array_equal(rebuilt_counts, arrays["element_count_by_block"]):
        raise ArtifactValidationError(f"{path} 保存元素数与逐预测复算不一致")
    metrics(arrays["target_ugm3"], arrays["prediction_ugm3"])
    arrays["rebuilt_block_label"] = rebuilt_blocks
    return arrays


def _same_support(left: dict[str, np.ndarray], right: dict[str, np.ndarray], context: str) -> None:
    for name in ("target_ugm3", "target_timestamp_ns", "lead", "rebuilt_block_label", "element_count_by_block"):
        if not np.array_equal(left[name], right[name]):
            raise ArtifactValidationError(f"{context} 两臂共享支持不一致: {name}")


def _discover_runs(input_dir: Path):
    records = []
    metadata_by_key = {}
    manifests = sorted(input_dir.rglob("run_manifest.csv"))
    if not manifests:
        raise ArtifactValidationError("未找到 run_manifest.csv")
    for manifest_path in manifests:
        task_dir = manifest_path.parent
        metadata_path = task_dir / "run_metadata.json"
        if not metadata_path.is_file():
            raise ArtifactValidationError(f"缺少 {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        history = int(metadata["config"]["history"])
        horizon = int(metadata["config"]["horizon"])
        station = int(metadata["center_station_id"])
        key = (history, horizon, station)
        if key in metadata_by_key:
            raise ArtifactValidationError(f"重复任务目录: {key}")
        metadata_by_key[key] = metadata
        manifest = pd.read_csv(manifest_path)
        needed = {
            "arm", "seed", "artifact", "config_fingerprint", "trained_epochs",
            "initialized_from_degraded", "backbone_frozen",
        }
        if not needed <= set(manifest.columns):
            raise ArtifactValidationError(f"{manifest_path} 缺列: {sorted(needed - set(manifest.columns))}")
        rows = manifest.to_dict("records")
        rebuilt_payload = build_frozen_run_payload(metadata)
        if rebuilt_payload != metadata.get("frozen_run_payload"):
            raise ArtifactValidationError(f"{metadata_path} frozen_run_payload 与独立重建不一致")
        expected_fingerprint = config_fingerprint(rebuilt_payload)
        if expected_fingerprint != metadata["config_fingerprint"]:
            raise ArtifactValidationError(f"{metadata_path} 配置指纹复算不一致")
        if not metadata.get("smoke_test"):
            if (history, horizon) not in TASKS or station not in B1_STATIONS:
                raise ArtifactValidationError(f"非预注册正式任务: {key}")
            if metadata["config"] != asdict(formal_config(history, horizon)):
                raise ArtifactValidationError(f"{metadata_path} 正式超参数偏离冻结配置")
            if set(metadata["opened_station_ids"]) != set(authorized_station_ids(station)):
                raise ArtifactValidationError(f"{metadata_path} 实际打开集合不等于授权集合")
            if set(metadata["opened_station_ids"]) & FORBIDDEN_STATIONS:
                raise ArtifactValidationError(f"{metadata_path} 实际打开集合触及禁用站点")
            if tuple(metadata["seeds"]) != FORMAL_SEEDS or tuple(metadata["arms"]) != ARMS:
                raise ArtifactValidationError(f"{metadata_path} 种子或实验臂偏离预注册")
            if len(metadata.get("selected_top5", [])) != 5:
                raise ArtifactValidationError(f"{metadata_path} 未固定选择 5 个邻站")
        for row in rows:
            if row["config_fingerprint"] != metadata["config_fingerprint"]:
                raise ArtifactValidationError(f"{manifest_path} 配置指纹不一致")
            if row["arm"] == BASELINE:
                if bool(row["initialized_from_degraded"]) or bool(row["backbone_frozen"]):
                    raise ArtifactValidationError("退化基线初始化/冻结标志非法")
            elif row["arm"] == SPATIAL:
                if not bool(row["initialized_from_degraded"]) or not bool(row["backbone_frozen"]):
                    raise ArtifactValidationError("ST 臂必须同种子退化初始化并冻结主干")
            else:
                raise ArtifactValidationError(f"未知实验臂: {row['arm']}")
            if not 1 <= int(row["trained_epochs"]) <= int(metadata["config"]["epochs"]):
                raise ArtifactValidationError("训练轮数越界")
            artifact_path = task_dir / str(row["artifact"])
            arrays = _load_artifact(artifact_path, metadata)
            records.append({
                "history": history,
                "horizon": horizon,
                "station": station,
                "arm": str(row["arm"]),
                "seed": int(row["seed"]),
                "trained_epochs": int(row["trained_epochs"]),
                "arrays": arrays,
                "metadata": metadata,
            })
    return records, metadata_by_key


def nominal_sign_tail(successes: int, non_ties: int) -> float:
    if non_ties == 0:
        return 1.0
    return float(sum(math.comb(non_ties, k) for k in range(successes, non_ties + 1)) / (2 ** non_ties))


def protocol_status(smoke: bool, complete: bool, gates_pass: bool) -> str:
    if smoke:
        return "SMOKE_NOT_ELIGIBLE"
    if not complete:
        return "INCOMPLETE"
    return "PASS" if gates_pass else "STOP"


def _aggregate_arm(records, selector):
    targets = np.concatenate([record["arrays"]["target_ugm3"][selector(record)] for record in records])
    predictions = np.concatenate([record["arrays"]["prediction_ugm3"][selector(record)] for record in records])
    return metrics(targets, predictions)


def summarize(input_dir: str | Path, output_dir: str | Path) -> dict:
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    records, metadata_by_key = _discover_runs(input_dir)
    smoke = any(bool(record["metadata"].get("smoke_test")) for record in records)
    keyed = {(r["history"], r["horizon"], r["station"], r["seed"], r["arm"]): r for r in records}
    if len(keyed) != len(records):
        raise ArtifactValidationError("发现重复的任务/站点/种子/实验臂")

    pair_rows, run_rows, station_rows, block_rows, lead_rows = [], [], [], [], []
    peak_rows, quantile_rows = [], []
    task_keys = sorted({(r["history"], r["horizon"]) for r in records})
    for history, horizon in task_keys:
        stations = sorted({r["station"] for r in records if (r["history"], r["horizon"]) == (history, horizon)})
        seeds = sorted({r["seed"] for r in records if (r["history"], r["horizon"]) == (history, horizon)})
        for station in stations:
            station_pair_rows = []
            for seed in seeds:
                pair = []
                for arm in ARMS:
                    key = (history, horizon, station, seed, arm)
                    if key not in keyed:
                        raise ArtifactValidationError(f"缺失实验臂: {key}")
                    record = keyed[key]
                    values = metrics(record["arrays"]["target_ugm3"], record["arrays"]["prediction_ugm3"])
                    run_rows.append({"history": history, "horizon": horizon, "station": station, "seed": seed, "arm": arm, **values})
                    pair.append(record)
                base, spatial = pair
                _same_support(base["arrays"], spatial["arrays"], f"{history}→{horizon}/s{station}/r{seed}")
                base_metric = metrics(base["arrays"]["target_ugm3"], base["arrays"]["prediction_ugm3"])
                spatial_metric = metrics(spatial["arrays"]["target_ugm3"], spatial["arrays"]["prediction_ugm3"])
                d = reduction_percent(base_metric["rmse"], spatial_metric["rmse"])
                pair_row = {"history": history, "horizon": horizon, "station": station, "seed": seed, "rmse_reduction_percent": d}
                pair_rows.append(pair_row)
                station_pair_rows.append(pair_row)

                threshold = float(base["metadata"]["peak_threshold_ugm3"])
                for arm_record in (base, spatial):
                    mask = arm_record["arrays"]["target_ugm3"] >= threshold
                    if mask.any():
                        peak_metric = metrics(arm_record["arrays"]["target_ugm3"][mask], arm_record["arrays"]["prediction_ugm3"][mask])
                    else:
                        peak_metric = {"rmse": math.nan, "mae": math.nan, "smape_percent": math.nan, "sse": 0.0, "element_count": 0}
                    peak_rows.append({"history": history, "horizon": horizon, "station": station, "seed": seed, "arm": arm_record["arm"], "peak_threshold_ugm3": threshold, "coverage_count": int(mask.sum()), **peak_metric})

                for block in range(1, 6):
                    mask = base["arrays"]["rebuilt_block_label"] == block
                    base_block = metrics(base["arrays"]["target_ugm3"][mask], base["arrays"]["prediction_ugm3"][mask])
                    spatial_block = metrics(spatial["arrays"]["target_ugm3"][mask], spatial["arrays"]["prediction_ugm3"][mask])
                    block_rows.append({"history": history, "horizon": horizon, "station": station, "seed": seed, "block": block, "base_rmse": base_block["rmse"], "base_mae": base_block["mae"], "base_smape_percent": base_block["smape_percent"], "spatial_rmse": spatial_block["rmse"], "spatial_mae": spatial_block["mae"], "spatial_smape_percent": spatial_block["smape_percent"], "rmse_reduction_percent": reduction_percent(base_block["rmse"], spatial_block["rmse"])})
                for lead in range(1, horizon + 1):
                    mask = base["arrays"]["lead"] == lead
                    base_lead = metrics(base["arrays"]["target_ugm3"][mask], base["arrays"]["prediction_ugm3"][mask])
                    spatial_lead = metrics(spatial["arrays"]["target_ugm3"][mask], spatial["arrays"]["prediction_ugm3"][mask])
                    lead_rows.append({"history": history, "horizon": horizon, "station": station, "seed": seed, "lead": lead, "base_rmse": base_lead["rmse"], "base_mae": base_lead["mae"], "base_smape_percent": base_lead["smape_percent"], "spatial_rmse": spatial_lead["rmse"], "spatial_mae": spatial_lead["mae"], "spatial_smape_percent": spatial_lead["smape_percent"], "rmse_reduction_percent": reduction_percent(base_lead["rmse"], spatial_lead["rmse"])})

                edges = np.asarray(base["metadata"]["fit_quantile_edges_ugm3"], dtype=float)
                bins = np.digitize(base["arrays"]["target_ugm3"], edges, right=True)
                for quantile_bin in range(5):
                    mask = bins == quantile_bin
                    if not mask.any():
                        continue
                    bm = metrics(base["arrays"]["target_ugm3"][mask], base["arrays"]["prediction_ugm3"][mask])
                    sm = metrics(spatial["arrays"]["target_ugm3"][mask], spatial["arrays"]["prediction_ugm3"][mask])
                    quantile_rows.append({"history": history, "horizon": horizon, "station": station, "seed": seed, "quantile_bin": quantile_bin + 1, "count": int(mask.sum()), "base_smape_percent": bm["smape_percent"], "spatial_smape_percent": sm["smape_percent"], "smape_change_percent_points": sm["smape_percent"] - bm["smape_percent"], "base_rmse": bm["rmse"], "spatial_rmse": sm["rmse"], "base_mae": bm["mae"], "spatial_mae": sm["mae"]})
            station_records = [
                record for record in records
                if (record["history"], record["horizon"], record["station"])
                == (history, horizon, station)
            ]
            arm_metrics = {}
            for arm in ARMS:
                arm_records = [record for record in station_records if record["arm"] == arm]
                arm_metrics[arm] = metrics(
                    np.concatenate([record["arrays"]["target_ugm3"] for record in arm_records]),
                    np.concatenate([record["arrays"]["prediction_ugm3"] for record in arm_records]),
                )
            station_rows.append({
                "history": history, "horizon": horizon, "station": station,
                "base_rmse": arm_metrics[BASELINE]["rmse"],
                "base_mae": arm_metrics[BASELINE]["mae"],
                "base_smape_percent": arm_metrics[BASELINE]["smape_percent"],
                "spatial_rmse": arm_metrics[SPATIAL]["rmse"],
                "spatial_mae": arm_metrics[SPATIAL]["mae"],
                "spatial_smape_percent": arm_metrics[SPATIAL]["smape_percent"],
                "mean_rmse_reduction_percent": float(np.mean([row["rmse_reduction_percent"] for row in station_pair_rows])),
            })

    pair_df = pd.DataFrame(pair_rows)
    block_df = pd.DataFrame(block_rows)
    gate_tasks = []
    for history, horizon in task_keys:
        pairs = pair_df[(pair_df.history == history) & (pair_df.horizon == horizon)]
        blocks = block_df[(block_df.history == history) & (block_df.horizon == horizon)]
        station_effects = pairs.groupby("station")["rmse_reduction_percent"].mean()
        block_effects = blocks.groupby("block")["rmse_reduction_percent"].mean()
        task_arm_metrics = {}
        for arm in ARMS:
            arm_records = [
                record for record in records
                if (record["history"], record["horizon"], record["arm"])
                == (history, horizon, arm)
            ]
            task_arm_metrics[arm] = metrics(
                np.concatenate([record["arrays"]["target_ugm3"] for record in arm_records]),
                np.concatenate([record["arrays"]["prediction_ugm3"] for record in arm_records]),
            )
        successes = int((pairs.rmse_reduction_percent > 0).sum())
        failures = int((pairs.rmse_reduction_percent < 0).sum())
        ties = int((pairs.rmse_reduction_percent == 0).sum())
        threshold = 1.0 if (history, horizon) == (24, 1) else 0.5
        pool_effect = float(pairs.rmse_reduction_percent.mean())
        complete = (
            len(pairs) == 40 and set(pairs.station) == set(B1_STATIONS)
            and set(pairs.seed) == set(FORMAL_SEEDS) and len(block_effects) == 5
        )
        gates = {
            "G1_mean_effect": bool(pool_effect >= threshold),
            "G2_pair_consistency": bool(successes >= 32),
            "G3_station_consistency": bool((station_effects > 0).sum() >= 6),
            "G4_block_robustness": bool((block_effects > 0).sum() >= 4),
        }
        task_pass = bool(complete and all(gates.values()) and not smoke)
        gate_tasks.append({
            "task": f"{history}h_{horizon}h", "complete_formal_matrix": complete,
            "pool_effect_percent": pool_effect, "G1_threshold_percent": threshold,
            "successes": successes, "failures": failures, "ties": ties,
            "nominal_one_sided_sign_tail_probability": nominal_sign_tail(successes, successes + failures),
            "sign_test_note": "Descriptive only: repeated seeds within stations are not independent; blocks are not additional samples.",
            "positive_station_count": int((station_effects > 0).sum()),
            "positive_block_pool_count": int((block_effects > 0).sum()),
            "station_effect_percent": {str(int(k)): float(v) for k, v in station_effects.items()},
            "block_pool_effect_percent": {str(int(k)): float(v) for k, v in block_effects.items()},
            "secondary_metrics": {
                "baseline": task_arm_metrics[BASELINE],
                "spatial": task_arm_metrics[SPATIAL],
                "smape_change_percent_points": (
                    task_arm_metrics[SPATIAL]["smape_percent"]
                    - task_arm_metrics[BASELINE]["smape_percent"]
                ),
                "quantile_decomposition_written": True,
            },
            "gates": gates, "task_pass": task_pass,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(run_rows).to_csv(output_dir / "run_metrics.csv", index=False)
    pair_df.to_csv(output_dir / "paired_effects.csv", index=False)
    pd.DataFrame(station_rows).to_csv(output_dir / "station_metrics.csv", index=False)
    block_df.to_csv(output_dir / "block_metrics.csv", index=False)
    pd.DataFrame(lead_rows).to_csv(output_dir / "lead_metrics.csv", index=False)
    pd.DataFrame(peak_rows).to_csv(output_dir / "peak_metrics.csv", index=False)
    pd.DataFrame(quantile_rows).to_csv(output_dir / "quantile_smape_decomposition.csv", index=False)
    complete_matrix = bool(
        not smoke
        and set(task_keys) == set(TASKS)
        and len(gate_tasks) == 2
        and all(row["complete_formal_matrix"] for row in gate_tasks)
    )
    gates_pass = bool(complete_matrix and all(row["task_pass"] for row in gate_tasks))
    gate_summary = {
        "status": protocol_status(smoke, complete_matrix, gates_pass),
        "conjunctive_two_task_pass": gates_pass,
        "artifact_validation": "PASS",
        "tasks": gate_tasks,
    }
    (output_dir / "gate_summary.json").write_text(json.dumps(gate_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return gate_summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    try:
        result = summarize(args.input_dir, output_dir)
    except Exception as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "gate_summary.json").write_text(
            json.dumps(
                {
                    "status": "INCOMPLETE",
                    "conjunctive_two_task_pass": False,
                    "artifact_validation": "FAIL",
                    "error": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (output_dir / "verification_failure.json").write_text(
            json.dumps(
                {"status": "INCOMPLETE", "artifact_validation": "FAIL", "error": str(exc)},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
