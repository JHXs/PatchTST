"""Independently recompute MC-CVLRA L1 metrics, diagnostics, and hard gates.

The summarizer consumes validation-level CSV artifacts rather than trusting runner
metric rows.  Smoke inputs are accepted only with ``--smoke`` and always produce
``NOT_EVALUABLE_SMOKE`` instead of a performance gate decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from cross_variable_lag_adapter import (
    assert_tensor_mappings_equal,
    expected_m_parameter_count,
    select_flat_width,
    tensor_mapping_hash,
)
from cross_variable_lag_l0_data import AUXILIARY_COLUMNS
from run_cross_variable_lag_l1 import (
    ALL_VARIANTS,
    FORMAL_OUTPUTS,
    FORMAL_SEEDS,
    KEY_L1_SOURCE_FILES,
    SHUFFLE_SEEDS,
    patchtst_checkpoint_substate,
)


REQUIRED_INPUT_FILES = (
    "run_manifest.json",
    "experiment_config.json",
    "dataset_metadata.json",
    "fit_ledger.csv",
    "station_selection.csv",
    "data_quality.csv",
    "target_time_sets.csv",
    "boundary_checks.csv",
    "parameter_audit.csv",
    "flat_width_candidates.csv",
    "run_records.csv",
    "checkpoint_mapping.csv",
    "patchtst_state_audit.csv",
    "validation_predictions.csv",
    "gradient_audit.csv",
    "gate_weight_diagnostics.csv",
    "group_mass_diagnostics.csv",
    "m_shuffle_predictions.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir-24", default=str(FORMAL_OUTPUTS[(24, 1)]))
    parser.add_argument("--dir-168", default=str(FORMAL_OUTPUTS[(168, 6)]))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--gpu-smoke-dir",
        default=(
            "experiments/results/cross_variable_lag_adapter/"
            "l1_168h_6h_batch512_gpu_smoke"
        ),
        help="Separate 168->6 batch-512 GPU resource smoke; checked only with --smoke.",
    )
    parser.add_argument(
        "--allow-missing-gpu-smoke",
        action="store_true",
        help=(
            "Write a fail-closed smoke recomputation when no GPU device is exposed; "
            "integrity remains false until the real GPU resource smoke exists."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Require _smoke inputs and suppress every formal L1 gate decision.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest_provenance(manifest: dict[str, Any], smoke: bool) -> None:
    required = {
        "git_commit",
        "git_dirty",
        "git_status_porcelain",
        "key_l1_source_sha256",
        "formal_clean_tree_required",
        "precommit_engineering_smoke",
        "provenance_statement",
    }
    if not required.issubset(manifest):
        raise AssertionError(
            f"Run manifest lacks provenance fields: {sorted(required - set(manifest))}"
        )
    commit = str(manifest["git_commit"])
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise AssertionError("Run manifest git commit is not a full SHA-1")
    if not bool(manifest["formal_clean_tree_required"]):
        raise AssertionError("Formal clean-tree requirement is not enabled")
    if smoke:
        if bool(manifest["git_dirty"]) != bool(manifest["precommit_engineering_smoke"]):
            raise AssertionError("Dirty smoke is not explicitly labelled pre-commit")
        if manifest["git_dirty"] and "PRE-COMMIT" not in manifest["provenance_statement"]:
            raise AssertionError("Dirty smoke provenance statement is ambiguous")
    elif manifest["git_dirty"] or manifest["precommit_engineering_smoke"]:
        raise AssertionError("Formal L1 artifact came from a dirty tree")

    source_hashes = manifest["key_l1_source_sha256"]
    repo_root = Path(__file__).resolve().parent
    expected_present = {
        name for name in KEY_L1_SOURCE_FILES if (repo_root / name).is_file()
    }
    if set(source_hashes) != expected_present:
        raise AssertionError("Manifest key L1 source file set differs from current contract")
    for relative_path, recorded_hash in source_hashes.items():
        if _sha256(repo_root / relative_path) != recorded_hash:
            raise AssertionError(f"L1 source hash mismatch: {relative_path}")


def load_checkpoint_patchtst(path: Path) -> dict[str, torch.Tensor]:
    full_state = torch.load(path, map_location="cpu", weights_only=True)
    return patchtst_checkpoint_substate(full_state)


def verify_patchtst_checkpoint_lineage(
    path: Path,
    checkpoints: pd.DataFrame,
    run_records: pd.DataFrame,
    smoke: bool,
) -> None:
    base_checkpoints = checkpoints[checkpoints["variant"].isin(["F00", "F10"])]
    expected_seeds = {2060} if smoke else set(FORMAL_SEEDS)
    if set(base_checkpoints["seed"].astype(int)) != expected_seeds:
        raise AssertionError("Base checkpoint seed set is incomplete")
    audits = pd.read_csv(path / "patchtst_state_audit.csv", keep_default_na=False)
    for seed in sorted(expected_seeds):
        paired = base_checkpoints[base_checkpoints["seed"].astype(int) == seed]
        if set(paired["variant"]) != {"F00", "F10"}:
            raise AssertionError(f"Seed {seed} lacks paired F00/F10 checkpoints")
        f00_row = paired[paired["variant"] == "F00"].iloc[0]
        f10_row = paired[paired["variant"] == "F10"].iloc[0]
        f00_patch = load_checkpoint_patchtst(Path(f00_row["checkpoint_path"]))
        f10_patch = load_checkpoint_patchtst(Path(f10_row["checkpoint_path"]))
        assert_tensor_mappings_equal(
            f10_patch,
            f00_patch,
            f"seed={seed} F10 checkpoint vs F00 initialization checkpoint",
        )
        f00_patch_hash = tensor_mapping_hash(f00_patch)
        f10_patch_hash = tensor_mapping_hash(f10_patch)
        if f00_patch_hash != f00_row["patchtst_checkpoint_state_sha256"]:
            raise AssertionError("F00 checkpoint PatchTST hash audit mismatch")
        f10_hash_fields = (
            "source_f00_checkpoint_patchtst_sha256",
            "patchtst_initial_state_sha256",
            "patchtst_checkpoint_state_sha256",
            "patchtst_final_loaded_state_sha256",
        )
        if any(str(f10_row[field]) != f00_patch_hash for field in f10_hash_fields):
            raise AssertionError("F10 PatchTST lineage hashes are not all F00-derived")
        if f10_patch_hash != f00_patch_hash:
            raise AssertionError("F10 checkpoint PatchTST hash differs from F00")
        if not _bool_mask(
            pd.Series([f10_row["patchtst_state_unchanged_from_f00_initialization"]])
        )[0]:
            raise AssertionError("F10 checkpoint mapping does not assert frozen PatchTST")

        f10_run = run_records[
            (run_records["seed"].astype(int) == seed)
            & (run_records["variant"] == "F10")
        ].iloc[0]
        if not (
            _bool_mask(pd.Series([f10_run["patchtst_state_unchanged"]]))[0]
            and _bool_mask(pd.Series([f10_run["patchtst_locked_eval"]]))[0]
        ):
            raise AssertionError("F10 run record failed the locked PatchTST contract")
        run_hash_fields = (
            "source_f00_checkpoint_patchtst_sha256",
            "patchtst_initial_state_sha256",
            "patchtst_checkpoint_state_sha256",
            "patchtst_final_loaded_state_sha256",
        )
        if any(str(f10_run[field]) != f00_patch_hash for field in run_hash_fields):
            raise AssertionError("F10 run lineage hashes differ from actual F00 checkpoint")

        seed_audits = audits[
            (audits["seed"].astype(int) == seed) & (audits["variant"] == "F10")
        ]
        required_stages = {
            "f10_after_f00_checkpoint_initialization",
            "epoch_after_validation",
            "best_checkpoint_saved",
            "final_best_checkpoint_loaded",
        }
        if not required_stages.issubset(set(seed_audits["stage"])):
            raise AssertionError(f"F10 seed={seed} PatchTST audit stages are incomplete")
        training_log = pd.read_csv(path / "training_logs" / f"F10_seed{seed}.csv")
        epoch_audits = seed_audits[seed_audits["stage"] == "epoch_after_validation"]
        if len(epoch_audits) != len(training_log):
            raise AssertionError("F10 per-epoch PatchTST audit count mismatch")
        if not (
            _bool_mask(seed_audits["invariant_required"]).all()
            and _bool_mask(seed_audits["persistent_state_equal"]).all()
            and _bool_mask(seed_audits["all_parameters_and_buffers_equal"]).all()
            and _bool_mask(seed_audits["patchtst_eval"]).all()
            and (seed_audits["persistent_changed_tensor_count"].astype(int) == 0).all()
            and (seed_audits["complete_changed_tensor_count"].astype(int) == 0).all()
        ):
            raise AssertionError("F10 per-stage PatchTST state audit failed")
        if not (seed_audits["patchtst_persistent_state_sha256"] == f00_patch_hash).all():
            raise AssertionError("F10 audit hash differs from actual F00 checkpoint")


def verify_gpu_resource_smoke(path: Path) -> dict[str, Any]:
    if not path.name.endswith("_gpu_smoke"):
        raise AssertionError("GPU resource smoke directory lacks _gpu_smoke suffix")
    resource_path = path / "resource_smoke.json"
    manifest_path = path / "run_manifest.json"
    if not resource_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("GPU resource smoke JSON artifacts are incomplete")
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if resource != manifest:
        raise AssertionError("GPU resource smoke manifest and result differ")
    verify_manifest_provenance(manifest, smoke=True)
    required_exact = {
        "task": "168h_6h",
        "variant": "F11",
        "history": 168,
        "horizon": 6,
        "batch_size_requested": 512,
        "batch_size_actual": 512,
        "device_type": "cuda",
        "forward_backward_completed": True,
        "loss_finite": True,
        "base_fully_frozen": True,
        "base_gradients_none": True,
        "base_eval": True,
        "patchtst_eval": True,
        "base_state_unchanged": True,
        "performance_interpretation_prohibited": True,
        "formal_selection_performed": False,
        "formal_test_used": False,
    }
    for field, expected in required_exact.items():
        if resource.get(field) != expected:
            raise AssertionError(
                f"GPU resource smoke field {field}={resource.get(field)!r}, expected {expected!r}"
            )
    if resource["base_state_before_sha256"] != resource["base_state_after_sha256"]:
        raise AssertionError("GPU smoke frozen base persistent state changed")
    if (
        resource["base_complete_state_before_sha256"]
        != resource["base_complete_state_after_sha256"]
    ):
        raise AssertionError("GPU smoke frozen base parameter/buffer state changed")
    if (
        resource["patchtst_complete_state_before_sha256"]
        != resource["patchtst_complete_state_after_sha256"]
    ):
        raise AssertionError("GPU smoke PatchTST parameter/buffer state changed")
    if int(resource["peak_memory_allocated_bytes"]) <= 0:
        raise AssertionError("GPU peak allocated memory was not recorded")
    checkpoint_path = Path(resource["checkpoint_path"])
    if not checkpoint_path.is_file() or _sha256(checkpoint_path) != resource[
        "checkpoint_sha256"
    ]:
        raise AssertionError("GPU smoke source F10 checkpoint hash mismatch")
    zero_keys = (
        "test_feature_rows_loaded",
        "test_target_values_loaded",
        "test_samples_constructed",
        "test_predictions_constructed",
        "test_caches_constructed",
        "test_metrics_constructed",
    )
    if any(int(resource["access_metadata"][key]) != 0 for key in zero_keys):
        raise AssertionError("GPU resource smoke touched formal test data")
    return {
        "path": str(path),
        "verified": True,
        "device": resource["device_name"],
        "batch_size": 512,
        "peak_memory_allocated_bytes": resource["peak_memory_allocated_bytes"],
        "peak_memory_reserved_bytes": resource["peak_memory_reserved_bytes"],
        "artifact_sha256": {
            "resource_smoke.json": _sha256(resource_path),
            "run_manifest.json": _sha256(manifest_path),
        },
    }


def regression_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if actual.shape != prediction.shape or actual.size == 0:
        raise ValueError("Metric arrays must have the same non-empty shape")
    error = prediction - actual
    denominator = np.abs(actual) + np.abs(prediction)
    return {
        "mse": float(np.mean(error ** 2)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "smape_percent": float(
            200.0 * np.mean(np.abs(error) / np.maximum(denominator, 1e-6))
        ),
        "bias": float(np.mean(error)),
    }


def _bool_mask(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)
    normalized = series.astype(str).str.lower()
    if not normalized.isin(["true", "false"]).all():
        raise ValueError(f"Unexpected boolean values in {series.name}")
    return normalized.eq("true").to_numpy()


def recompute_prediction_metrics(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {
        "task",
        "seed",
        "variant",
        "split",
        "sample_index",
        "origin_time",
        "horizon_step",
        "target_time",
        "target_scaled",
        "prediction_scaled",
        "target_ugm3",
        "prediction_ugm3",
        "high_pollution",
        "rapid_rise",
        "decline",
        "ordinary",
    }
    if not required.issubset(predictions.columns):
        raise ValueError(f"Validation prediction schema missing {sorted(required - set(predictions.columns))}")
    if set(predictions["split"].unique()) != {"validation"}:
        raise AssertionError("Only validation predictions may enter L1 summarization")
    key_columns = ["task", "seed", "variant", "sample_index", "horizon_step"]
    if predictions.duplicated(key_columns).any():
        raise AssertionError("Duplicate validation prediction keys")

    overall_rows = []
    horizon_rows = []
    scenario_rows = []
    for (task, seed, variant), group in predictions.groupby(
        ["task", "seed", "variant"], sort=False
    ):
        physical = regression_metrics(group["target_ugm3"], group["prediction_ugm3"])
        scaled = regression_metrics(group["target_scaled"], group["prediction_scaled"])
        overall_rows.append(
            {
                "task": task,
                "seed": int(seed),
                "variant": variant,
                "cell_count": len(group),
                "origin_count": group["sample_index"].nunique(),
                "mse_scaled": scaled["mse"],
                "rmse_scaled": scaled["rmse"],
                "mae_scaled": scaled["mae"],
                "rmse_ugm3": physical["rmse"],
                "mae_ugm3": physical["mae"],
                "smape_percent": physical["smape_percent"],
                "bias_ugm3": physical["bias"],
            }
        )
        for horizon_step, horizon_group in group.groupby("horizon_step", sort=True):
            values = regression_metrics(
                horizon_group["target_ugm3"], horizon_group["prediction_ugm3"]
            )
            horizon_rows.append(
                {
                    "task": task,
                    "seed": int(seed),
                    "variant": variant,
                    "horizon_step": int(horizon_step),
                    "cell_count": len(horizon_group),
                    "rmse_ugm3": values["rmse"],
                    "mae_ugm3": values["mae"],
                    "smape_percent": values["smape_percent"],
                    "bias_ugm3": values["bias"],
                }
            )
        for scenario in ("high_pollution", "rapid_rise", "decline", "ordinary"):
            mask = _bool_mask(group[scenario])
            if not mask.any():
                continue
            values = regression_metrics(
                group.loc[mask, "target_ugm3"], group.loc[mask, "prediction_ugm3"]
            )
            scenario_rows.append(
                {
                    "task": task,
                    "seed": int(seed),
                    "variant": variant,
                    "scenario": scenario,
                    "cell_count": int(mask.sum()),
                    "rmse_ugm3": values["rmse"],
                    "mae_ugm3": values["mae"],
                    "smape_percent": values["smape_percent"],
                    "bias_ugm3": values["bias"],
                }
            )
    return (
        pd.DataFrame(overall_rows),
        pd.DataFrame(horizon_rows),
        pd.DataFrame(scenario_rows),
    )


def compute_factorial_metrics(overall: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_columns = ("rmse_ugm3", "mae_ugm3", "smape_percent")
    for (task, seed), group in overall.groupby(["task", "seed"], sort=False):
        indexed = group.set_index("variant")
        missing = {"F00", "F01", "F10", "F11"} - set(indexed.index)
        if missing:
            raise ValueError(f"2x2 cells missing for {task} seed={seed}: {sorted(missing)}")
        for metric in metric_columns:
            f00 = float(indexed.loc["F00", metric])
            f01 = float(indexed.loc["F01", metric])
            f10 = float(indexed.loc["F10", metric])
            f11 = float(indexed.loc["F11", metric])
            gain_a_s0 = f00 - f01
            gain_a_s1 = f10 - f11
            rows.append(
                {
                    "task": task,
                    "seed": int(seed),
                    "metric": metric,
                    "F00": f00,
                    "F01": f01,
                    "F10": f10,
                    "F11": f11,
                    "G_A_given_S0": gain_a_s0,
                    "G_A_given_S1": gain_a_s1,
                    "G_S_given_A0": f00 - f10,
                    "G_comb": f00 - f11,
                    "I": gain_a_s0 - gain_a_s1,
                    "R_attn": (
                        1.0 - gain_a_s1 / gain_a_s0 if gain_a_s0 > 0 else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def compute_paired_metrics(overall: pd.DataFrame) -> pd.DataFrame:
    comparisons = (
        ("F01", "F00", "adapter_gain_without_spatial"),
        ("F10", "F00", "spatial_gain_without_adapter"),
        ("F11", "F10", "M_vs_locked_ST"),
        ("F11", "F00", "combined_vs_degraded"),
        ("F11", "B1", "M_vs_self_history"),
        ("F11", "B2", "M_vs_recent_only"),
        ("F11", "B_flat", "M_vs_flat"),
        ("F11", "B3", "M_vs_uniform_gate"),
    )
    rows = []
    for (task, seed), group in overall.groupby(["task", "seed"], sort=False):
        indexed = group.set_index("variant")
        for candidate, baseline, comparison in comparisons:
            if candidate not in indexed.index or baseline not in indexed.index:
                raise ValueError(f"Missing {candidate}/{baseline} for {task} seed={seed}")
            for metric in ("rmse_ugm3", "mae_ugm3", "smape_percent"):
                baseline_value = float(indexed.loc[baseline, metric])
                candidate_value = float(indexed.loc[candidate, metric])
                rows.append(
                    {
                        "task": task,
                        "seed": int(seed),
                        "comparison": comparison,
                        "candidate": candidate,
                        "baseline": baseline,
                        "metric": metric,
                        "baseline_error": baseline_value,
                        "candidate_error": candidate_value,
                        "absolute_improvement": baseline_value - candidate_value,
                        "relative_improvement_percent": 100.0
                        * (baseline_value - candidate_value)
                        / baseline_value,
                        "improved": candidate_value < baseline_value,
                    }
                )
    return pd.DataFrame(rows)


def recompute_shuffle_metrics(
    predictions: pd.DataFrame,
    shuffle_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    permutation_rows = []
    summary_rows = []
    for (task, train_seed), group in shuffle_predictions.groupby(
        ["task", "train_seed"], sort=False
    ):
        diagnostic_count_values = group["diagnostic_origin_count"].unique()
        if len(diagnostic_count_values) != 1:
            raise AssertionError("M-shuffle diagnostic count differs across permutations")
        diagnostic_count = int(diagnostic_count_values[0])
        main = predictions[
            (predictions["task"] == task)
            & (predictions["seed"] == train_seed)
            & (predictions["sample_index"] < diagnostic_count)
            & (predictions["variant"].isin(["F10", "F11"]))
        ]
        pivot = main.pivot(
            index=["sample_index", "horizon_step"],
            columns="variant",
            values=["target_ugm3", "prediction_ugm3"],
        ).sort_index()
        if len(pivot) != diagnostic_count * int(main["horizon_step"].nunique()):
            raise AssertionError("M-shuffle original diagnostic subset is incomplete")
        target_f10 = pivot[("target_ugm3", "F10")].to_numpy()
        target_f11 = pivot[("target_ugm3", "F11")].to_numpy()
        if not np.array_equal(target_f10, target_f11):
            raise AssertionError("F10/F11 diagnostic targets differ")
        base_prediction = pivot[("prediction_ugm3", "F10")].to_numpy()
        original_prediction = pivot[("prediction_ugm3", "F11")].to_numpy()
        rmse_base = regression_metrics(target_f10, base_prediction)["rmse"]
        rmse_original = regression_metrics(target_f10, original_prediction)["rmse"]
        gain_original = rmse_base - rmse_original
        shuffled_gains = []
        for permutation_seed, permutation_group in group.groupby(
            "permutation_seed", sort=True
        ):
            expected_rows = diagnostic_count * int(
                permutation_group["horizon_step"].nunique()
            )
            if len(permutation_group) != expected_rows:
                raise AssertionError("Incomplete M-shuffle permutation predictions")
            mapping = permutation_group[
                ["diagnostic_sample_index", "source_sample_index"]
            ].drop_duplicates()
            if len(mapping) != diagnostic_count:
                raise AssertionError("M-shuffle source mapping is not sample-complete")
            mapping = mapping.sort_values("diagnostic_sample_index")
            receiver = mapping["diagnostic_sample_index"].to_numpy(dtype=np.int64)
            source = mapping["source_sample_index"].to_numpy(dtype=np.int64)
            if not np.array_equal(receiver, np.arange(diagnostic_count)):
                raise AssertionError("M-shuffle receiver order is not complete")
            if not np.array_equal(receiver % 24, source % 24):
                raise AssertionError("M-shuffle did not preserve within-block positions")
            receiver_blocks = receiver.reshape(-1, 24)[:, 0] // 24
            source_block_matrix = (source // 24).reshape(-1, 24)
            if not np.all(source_block_matrix == source_block_matrix[:, :1]):
                raise AssertionError("M-shuffle split a source block")
            if set(source_block_matrix[:, 0]) != set(receiver_blocks):
                raise AssertionError("M-shuffle source blocks are not a permutation")
            ordered = permutation_group.sort_values(
                ["diagnostic_sample_index", "horizon_step"]
            )
            target = ordered["target_ugm3"].to_numpy()
            shuffled_prediction = ordered["shuffled_prediction_ugm3"].to_numpy()
            shuffled_base = ordered["base_prediction_ugm3"].to_numpy()
            if not np.array_equal(target, target_f10):
                raise AssertionError("M-shuffle targets differ from original subset")
            if not np.allclose(shuffled_base, base_prediction, rtol=0, atol=1e-6):
                raise AssertionError("M-shuffle changed the frozen base prediction")
            rmse_shuffle = regression_metrics(target, shuffled_prediction)["rmse"]
            gain_shuffle = rmse_base - rmse_shuffle
            shuffled_gains.append(gain_shuffle)
            permutation_rows.append(
                {
                    "task": task,
                    "train_seed": int(train_seed),
                    "permutation_seed": int(permutation_seed),
                    "diagnostic_origin_count": diagnostic_count,
                    "excluded_tail_origin_count": int(
                        permutation_group["excluded_tail_origin_count"].iloc[0]
                    ),
                    "rmse_base": rmse_base,
                    "rmse_original_M": rmse_original,
                    "rmse_shuffled_M": rmse_shuffle,
                    "G_orig": gain_original,
                    "G_shuffle": gain_shuffle,
                }
            )
        if set(group["permutation_seed"].astype(int)) != set(SHUFFLE_SEEDS):
            raise AssertionError("M-shuffle permutation seeds differ from 3101-3105")
        mean_shuffled_gain = float(np.mean(shuffled_gains))
        loss_fraction = (
            1.0 - mean_shuffled_gain / gain_original
            if gain_original > 0
            else float("nan")
        )
        summary_rows.append(
            {
                "task": task,
                "train_seed": int(train_seed),
                "diagnostic_origin_count": diagnostic_count,
                "G_orig": gain_original,
                "mean_G_shuffle": mean_shuffled_gain,
                "L_s": loss_fraction,
                "G_orig_positive": gain_original > 0,
                "shuffle_loss_at_least_50_percent": (
                    bool(loss_fraction >= 0.50) if np.isfinite(loss_fraction) else False
                ),
            }
        )
    return pd.DataFrame(permutation_rows), pd.DataFrame(summary_rows)


def _comparison_observation(
    overall: pd.DataFrame,
    task: str,
    baseline: str,
    candidate: str,
) -> dict[str, Any]:
    subset = overall[
        (overall["task"] == task)
        & (overall["variant"].isin([baseline, candidate]))
    ]
    pivot = subset.pivot(index="seed", columns="variant", values="rmse_ugm3")
    if baseline not in pivot or candidate not in pivot:
        raise ValueError(f"Gate comparison missing {baseline}/{candidate} for {task}")
    pivot = pivot.dropna()
    improvements = 100.0 * (pivot[baseline] - pivot[candidate]) / pivot[baseline]
    return {
        "seeds": [int(value) for value in pivot.index],
        "improvements_percent": [float(value) for value in improvements],
        "improved_seed_count": int((improvements > 0).sum()),
        "seed_count": len(improvements),
        "mean_improvement_percent": float(improvements.mean()),
    }


def compute_l1_gate(
    overall: pd.DataFrame,
    horizon: pd.DataFrame,
    scenario: pd.DataFrame,
    shuffle_summary: pd.DataFrame,
    smoke: bool,
) -> dict[str, Any]:
    if smoke:
        return {
            "status": "NOT_EVALUABLE_SMOKE",
            "passed": False,
            "formal_selection_performed": False,
            "reason": "Smoke artifacts are permanently excluded from L1 selection and paper evidence.",
        }
    required_tasks = {"24h_1h", "168h_6h"}
    if set(overall["task"].unique()) != required_tasks:
        raise ValueError("Formal L1 gate requires both frozen tasks")
    for task in required_tasks:
        task_seeds = set(overall.loc[overall["task"] == task, "seed"].astype(int))
        if task_seeds != set(FORMAL_SEEDS):
            raise ValueError(f"Formal {task} seeds differ from 2060-2062")

    versus_f10 = _comparison_observation(overall, "168h_6h", "F10", "F11")
    versus_b1 = _comparison_observation(overall, "168h_6h", "B1", "F11")
    versus_b2 = _comparison_observation(overall, "168h_6h", "B2", "F11")
    versus_flat = _comparison_observation(overall, "168h_6h", "B_flat", "F11")
    noninferior_24 = _comparison_observation(overall, "24h_1h", "F10", "F11")

    horizon_168 = horizon[horizon["task"] == "168h_6h"]
    horizon_mean = horizon_168.groupby(
        ["variant", "horizon_step"], as_index=False
    )["rmse_ugm3"].mean()
    horizon_pivot = horizon_mean.pivot(
        index="horizon_step", columns="variant", values="rmse_ugm3"
    )
    improved_horizons = int((horizon_pivot["F11"] < horizon_pivot["F10"]).sum())

    high = scenario[
        (scenario["task"] == "168h_6h")
        & (scenario["scenario"] == "high_pollution")
        & (scenario["variant"].isin(["F10", "F11"]))
    ]
    high_pivot = high.pivot(index="seed", columns="variant", values="rmse_ugm3").dropna()
    if len(high_pivot) != 3:
        raise ValueError("High-pollution safety gate requires three paired seeds")
    high_relative_change = 100.0 * (high_pivot["F11"] / high_pivot["F10"] - 1.0)

    required_shuffle = shuffle_summary[shuffle_summary["task"] == "168h_6h"]
    if set(required_shuffle["train_seed"].astype(int)) != set(FORMAL_SEEDS):
        raise ValueError("Formal M-shuffle requires separate 2060-2062 rows")
    shuffle_pass = bool(
        required_shuffle["G_orig_positive"].astype(bool).all()
        and required_shuffle["shuffle_loss_at_least_50_percent"].astype(bool).all()
    )
    gates = {
        "M_vs_F10": {
            "passed": versus_f10["improved_seed_count"] == 3
            and versus_f10["mean_improvement_percent"] >= 0.5,
            "required": "3/3 improved and mean paired RMSE improvement >=0.5%",
            "observed": versus_f10,
        },
        "M_vs_B1": {
            "passed": versus_b1["improved_seed_count"] == 3
            and versus_b1["mean_improvement_percent"] >= 0.2,
            "required": "3/3 improved and mean paired RMSE improvement >=0.2%",
            "observed": versus_b1,
        },
        "M_vs_B2": {
            "passed": versus_b2["improved_seed_count"] >= 2
            and versus_b2["mean_improvement_percent"] >= 0.2,
            "required": ">=2/3 improved and mean paired RMSE improvement >=0.2%",
            "observed": versus_b2,
        },
        "M_vs_B_flat": {
            "passed": versus_flat["improved_seed_count"] >= 2
            and versus_flat["mean_improvement_percent"] >= 0.2,
            "required": ">=2/3 improved and mean paired RMSE improvement >=0.2%",
            "observed": versus_flat,
        },
        "horizon_direction": {
            "passed": improved_horizons >= 4,
            "required": ">=4/6 mean horizon RMSE values improve over F10",
            "observed_improved_horizons": improved_horizons,
        },
        "high_pollution_safety": {
            "passed": float(high_relative_change.mean()) <= 1.0,
            "required": "mean paired high-pollution RMSE worsening <=1.0%",
            "observed_relative_changes_percent": [
                float(value) for value in high_relative_change
            ],
            "observed_mean_relative_change_percent": float(
                high_relative_change.mean()
            ),
        },
        "task_24_noninferiority": {
            "passed": -noninferior_24["mean_improvement_percent"] <= 0.2,
            "required": "mean paired 24->1 RMSE worsening <=0.2%",
            "observed": noninferior_24,
        },
        "M_shuffle": {
            "passed": shuffle_pass,
            "required": "each seed G_orig>0 and L_s>=0.50",
            "observed": required_shuffle.to_dict(orient="records"),
        },
        "factorial_combination": {
            "passed": versus_f10["improved_seed_count"] == 3
            and versus_f10["mean_improvement_percent"] >= 0.5,
            "required": "G_A|S=1 satisfies M_vs_F10 gate",
            "observed": versus_f10,
        },
    }
    passed = all(bool(value["passed"]) for value in gates.values())
    return {
        "status": "PASS" if passed else "STOP",
        "passed": passed,
        "formal_selection_performed": True,
        "gates": gates,
    }


def verify_input_directory(path: Path, smoke: bool) -> dict[str, Any]:
    missing = [name for name in REQUIRED_INPUT_FILES if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing L1 artifacts in {path}: {missing}")
    manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
    config = json.loads((path / "experiment_config.json").read_text(encoding="utf-8"))
    metadata = json.loads((path / "dataset_metadata.json").read_text(encoding="utf-8"))
    if bool(manifest["smoke"]) != smoke:
        raise AssertionError("Summarizer smoke mode differs from runner manifest")
    verify_manifest_provenance(manifest, smoke)
    if smoke and not path.name.endswith("_smoke"):
        raise AssertionError("Smoke input directory lacks _smoke suffix")
    if not smoke and path.name.endswith("_smoke"):
        raise AssertionError("Formal summarization rejected a _smoke input")
    access = metadata["access_metadata"]
    zero_keys = (
        "test_feature_rows_loaded",
        "test_target_values_loaded",
        "test_samples_constructed",
        "test_predictions_constructed",
        "test_caches_constructed",
        "test_metrics_constructed",
    )
    if any(int(access[key]) != 0 for key in zero_keys):
        raise AssertionError("A formal test access counter is non-zero")
    if manifest["formal_test_used"] or manifest["formal_test_features_or_targets_loaded"]:
        raise AssertionError("Runner manifest reports formal test use")

    boundaries = pd.read_csv(path / "boundary_checks.csv")
    if not (
        (boundaries["target_time_intersection_count"] == 0).all()
        and boundaries["assertion_passed"].astype(bool).all()
    ):
        raise AssertionError("Official target-time disjointness failed")
    station = pd.read_csv(path / "station_selection.csv")
    if set(station["station_id"].astype(int)) != set(range(1001, 1037)):
        raise AssertionError("Station pool is not exactly 1001-1036")
    center_row = station[station["station_id"] == 1013].iloc[0]
    if bool(center_row["candidate_neighbor"]) or bool(center_row["selected_neighbor"]):
        raise AssertionError("Center station entered the neighbor pool")
    selected = station[station["selected_neighbor"].astype(bool)]
    if not (selected["raw_completeness"] == 1.0).all():
        raise AssertionError("A selected station is incomplete in raw training rows")
    if not (selected["pearson_to_center"] > 0.85).all():
        raise AssertionError("A selected station failed the frozen Pearson threshold")

    if metadata["auxiliary_variable_order"] != list(AUXILIARY_COLUMNS):
        raise AssertionError("P=7 variable order differs from the signed protocol")
    if metadata["lags"] != [0, 1, 2, 3, 6, 12]:
        raise AssertionError("Lag order differs from the frozen protocol")
    if not metadata["all_target_time_sets_disjoint"]:
        raise AssertionError("Dataset metadata reports target-time leakage")

    parameter = pd.read_csv(path / "parameter_audit.csv")
    horizon = int(config["horizon"])
    m_count = expected_m_parameter_count(horizon)
    if not (
        parameter.loc[parameter["variant"] != "B_flat", "trainable_parameter_count"]
        == m_count
    ).all():
        raise AssertionError("M-family trainable parameter count mismatch")
    flat = parameter[parameter["variant"] == "B_flat"].iloc[0]
    flat_info = select_flat_width(horizon)
    if int(flat["flat_width"]) != int(flat_info["selected_width"]):
        raise AssertionError("B-flat width differs from count-only selection")
    if float(flat["relative_to_m_percent"]) > 3.0:
        raise AssertionError("B-flat parameter difference exceeds 3%")

    checkpoints = pd.read_csv(path / "checkpoint_mapping.csv", keep_default_na=False)
    for row in checkpoints.to_dict(orient="records"):
        checkpoint_path = Path(row["checkpoint_path"])
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        if _sha256(checkpoint_path) != row["checkpoint_sha256"]:
            raise AssertionError(f"Checkpoint hash mismatch: {checkpoint_path}")
    run_records = pd.read_csv(path / "run_records.csv")
    adapter_records = run_records[
        run_records["variant"].isin(["F01", "F11", "B1", "B2", "B_flat", "B3"])
    ]
    if not _bool_mask(adapter_records["base_state_unchanged"]).all():
        raise AssertionError("A frozen base state hash changed")
    if not (
        _bool_mask(adapter_records["base_eval_during_adapter"]).all()
        and (~_bool_mask(adapter_records["base_parameters_require_grad"])).all()
        and _bool_mask(adapter_records["base_gradients_none"]).all()
    ):
        raise AssertionError("An adapter base was not held in eval/no-grad state")
    verify_patchtst_checkpoint_lineage(path, checkpoints, run_records, smoke)
    expected_seeds = {2060} if smoke else set(FORMAL_SEEDS)
    if set(run_records["seed"].astype(int)) != expected_seeds:
        raise AssertionError("Actual run seeds differ from frozen smoke/formal set")
    for seed, group in run_records.groupby("seed"):
        if set(group["variant"]) != set(ALL_VARIANTS):
            raise AssertionError(f"Seed {seed} lacks an L1 variant")
        m_family = group[group["variant"].isin(["F01", "F11", "B1", "B2", "B3"])]
        if m_family["initial_adapter_state_sha256"].nunique() != 1:
            raise AssertionError("M-family variants did not share the same initialization")
        checkpoint_by_variant = group.set_index("variant")["checkpoint_sha256"]
        f01_base = group.loc[group["variant"] == "F01", "base_checkpoint_sha256"].iloc[0]
        if f01_base != checkpoint_by_variant["F00"]:
            raise AssertionError("F01 did not share the paired F00 checkpoint")
        f10_hash = checkpoint_by_variant["F10"]
        other_bases = group.loc[
            group["variant"].isin(["F11", "B1", "B2", "B_flat", "B3"]),
            "base_checkpoint_sha256",
        ]
        if not (other_bases == f10_hash).all():
            raise AssertionError("F10-based adapter controls did not share one checkpoint")

    gradients = pd.read_csv(path / "gradient_audit.csv")
    first = gradients[gradients["backward_stage"] == "first_backward_before_update"]
    later = gradients[
        gradients["backward_stage"] == "after_first_update_probe_no_optimizer_step"
    ]
    if not (
        first["output_layer_nonzero"].astype(bool).all()
        and (~first["upstream_nonzero"].astype(bool)).all()
        and (~first["beta_nonzero"].astype(bool)).all()
        and later["output_layer_nonzero"].astype(bool).all()
        and later["upstream_nonzero"].astype(bool).all()
        and later["beta_nonzero"].astype(bool).all()
    ):
        raise AssertionError("Adapter gradient audit failed")
    groups = pd.read_csv(path / "group_mass_diagnostics.csv")
    if float(groups["mass_sum_max_abs_error"].max()) > 1e-6:
        raise AssertionError("Pollutant/meteorology gate mass does not sum to one")
    predictions = pd.read_csv(path / "validation_predictions.csv")
    if set(predictions["variant"].unique()) != set(ALL_VARIANTS):
        raise AssertionError("Validation predictions lack an L1 variant")
    return {
        "path": str(path),
        "task": manifest["task"],
        "smoke": smoke,
        "required_files_present": True,
        "test_access_counts_all_zero": True,
        "target_time_sets_disjoint": True,
        "training_only_station_selection_verified": True,
        "p7_and_lags_verified": True,
        "parameter_counts_verified": True,
        "checkpoint_hashes_verified": True,
        "adapter_base_state_hashes_unchanged": True,
        "f00_to_f10_patchtst_checkpoint_lineage_verified": True,
        "f10_patchtst_per_epoch_parameter_buffer_identity_verified": True,
        "base_eval_no_grad_verified": True,
        "git_and_source_provenance_verified": True,
        "shared_checkpoint_and_initialization_verified": True,
        "gradient_contract_verified": True,
        "group_mass_verified": True,
        "prediction_scope_validation_only": True,
        "artifact_sha256": {
            name: _sha256(path / name) for name in REQUIRED_INPUT_FILES
        },
    }


def write_chinese_report(
    output_dir: Path,
    smoke: bool,
    overall: pd.DataFrame,
    factorial: pd.DataFrame,
    shuffle_summary: pd.DataFrame,
    gate: dict[str, Any],
    integrity: dict[str, Any],
) -> None:
    lines = [
        "# MC-CVLRA L1 实现与复算报告",
        "",
        "> SMOKE FLOW ONLY：本报告中的数值只验证软件链路，永不进入正式选型、阶段门或论文证据。"
        if smoke
        else "> L1 TRAIN+VALIDATION SELECTION：本报告不包含正式 test 结果。",
        "",
        "## 范围与隔离",
        "",
        "- 北京中心站固定为1013；数据按原始时间戳精确70/10/20切分。",
        "- 特征只读取train+validation；test特征、目标值、样本、预测、缓存和指标计数均为0。",
        "- 站点池为1001–1036且排除中心；筛站、标准化和场景阈值只接收训练片。",
        "- P=7清洗与变量顺序复用L0签字实现；无bfill。",
        "",
        "## 实现验收",
        "",
        "- M参数数为24→1的1532与168→6的6547；B-flat固定宽度15/47且差异不超过3%。",
        "- F10每轮训练均强制PatchTST子模块eval；F00来源、F10初始化、逐epoch、最佳checkpoint和最终加载的参数/全部buffer均独立复核并逐元素一致。",
        "- F01共享F00 checkpoint；F11/B1/B2/B-flat/B3共享F10 checkpoint。基础模型在适配器训练期间保持eval/no-grad，前后状态hash一致。",
        "- 首次backward只有零初始化输出层非零；第一次更新后的探测backward已向上游和beta传播。",
        "- 污染物与气象质量来自同一次全局softmax且逐样本逐horizon和为1。",
        "",
        "## 复算状态",
        "",
        f"- 输入任务数：{overall['task'].nunique()}；逐任务变体数：{overall.groupby('task')['variant'].nunique().to_dict()}。",
        f"- 2×2逐种子/指标行数：{len(factorial)}；M-shuffle逐训练种子汇总行数：{len(shuffle_summary)}。",
        f"- gate状态：`{gate['status']}`。",
        f"- 完整性验收：{'PASS' if integrity['passed'] else 'FAIL'}。",
    ]
    if smoke:
        lines.extend(
            [
                "",
                (
                    "168→6真实GPU batch512资源smoke已独立核验完整F11前向/反向、冻结基础状态和显存记录；只作资源验收，不解释性能。"
                    if integrity["checks"]["gpu_batch512_resource_smoke_verified"]
                    else "168→6真实GPU batch512资源smoke尚未完成：当前复算以fail-closed状态记录该资源阻断，不得据此启动正式运行。"
                ),
                "",
                "smoke指标虽然已从逐预测CSV机械复算，但本报告故意不展示模型优劣、相对改善或门结论；不得据此改变结构、lag、beta、门槛或训练预算。",
            ]
        )
    (output_dir / "L1实现与复算报告.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.allow_missing_gpu_smoke and not args.smoke:
        raise ValueError("Missing GPU smoke may only be recorded in --smoke mode")
    input_dirs = [Path(args.dir_24), Path(args.dir_168)]
    if args.smoke:
        input_dirs = [
            path if path.name.endswith("_smoke") else path.with_name(path.name + "_smoke")
            for path in input_dirs
        ]
    output_dir = Path(args.output_dir) if args.output_dir else Path(
        "experiments/results/cross_variable_lag_adapter/"
        + ("l1_selection_summary_smoke" if args.smoke else "l1_selection_summary")
    )
    if args.smoke and not output_dir.name.endswith("_smoke"):
        output_dir = output_dir.with_name(output_dir.name + "_smoke")
    if not args.smoke and output_dir.name.endswith("_smoke"):
        raise ValueError("Formal summary output cannot have a _smoke suffix")
    output_dir.mkdir(parents=True, exist_ok=True)

    verifications = [verify_input_directory(path, args.smoke) for path in input_dirs]
    gpu_verification = None
    if args.smoke:
        gpu_path = Path(args.gpu_smoke_dir)
        if gpu_path.is_dir():
            gpu_verification = verify_gpu_resource_smoke(gpu_path)
        elif args.allow_missing_gpu_smoke:
            gpu_verification = {
                "path": str(gpu_path),
                "verified": False,
                "status": "BLOCKED_NO_EXPOSED_GPU_DEVICE",
            }
        else:
            raise FileNotFoundError(
                f"Required GPU resource smoke is absent: {gpu_path}"
            )
    prediction_frames = [pd.read_csv(path / "validation_predictions.csv") for path in input_dirs]
    shuffle_frames = [pd.read_csv(path / "m_shuffle_predictions.csv") for path in input_dirs]
    predictions = pd.concat(prediction_frames, ignore_index=True)
    shuffle_predictions = pd.concat(shuffle_frames, ignore_index=True)
    overall, horizon, scenario = recompute_prediction_metrics(predictions)
    factorial = compute_factorial_metrics(overall)
    paired = compute_paired_metrics(overall)
    shuffle_permutation, shuffle_summary = recompute_shuffle_metrics(
        predictions, shuffle_predictions
    )
    gate = compute_l1_gate(
        overall, horizon, scenario, shuffle_summary, smoke=args.smoke
    )

    overall.to_csv(output_dir / "raw_metrics_recomputed.csv", index=False)
    horizon.to_csv(output_dir / "horizon_metrics_recomputed.csv", index=False)
    scenario.to_csv(output_dir / "scenario_metrics_recomputed.csv", index=False)
    factorial.to_csv(output_dir / "factorial_metrics.csv", index=False)
    paired.to_csv(output_dir / "paired_metrics.csv", index=False)
    shuffle_permutation.to_csv(output_dir / "shuffle_metrics.csv", index=False)
    shuffle_summary.to_csv(output_dir / "shuffle_summary.csv", index=False)
    (output_dir / "gate_status.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    integrity = {
        "passed": True,
        "smoke": args.smoke,
        "formal_gate_evaluated": not args.smoke,
        "gate_status": gate["status"],
        "input_verification": verifications,
        "gpu_resource_smoke_verification": gpu_verification,
        "checks": {
            "both_tasks_present": set(overall["task"]) == {"24h_1h", "168h_6h"},
            "metrics_recomputed_from_validation_predictions": True,
            "factorial_formulas_recomputed": True,
            "shuffle_same_subset_and_block_mapping_verified": True,
            "formal_test_access_counts_all_zero": True,
            "smoke_excluded_from_formal_gate": (
                gate["status"] == "NOT_EVALUABLE_SMOKE" if args.smoke else True
            ),
            "gpu_batch512_resource_smoke_verified": (
                bool(gpu_verification and gpu_verification["verified"])
                if args.smoke
                else True
            ),
        },
    }
    integrity["passed"] = all(integrity["checks"].values())
    (output_dir / "integrity_verification.json").write_text(
        json.dumps(integrity, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_chinese_report(
        output_dir,
        args.smoke,
        overall,
        factorial,
        shuffle_summary,
        gate,
        integrity,
    )
    print(
        f"Independent L1 recomputation written to {output_dir.resolve()} "
        f"with status {gate['status']}"
    )


if __name__ == "__main__":
    main()
