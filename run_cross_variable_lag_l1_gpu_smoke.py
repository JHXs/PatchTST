"""Run the isolated 168->6, batch-512 F11 GPU resource acceptance smoke.

This command performs exactly one full forward/backward batch with the real L1
maximum task structure and a frozen F10 base.  It records resource facts only;
it does not train, compare performance, construct test data, or enter selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

from cross_variable_lag_adapter import (
    assert_tensor_mappings_equal,
    clone_module_parameters_and_buffers,
    module_state_hash,
    tensor_mapping_hash,
)
from cross_variable_lag_l1_data import L1TaskConfig, prepare_l1_data
from run_cross_variable_lag_l1 import (
    L1RunConfig,
    build_adapter,
    build_l1_base_model,
    collect_git_provenance,
    file_sha256,
    gradient_contract,
)


DEFAULT_BASE_CHECKPOINT = Path(
    "experiments/results/cross_variable_lag_adapter/"
    "l1_selection_168h_6h_smoke/checkpoints/F10_seed2060.pt"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/results/cross_variable_lag_adapter/"
    "l1_168h_6h_batch512_gpu_smoke"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="tsai/data/stations_data")
    parser.add_argument("--base-checkpoint", default=str(DEFAULT_BASE_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = Path(args.output_dir)
    if not output_dir.name.endswith("_gpu_smoke"):
        raise ValueError("GPU resource smoke output must end in _gpu_smoke")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The batch-512 resource acceptance requires a real CUDA/ROCm GPU")
    checkpoint_path = Path(args.base_checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    config = L1RunConfig(
        history=168,
        horizon=6,
        epochs=1,
        patience=1,
        batch_size=512,
    )
    data = prepare_l1_data(
        args.data_dir,
        L1TaskConfig(history=168, horizon=6),
        train_sample_limit=512,
        validation_sample_limit=24,
    )
    if len(data.train.origin_rows) != 512:
        raise AssertionError("GPU resource smoke did not construct exactly 512 train samples")
    zero_access_keys = (
        "test_feature_rows_loaded",
        "test_target_values_loaded",
        "test_samples_constructed",
        "test_predictions_constructed",
        "test_caches_constructed",
        "test_metrics_constructed",
    )
    if any(int(data.metadata["access_metadata"][key]) != 0 for key in zero_access_keys):
        raise AssertionError("GPU resource smoke touched formal test data")

    base = build_l1_base_model(
        config,
        "F10",
        num_stations=len(data.metadata["selected_station_ids"]),
        center_station_idx=int(data.metadata["center_station_idx"]),
    )
    checkpoint_state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    base.load_state_dict(checkpoint_state)
    base.requires_grad_(False)
    base.eval()
    if base.patch_tst.training or any(
        parameter.requires_grad for parameter in base.parameters()
    ):
        raise AssertionError("GPU smoke base is not fully frozen/eval")
    adapter = build_adapter(config, "F11")
    adapter.train()
    base_state_before = module_state_hash(base)
    base_complete_before = clone_module_parameters_and_buffers(base)
    patchtst_complete_before = clone_module_parameters_and_buffers(base.patch_tst)

    base = base.to(device)
    adapter = adapter.to(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    x_st = torch.from_numpy(data.train.x_st).to(device, non_blocking=False)
    auxiliary = torch.from_numpy(data.train.auxiliary_tokens).to(
        device, non_blocking=False
    )
    context = torch.from_numpy(data.train.context).to(device, non_blocking=False)
    target = torch.from_numpy(data.train.targets_scaled).to(device, non_blocking=False)
    if x_st.shape[0] != 512:
        raise AssertionError("Transferred GPU batch is not 512 samples")

    adapter.zero_grad(set_to_none=True)
    with torch.no_grad():
        base_prediction = base(x_st)
    prediction = adapter(base_prediction, auxiliary, context)
    loss = nn.functional.mse_loss(prediction, target)
    if not torch.isfinite(loss):
        raise FloatingPointError("GPU resource smoke produced a non-finite loss")
    loss.backward()
    torch.cuda.synchronize(device)
    gradients = gradient_contract(adapter)
    if not gradients["output_layer_nonzero"]:
        raise AssertionError("F11 output head did not receive a GPU backward gradient")
    if any(parameter.grad is not None for parameter in base.parameters()):
        raise AssertionError("Frozen F10 base received gradients in GPU smoke")

    base_state_after = module_state_hash(base)
    base_complete_after = clone_module_parameters_and_buffers(base)
    patchtst_complete_after = clone_module_parameters_and_buffers(base.patch_tst)
    assert_tensor_mappings_equal(
        base_complete_after,
        base_complete_before,
        "GPU smoke frozen F10 base parameters/buffers",
    )
    assert_tensor_mappings_equal(
        patchtst_complete_after,
        patchtst_complete_before,
        "GPU smoke frozen PatchTST parameters/buffers",
    )
    if base_state_after != base_state_before:
        raise AssertionError("GPU smoke changed the frozen F10 persistent state")

    device_index = torch.cuda.current_device() if device.index is None else device.index
    properties = torch.cuda.get_device_properties(device_index)
    provenance = collect_git_provenance(Path(__file__).resolve().parent)
    result: dict[str, object] = {
        "scope": "L1_168H_6H_BATCH512_GPU_RESOURCE_SMOKE_ONLY",
        "performance_interpretation_prohibited": True,
        "formal_selection_performed": False,
        "formal_test_used": False,
        "formal_clean_tree_required": True,
        "precommit_engineering_smoke": bool(provenance["git_dirty"]),
        "provenance_statement": (
            "PRE-COMMIT DIRTY-TREE ENGINEERING GPU RESOURCE SMOKE; NOT FORMAL EVIDENCE"
            if provenance["git_dirty"]
            else "CLEAN-TREE ENGINEERING GPU RESOURCE SMOKE; NOT FORMAL EVIDENCE"
        ),
        "command": sys.argv,
        "output_dir": str(output_dir),
        "task": config.task_name,
        "variant": "F11",
        "history": 168,
        "horizon": 6,
        "batch_size_requested": 512,
        "batch_size_actual": int(x_st.shape[0]),
        "x_st_shape": list(x_st.shape),
        "auxiliary_token_shape": list(auxiliary.shape),
        "target_shape": list(target.shape),
        "device_type": device.type,
        "device_index": int(device_index),
        "device_name": torch.cuda.get_device_name(device_index),
        "device_total_memory_bytes": int(properties.total_memory),
        "torch_version": torch.__version__,
        "torch_hip_version": torch.version.hip,
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "current_memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "forward_backward_completed": True,
        "loss_finite": True,
        "base_fully_frozen": True,
        "base_gradients_none": True,
        "base_eval": not base.training,
        "patchtst_eval": not base.patch_tst.training,
        "base_state_before_sha256": base_state_before,
        "base_state_after_sha256": base_state_after,
        "base_state_unchanged": base_state_before == base_state_after,
        "base_complete_state_before_sha256": tensor_mapping_hash(base_complete_before),
        "base_complete_state_after_sha256": tensor_mapping_hash(base_complete_after),
        "patchtst_complete_state_before_sha256": tensor_mapping_hash(
            patchtst_complete_before
        ),
        "patchtst_complete_state_after_sha256": tensor_mapping_hash(
            patchtst_complete_after
        ),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "gradient_contract": gradients,
        "access_metadata": data.metadata["access_metadata"],
        **provenance,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resource_smoke.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    result = run(parse_args())
    print(
        "GPU resource smoke completed: "
        f"device={result['device_name']}, batch={result['batch_size_actual']}, "
        f"peak_allocated_bytes={result['peak_memory_allocated_bytes']}"
    )
    print("Resource facts only; performance interpretation is prohibited.")


if __name__ == "__main__":
    main()
