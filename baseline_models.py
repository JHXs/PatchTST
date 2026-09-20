"""Neural baseline definitions for the frozen baseline-comparison protocol.

All public models accept ``[batch, stations, history]`` and return
``[batch, 1, horizon]``.  Channel selection is part of the wrapper so the
single-station and multi-station information sets are explicit and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from tsai.models.MLP import MLP
from tsai.models.PatchTST import PatchTST
from tsai.models.RNN import GRU, LSTM
from tsai.models.ResNet import ResNet
from tsai.models.TCN import TCN
from tsai.models.TST import TST


TARGET_PARAMETER_COUNT = 21_233
MATCHED_LOWER = int(TARGET_PARAMETER_COUNT * 0.8)
MATCHED_UPPER = int(TARGET_PARAMETER_COUNT * 1.2)

NEURAL_ARMS = (
    "center_mlp",
    "center_gru",
    "center_lstm",
    "center_tcn",
    "center_resnet",
    "center_tst",
    "plain_patchtst_all",
    "plain_patchtst_top5",
    "multi_tst",
    "multi_gru",
)
CAPACITY_TIERS = ("default", "matched")


@dataclass(frozen=True)
class ModelRegistration:
    arm: str
    layer: str
    requested_capacity: str
    capacity_status: str
    input_channels: int
    selected_channel_indices: tuple[int, ...]
    parameter_count: int
    hyperparameters: dict


class TsaiForecastAdapter(nn.Module):
    """Select an information set and normalize tsai output shapes."""

    def __init__(
        self,
        model: nn.Module,
        channel_indices: tuple[int, ...],
        horizon: int,
        patchtst_center_output: int | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.channel_indices = tuple(int(index) for index in channel_indices)
        self.horizon = int(horizon)
        self.patchtst_center_output = patchtst_center_output
        self.register_buffer(
            "channel_index_tensor",
            torch.tensor(self.channel_indices, dtype=torch.long),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, L]; selected: [B, C, L]
        selected = x.index_select(1, self.channel_index_tensor)
        output = self.model(selected)
        if output.ndim == 2:
            output = output.unsqueeze(1)
        elif output.ndim == 3 and self.patchtst_center_output is not None:
            # tsai 1.0.1 PatchTST is channel-independent and returns
            # [B, c_in, pred_dim] even when c_out=1.
            output = output[:, self.patchtst_center_output:self.patchtst_center_output + 1]
        elif output.ndim == 3 and output.shape[1] == self.horizon and output.shape[2] == 1:
            output = output.transpose(1, 2)
        if output.ndim != 3 or output.shape[1:] != (1, self.horizon):
            raise RuntimeError(
                f"Unexpected tsai output shape {tuple(output.shape)}; "
                f"expected [B, 1, {self.horizon}]"
            )
        return output


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def top5_channel_indices(metadata: dict) -> tuple[int, ...]:
    """Return center plus the five highest training-period correlations."""
    station_ids = [int(value) for value in metadata["station_ids"]]
    center_idx = int(metadata["center_station_idx"])
    correlations = metadata.get("station_correlations", {})
    neighbor_indices = [index for index in range(len(station_ids)) if index != center_idx]
    neighbor_indices.sort(
        key=lambda index: (
            -float(correlations.get(str(station_ids[index]), float("-inf"))),
            station_ids[index],
        )
    )
    return (center_idx, *neighbor_indices[:5])


def channel_indices_for_arm(arm: str, metadata: dict) -> tuple[int, ...]:
    station_count = len(metadata["station_ids"])
    center_idx = int(metadata["center_station_idx"])
    if arm.startswith("center_"):
        return (center_idx,)
    if arm == "plain_patchtst_top5":
        return top5_channel_indices(metadata)
    if arm in {"plain_patchtst_all", "multi_tst", "multi_gru"}:
        return tuple(range(station_count))
    raise ValueError(f"Unknown neural baseline arm: {arm}")


def _closest_builder(
    builders: list[tuple[dict, Callable[[], nn.Module]]],
) -> tuple[dict, Callable[[], nn.Module], int]:
    """Choose the closest parameter count without advancing caller RNG state."""
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    counts: list[int] = []
    with torch.random.fork_rng(devices=devices):
        for _, builder in builders:
            counts.append(count_parameters(builder()))
    best = min(
        range(len(builders)),
        key=lambda index: (abs(counts[index] - TARGET_PARAMETER_COUNT), counts[index]),
    )
    hyperparameters, builder = builders[best]
    return hyperparameters, builder, counts[best]


def _model_builder(
    family: str,
    c_in: int,
    history: int,
    horizon: int,
    capacity: str,
) -> tuple[Callable[[], nn.Module], dict, str]:
    if capacity not in CAPACITY_TIERS:
        raise ValueError(f"Unknown capacity tier: {capacity}")

    if family == "patchtst":
        kwargs = {
            "c_in": c_in,
            "c_out": 1,
            "seq_len": history,
            "pred_dim": horizon,
            "n_layers": 3,
            "n_heads": 4,
            "d_model": 16,
            "d_ff": 128,
            "dropout": 0.2,
            "patch_len": 4,
            "stride": 2,
            "padding_patch": True,
        }
        builder = lambda: PatchTST(**kwargs)
        count = _count_builder(builder)
        status = "matched" if MATCHED_LOWER <= count <= MATCHED_UPPER else "matched_nearest"
        return builder, kwargs, "default" if capacity == "default" else status

    default_builders: dict[str, tuple[dict, Callable[[], nn.Module]]] = {
        "mlp": ({"layers": [500, 500, 500]}, lambda: MLP(c_in, horizon, history)),
        "gru": ({"hidden_size": 100}, lambda: GRU(c_in, horizon)),
        "lstm": ({"hidden_size": 100}, lambda: LSTM(c_in, horizon)),
        "tcn": ({"layers": [25] * 8, "ks": 7}, lambda: TCN(c_in, horizon)),
        "resnet": ({"library_fixed": True}, lambda: ResNet(c_in, horizon)),
        "tst": (
            {"n_layers": 3, "d_model": 128, "n_heads": 16, "d_ff": 256},
            lambda: TST(c_in, horizon, history),
        ),
    }
    if capacity == "default":
        hyperparameters, builder = default_builders[family]
        return builder, hyperparameters, "default"

    if family == "mlp":
        candidates = [
            (
                {"layers": [width], "ps": [0.1]},
                lambda width=width: MLP(
                    c_in, horizon, history, layers=[width], ps=[0.1]
                ),
            )
            for width in range(16, 1025, 8)
        ]
    elif family == "gru":
        candidates = [
            ({"hidden_size": width}, lambda width=width: GRU(c_in, horizon, hidden_size=width))
            for width in range(8, 129)
        ]
    elif family == "lstm":
        candidates = [
            ({"hidden_size": width}, lambda width=width: LSTM(c_in, horizon, hidden_size=width))
            for width in range(8, 129)
        ]
    elif family == "tcn":
        candidates = [
            (
                {"layers": [width] * depth, "ks": kernel},
                lambda width=width, depth=depth, kernel=kernel: TCN(
                    c_in, horizon, layers=[width] * depth, ks=kernel
                ),
            )
            for depth in (2, 3, 4, 6, 8)
            for width in (8, 12, 16, 20, 24, 28, 32)
            for kernel in (3, 5, 7)
        ]
    elif family == "tst":
        candidates = [
            (
                {"n_layers": layers, "d_model": model_dim, "n_heads": heads, "d_ff": ff_dim},
                lambda layers=layers, model_dim=model_dim, heads=heads, ff_dim=ff_dim: TST(
                    c_in,
                    horizon,
                    history,
                    n_layers=layers,
                    d_model=model_dim,
                    n_heads=heads,
                    d_ff=ff_dim,
                ),
            )
            for layers in (1, 2, 3)
            for model_dim in (8, 16, 24, 32, 48)
            for heads in (1, 2, 4)
            if model_dim % heads == 0
            for ff_dim in (32, 64, 128)
        ]
    elif family == "resnet":
        hyperparameters, builder = default_builders[family]
        return builder, hyperparameters, "matched_nearest"
    else:
        raise ValueError(f"Unknown family: {family}")

    hyperparameters, builder, count = _closest_builder(candidates)
    status = "matched" if MATCHED_LOWER <= count <= MATCHED_UPPER else "matched_nearest"
    return builder, hyperparameters, status


def _count_builder(builder: Callable[[], nn.Module]) -> int:
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        return count_parameters(builder())


def build_baseline_model(
    arm: str,
    capacity: str,
    config,
    metadata: dict,
) -> tuple[nn.Module, ModelRegistration]:
    """Build one registered neural arm without moving it to a device."""
    if arm not in NEURAL_ARMS:
        raise ValueError(f"Unknown neural baseline arm: {arm}")
    indices = channel_indices_for_arm(arm, metadata)
    family = {
        "center_mlp": "mlp",
        "center_gru": "gru",
        "center_lstm": "lstm",
        "center_tcn": "tcn",
        "center_resnet": "resnet",
        "center_tst": "tst",
        "plain_patchtst_all": "patchtst",
        "plain_patchtst_top5": "patchtst",
        "multi_tst": "tst",
        "multi_gru": "gru",
    }[arm]
    builder, hyperparameters, capacity_status = _model_builder(
        family, len(indices), config.history, config.horizon, capacity
    )
    inner = builder()
    local_center = indices.index(int(metadata["center_station_idx"]))
    model = TsaiForecastAdapter(
        inner,
        indices,
        config.horizon,
        patchtst_center_output=local_center if family == "patchtst" else None,
    )
    parameter_count = count_parameters(model)
    registration = ModelRegistration(
        arm=arm,
        layer="B" if arm.startswith("center_") else "C",
        requested_capacity=capacity,
        capacity_status=capacity_status,
        input_channels=len(indices),
        selected_channel_indices=indices,
        parameter_count=parameter_count,
        hyperparameters=hyperparameters,
    )
    return model, registration
