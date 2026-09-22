"""Single-station neural baselines for the baseline comparison protocol.

Every public model is registered with exactly one input channel.  The adapter
accepts the repository's shared ``[B, S, L]`` tensors, selects only the centre
station, and returns ``[B, 1, H]``.  Neighbour channels therefore cannot affect
any baseline prediction.
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


class SingleStationAdapter(nn.Module):
    """Select the centre channel and normalize tsai output to ``[B, 1, H]``."""

    def __init__(
        self,
        model: nn.Module,
        center_station_idx: int,
        horizon: int,
    ) -> None:
        super().__init__()
        self.model = model
        self.center_station_idx = int(center_station_idx)
        self.horizon = int(horizon)
        self.input_channels = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected [B,S,L], got {tuple(x.shape)}")
        if not 0 <= self.center_station_idx < x.shape[1]:
            raise ValueError("center_station_idx is outside the input channel range")
        # [B, S, L] -> [B, 1, L].  No neighbour value crosses this boundary.
        center = x[:, self.center_station_idx:self.center_station_idx + 1, :]
        output = self.model(center)
        if output.ndim == 2:
            output = output.unsqueeze(1)
        elif output.ndim == 3 and output.shape[1] == self.horizon and output.shape[2] == 1:
            output = output.transpose(1, 2)
        if output.ndim != 3 or tuple(output.shape[1:]) != (1, self.horizon):
            raise RuntimeError(
                f"unexpected tsai output {tuple(output.shape)}; "
                f"expected [B,1,{self.horizon}]"
            )
        return output


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _count_builder(builder: Callable[[], nn.Module]) -> int:
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        return count_parameters(builder())


def _closest_builder(
    builders: list[tuple[dict, Callable[[], nn.Module]]],
) -> tuple[dict, Callable[[], nn.Module], int]:
    counts = [_count_builder(builder) for _, builder in builders]
    best = min(
        range(len(builders)),
        key=lambda index: (abs(counts[index] - TARGET_PARAMETER_COUNT), counts[index]),
    )
    hyperparameters, builder = builders[best]
    return hyperparameters, builder, counts[best]


def _model_builder(
    family: str,
    history: int,
    horizon: int,
    capacity: str,
) -> tuple[Callable[[], nn.Module], dict, str]:
    if capacity not in CAPACITY_TIERS:
        raise ValueError(f"unknown capacity tier: {capacity}")

    default_builders: dict[str, tuple[dict, Callable[[], nn.Module]]] = {
        "mlp": ({"layers": [500, 500, 500]}, lambda: MLP(1, horizon, history)),
        "gru": ({"hidden_size": 100}, lambda: GRU(1, horizon)),
        "lstm": ({"hidden_size": 100}, lambda: LSTM(1, horizon)),
        "tcn": ({"layers": [25] * 8, "ks": 7}, lambda: TCN(1, horizon)),
        "resnet": ({"library_fixed": True}, lambda: ResNet(1, horizon)),
        "tst": (
            {"n_layers": 3, "d_model": 128, "n_heads": 16, "d_ff": 256},
            lambda: TST(1, horizon, history),
        ),
        "patchtst": (
            {
                "n_layers": 3,
                "n_heads": 4,
                "d_model": 16,
                "d_ff": 128,
                "dropout": 0.2,
                "patch_len": 4,
                "stride": 2,
            },
            lambda: PatchTST(
                c_in=1,
                c_out=1,
                seq_len=history,
                pred_dim=horizon,
                n_layers=3,
                n_heads=4,
                d_model=16,
                d_ff=128,
                dropout=0.2,
                patch_len=4,
                stride=2,
                padding_patch=True,
            ),
        ),
    }
    if capacity == "default" or family in {"resnet", "patchtst"}:
        hyperparameters, builder = default_builders[family]
        count = _count_builder(builder)
        if capacity == "default":
            status = "default"
        else:
            status = "matched" if MATCHED_LOWER <= count <= MATCHED_UPPER else "matched_nearest"
        return builder, hyperparameters, status

    if family == "mlp":
        candidates = [
            (
                {"layers": [width], "ps": [0.1]},
                lambda width=width: MLP(1, horizon, history, layers=[width], ps=[0.1]),
            )
            for width in range(16, 1025, 8)
        ]
    elif family == "gru":
        candidates = [
            ({"hidden_size": width}, lambda width=width: GRU(1, horizon, hidden_size=width))
            for width in range(8, 129)
        ]
    elif family == "lstm":
        candidates = [
            ({"hidden_size": width}, lambda width=width: LSTM(1, horizon, hidden_size=width))
            for width in range(8, 129)
        ]
    elif family == "tcn":
        candidates = [
            (
                {"layers": [width] * depth, "ks": kernel},
                lambda width=width, depth=depth, kernel=kernel: TCN(
                    1, horizon, layers=[width] * depth, ks=kernel
                ),
            )
            for depth in (2, 3, 4, 6, 8)
            for width in (8, 12, 16, 20, 24, 28, 32)
            for kernel in (3, 5, 7)
        ]
    elif family == "tst":
        candidates = [
            (
                {
                    "n_layers": layers,
                    "d_model": model_dim,
                    "n_heads": heads,
                    "d_ff": ff_dim,
                },
                lambda layers=layers, model_dim=model_dim, heads=heads, ff_dim=ff_dim: TST(
                    1,
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
    else:
        raise ValueError(f"unknown family: {family}")

    hyperparameters, builder, count = _closest_builder(candidates)
    status = "matched" if MATCHED_LOWER <= count <= MATCHED_UPPER else "matched_nearest"
    return builder, hyperparameters, status


def build_single_station_model(
    arm: str,
    capacity: str,
    config,
    metadata: dict,
) -> tuple[nn.Module, ModelRegistration]:
    """Build a registered centre-only model without moving it to a device."""
    if arm not in NEURAL_ARMS:
        raise ValueError(f"unknown single-station neural arm: {arm}")
    family = {
        "center_mlp": "mlp",
        "center_gru": "gru",
        "center_lstm": "lstm",
        "center_tcn": "tcn",
        "center_resnet": "resnet",
        "center_tst": "tst",
    }[arm]
    builder, hyperparameters, capacity_status = _model_builder(
        family, int(config.history), int(config.horizon), capacity
    )
    model = SingleStationAdapter(
        builder(), int(metadata["center_station_idx"]), int(config.horizon)
    )
    registration = ModelRegistration(
        arm=arm,
        layer="single_station_neural",
        requested_capacity=capacity,
        capacity_status=capacity_status,
        input_channels=1,
        selected_channel_indices=(int(metadata["center_station_idx"]),),
        parameter_count=count_parameters(model),
        hyperparameters=hyperparameters,
    )
    if registration.input_channels != 1:
        raise AssertionError("single-station baseline must register exactly one channel")
    return model, registration
