"""Registered center-station backbones for the frozen backbone-upgrade protocol.

The historical baseline checkpoints store the state of a small adapter under a
``model.`` prefix.  :class:`BackboneAdapter` deliberately uses the same prefix,
so those checkpoints can be loaded strictly without rewriting tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from tsai.models.MLP import MLP
from tsai.models.PatchTST import PatchTST
from tsai.models.RNN import GRU, LSTM
from tsai.models.ResNet import ResNet
from tsai.models.TCN import TCN
from tsai.models.TST import TST


CENTER_ARMS = (
    "center_gru",
    "center_lstm",
    "center_tcn",
    "center_mlp",
    "center_resnet",
    "center_tst",
)
HEADLINE_TASKS = ((24, 1), (168, 6))


@dataclass(frozen=True)
class CandidateSpec:
    """One protocol-level candidate, including its capacity tier."""

    arm: str
    capacity: str

    @property
    def variant(self) -> str:
        if self.arm == "degraded_patchtst":
            return self.arm
        return f"{self.arm}_{self.capacity}"


class BackboneAdapter(nn.Module):
    """Normalize any center-only forecasting backbone to ``[B, 1, H]``."""

    def __init__(self, model: nn.Module, horizon: int) -> None:
        super().__init__()
        self.model = model
        self.horizon = int(horizon)

    def forward(self, center_x: torch.Tensor) -> torch.Tensor:
        # center_x: [B, 1, L]
        output = self.model(center_x)
        if output.ndim == 2:
            output = output.unsqueeze(1)
        elif output.ndim == 3 and output.shape[1:] == (self.horizon, 1):
            output = output.transpose(1, 2)
        if output.ndim != 3 or output.shape[1:] != (1, self.horizon):
            raise RuntimeError(
                f"主干输出形状 {tuple(output.shape)}，期望 [B, 1, {self.horizon}]"
            )
        return output


def expected_candidates(city: str, history: int, horizon: int) -> tuple[CandidateSpec, ...]:
    """Return the full registered universe so reduced coverage is explicit."""
    if city not in {"beijing", "guangzhou"}:
        raise ValueError(f"未知城市: {city}")
    neural = tuple(
        CandidateSpec(arm, capacity)
        for arm in CENTER_ARMS
        for capacity in ("default", "matched")
    )
    return (*neural, CandidateSpec("degraded_patchtst", "protocol"))


def candidate_in_protocol_coverage(
    city: str, history: int, horizon: int, spec: CandidateSpec
) -> bool:
    """Whether a universe member was scheduled for this protocol stratum."""
    if spec.arm == "degraded_patchtst":
        return True
    if city == "guangzhou":
        return spec.arm == "center_gru"
    if city == "beijing" and (int(history), int(horizon)) in HEADLINE_TASKS:
        return True
    if city == "beijing":
        return spec.arm in {"center_gru", "center_lstm", "center_tcn"} and spec.capacity == "default"
    raise ValueError(f"未知城市: {city}")


def build_center_backbone(
    arm: str,
    history: int,
    horizon: int,
    hyperparameters: dict[str, Any] | None = None,
) -> nn.Module:
    """Reconstruct a tsai center-only model from its recorded hyperparameters."""
    hp = dict(hyperparameters or {})
    if arm == "center_mlp":
        kwargs = {key: hp[key] for key in ("layers", "ps") if key in hp}
        return MLP(1, horizon, history, **kwargs)
    if arm == "center_gru":
        return GRU(1, horizon, hidden_size=int(hp.get("hidden_size", 100)))
    if arm == "center_lstm":
        return LSTM(1, horizon, hidden_size=int(hp.get("hidden_size", 100)))
    if arm == "center_tcn":
        return TCN(
            1,
            horizon,
            layers=list(hp.get("layers", [25] * 8)),
            ks=int(hp.get("ks", 7)),
        )
    if arm == "center_resnet":
        return ResNet(1, horizon)
    if arm == "center_tst":
        kwargs = {
            key: int(hp[key])
            for key in ("n_layers", "d_model", "n_heads", "d_ff")
            if key in hp
        }
        return TST(1, horizon, history, **kwargs)
    raise ValueError(f"不支持的中心站主干: {arm}")


def build_patchtst_backbone(history: int, horizon: int) -> nn.Module:
    """Reconstruct the exact PatchTST used by the current degraded baseline."""
    return PatchTST(
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
    )


def load_candidate_adapter(
    arm: str,
    history: int,
    horizon: int,
    checkpoint_path: str,
    hyperparameters: dict[str, Any] | None = None,
    map_location: str | torch.device = "cpu",
) -> BackboneAdapter:
    """Build and strictly load one historical candidate checkpoint."""
    state = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    if arm == "degraded_patchtst":
        prefix = "patch_tst."
        state = {
            key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)
        }
        model = build_patchtst_backbone(history, horizon)
        model.load_state_dict(state, strict=True)
        return BackboneAdapter(model, horizon).to(map_location)

    model = build_center_backbone(arm, history, horizon, hyperparameters)
    adapter = BackboneAdapter(model, horizon)
    adapter.load_state_dict(state, strict=True)
    # tsai TST constructs its positional parameter on default_device(), which
    # can differ from map_location on a GPU host running a CPU experiment.
    return adapter.to(map_location)
