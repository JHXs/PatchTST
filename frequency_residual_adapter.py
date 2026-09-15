"""Capacity-matched forecast residual adapters for P1 frequency experiments."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from ST_PatchTST_model import ST_PatchTST


def _descending_cosine(
    frequencies: torch.Tensor,
    cutoff: float,
    half_width: float,
) -> torch.Tensor:
    lower = cutoff - half_width
    upper = cutoff + half_width
    position = ((frequencies - lower) / max(upper - lower, 1e-12)).clamp(0, 1)
    transition = 0.5 * (1 + torch.cos(math.pi * position))
    return torch.where(
        frequencies <= lower,
        torch.ones_like(frequencies),
        torch.where(frequencies >= upper, torch.zeros_like(frequencies), transition),
    )


def build_fixed_band_masks(
    seq_len: int,
    low_period_hours: float = 48.0,
    high_period_hours: float = 12.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build smooth low/mid/high masks forming a partition of unity."""

    if seq_len < 2:
        raise ValueError("seq_len 必须至少为 2")
    if not 0 < high_period_hours < low_period_hours:
        raise ValueError("周期边界必须满足 0 < high_period < low_period")
    frequencies = torch.fft.rfftfreq(seq_len, d=1.0, dtype=torch.float32)
    low_cutoff = 1.0 / low_period_hours
    high_cutoff = 1.0 / high_period_hours
    frequency_bin = 1.0 / seq_len
    low_width = min(frequency_bin, low_cutoff * 0.5)
    high_width = min(frequency_bin, (high_cutoff - low_cutoff) * 0.25)

    low = _descending_cosine(frequencies, low_cutoff, low_width)
    high = 1.0 - _descending_cosine(
        frequencies, high_cutoff, high_width
    )
    mid = (1.0 - low) * (1.0 - high)
    masks = torch.stack([low, mid, high], dim=0).clamp_min(0)
    masks = masks / masks.sum(dim=0, keepdim=True).clamp_min(1e-12)
    # The DC component is trend by definition and must not be split on short windows.
    masks[:, 0] = 0
    masks[0, 0] = 1
    return frequencies, masks


class FixedBandDecomposition(nn.Module):
    """Window-local differentiable rFFT decomposition into three fixed bands."""

    def __init__(
        self,
        seq_len: int,
        low_period_hours: float = 48.0,
        high_period_hours: float = 12.0,
    ) -> None:
        super().__init__()
        frequencies, masks = build_fixed_band_masks(
            seq_len, low_period_hours, high_period_hours
        )
        self.seq_len = int(seq_len)
        self.num_bands = int(masks.shape[0])
        self.low_period_hours = float(low_period_hours)
        self.high_period_hours = float(high_period_hours)
        self.register_buffer("frequencies", frequencies, persistent=True)
        self.register_buffer("masks", masks, persistent=True)

    def forward(self, center_x: torch.Tensor) -> torch.Tensor:
        if center_x.ndim != 3 or center_x.shape[1] != 1:
            raise ValueError(
                f"center_x 必须为 [B,1,L]，实际形状为 {tuple(center_x.shape)}"
            )
        if center_x.shape[-1] != self.seq_len:
            raise ValueError(
                f"历史长度应为 {self.seq_len}，实际为 {center_x.shape[-1]}"
            )
        if center_x.dtype not in (torch.float32, torch.float64):
            center_x = center_x.float()
        spectrum = torch.fft.rfft(center_x, dim=-1)
        masked_spectrum = (
            spectrum.unsqueeze(1)
            * self.masks.to(dtype=center_x.dtype).unsqueeze(0).unsqueeze(2)
        )
        # [B,K,1,F] -> [B,K,L]
        bands = torch.fft.irfft(
            masked_spectrum, n=self.seq_len, dim=-1
        ).squeeze(2)
        return bands


class CapacityMatchedResidualAdapter(nn.Module):
    """Shared trainable head used by time and frequency controls."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        hidden_dim: int = 32,
        pool_bins: int = 4,
        dropout: float = 0.2,
        num_streams: int = 3,
    ) -> None:
        super().__init__()
        if hidden_dim < 1 or pool_bins < 1 or num_streams < 1:
            raise ValueError("hidden_dim、pool_bins 和 num_streams 必须为正整数")
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.hidden_dim = int(hidden_dim)
        self.pool_bins = int(pool_bins)
        self.num_streams = int(num_streams)
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal_pool = nn.AdaptiveAvgPool1d(pool_bins)
        stream_feature_dim = hidden_dim * (pool_bins + 1)
        self.forecast_out = nn.Linear(num_streams * stream_feature_dim, pred_len)
        nn.init.zeros_(self.forecast_out.weight)
        nn.init.zeros_(self.forecast_out.bias)

    def build_streams(self, center_x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward_components(self, center_x: torch.Tensor) -> dict:
        streams = self.build_streams(center_x)
        if streams.shape != (
            center_x.shape[0],
            self.num_streams,
            self.seq_len,
        ):
            raise AssertionError(f"适配器流形状异常: {tuple(streams.shape)}")
        batch_size = streams.shape[0]
        encoded = self.temporal_encoder(
            streams.reshape(batch_size * self.num_streams, 1, self.seq_len)
        )
        pooled = self.temporal_pool(encoded).flatten(1)
        recent = encoded[..., -1]
        features = torch.cat([pooled, recent], dim=1).reshape(
            batch_size, self.num_streams, -1
        )
        # P1 has no learned band gate: every stream receives the same 1/K scale.
        equal_weighted_features = features / self.num_streams
        raw_residual = self.forecast_out(
            equal_weighted_features.flatten(1)
        ).unsqueeze(1)
        return {
            "streams": streams,
            "encoded_streams": encoded.reshape(
                batch_size, self.num_streams, self.hidden_dim, self.seq_len
            ),
            "stream_features": features,
            "raw_residual": raw_residual,
        }

    def forward(self, center_x: torch.Tensor) -> torch.Tensor:
        return self.forward_components(center_x)["raw_residual"]


class TimeResidualAdapter(CapacityMatchedResidualAdapter):
    """Equal-capacity control that directly encodes the raw center history."""

    adapter_kind = "time"

    def build_streams(self, center_x: torch.Tensor) -> torch.Tensor:
        if center_x.ndim != 3 or center_x.shape[1:] != (1, self.seq_len):
            raise ValueError(
                f"center_x 必须为 [B,1,{self.seq_len}]，实际为 {tuple(center_x.shape)}"
            )
        return center_x.expand(-1, self.num_streams, -1)


class FixedFrequencyResidualAdapter(CapacityMatchedResidualAdapter):
    """Three-band rFFT residual adapter with no learnable band boundaries."""

    adapter_kind = "fixed_frequency"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.decomposition = FixedBandDecomposition(self.seq_len)
        if self.decomposition.num_bands != self.num_streams:
            raise ValueError("固定频带数必须与适配器流数量一致")

    def build_streams(self, center_x: torch.Tensor) -> torch.Tensor:
        return self.decomposition(center_x)


class FrozenSTForecastResidual(nn.Module):
    """Add a bounded adapter residual after a completely frozen ST forecast."""

    def __init__(
        self,
        base_model: ST_PatchTST,
        adapter: CapacityMatchedResidualAdapter,
        beta_max: float = 0.5,
        beta_init: float = 0.1,
    ) -> None:
        super().__init__()
        if not 0 < beta_max:
            raise ValueError("beta_max 必须大于 0")
        if not 0 <= beta_init <= beta_max:
            raise ValueError("beta_init 必须位于 [0,beta_max]")
        self.base_model = base_model
        self.adapter = adapter
        self.max_beta = float(beta_max)
        ratio = min(max(beta_init / beta_max, 1e-6), 1 - 1e-6)
        self.beta_logit = nn.Parameter(
            torch.tensor(np.log(ratio / (1 - ratio)), dtype=torch.float32)
        )
        self.freeze_base_model()

    def freeze_base_model(self) -> None:
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False
        self.base_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Preserve every parameter and running state of the locked ST base.
        self.base_model.eval()
        return self

    def forward_components(
        self,
        x: torch.Tensor,
        disable_frequency: bool = False,
    ) -> dict:
        self.base_model.eval()
        with torch.no_grad():
            base_prediction = self.base_model(x)
        center_x = x[:, self.base_model.center_station_idx:self.base_model.center_station_idx + 1]
        adapter_components = self.adapter.forward_components(center_x)
        beta = self.max_beta * torch.sigmoid(self.beta_logit)
        effective_beta = x.new_tensor(0.0) if disable_frequency else beta
        forecast_residual = effective_beta * torch.tanh(
            adapter_components["raw_residual"]
        )
        return {
            "base_prediction": base_prediction,
            "forecast_residual": forecast_residual,
            "prediction": base_prediction + forecast_residual,
            "beta": beta,
            "frequency_disabled": bool(disable_frequency),
            **adapter_components,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_components(x)["prediction"]


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in module.parameters()
        if parameter.requires_grad
    )
