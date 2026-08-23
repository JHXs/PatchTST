"""Pre-registered causal local multiscale alternative after the rFFT F3 failure."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from frequency_residual_adapter import CapacityMatchedResidualAdapter


class CausalLocalFilterBank(nn.Module):
    """Fixed 12/48-hour trailing-mean filter bank with exact reconstruction."""

    def __init__(self, seq_len: int, short_hours: int = 12, long_hours: int = 48):
        super().__init__()
        if not 1 < short_hours < long_hours:
            raise ValueError("滤波尺度必须满足 1 < short_hours < long_hours")
        if seq_len < long_hours:
            raise ValueError("历史窗口必须不短于 long_hours")
        self.seq_len = int(seq_len)
        self.short_hours = int(short_hours)
        self.long_hours = int(long_hours)

    @staticmethod
    def trailing_mean(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        # Replicate only the earliest already observed value on the left.  There
        # is no right padding and no access to a later position in the window.
        padded = F.pad(x, (kernel_size - 1, 0), mode="replicate")
        return F.avg_pool1d(padded, kernel_size=kernel_size, stride=1)

    def forward(self, center_x: torch.Tensor) -> torch.Tensor:
        if center_x.ndim != 3 or center_x.shape[1:] != (1, self.seq_len):
            raise ValueError(
                f"center_x 必须为 [B,1,{self.seq_len}]，实际为 {tuple(center_x.shape)}"
            )
        mean_short = self.trailing_mean(center_x, self.short_hours)
        mean_long = self.trailing_mean(center_x, self.long_hours)
        low = mean_long
        mid = mean_short - mean_long
        high = center_x - mean_short
        return torch.cat([low, mid, high], dim=1)


class CausalFilterBankResidualAdapter(CapacityMatchedResidualAdapter):
    """Capacity-matched adapter using the sole pre-registered local filter bank."""

    adapter_kind = "causal_local_filterbank"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.num_streams != 3:
            raise ValueError("因果局部滤波器组固定为三个流")
        self.decomposition = CausalLocalFilterBank(self.seq_len, 12, 48)

    def build_streams(self, center_x: torch.Tensor) -> torch.Tensor:
        return self.decomposition(center_x)
