"""Thin constructors for the preregistered single-station Informer arms."""

from __future__ import annotations

from torch import nn

from Informer_model import LTSF_Informer


def build_informer_arm(
    d_model: int,
    e_layers: int,
    seq_len: int,
    horizon: int,
    center_station_idx: int,
) -> nn.Module:
    """Build the locked Informer backbone used by ``SingleStationAdapter``.

    ``center_station_idx`` is accepted deliberately so every arm constructor has
    the same audited signature.  Channel selection itself remains the sole
    responsibility of ``SingleStationAdapter``.
    """
    if int(center_station_idx) < 0:
        raise ValueError("center_station_idx 必须为非负整数")
    return LTSF_Informer(
        c_in=1,
        c_out=1,
        seq_len=int(seq_len),
        pred_dim=int(horizon),
        label_len=min(48, int(seq_len) // 2),
        factor=5,
        attn="prob",
        distil=True,
        dropout=0.1,
        attn_dropout=0.1,
        d_ff=2 * int(d_model),
        d_layers=1,
        n_heads=1,
        d_model=int(d_model),
        e_layers=int(e_layers),
    )
