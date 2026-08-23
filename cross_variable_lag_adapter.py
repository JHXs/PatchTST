"""Frozen MC-CVLRA adapter and preregistered L1 controls.

The module is deliberately independent of data loading and ST-PatchTST training.
It accepts a frozen base prediction, the fixed seven-variable/six-lag token tensor,
and four PM2.5 history summaries.  No test-split concepts exist in this module.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


NUM_AUXILIARY_VARIABLES = 7
NUM_LAGS = 6
TOKEN_COMPONENTS = 2
ADAPTER_DIM = 16
CONTEXT_DIM = 4
POLLUTANT_COUNT = 4
BETA_MAX = 0.2
BETA_INIT = 0.02

ADAPTER_VARIANTS = ("F01", "F11", "B1", "B2", "B_flat", "B3")


def _bounded_logit(initial: float, maximum: float) -> float:
    if maximum <= 0:
        raise ValueError("maximum must be positive when constructing a bounded logit")
    if not 0 <= initial <= maximum:
        raise ValueError(f"initial={initial} must be in [0, maximum={maximum}]")
    ratio = min(max(initial / maximum, 1e-6), 1 - 1e-6)
    return math.log(ratio / (1 - ratio))


def tensor_mapping_hash(values: Mapping[str, torch.Tensor]) -> str:
    """Hash a named tensor mapping without relying on object identity."""
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def module_state_hash(module: nn.Module) -> str:
    """Hash every parameter and persistent buffer in a module state dict."""
    return tensor_mapping_hash(module.state_dict())


def clone_module_parameters_and_buffers(module: nn.Module) -> dict[str, torch.Tensor]:
    """Clone every live parameter and buffer, including non-persistent buffers."""
    snapshot: dict[str, torch.Tensor] = {}
    for name, parameter in module.named_parameters(recurse=True):
        snapshot[f"parameter::{name}"] = parameter.detach().cpu().clone()
    for name, buffer in module.named_buffers(recurse=True):
        snapshot[f"buffer::{name}"] = buffer.detach().cpu().clone()
    return snapshot


def compare_tensor_mappings(
    actual: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
) -> list[str]:
    """Return every missing, unexpected, shape/dtype, or value mismatch."""
    differences = []
    actual_keys = set(actual)
    expected_keys = set(expected)
    differences.extend(f"missing::{name}" for name in sorted(expected_keys - actual_keys))
    differences.extend(
        f"unexpected::{name}" for name in sorted(actual_keys - expected_keys)
    )
    for name in sorted(actual_keys & expected_keys):
        actual_tensor = actual[name].detach().cpu()
        expected_tensor = expected[name].detach().cpu()
        if actual_tensor.dtype != expected_tensor.dtype:
            differences.append(f"dtype::{name}")
        elif actual_tensor.shape != expected_tensor.shape:
            differences.append(f"shape::{name}")
        elif not torch.equal(actual_tensor, expected_tensor):
            differences.append(f"value::{name}")
    return differences


def assert_tensor_mappings_equal(
    actual: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
    label: str,
) -> None:
    """Require exact equality and report all changed tensor names."""
    differences = compare_tensor_mappings(actual, expected)
    if differences:
        raise AssertionError(
            f"{label}: {len(differences)} tensor mismatches: {differences}"
        )


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def expected_m_parameter_count(horizon: int) -> int:
    if horizon < 1:
        raise ValueError("horizon must be positive")
    return 529 + 1003 * int(horizon)


def flat_parameter_count(width: int, horizon: int) -> int:
    if width < 1 or horizon < 1:
        raise ValueError("width and horizon must be positive")
    return width ** 2 + (86 + horizon) * width + horizon + 1


def select_flat_width(horizon: int) -> dict[str, Any]:
    """Select B-flat width only by distance to the frozen M parameter count."""
    target = expected_m_parameter_count(horizon)
    candidates = [
        {
            "width": width,
            "parameter_count": flat_parameter_count(width, horizon),
            "absolute_difference": abs(flat_parameter_count(width, horizon) - target),
            "relative_difference_percent": (
                100.0 * abs(flat_parameter_count(width, horizon) - target) / target
            ),
        }
        for width in range(1, 257)
    ]
    selected = min(candidates, key=lambda row: (row["absolute_difference"], row["width"]))
    return {
        "target_parameter_count": target,
        "selected_width": selected["width"],
        "selected_parameter_count": selected["parameter_count"],
        "relative_difference_percent": selected["relative_difference_percent"],
        "accepted": selected["relative_difference_percent"] <= 3.0,
        "candidate_widths": list(range(1, 257)),
        "candidates": candidates,
    }


def controlled_token_input(
    auxiliary_tokens: torch.Tensor,
    self_tokens: torch.Tensor,
    variant: str,
) -> torch.Tensor:
    """Apply only the preregistered B1/B2 input replacements.

    ``auxiliary_tokens`` has shape [B,7,6,2]. ``self_tokens`` may be either
    [B,1,6,2] or [B,7,6,2].
    """
    if auxiliary_tokens.ndim != 4 or tuple(auxiliary_tokens.shape[1:]) != (
        NUM_AUXILIARY_VARIABLES,
        NUM_LAGS,
        TOKEN_COMPONENTS,
    ):
        raise ValueError("auxiliary_tokens must have shape [B,7,6,2]")
    if self_tokens.ndim != 4 or self_tokens.shape[0] != auxiliary_tokens.shape[0]:
        raise ValueError("self_tokens must have shape [B,1|7,6,2]")
    if tuple(self_tokens.shape[2:]) != (NUM_LAGS, TOKEN_COMPONENTS):
        raise ValueError("self_tokens must have six lags and two components")

    if variant == "B1":
        if self_tokens.shape[1] == 1:
            return self_tokens.expand(-1, NUM_AUXILIARY_VARIABLES, -1, -1)
        if self_tokens.shape[1] == NUM_AUXILIARY_VARIABLES:
            return self_tokens
        raise ValueError("B1 self_tokens station dimension must be 1 or 7")
    if variant == "B2":
        return auxiliary_tokens[:, :, :1, :].expand(-1, -1, NUM_LAGS, -1)
    if variant in {"F01", "F11", "B_flat", "B3"}:
        return auxiliary_tokens
    raise ValueError(f"Unknown adapter variant: {variant}")


class CrossVariableLagAdapter(nn.Module):
    """The fixed M adapter, with an optional B3 uniform-gate intervention."""

    def __init__(
        self,
        horizon: int,
        beta_max: float = BETA_MAX,
        beta_init: float = BETA_INIT,
        uniform_gate: bool = False,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.beta_max = float(beta_max)
        self.beta_init = float(beta_init)
        self.uniform_gate = bool(uniform_gate)
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if self.beta_max < 0:
            raise ValueError("beta_max must be non-negative")

        self.phi_shared = nn.Sequential(
            nn.Linear(TOKEN_COMPONENTS, ADAPTER_DIM, bias=True),
            nn.GELU(),
            nn.Linear(ADAPTER_DIM, ADAPTER_DIM, bias=True),
        )
        self.variable_embedding = nn.Embedding(NUM_AUXILIARY_VARIABLES, ADAPTER_DIM)
        self.lag_embedding = nn.Embedding(NUM_LAGS, ADAPTER_DIM)
        self.horizon_queries = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(CONTEXT_DIM, ADAPTER_DIM, bias=True),
                    nn.GELU(),
                    nn.Linear(ADAPTER_DIM, ADAPTER_DIM, bias=True),
                )
                for _ in range(self.horizon)
            ]
        )
        # [P,K,H], matching the protocol's bias[p,k,h].
        self.score_bias = nn.Parameter(
            torch.zeros(NUM_AUXILIARY_VARIABLES, NUM_LAGS, self.horizon)
        )
        self.output_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * ADAPTER_DIM + CONTEXT_DIM, ADAPTER_DIM, bias=True),
                    nn.GELU(),
                    nn.Linear(ADAPTER_DIM, 1, bias=True),
                )
                for _ in range(self.horizon)
            ]
        )
        for head in self.output_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

        if self.beta_max == 0:
            if self.beta_init != 0:
                raise ValueError("beta_init must be zero when beta_max is zero")
            self.register_parameter("beta_logit", None)
        else:
            self.beta_logit = nn.Parameter(
                torch.tensor(
                    _bounded_logit(self.beta_init, self.beta_max),
                    dtype=torch.float32,
                )
            )

        self.register_buffer(
            "pollutant_mask",
            torch.tensor([True] * POLLUTANT_COUNT + [False] * 3),
            persistent=False,
        )

    def _validate_inputs(
        self,
        base_prediction: torch.Tensor,
        tokens: torch.Tensor,
        context: torch.Tensor,
    ) -> None:
        if base_prediction.ndim != 3 or tuple(base_prediction.shape[1:]) != (
            1,
            self.horizon,
        ):
            raise ValueError(f"base_prediction must have shape [B,1,{self.horizon}]")
        if tokens.ndim != 4 or tuple(tokens.shape[1:]) != (
            NUM_AUXILIARY_VARIABLES,
            NUM_LAGS,
            TOKEN_COMPONENTS,
        ):
            raise ValueError("tokens must have shape [B,7,6,2]")
        if context.ndim != 2 or tuple(context.shape[1:]) != (CONTEXT_DIM,):
            raise ValueError("context must have shape [B,4]")
        if not (base_prediction.shape[0] == tokens.shape[0] == context.shape[0]):
            raise ValueError("adapter batch dimensions do not match")

    def forward_components(
        self,
        base_prediction: torch.Tensor,
        tokens: torch.Tensor,
        context: torch.Tensor,
        disable_auxiliary: bool = False,
    ) -> dict[str, torch.Tensor | bool]:
        self._validate_inputs(base_prediction, tokens, context)
        if disable_auxiliary or self.beta_logit is None:
            zero_residual = torch.zeros_like(base_prediction)
            return {
                "base_prediction": base_prediction,
                "prediction": base_prediction,
                "residual": zero_residual,
                "raw_correction": zero_residual,
                "beta": base_prediction.new_tensor(0.0),
                "gate_weights": base_prediction.new_zeros(
                    base_prediction.shape[0],
                    NUM_AUXILIARY_VARIABLES,
                    NUM_LAGS,
                    self.horizon,
                ),
                "pollutant_mass": base_prediction.new_zeros(
                    base_prediction.shape[0], self.horizon
                ),
                "meteorology_mass": base_prediction.new_zeros(
                    base_prediction.shape[0], self.horizon
                ),
                "auxiliary_disabled": True,
            }

        embeddings = self.phi_shared(tokens)
        embeddings = (
            embeddings
            + self.variable_embedding.weight[None, :, None, :]
            + self.lag_embedding.weight[None, None, :, :]
        )
        if self.uniform_gate:
            weights = embeddings.new_full(
                (
                    embeddings.shape[0],
                    NUM_AUXILIARY_VARIABLES,
                    NUM_LAGS,
                    self.horizon,
                ),
                1.0 / (NUM_AUXILIARY_VARIABLES * NUM_LAGS),
            )
        else:
            queries = torch.stack(
                [query(context) for query in self.horizon_queries], dim=1
            )  # [B,H,d]
            scores = torch.einsum("bpkd,bhd->bpkh", embeddings, queries)
            scores = scores + self.score_bias.unsqueeze(0)
            weights = torch.softmax(
                scores.reshape(scores.shape[0], -1, self.horizon), dim=1
            ).reshape_as(scores)

        pollutant_weights = weights[:, :POLLUTANT_COUNT]
        meteorology_weights = weights[:, POLLUTANT_COUNT:]
        pollutant_embeddings = embeddings[:, :POLLUTANT_COUNT]
        meteorology_embeddings = embeddings[:, POLLUTANT_COUNT:]
        z_pollutant = torch.einsum(
            "bpkh,bpkd->bhd", pollutant_weights, pollutant_embeddings
        )
        z_meteorology = torch.einsum(
            "bpkh,bpkd->bhd", meteorology_weights, meteorology_embeddings
        )
        pollutant_mass = pollutant_weights.sum(dim=(1, 2))
        meteorology_mass = meteorology_weights.sum(dim=(1, 2))

        raw_by_horizon = []
        for horizon_index, head in enumerate(self.output_heads):
            head_input = torch.cat(
                [
                    z_pollutant[:, horizon_index],
                    z_meteorology[:, horizon_index],
                    context,
                ],
                dim=1,
            )
            raw_by_horizon.append(head(head_input))
        raw_correction = torch.stack(raw_by_horizon, dim=-1)  # [B,1,H]
        beta = self.beta_max * torch.sigmoid(self.beta_logit)
        residual = beta * torch.tanh(raw_correction)
        prediction = base_prediction + residual
        return {
            "base_prediction": base_prediction,
            "prediction": prediction,
            "residual": residual,
            "raw_correction": raw_correction,
            "beta": beta,
            "gate_weights": weights,
            "pollutant_mass": pollutant_mass,
            "meteorology_mass": meteorology_mass,
            "auxiliary_disabled": False,
        }

    def forward(
        self,
        base_prediction: torch.Tensor,
        tokens: torch.Tensor,
        context: torch.Tensor,
        disable_auxiliary: bool = False,
    ) -> torch.Tensor:
        return self.forward_components(
            base_prediction,
            tokens,
            context,
            disable_auxiliary=disable_auxiliary,
        )["prediction"]


class FlatResidualAdapter(nn.Module):
    """The fixed three-layer, same-input B-flat control."""

    def __init__(
        self,
        horizon: int,
        width: int,
        beta_max: float = BETA_MAX,
        beta_init: float = BETA_INIT,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.width = int(width)
        self.beta_max = float(beta_max)
        self.beta_init = float(beta_init)
        if self.horizon < 1 or self.width < 1:
            raise ValueError("horizon and width must be positive")
        if self.beta_max < 0:
            raise ValueError("beta_max must be non-negative")
        self.network = nn.Sequential(
            nn.Linear(
                NUM_AUXILIARY_VARIABLES * NUM_LAGS * TOKEN_COMPONENTS,
                self.width,
                bias=True,
            ),
            nn.GELU(),
            nn.Linear(self.width, self.width, bias=True),
            nn.GELU(),
            nn.Linear(self.width, self.horizon, bias=True),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        if self.beta_max == 0:
            if self.beta_init != 0:
                raise ValueError("beta_init must be zero when beta_max is zero")
            self.register_parameter("beta_logit", None)
        else:
            self.beta_logit = nn.Parameter(
                torch.tensor(
                    _bounded_logit(self.beta_init, self.beta_max),
                    dtype=torch.float32,
                )
            )

    def forward_components(
        self,
        base_prediction: torch.Tensor,
        tokens: torch.Tensor,
        context: torch.Tensor | None = None,
        disable_auxiliary: bool = False,
    ) -> dict[str, torch.Tensor | bool]:
        if base_prediction.ndim != 3 or tuple(base_prediction.shape[1:]) != (
            1,
            self.horizon,
        ):
            raise ValueError(f"base_prediction must have shape [B,1,{self.horizon}]")
        if tokens.ndim != 4 or tuple(tokens.shape[1:]) != (
            NUM_AUXILIARY_VARIABLES,
            NUM_LAGS,
            TOKEN_COMPONENTS,
        ):
            raise ValueError("tokens must have shape [B,7,6,2]")
        if disable_auxiliary or self.beta_logit is None:
            zero = torch.zeros_like(base_prediction)
            return {
                "base_prediction": base_prediction,
                "prediction": base_prediction,
                "residual": zero,
                "raw_correction": zero,
                "beta": base_prediction.new_tensor(0.0),
                "auxiliary_disabled": True,
            }
        raw_correction = self.network(tokens.flatten(start_dim=1)).unsqueeze(1)
        beta = self.beta_max * torch.sigmoid(self.beta_logit)
        residual = beta * torch.tanh(raw_correction)
        return {
            "base_prediction": base_prediction,
            "prediction": base_prediction + residual,
            "residual": residual,
            "raw_correction": raw_correction,
            "beta": beta,
            "auxiliary_disabled": False,
        }

    def forward(
        self,
        base_prediction: torch.Tensor,
        tokens: torch.Tensor,
        context: torch.Tensor | None = None,
        disable_auxiliary: bool = False,
    ) -> torch.Tensor:
        return self.forward_components(
            base_prediction,
            tokens,
            context,
            disable_auxiliary=disable_auxiliary,
        )["prediction"]


class FrozenBaseAdapterModel(nn.Module):
    """Keep an arbitrary forecasting base in eval/no-grad behind an adapter."""

    def __init__(self, base_model: nn.Module, adapter: nn.Module) -> None:
        super().__init__()
        self.base_model = base_model
        self.adapter = adapter
        self.base_model.requires_grad_(False)
        for parameter in self.base_model.parameters():
            parameter.grad = None
        self.base_model.eval()

    def train(self, mode: bool = True) -> "FrozenBaseAdapterModel":
        super().train(mode)
        self.base_model.eval()
        return self

    def forward_components(
        self,
        x_st: torch.Tensor,
        tokens: torch.Tensor,
        context: torch.Tensor,
        disable_auxiliary: bool = False,
    ) -> dict[str, torch.Tensor | bool]:
        self.base_model.eval()
        with torch.no_grad():
            base_prediction = self.base_model(x_st)
        return self.adapter.forward_components(
            base_prediction,
            tokens,
            context,
            disable_auxiliary=disable_auxiliary,
        )

    def forward(
        self,
        x_st: torch.Tensor,
        tokens: torch.Tensor,
        context: torch.Tensor,
        disable_auxiliary: bool = False,
    ) -> torch.Tensor:
        return self.forward_components(
            x_st,
            tokens,
            context,
            disable_auxiliary=disable_auxiliary,
        )["prediction"]
