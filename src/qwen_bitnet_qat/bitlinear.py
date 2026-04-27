from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

BitScheme = Literal["b1_58", "b1_125"]
ScaleMethod = Literal["absmean", "group_absmean"]


@dataclass
class BitLinearConfig:
    scheme: BitScheme = "b1_58"
    activation_bits: int = 8
    quantize_activations: bool = True
    scale_method: ScaleMethod = "absmean"
    group_size: int = 128
    eps: float = 1e-6

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict | None) -> "BitLinearConfig":
        return cls(**(payload or {}))


def _round_ste(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.round(x) - x).detach()


def _clamp_ste(x: torch.Tensor, min_value: float, max_value: float) -> torch.Tensor:
    return x + (torch.clamp(x, min_value, max_value) - x).detach()


def _pad_last_dim(x: torch.Tensor, group_size: int) -> tuple[torch.Tensor, int]:
    remainder = x.shape[-1] % group_size
    pad = 0 if remainder == 0 else group_size - remainder
    if pad:
        x = F.pad(x, (0, pad))
    return x, pad


def quantize_activation_absmax(x: torch.Tensor, bits: int = 8, eps: float = 1e-6) -> torch.Tensor:
    if bits <= 0:
        return x
    qmax = float(2 ** (bits - 1) - 1)
    scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp_min(eps) / qmax
    x_int = _clamp_ste(_round_ste(x / scale), -qmax - 1, qmax)
    return x_int * scale


def ternary_weight_absmean(w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    scale = w.detach().abs().mean().clamp_min(eps)
    q = _clamp_ste(_round_ste(w / scale), -1.0, 1.0)
    return q * scale


def ternary_weight_group_absmean(w: torch.Tensor, group_size: int = 128, eps: float = 1e-6) -> torch.Tensor:
    original_shape = w.shape
    flat = w.reshape(-1, original_shape[-1])
    padded, pad = _pad_last_dim(flat, group_size)
    grouped = padded.reshape(flat.shape[0], -1, group_size)
    scale = grouped.detach().abs().mean(dim=-1, keepdim=True).clamp_min(eps)
    q = _clamp_ste(_round_ste(grouped / scale), -1.0, 1.0)
    out = (q * scale).reshape(flat.shape[0], -1)
    if pad:
        out = out[:, :-pad]
    return out.reshape(original_shape)


def binary_weight_group_absmean(w: torch.Tensor, group_size: int = 128, eps: float = 1e-6) -> torch.Tensor:
    original_shape = w.shape
    flat = w.reshape(-1, original_shape[-1])
    padded, pad = _pad_last_dim(flat, group_size)
    grouped = padded.reshape(flat.shape[0], -1, group_size)
    scale = grouped.detach().abs().mean(dim=-1, keepdim=True).clamp_min(eps)
    q_hard = torch.where(grouped >= 0, torch.ones_like(grouped), -torch.ones_like(grouped))
    q = grouped + (q_hard - grouped).detach()
    out = (q * scale).reshape(flat.shape[0], -1)
    if pad:
        out = out[:, :-pad]
    return out.reshape(original_shape)


def bitnet_weight_quantize(w: torch.Tensor, cfg: BitLinearConfig) -> torch.Tensor:
    if cfg.scheme == "b1_58":
        if cfg.scale_method == "group_absmean":
            return ternary_weight_group_absmean(w, group_size=cfg.group_size, eps=cfg.eps)
        return ternary_weight_absmean(w, eps=cfg.eps)
    if cfg.scheme == "b1_125":
        return binary_weight_group_absmean(w, group_size=cfg.group_size, eps=cfg.eps)
    raise ValueError(f"Unsupported BitLinear scheme: {cfg.scheme}")


class BitLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        config: BitLinearConfig | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or BitLinearConfig()
        self.weight = nn.Parameter(torch.empty((out_features, in_features), dtype=dtype, device=device))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype, device=device)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / fan_in**0.5 if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    def from_linear(cls, module: nn.Linear, config: BitLinearConfig | None = None) -> "BitLinear":
        out = cls(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            config=config,
            dtype=module.weight.dtype,
            device=module.weight.device,
        )
        with torch.no_grad():
            out.weight.copy_(module.weight)
            if module.bias is not None and out.bias is not None:
                out.bias.copy_(module.bias)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.quantize_activations:
            x = quantize_activation_absmax(x, bits=self.config.activation_bits, eps=self.config.eps)
        w = bitnet_weight_quantize(self.weight, self.config)
        return F.linear(x, w, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, scheme={self.config.scheme}, "
            f"activation_bits={self.config.activation_bits}, scale_method={self.config.scale_method}, "
            f"group_size={self.config.group_size}"
        )
