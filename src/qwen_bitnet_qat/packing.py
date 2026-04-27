from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def pack_bits(bits: torch.Tensor) -> tuple[torch.Tensor, int]:
    bits = bits.to(torch.uint8).flatten()
    original_numel = bits.numel()
    pad = (-original_numel) % 8
    if pad:
        bits = F.pad(bits, (0, pad))
    bits = bits.reshape(-1, 8)
    shifts = torch.arange(8, dtype=torch.uint8, device=bits.device)
    packed = torch.sum(bits << shifts, dim=-1).to(torch.uint8)
    return packed.cpu(), original_numel


def unpack_bits(packed: torch.Tensor, original_numel: int) -> torch.Tensor:
    packed = packed.to(torch.uint8).flatten()
    shifts = torch.arange(8, dtype=torch.uint8, device=packed.device)
    bits = ((packed[:, None] >> shifts) & 1).reshape(-1)
    return bits[:original_numel].to(torch.uint8)


def pack_trits_base3(trits: torch.Tensor) -> tuple[torch.Tensor, int]:
    trits = trits.to(torch.uint8).flatten()
    original_numel = trits.numel()
    pad = (-original_numel) % 5
    if pad:
        trits = F.pad(trits, (0, pad))
    trits = trits.reshape(-1, 5)
    powers = torch.tensor([1, 3, 9, 27, 81], dtype=torch.uint8, device=trits.device)
    packed = torch.sum(trits * powers, dim=-1).to(torch.uint8)
    return packed.cpu(), original_numel


def unpack_trits_base3(packed: torch.Tensor, original_numel: int) -> torch.Tensor:
    values = packed.to(torch.long).flatten()
    outs = []
    for _ in range(5):
        outs.append((values % 3).to(torch.uint8))
        values = values // 3
    trits = torch.stack(outs, dim=1).reshape(-1)
    return trits[:original_numel]


def hard_ternary_absmean(weight: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    scale = weight.abs().mean().clamp_min(eps).to(torch.float16)
    q = torch.clamp(torch.round(weight / scale.to(weight.dtype)), -1, 1).to(torch.int8)
    return q, scale


def hard_ternary_group_absmean(weight: torch.Tensor, group_size: int = 128, eps: float = 1e-6):
    original_shape = tuple(weight.shape)
    flat = weight.reshape(-1, original_shape[-1])
    pad = (-flat.shape[-1]) % group_size
    if pad:
        flat = F.pad(flat, (0, pad))
    grouped = flat.reshape(flat.shape[0], -1, group_size)
    scale = grouped.abs().mean(dim=-1, keepdim=True).clamp_min(eps)
    q = torch.clamp(torch.round(grouped / scale), -1, 1).to(torch.int8)
    if pad:
        q_flat = q.reshape(flat.shape[0], -1)[:, :-pad]
    else:
        q_flat = q.reshape(flat.shape[0], -1)
    return q_flat.reshape(original_shape), scale.squeeze(-1).to(torch.float16), pad


def hard_binary_group_absmean(weight: torch.Tensor, group_size: int = 128, eps: float = 1e-6):
    original_shape = tuple(weight.shape)
    flat = weight.reshape(-1, original_shape[-1])
    pad = (-flat.shape[-1]) % group_size
    if pad:
        flat = F.pad(flat, (0, pad))
    grouped = flat.reshape(flat.shape[0], -1, group_size)
    scale = grouped.abs().mean(dim=-1, keepdim=True).clamp_min(eps)
    bits = (grouped >= 0).to(torch.uint8)
    if pad:
        bits_flat = bits.reshape(flat.shape[0], -1)[:, :-pad]
    else:
        bits_flat = bits.reshape(flat.shape[0], -1)
    return bits_flat.reshape(original_shape), scale.squeeze(-1).to(torch.float16), pad


def effective_bits_per_weight(scheme: str, group_size: int = 128) -> float:
    if scheme == "b1_125":
        return 1.0 + 16.0 / group_size
    if scheme == "b1_58":
        return math.log2(3.0)
    raise ValueError(scheme)
