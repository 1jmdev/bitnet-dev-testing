from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from .bitlinear import BitLinear, BitLinearConfig

DEFAULT_EXCLUDE = (
    r"(^|\.)lm_head$",
    r"(^|\.)embed_tokens$",
    r"(^|\.)wte$",
    r"(^|\.)vision",
    r"(^|\.)visual",
    r"patch_embed",
    r"merger",
)


def _matches_any(name: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, name) for pattern in patterns)


def _iter_named_linears(module: nn.Module, prefix: str = ""):
    for child_name, child in module.named_children():
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Linear):
            yield module, child_name, full_name, child
        else:
            yield from _iter_named_linears(child, full_name)


def patch_model_with_bitlinear(
    model: nn.Module,
    bitlinear_config: BitLinearConfig | dict | None = None,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
    include_vision: bool = False,
    verbose: bool = False,
) -> dict:
    cfg = BitLinearConfig.from_dict(bitlinear_config) if isinstance(bitlinear_config, dict) else (bitlinear_config or BitLinearConfig())
    include_patterns = tuple(include or (r".*",))
    exclude_patterns = tuple(exclude or DEFAULT_EXCLUDE)
    if include_vision:
        exclude_patterns = tuple(p for p in exclude_patterns if p not in (r"(^|\.)vision", r"(^|\.)visual"))

    replaced: list[str] = []
    skipped: list[str] = []
    for parent, attr, full_name, child in list(_iter_named_linears(model)):
        if _matches_any(full_name, include_patterns) and not _matches_any(full_name, exclude_patterns):
            setattr(parent, attr, BitLinear.from_linear(child, cfg))
            replaced.append(full_name)
        else:
            skipped.append(full_name)

    model._qwen_bitnet_qat_config = cfg.to_dict()  # type: ignore[attr-defined]
    stats = {"replaced": replaced, "skipped": skipped, "num_replaced": len(replaced), "num_skipped": len(skipped)}
    if verbose:
        print(json.dumps(stats, indent=2))
    return stats


def count_bitlinear_modules(model: nn.Module) -> int:
    return sum(1 for module in model.modules() if isinstance(module, BitLinear))


def set_trainable_policy(
    model: nn.Module,
    policy: str = "all",
    train_norms: bool = True,
    train_lm_head: bool = False,
    train_embeddings: bool = False,
) -> dict:
    for param in model.parameters():
        param.requires_grad_(False)

    policy = policy.lower()
    trainable = 0
    total = 0
    for name, module in model.named_modules():
        enable = False
        if policy == "all":
            enable = True
        elif policy == "bitlinear":
            enable = isinstance(module, BitLinear)
        elif policy == "bitlinear_norm_lmhead":
            enable = isinstance(module, BitLinear) or (train_norms and _is_norm(module)) or (train_lm_head and name.endswith("lm_head"))
        else:
            raise ValueError(f"Unknown trainable policy: {policy}")

        if train_norms and _is_norm(module):
            enable = True
        if train_lm_head and name.endswith("lm_head"):
            enable = True
        if train_embeddings and isinstance(module, nn.Embedding):
            enable = True

        if enable:
            for param in module.parameters(recurse=False):
                param.requires_grad_(True)

    for param in model.parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    return {"trainable_parameters": trainable, "total_parameters": total, "trainable_fraction": trainable / max(total, 1)}


def _is_norm(module: nn.Module) -> bool:
    name = module.__class__.__name__.lower()
    return "norm" in name or isinstance(module, (nn.LayerNorm, nn.RMSNorm))


def write_patch_manifest(output_dir: str | Path, stats: dict, cfg: BitLinearConfig | dict) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "qwen_bitnet_qat_manifest.json"
    payload = {"bitlinear_config": cfg.to_dict() if isinstance(cfg, BitLinearConfig) else cfg, "patch_stats": stats}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_patch_manifest(path_or_dir: str | Path) -> dict:
    path = Path(path_or_dir)
    if path.is_dir():
        path = path / "qwen_bitnet_qat_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))
