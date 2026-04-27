from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from .bitlinear import BitLinear, BitLinearConfig
from .model_loading import load_auto_model, load_checkpoint_state_dict
from .packing import (
    effective_bits_per_weight,
    hard_binary_group_absmean,
    hard_ternary_absmean,
    hard_ternary_group_absmean,
    pack_bits,
    pack_trits_base3,
)
from .patch import patch_model_with_bitlinear, read_patch_manifest, write_patch_manifest


def export_packed_model(
    base_model: str,
    checkpoint: str | Path,
    out_dir: str | Path,
    scheme: str = "b1_58",
    group_size: int = 128,
    scale_method: str = "absmean",
    dtype: str = "bfloat16",
    include_vision: bool = False,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = BitLinearConfig(scheme=scheme, group_size=group_size, scale_method=scale_method)
    model = load_auto_model(base_model, dtype=dtype, device_map=None, trust_remote_code=True)
    stats = patch_model_with_bitlinear(model, cfg, include_vision=include_vision)
    state = load_checkpoint_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Missing keys while loading checkpoint: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys while loading checkpoint: {len(unexpected)}")

    tensors: dict[str, torch.Tensor] = {}
    metadata: dict[str, dict] = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, BitLinear):
            continue
        w = module.weight.detach().float().cpu()
        key_base = module_name.replace(".", "__")
        if scheme == "b1_58":
            if scale_method == "group_absmean":
                q, scale, pad = hard_ternary_group_absmean(w, group_size=group_size)
                trits = (q + 1).to(torch.uint8)
                packed, original_numel = pack_trits_base3(trits)
                tensors[f"{key_base}.packed_trits"] = packed
                tensors[f"{key_base}.scale"] = scale.cpu()
                metadata[module_name] = {
                    "shape": list(w.shape),
                    "scheme": scheme,
                    "encoding": "base3_5trits_per_uint8",
                    "scale_method": scale_method,
                    "group_size": group_size,
                    "pad_last_dim": pad,
                    "original_numel": original_numel,
                }
            else:
                q, scale = hard_ternary_absmean(w)
                trits = (q + 1).to(torch.uint8)
                packed, original_numel = pack_trits_base3(trits)
                tensors[f"{key_base}.packed_trits"] = packed
                tensors[f"{key_base}.scale"] = scale.reshape(1).cpu()
                metadata[module_name] = {
                    "shape": list(w.shape),
                    "scheme": scheme,
                    "encoding": "base3_5trits_per_uint8",
                    "scale_method": scale_method,
                    "group_size": None,
                    "original_numel": original_numel,
                }
        elif scheme == "b1_125":
            bits, scale, pad = hard_binary_group_absmean(w, group_size=group_size)
            packed, original_numel = pack_bits(bits)
            tensors[f"{key_base}.packed_bits"] = packed
            tensors[f"{key_base}.scale"] = scale.cpu()
            metadata[module_name] = {
                "shape": list(w.shape),
                "scheme": scheme,
                "encoding": "1bit_sign_plus_fp16_group_scale",
                "scale_method": "group_absmean",
                "group_size": group_size,
                "pad_last_dim": pad,
                "original_numel": original_numel,
            }
        else:
            raise ValueError(f"Unsupported scheme: {scheme}")

    save_file(tensors, str(out / "model.packed.safetensors"), metadata={"format": "qwen_bitnet_qat_packed"})
    payload = {
        "base_model": base_model,
        "checkpoint": str(checkpoint),
        "scheme": scheme,
        "effective_bits_per_weight": effective_bits_per_weight(scheme, group_size=group_size),
        "scale_method": scale_method,
        "group_size": group_size,
        "num_packed_bitlinear_layers": len(metadata),
        "layers": metadata,
        "warning": "This is a research packed artifact for this codebase. It is not a bitnet.cpp GGUF export.",
    }
    (out / "packed_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_patch_manifest(out, stats, cfg)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="Qwen/Qwen3.5-0.8B-Base")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--scheme", choices=["b1_58", "b1_125"], default="b1_58")
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--scale-method", choices=["absmean", "group_absmean"], default="absmean")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--include-vision", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = export_packed_model(
        base_model=args.base_model,
        checkpoint=args.checkpoint,
        out_dir=args.out,
        scheme=args.scheme,
        group_size=args.group_size,
        scale_method=args.scale_method,
        dtype=args.dtype,
        include_vision=args.include_vision,
    )
    print(json.dumps(payload, indent=2)[:4000])


if __name__ == "__main__":
    main()
