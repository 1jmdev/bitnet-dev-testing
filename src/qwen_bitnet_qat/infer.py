from __future__ import annotations

import argparse

import torch
from safetensors.torch import load_file

from .bitlinear import BitLinearConfig
from .model_loading import load_auto_model, load_tokenizer
from .patch import patch_model_with_bitlinear


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="Qwen/Qwen3.5-0.8B-Base")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--prompt", required=True)
    p.add_argument("--scheme", choices=["b1_58", "b1_125"], default="b1_58")
    p.add_argument("--scale-method", choices=["absmean", "group_absmean"], default="absmean")
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args.base_model)
    model = load_auto_model(args.base_model, dtype="bfloat16" if torch.cuda.is_available() else "float32", device_map=None)
    patch_model_with_bitlinear(
        model,
        BitLinearConfig(scheme=args.scheme, scale_method=args.scale_method, group_size=args.group_size),
    )
    if args.checkpoint:
        state = load_file(args.checkpoint) if args.checkpoint.endswith(".safetensors") else torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=False)
    model.to(args.device)
    model.eval()
    inputs = tokenizer(args.prompt, return_tensors="pt").to(args.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    print(tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
