from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    p.add_argument("--out", default="metadata/qwen3_5_0_8b_base")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    found = []
    for name in FILES:
        try:
            src = Path(hf_hub_download(args.model, name))
        except Exception as exc:
            print(f"skip {name}: {exc}")
            continue
        dst = out / name
        dst.write_bytes(src.read_bytes())
        found.append(name)
    summary = {"model": args.model, "files": found}
    if (out / "config.json").exists():
        config = json.loads((out / "config.json").read_text(encoding="utf-8"))
        text = config.get("text_config", config)
        summary["model_type"] = config.get("model_type")
        summary["text_config"] = {
            "hidden_size": text.get("hidden_size"),
            "intermediate_size": text.get("intermediate_size"),
            "num_hidden_layers": text.get("num_hidden_layers"),
            "num_attention_heads": text.get("num_attention_heads"),
            "num_key_value_heads": text.get("num_key_value_heads"),
            "vocab_size": text.get("vocab_size"),
            "max_position_embeddings": text.get("max_position_embeddings"),
            "layer_types": text.get("layer_types"),
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
