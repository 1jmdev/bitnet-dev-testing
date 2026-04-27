from __future__ import annotations

import argparse
import json
from pathlib import Path

ROWS = [
    "BitNet-style training constrains linear projections during forward passes while preserving trainable master weights.",
    "Qwen3.5-0.8B uses a hybrid language architecture and includes a vision encoder, so deployment exports need model-specific kernels.",
    "Quantization-aware training is preferred over one-shot conversion for extreme 1-bit and ternary models.",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/toy.jsonl")
    p.add_argument("--repeat", type=int, default=128)
    args = p.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i in range(args.repeat):
            f.write(json.dumps({"text": ROWS[i % len(ROWS)]}) + "\n")
    print(out)


if __name__ == "__main__":
    main()
