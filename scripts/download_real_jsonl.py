from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path


def fetch_rows(dataset: str, config: str, split: str, offset: int, length: int):
    qs = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    url = f"https://datasets-server.huggingface.co/rows?{qs}"
    with urllib.request.urlopen(url, timeout=60) as r:
        payload = json.loads(r.read().decode("utf-8"))
    return payload.get("rows", [])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="Salesforce/wikitext")
    p.add_argument("--name", default="wikitext-2-raw-v1")
    p.add_argument("--split", default="train")
    p.add_argument("--out", default="data/wikitext2_train.jsonl")
    p.add_argument("--limit", type=int, default=2000)
    p.add_argument("--page-size", type=int, default=100)
    p.add_argument("--text-field", default="text")
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    offset = 0

    with out.open("w", encoding="utf-8") as f:
        while written < args.limit:
            rows = fetch_rows(args.dataset, args.name, args.split, offset, min(args.page_size, args.limit - written))
            if not rows:
                break

            for item in rows:
                row = item.get("row", item)
                text = str(row.get(args.text_field, "")).strip()
                if text:
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    written += 1
                    if written >= args.limit:
                        break

            offset += len(rows)
            time.sleep(0.05)

    print({"out": str(out), "rows": written})


if __name__ == "__main__":
    main()
