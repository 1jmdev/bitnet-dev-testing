from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class TextRowsDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]

    @property
    def column_names(self):
        keys = set()
        for row in self.rows[:100]:
            keys.update(row.keys())
        return list(keys)


class TokenRowsDataset(Dataset):
    def __init__(self, rows: list[dict[str, list[int]]]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


@dataclass
class CausalBatchCollator:
    pad_token_id: int
    label_pad_token_id: int = -100

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = [torch.tensor(row["input_ids"], dtype=torch.long) for row in rows]
        labels = [torch.tensor(row.get("labels", row["input_ids"]), dtype=torch.long) for row in rows]
        attention_mask = [torch.ones_like(x) for x in input_ids]
        return {
            "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id),
            "attention_mask": pad_sequence(attention_mask, batch_first=True, padding_value=0),
            "labels": pad_sequence(labels, batch_first=True, padding_value=self.label_pad_token_id),
        }


def load_text_dataset(path_or_name: str, split: str = "train", data_files=None) -> TextRowsDataset:
    path = Path(path_or_name)
    rows: list[dict[str, Any]] = []

    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    elif path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            rows = payload.get(split, payload.get("data", []))
        else:
            raise TypeError("Unsupported JSON dataset format")
    elif path.suffix == ".txt":
        rows = [{"text": x.strip()} for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    else:
        raise ValueError(
            "This Python 3.14-safe loader only supports local .jsonl, .json, or .txt files. "
            "Download HF datasets to JSONL first."
        )

    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    return TextRowsDataset(rows)


def _row_to_text(row: dict[str, Any], tokenizer, text_field: str, messages_field: str | None) -> str:
    if messages_field and messages_field in row and row[messages_field] is not None:
        return tokenizer.apply_chat_template(row[messages_field], tokenize=False, add_generation_prompt=False)
    if text_field in row and row[text_field] is not None:
        return str(row[text_field])
    for key in ("prompt", "content", "completion"):
        if key in row and row[key] is not None:
            return str(row[key])
    raise KeyError(f"No text field found. Tried {text_field}, messages, prompt, content, completion")


def tokenize_for_causal_lm(
    dataset: Dataset,
    tokenizer,
    text_field: str = "text",
    messages_field: str | None = "messages",
    max_seq_len: int = 2048,
    num_proc: int | None = None,
) -> TokenRowsDataset:
    eos = tokenizer.eos_token or ""
    rows = []

    for row in dataset:
        text = _row_to_text(row, tokenizer, text_field=text_field, messages_field=messages_field)
        if eos and not text.endswith(eos):
            text += eos
        ids = tokenizer(text, truncation=True, max_length=max_seq_len, add_special_tokens=True)["input_ids"]
        if len(ids) > 1:
            rows.append({"input_ids": ids, "labels": list(ids)})

    if not rows:
        raise ValueError("Tokenization produced zero usable rows")
    return TokenRowsDataset(rows)


def constant_length_tokenize(
    dataset: Dataset,
    tokenizer,
    text_field: str = "text",
    messages_field: str | None = "messages",
    seq_len: int = 2048,
    num_proc: int | None = None,
) -> TokenRowsDataset:
    eos_id = tokenizer.eos_token_id
    all_ids: list[int] = []

    for row in dataset:
        text = _row_to_text(row, tokenizer, text_field=text_field, messages_field=messages_field)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        all_ids.extend(ids)
        if eos_id is not None:
            all_ids.append(eos_id)

    total = (len(all_ids) // seq_len) * seq_len
    rows = [
        {"input_ids": all_ids[i : i + seq_len], "labels": all_ids[i : i + seq_len]}
        for i in range(0, total, seq_len)
    ]

    if not rows:
        raise ValueError(f"Not enough tokens for one constant-length sequence of {seq_len}")
    return TokenRowsDataset(rows)
