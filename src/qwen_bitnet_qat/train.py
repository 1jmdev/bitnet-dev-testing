from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup, set_seed

from .bitlinear import BitLinearConfig
from .config_utils import ensure_dir, load_yaml
from .data import CausalBatchCollator, constant_length_tokenize, load_text_dataset, tokenize_for_causal_lm
from .model_loading import load_auto_model, load_tokenizer
from .patch import count_bitlinear_modules, patch_model_with_bitlinear, set_trainable_policy, write_patch_manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    return p.parse_args()


def _cfg(payload: dict, dotted: str, default=None):
    cur = payload
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _loss_with_optional_distillation(student_out, labels, teacher_logits=None, distill_weight=0.0, temperature=1.0):
    ce_loss = student_out.loss
    if teacher_logits is None or distill_weight <= 0:
        return ce_loss
    s = student_out.logits[:, :-1, :] / temperature
    t = teacher_logits[:, :-1, :] / temperature
    mask = labels[:, 1:] != -100
    kl = F.kl_div(F.log_softmax(s, dim=-1), F.softmax(t, dim=-1), reduction="none").sum(dim=-1)
    kl = (kl * mask).sum() / mask.sum().clamp_min(1)
    return ce_loss + distill_weight * (temperature**2) * kl


def train_from_config(config_path: str | Path) -> None:
    cfg = load_yaml(config_path)
    set_seed(int(_cfg(cfg, "training.seed", 42)))

    accelerator = Accelerator(
        gradient_accumulation_steps=int(_cfg(cfg, "training.gradient_accumulation_steps", 1)),
        mixed_precision=_cfg(cfg, "training.mixed_precision", "bf16"),
    )

    model_id = _cfg(cfg, "model.id", "Qwen/Qwen3.5-0.8B-Base")
    output_dir = ensure_dir(_cfg(cfg, "training.output_dir", "outputs/qwen3_5_0_8b_bitnet"))
    tokenizer = load_tokenizer(model_id, trust_remote_code=bool(_cfg(cfg, "model.trust_remote_code", True)))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    accelerator.print(f"Loading model: {model_id}")
    model = load_auto_model(
        model_id,
        dtype=_cfg(cfg, "model.dtype", "bfloat16"),
        device_map=None,
        trust_remote_code=bool(_cfg(cfg, "model.trust_remote_code", True)),
    )
    if hasattr(model, "config"):
        model.config.use_cache = False
    if bool(_cfg(cfg, "training.gradient_checkpointing", True)) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    bit_cfg = BitLinearConfig(
        scheme=_cfg(cfg, "bitlinear.scheme", "b1_58"),
        activation_bits=int(_cfg(cfg, "bitlinear.activation_bits", 8)),
        quantize_activations=bool(_cfg(cfg, "bitlinear.quantize_activations", True)),
        scale_method=_cfg(cfg, "bitlinear.scale_method", "absmean"),
        group_size=int(_cfg(cfg, "bitlinear.group_size", 128)),
        eps=float(_cfg(cfg, "bitlinear.eps", 1e-6)),
    )
    patch_stats = patch_model_with_bitlinear(
        model,
        bitlinear_config=bit_cfg,
        include=_cfg(cfg, "patch.include", None),
        exclude=_cfg(cfg, "patch.exclude", None),
        include_vision=bool(_cfg(cfg, "patch.include_vision", False)),
        verbose=bool(_cfg(cfg, "patch.verbose", False)),
    )
    train_stats = set_trainable_policy(
        model,
        policy=_cfg(cfg, "training.trainable_policy", "all"),
        train_norms=bool(_cfg(cfg, "training.train_norms", True)),
        train_lm_head=bool(_cfg(cfg, "training.train_lm_head", False)),
        train_embeddings=bool(_cfg(cfg, "training.train_embeddings", False)),
    )
    accelerator.print(f"Patched BitLinear modules: {count_bitlinear_modules(model)}")
    accelerator.print(json.dumps(train_stats, indent=2))

    teacher = None
    distill_weight = float(_cfg(cfg, "distillation.weight", 0.0))
    if distill_weight > 0:
        accelerator.print("Loading teacher model for KL distillation")
        teacher = load_auto_model(
            model_id,
            dtype=_cfg(cfg, "model.dtype", "bfloat16"),
            device_map=None,
            trust_remote_code=bool(_cfg(cfg, "model.trust_remote_code", True)),
        )
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    ds = load_text_dataset(
        _cfg(cfg, "data.dataset", "data/train.jsonl"),
        split=_cfg(cfg, "data.split", "train"),
        data_files=_cfg(cfg, "data.data_files", None),
    )
    if bool(_cfg(cfg, "data.constant_length", False)):
        tokenized = constant_length_tokenize(
            ds,
            tokenizer,
            text_field=_cfg(cfg, "data.text_field", "text"),
            messages_field=_cfg(cfg, "data.messages_field", "messages"),
            seq_len=int(_cfg(cfg, "data.max_seq_len", 2048)),
            num_proc=_cfg(cfg, "data.num_proc", None),
        )
    else:
        tokenized = tokenize_for_causal_lm(
            ds,
            tokenizer,
            text_field=_cfg(cfg, "data.text_field", "text"),
            messages_field=_cfg(cfg, "data.messages_field", "messages"),
            max_seq_len=int(_cfg(cfg, "data.max_seq_len", 2048)),
            num_proc=_cfg(cfg, "data.num_proc", None),
        )

    collator = CausalBatchCollator(pad_token_id=tokenizer.pad_token_id)
    dataloader = DataLoader(
        tokenized,
        batch_size=int(_cfg(cfg, "training.micro_batch_size", 1)),
        shuffle=True,
        collate_fn=collator,
        num_workers=int(_cfg(cfg, "data.num_workers", 0)),
    )

    optim_name = str(_cfg(cfg, "training.optimizer", "adamw")).lower()
    optim_params = [p for p in model.parameters() if p.requires_grad]

    if optim_name in {"adamw8bit", "paged_adamw8bit", "paged_adamw_8bit"}:
        import bitsandbytes as bnb

        opt_cls = bnb.optim.PagedAdamW8bit if "paged" in optim_name else bnb.optim.AdamW8bit
        optim = opt_cls(
            optim_params,
            lr=float(_cfg(cfg, "training.learning_rate", 2e-5)),
            betas=tuple(_cfg(cfg, "training.betas", [0.9, 0.95])),
            weight_decay=float(_cfg(cfg, "training.weight_decay", 0.1)),
        )
    else:
        optim = AdamW(
            optim_params,
            lr=float(_cfg(cfg, "training.learning_rate", 2e-5)),
            betas=tuple(_cfg(cfg, "training.betas", [0.9, 0.95])),
            weight_decay=float(_cfg(cfg, "training.weight_decay", 0.1)),
        )
    epochs = int(_cfg(cfg, "training.epochs", 1))
    max_steps = int(_cfg(cfg, "training.max_steps", 0))
    updates_per_epoch = math.ceil(len(dataloader) / accelerator.gradient_accumulation_steps)
    total_steps = max_steps if max_steps > 0 else epochs * updates_per_epoch
    scheduler = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=int(_cfg(cfg, "training.warmup_steps", 100)),
        num_training_steps=total_steps,
    )

    if teacher is not None:
        model, teacher, optim, dataloader, scheduler = accelerator.prepare(model, teacher, optim, dataloader, scheduler)
    else:
        model, optim, dataloader, scheduler = accelerator.prepare(model, optim, dataloader, scheduler)

    model.train()
    global_step = 0
    save_every = int(_cfg(cfg, "training.save_every_steps", 1000))
    max_grad_norm = float(_cfg(cfg, "training.max_grad_norm", 1.0))
    progress = tqdm(total=total_steps, disable=not accelerator.is_local_main_process)

    for _epoch in range(epochs):
        for batch in dataloader:
            with accelerator.accumulate(model):
                teacher_logits = None
                if teacher is not None:
                    with torch.no_grad():
                        teacher_logits = teacher(**batch).logits.detach()
                out = model(**batch)
                loss = _loss_with_optional_distillation(
                    out,
                    batch["labels"],
                    teacher_logits=teacher_logits,
                    distill_weight=distill_weight,
                    temperature=float(_cfg(cfg, "distillation.temperature", 1.0)),
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                optim.step()
                scheduler.step()
                optim.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                progress.set_postfix(loss=f"{loss.detach().float().item():.4f}")
                if save_every > 0 and global_step % save_every == 0:
                    save_checkpoint(accelerator, model, tokenizer, output_dir / f"step_{global_step}", patch_stats, bit_cfg, cfg)
                if global_step >= total_steps:
                    break
        if global_step >= total_steps:
            break

    progress.close()
    save_checkpoint(accelerator, model, tokenizer, output_dir, patch_stats, bit_cfg, cfg)


def save_checkpoint(accelerator, model, tokenizer, out_dir: Path, patch_stats: dict, bit_cfg: BitLinearConfig, cfg: dict) -> None:
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    write_patch_manifest(out_dir, patch_stats, bit_cfg)
    (out_dir / "train_config_resolved.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    accelerator.print(f"Saved checkpoint: {out_dir}")


def main() -> None:
    args = parse_args()
    train_from_config(args.config)


if __name__ == "__main__":
    main()
