from __future__ import annotations

import inspect
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer


def _get_auto_model_class():
    import transformers

    candidates = [
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModelForCausalLM",
    ]
    for name in candidates:
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls
    raise RuntimeError("No compatible Transformers AutoModel class found.")


def load_tokenizer(model_id: str, trust_remote_code: bool = True):
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code, use_fast=True)


def load_auto_model(
    model_id: str,
    dtype: str | torch.dtype = "bfloat16",
    device_map: str | dict | None = None,
    trust_remote_code: bool = True,
    low_cpu_mem_usage: bool = True,
):
    auto_cls = _get_auto_model_class()
    kwargs = {
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": low_cpu_mem_usage,
    }
    if device_map is not None:
        kwargs["device_map"] = device_map
    if dtype is not None:
        dtype_value = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        sig = inspect.signature(auto_cls.from_pretrained)
        if "torch_dtype" in sig.parameters:
            kwargs["torch_dtype"] = dtype_value
        else:
            kwargs["dtype"] = dtype_value
    return auto_cls.from_pretrained(model_id, **kwargs)


def load_config_only(model_id: str, trust_remote_code: bool = True):
    return AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)


def load_checkpoint_state_dict(checkpoint_dir: str | Path) -> dict[str, torch.Tensor]:
    path = Path(checkpoint_dir)
    if path.is_file():
        if path.suffix == ".safetensors":
            return load_file(str(path))
        return torch.load(path, map_location="cpu")

    safetensor_files = sorted(path.glob("*.safetensors"))
    if len(safetensor_files) == 1:
        return load_file(str(safetensor_files[0]))
    if (path / "model.safetensors").exists():
        return load_file(str(path / "model.safetensors"))
    if (path / "pytorch_model.bin").exists():
        return torch.load(path / "pytorch_model.bin", map_location="cpu")
    raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found in {checkpoint_dir}")
