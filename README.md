# qwen-bitnet-qat

Research codebase for adapting Qwen/Qwen3.5-0.8B or Qwen/Qwen3.5-0.8B-Base into a BitNet-like model through quantization-aware post-training.

This does not pretend that a full-precision Qwen checkpoint can be losslessly converted into a production BitNet checkpoint in one command. Extreme 1-bit and 1.58-bit models need QAT or native training. This repo provides that QAT path:

- `b1_58`: ternary weights `{-1, 0, +1}` with absmean scaling and 8-bit activation fake quantization.
- `b1_125`: binary weights `{−scale, +scale}` with FP16 group scales, group size 128, giving 1 + 16/128 = 1.125 effective bits per weight before container overhead.
- BF16/FP32 master weights remain trainable during post-training.
- The exported packed format is for this repo. It is not a bitnet.cpp GGUF converter.

## Recommended source model

Use `Qwen/Qwen3.5-0.8B-Base` for the first QAT stage. It is the pre-trained-only base checkpoint and is better suited for continued pretraining under a new quantization constraint. Use `Qwen/Qwen3.5-0.8B` only when you specifically want to preserve chat/post-trained behavior and are willing to run SFT after the QAT adaptation.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

For Qwen3.5 support, use a Transformers build new enough to include Qwen3.5. If PyPI lags, install the main branch:

```bash
pip install 'transformers @ git+https://github.com/huggingface/transformers.git@main'
```

## Fetch config only

```bash
python scripts/fetch_hf_config.py \
  --model Qwen/Qwen3.5-0.8B-Base \
  --out metadata/qwen3_5_0_8b_base
```

This downloads small metadata files, not model weights.

## Prepare a toy dataset

```bash
python scripts/make_toy_jsonl.py --out data/train.jsonl --repeat 256
```

Real data should be JSONL with either:

```json
{"text": "raw causal LM text"}
```

or chat messages:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

## Train b1.58 QAT

```bash
accelerate launch -m qwen_bitnet_qat.train \
  --config configs/qwen3_5_0_8b_base_b158.yaml
```

## Train 1.125-bit binary/group-scale QAT

```bash
accelerate launch -m qwen_bitnet_qat.train \
  --config configs/qwen3_5_0_8b_base_b1125.yaml
```

## Export packed research artifact

For b1.58 ternary:

```bash
python -m qwen_bitnet_qat.export \
  --base-model Qwen/Qwen3.5-0.8B-Base \
  --checkpoint outputs/qwen3_5_0_8b_base_b158_qat/model.safetensors \
  --scheme b1_58 \
  --scale-method absmean \
  --out packed/qwen3_5_0_8b_base_b158
```

For 1.125-bit binary group scale:

```bash
python -m qwen_bitnet_qat.export \
  --base-model Qwen/Qwen3.5-0.8B-Base \
  --checkpoint outputs/qwen3_5_0_8b_base_b1125_qat/model.safetensors \
  --scheme b1_125 \
  --group-size 128 \
  --out packed/qwen3_5_0_8b_base_b1125
```

The export writes:

- `model.packed.safetensors`: packed uint8 weights plus FP16 scales.
- `packed_config.json`: layer metadata, original shapes, scale format, and effective bits per weight.
- `qwen_bitnet_qat_manifest.json`: patch/BitLinear metadata.

## Inference through fake-quantized BitLinear

```bash
python -m qwen_bitnet_qat.infer \
  --base-model Qwen/Qwen3.5-0.8B-Base \
  --checkpoint outputs/qwen3_5_0_8b_base_b158_qat/model.safetensors \
  --scheme b1_58 \
  --prompt "Explain BitNet in one paragraph."
```

## Practical training recipe

1. Start with `Qwen/Qwen3.5-0.8B-Base`.
2. Patch only language projection layers first. Leave token embeddings, LM head, and vision encoder unquantized for stability.
3. Run QAT continuation on a broad text/code/math corpus. For 0.8B, start with 10B+ tokens if quality matters; 100M–1B tokens is only a smoke/ablation run.
4. Run SFT after QAT if the final target is a chat model.
5. Consider a final DPO/preference stage only after SFT quality is recovered.
6. Export the packed research format, then write model-specific kernels if you need real speedups.

## Important limitations

Qwen3.5-0.8B is a hybrid multimodal model, not the same architecture as Microsoft BitNet b1.58 2B4T. This code replaces `torch.nn.Linear` projections with BitLinear-style fake-quantized projections in the loaded model. It does not rewrite Qwen3.5's Gated DeltaNet kernels, MTP path, vision encoder, or serving runtime.

Standard PyTorch/Transformers inference with fake quantization is useful for correctness testing. It is not where BitNet speedups come from. Production inference requires dedicated kernels and a runtime export matched to the target architecture.
