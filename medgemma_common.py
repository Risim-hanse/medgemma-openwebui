"""Shared MedGemma loading logic for the T4 (SM 7.5).

Hardware notes for this box:
  - Tesla T4, 15 GB VRAM, compute capability 7.5 (Turing).
  - bf16 is EMULATED on the T4 (no native bf16 tensor cores). A raw matmul
    benchmark looks damning: fp16 ~44 TFLOPS vs bf16 ~2.8 TFLOPS.
    That benchmark is misleading for this workload. Autoregressive decode is
    memory-bandwidth bound, and bf16 costs the same 2 bytes per weight as
    fp16, so real decode speed is close (~50 tok/s either way).
  - CRITICAL: this model does NOT work in fp16. Measured on the T4 with the
    real medgemma-4b-it weights:
        bfloat16 -> no NaN, correct output ("Paris"), ~54 tok/s
        float16  -> NaN logits, empty/garbage output
    Gemma 3's attention path overflows fp16's narrow exponent range
    (max 65504, min normal ~6e-5). The checkpoint's native dtype is
    bfloat16. Use bf16 despite the emulated-math penalty.
  - FlashAttention-2 requires SM 8.0+ and cannot be used. We rely on
    PyTorch SDPA, whose mem-efficient backend supports SM 7.5.
"""

import os

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

MODEL_ID = os.environ.get("MEDGEMMA_MODEL_ID", "google/medgemma-4b-it")

# bfloat16, NOT float16 -- see module docstring. fp16 produces NaNs here.
DTYPE = torch.bfloat16


def device_note() -> str:
    cap = torch.cuda.get_device_capability(0)
    return (
        f"{torch.cuda.get_device_name(0)} "
        f"(sm_{cap[0]}{cap[1]}), dtype={DTYPE}"
    )


def load_model(model_id: str = MODEL_ID, load_in_4bit: bool = False):
    """Load MedGemma + processor onto the T4 in float16.

    load_in_4bit is off by default: the 4B model fits comfortably in
    fp16 (~8 GB of 15 GB), and quantization costs accuracy for no gain.
    """
    kwargs = dict(
        dtype=DTYPE,
        device_map="cuda:0",
        attn_implementation="sdpa",  # flash-attn is impossible on SM 7.5
        low_cpu_mem_usage=True,
    )
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=DTYPE,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        # device_map must not be set alongside a 4bit quant config.
        kwargs.pop("device_map")

    model = Gemma3ForConditionalGeneration.from_pretrained(model_id, **kwargs)
    model.eval()

    processor = AutoProcessor.from_pretrained(model_id)
    # Keep image tokens on GPU in fp16 rather than letting the processor
    # hand back fp32 pixel values.
    if hasattr(processor, "image_processor"):
        try:
            processor.image_processor.torch_dtype = DTYPE
        except Exception:
            pass

    return model, processor


def _normalize_messages(messages):
    """Gemma3's chat template iterates message["content"] as a list of typed
    parts. transformers >=4.50 requires that structure, so plain string
    content (as used in Google's model-card examples) must be wrapped."""
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            content = [
                c if isinstance(c, dict) else {"type": "text", "text": str(c)}
                for c in content
            ]
        out.append({"role": m["role"], "content": content})
    return out


def build_inputs(processor, model, messages, device=None):
    """Tokenize a chat-template message list and move it onto the GPU."""
    device = device or model.device
    messages = _normalize_messages(messages)
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    # Cast floating point tensors (pixel_values) to fp16.
    for k, v in inputs.items():
        if torch.is_floating_point(v):
            inputs[k] = v.to(DTYPE)
    return inputs


@torch.inference_mode()
def generate(model, processor, messages, max_new_tokens=256):
    inputs = build_inputs(processor, model, messages)
    prompt_len = inputs["input_ids"].shape[-1]
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return processor.decode(out[0][prompt_len:], skip_special_tokens=True)