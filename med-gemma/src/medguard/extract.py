"""MedGemma model loading + prescription extraction (vision and text).

Measured-good recipe for Tesla T4 (sm_75), verified on this box:
  - torch.bfloat16 (fp16 produces NaN logits — Gemma 3 attention overflows
    fp16's exponent range; the checkpoint's native dtype is bf16)
  - attn_implementation="sdpa" (flash-attn needs SM 8.0+)
  - Gemma3ForConditionalGeneration, device_map="cuda:0"
  - pixel_values cast to bf16 (processor otherwise hands back fp32)

Default model is google/medgemma-4b-it (fully cached locally).
Override: MEDGEMMA_MODEL_ID=google/medgemma-1.5-4b-it (needs download + access).

Requires gated access unless weights are cached:
https://huggingface.co/google/medgemma-4b-it
"""

from __future__ import annotations

import json
import os
import re

from PIL import Image

from .config import MODEL_ID
from .models import Medication
from .prompts import EXTRACTION_PROMPT, EXTRACTION_PROMPT_TEXTONLY

_model = None
_processor = None


def load_model(model_id: str = MODEL_ID):
    """Load MedGemma once per process. Raises with a helpful message if gated."""
    global _model, _processor
    if _model is not None:
        return _model, _processor
    import torch
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    try:
        _model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        _model.eval()
        _processor = AutoProcessor.from_pretrained(model_id)
        if hasattr(_processor, "image_processor"):
            try:
                _processor.image_processor.torch_dtype = torch.bfloat16
            except Exception:
                pass
    except Exception as e:
        raise RuntimeError(
            f"Could not load {model_id}. Gated repo: request access at "
            f"https://huggingface.co/{model_id} and run `hf auth login`. "
            f"Original error: {e}") from e
    return _model, _processor


def normalize_messages(messages: list[dict]) -> list[dict]:
    """Wrap plain-string content into typed parts (transformers>=4.50 needs it)."""
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            content = [c if isinstance(c, dict) else {"type": "text", "text": str(c)}
                       for c in content]
        out.append({"role": m["role"], "content": content})
    return out


def build_inputs(messages: list[dict]):
    import torch

    model, processor = load_model()
    inputs = processor.apply_chat_template(
        normalize_messages(messages), add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
    for k, v in inputs.items():
        if torch.is_floating_point(v):
            inputs[k] = v.to(torch.bfloat16)
    return inputs


def _generate_local(messages: list[dict], max_new_tokens: int = 1024) -> str:
    import torch

    model, processor = load_model()
    inputs = build_inputs(messages)
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.decode(out[0][input_len:], skip_special_tokens=True)


def _use_gateway() -> bool:
    return os.environ.get("MEDGUARD_BACKEND", "local").lower() == "openwebui"


def _generate(messages: list[dict], max_new_tokens: int = 1024) -> str:
    if _use_gateway():
        from .gateway import complete
        return complete(messages, max_tokens=max_new_tokens)
    return _generate_local(messages, max_new_tokens)


def _parse_extraction(raw: str) -> tuple[str, list[Medication]]:
    """Pull the JSON object out of possibly-chatty output; strict-validate."""
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise ValueError("Model did not return JSON")
    data = json.loads(m.group(0))
    meds = [Medication(**{k: (v or "") for k, v in med.items()
                           if k in Medication.model_fields})
            for med in data.get("medications", [])]
    return data.get("raw_text", ""), meds


def extract_from_image(image: Image.Image) -> tuple[str, list[Medication]]:
    image = image.convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": EXTRACTION_PROMPT}]}]
    return _parse_extraction(_generate(messages, max_new_tokens=1500))


def extract_from_text(text: str) -> tuple[str, list[Medication]]:
    messages = [{"role": "user", "content": [
        {"type": "text", "text": EXTRACTION_PROMPT_TEXTONLY.format(text=text)}]}]
    return _parse_extraction(_generate(messages, max_new_tokens=1500))


def model_loaded() -> bool:
    return _model is not None


def device_info() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            return (f"{torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]}), "
                    "dtype=bfloat16, attn=sdpa")
        return "cpu"
    except Exception:
        return "unknown"


if __name__ == "__main__":
    # Smoke test: python -m medguard.extract
    m, p = load_model()
    print("loaded:", device_info())
    print(_generate([{"role": "user", "content": "Reply with exactly: Paris"}],
                    max_new_tokens=16))
