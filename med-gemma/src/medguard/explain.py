"""Findings-grounded plain-language explanation via MedGemma.

The model receives the deterministic Review as JSON plus a grounding preamble
that forbids adding any claim. Streaming is supported for the Chainlit UI.
"""

from __future__ import annotations

from threading import Thread
from typing import Iterator

from .config import DISCLAIMER
from .extract import load_model
from .models import Review
from .prompts import GROUNDING_PREAMBLE, MEDGUARD_SYSTEM_PROMPT

LANG_NAMES = {"en": "English", "hi": "Hindi (Devanagari script, simple words)"}


def build_messages(review: Review, language: str = "en") -> list[dict]:
    lang = LANG_NAMES.get(language, "English")
    return [
        {"role": "system", "content": [{"type": "text", "text": MEDGUARD_SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text":
            GROUNDING_PREAMBLE.format(language=lang)
            + "\nVERIFIED FINDINGS JSON:\n"
            + review.model_dump_json(indent=2)}]},
    ]


def explain_review(review: Review, language: str = "en",
                   max_new_tokens: int = 1200) -> str:
    from .extract import _use_gateway

    if _use_gateway():
        from . import gateway
        text = gateway.complete(build_messages(review, language),
                                max_tokens=max_new_tokens)
    else:
        text = _explain_local(review, language, max_new_tokens)
    if "IMPORTANT DISCLAIMER" not in text:
        text += "\n\n6. IMPORTANT DISCLAIMER\n\"" + DISCLAIMER + "\""
    return text


def _explain_local(review: Review, language: str = "en",
                   max_new_tokens: int = 1200) -> str:
    import torch

    model, processor = load_model()
    inputs = processor.apply_chat_template(
        build_messages(review, language), add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = processor.decode(out[0][input_len:], skip_special_tokens=True)
    if "IMPORTANT DISCLAIMER" not in text:
        text += "\n\n6. IMPORTANT DISCLAIMER\n\"" + DISCLAIMER + "\""
    return text


def explain_stream(review: Review, language: str = "en",
                   max_new_tokens: int = 1200) -> Iterator[str]:
    """Yield text chunks (Chainlit / st.write_stream compatible)."""
    from .extract import _use_gateway

    if _use_gateway():
        from . import gateway
        yield from gateway.complete_stream(build_messages(review, language),
                                           max_tokens=max_new_tokens)
        return
    import torch
    from transformers import TextIteratorStreamer

    model, processor = load_model()
    inputs = processor.apply_chat_template(
        build_messages(review, language), add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(processor, skip_special_tokens=True)
    kwargs = {**inputs, "max_new_tokens": max_new_tokens,
              "do_sample": False, "streamer": streamer}
    Thread(target=model.generate, kwargs=kwargs, daemon=True).start()
    yield from streamer
