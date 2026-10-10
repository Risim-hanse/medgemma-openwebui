"""MedGuard - on-device medical prescription & drug-interaction inspector.

Chainlit front end for MedGemma 4B running locally on a Tesla T4.

Why Chainlit over Streamlit:
  - Real token-by-token streaming. On a T4 generation is slow (single-digit
    tok/s), so streaming is the difference between a responsive UI and one
    that looks hung. Streamlit reruns the entire script per interaction and
    cannot stream tokens.
  - WebSocket transport instead of full-page rerenders.
  - Native image upload, which MedGemma needs for prescription photos and
    clinical imagery (multimodal SigLIP encoder).

Why not Open WebUI / LibreChat / Dify: each wants 12GB+ RAM of its own, and
this box has a ~13.4GiB cgroup cap shared with an ~8GB model load. Not viable.

Run:  ./venv/bin/chainlit run ui/chainlit_app.py --port 8001
"""

import os
import sys
import threading
import time

# Chainlit execs this file by path with ui/ as the module dir, so the repo
# root (which holds medgemma_common.py and the venv layout) is not on
# sys.path. Add it before importing our own modules.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import chainlit as cl
import torch
from PIL import Image
from transformers import TextIteratorStreamer

from medgemma_common import MODEL_ID, build_inputs, device_note, load_model

STATE = {"model": None, "processor": None}

# Deliberately conservative. MedGemma is an assistive inspector, not a
# prescriber, and its own model card says it must not inform clinical
# decisions directly.
SYSTEM_PROMPT = """You are MedGuard, a cautious medical assistant that inspects \
prescriptions and checks for drug-drug interactions.

You are NOT a doctor and NOT a prescriber. Your role is to:
- Transcribe medications, doses, and frequencies from prescriptions or images.
- Flag potential drug-drug interactions and contraindications you recognize.
- Note anything clinically unusual about the regimen.
- Recommend the pharmacist or prescribing doctor confirm anything you raise.

Always:
- State clearly that output is informational and requires pharmacist review.
- Never tell the user to start, stop, or change a dose themselves.
- If you are unsure whether an interaction exists, say so rather than guessing.
- Prefer flagging for human review over false reassurance.

Respond concisely in plain language. Where severity is genuinely unclear, \
say "needs pharmacist review" rather than inventing a confidence level."""


@cl.on_chat_start
async def start():
    cl.user_session.set("history", [])
    if STATE["model"] is None:
        await cl.Message(
            content=f"Loading MedGemma ({MODEL_ID}) onto the GPU..."
        ).send()
        # Off the event loop: the load is synchronous and takes ~30s.
        STATE["model"], STATE["processor"] = await cl.make_async(load_model)()
    await cl.Message(
        author="MedGuard",
        content=(
            f"Ready. Running on {device_note()}.\n\n"
            "Paste a prescription or upload a photo, and I'll inspect it."
        ),
    ).send()
    await _send_clear_button()


@cl.on_message
async def on_message(message: cl.Message):
    model, processor = STATE["model"], STATE["processor"]
    if model is None:
        await cl.Message(content="Model is still loading, one moment.").send()
        return

    images = []
    for element in message.elements:
        if isinstance(element, cl.Image):
            images.append(Image.open(element.path).convert("RGB"))
        elif isinstance(element, Image):
            images.append(element.convert("RGB"))

    if images:
        content = [{"type": "image", "image": img} for img in images]
        content.append(
            {
                "type": "text",
                "text": message.content
                or "Transcribe and inspect this prescription. List every "
                "medication with dose and frequency, then flag interactions.",
            }
        )
    else:
        content = message.content

    history = cl.user_session.get("history") or []
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": content})

    inputs = build_inputs(processor, model, messages)
    prompt_len = inputs["input_ids"].shape[-1]

    streamer = TextIteratorStreamer(
        processor.tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    # generate() blocks, so run it on a thread and consume the streamer.
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=int(cl.user_session.get("max_tokens") or 512),
        do_sample=False,
        streamer=streamer,
    )
    error: list[BaseException] = []

    def run():
        try:
            with torch.inference_mode():
                model.generate(**gen_kwargs)
        except BaseException as exc:  # surfaced in the UI, not swallowed
            error.append(exc)
            streamer.end()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    msg = cl.Message(content="", author="MedGuard")
    t0 = time.time()
    n = 0
    for chunk in streamer:
        if chunk:
            n += 1
            await msg.stream_token(chunk)
    thread.join()
    await msg.send()

    if error:
        await cl.Message(
            content=f"Generation failed: `{type(error[0]).__name__}: {error[0]}`"
        ).send()
        return

    dt = time.time() - t0
    if n and dt > 0:
        await cl.Message(
            content=f"_{n} tokens in {dt:.1f}s ({n/dt:.1f} tok/s, T4 fp16)_"
        ).send()

    text = msg.content
    history.append({"role": "user", "content": content})
    history.append({"role": "assistant", "content": text})
    # Bound context: images are PIL objects, so keep history short.
    cl.user_session.set("history", history[-6:])


@cl.action_callback("clear")
async def _on_clear():
    cl.user_session.set("history", [])
    await cl.Message(content="Cleared.").send()


async def _send_clear_button():
    """Chainlit 2.x: actions are dataclasses emitted to the UI, and the
    handler is registered via @cl.action_callback rather than a decorator
    on the button itself."""
    await cl.Action(
        name="clear",
        label="Clear conversation",
        payload={},
    ).send(for_id="clear")