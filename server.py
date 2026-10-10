"""OpenAI-compatible MedGemma server for the T4.

Run:  ./venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000

The 4B model occupies ~8 GB of the T4's 15 GB, leaving little room for
concurrency, so requests are serialized behind a lock.
"""

import base64
import io
import threading
import time
import uuid

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel
import json

from medgemma_common import MODEL_ID, device_note, load_model

app = FastAPI(title="MedGemma 4B (T4 fp16)")

# Default/cap for generated tokens. 256 cut answers off mid-sentence, and even
# 1024 truncates long essays, so default == cap: answers end at natural EOS.
# Worst case on the T4 (~6 tok/s) is ~6 min for a full 2048-token ramble.
DEFAULT_MAX_TOKENS = 2048
MAX_NEW_TOKENS_CAP = 2048


def _resolve_limit(req: "ChatRequest") -> int:
    limit = req.max_completion_tokens or req.max_tokens or DEFAULT_MAX_TOKENS
    return max(1, min(int(limit), MAX_NEW_TOKENS_CAP))

# Single GPU, limited VRAM -> serialize generation.
_LOCK = threading.Lock()
_STATE = {"model": None, "processor": None, "ready": False, "error": None}


@app.on_event("startup")
def startup():
    try:
        model, processor = load_model()
        _STATE["model"], _STATE["processor"] = model, processor
        _STATE["ready"] = True
    except Exception as exc:  # keep /health alive even if load fails
        _STATE["error"] = f"{type(exc).__name__}: {exc}"
        raise


class Message(BaseModel):
    role: str
    content: str | list[dict] | None = None


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float = 0.0
    stream: bool = False

    model_config = {"extra": "ignore"}


def _decode_image(url: str) -> Image.Image:
    """Load an image from an http(s) URL or a base64 data URI."""
    import requests

    if url.startswith("data:"):
        b64 = url.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(b64)))
    resp = requests.get(
        url,
        # Wikimedia (and some CDNs) reject generic agents outright.
        headers={"User-Agent": "medgemma-server/1.0 (local clinical tooling)"},
        timeout=30,
    )
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def _to_chat_messages(req: ChatRequest):
    """Convert OpenAI-style messages, inlining any images."""
    messages = []
    for m in req.messages:
        if isinstance(m.content, str):
            messages.append({"role": m.role, "content": m.content})
            continue

        parts, texts = [], []
        for chunk in m.content:
            ctype = chunk.get("type")
            if ctype == "text":
                texts.append(chunk.get("text", ""))
            elif ctype in ("image_url", "image"):
                url = (chunk.get("image_url") or {}).get("url") or chunk.get("image")
                if not url:
                    raise HTTPException(400, "image chunk missing url")
                parts.append({"type": "image", "image": _decode_image(url)})
        if texts:
            parts.insert(0, {"type": "text", "text": "\n".join(texts)})
        messages.append({"role": m.role, "content": parts})
    return messages


@app.get("/health")
def health():
    if not _STATE["ready"]:
        return JSONResponse(status_code=503, content={"status": "loading", "error": _STATE["error"]})
    free, total = torch.cuda.mem_get_info()
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": device_note(),
        "vram_free_gib": round(free / 2**30, 2),
        "vram_total_gib": round(total / 2**30, 2),
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "google",
            }
        ],
    }


@app.get("/v1/models/{model_id:path}")
def get_model(model_id: str):
    return {
        "id": MODEL_ID,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "google",
    }


def _generate(req: ChatRequest, messages):
    """Run generation under the single-GPU lock. Returns (text, prompt_len, new_tokens, limit)."""
    model, processor = _STATE["model"], _STATE["processor"]
    limit = _resolve_limit(req)

    t0 = time.time()
    with _LOCK:
        from medgemma_common import build_inputs

        inputs = build_inputs(processor, model, messages)
        prompt_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=limit,
                do_sample=req.temperature > 0,
                temperature=req.temperature if req.temperature > 0 else None,
            )
        text = processor.decode(out[0][prompt_len:], skip_special_tokens=True)
    dt = time.time() - t0

    new_tokens = out.shape[-1] - prompt_len
    print(f"[generate] {new_tokens} tok in {dt:.1f}s = {new_tokens/dt:.2f} tok/s")
    return text, prompt_len, new_tokens, limit


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    if not _STATE["ready"]:
        raise HTTPException(503, f"model not ready: {_STATE['error']}")

    messages = _to_chat_messages(req)
    model_name = req.model or MODEL_ID

    if req.stream:
        from medgemma_common import build_inputs
        from transformers import TextIteratorStreamer

        model, processor = _STATE["model"], _STATE["processor"]
        limit = _resolve_limit(req)
        try:
            inputs = build_inputs(processor, model, messages)
        except Exception as exc:
            raise HTTPException(400, f"input build failed: {exc}")
        prompt_len = inputs["input_ids"].shape[-1]
        # AutoProcessor wraps a tokenizer; streamer needs the tokenizer.
        tok = getattr(processor, "tokenizer", processor)
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=limit,
            do_sample=req.temperature > 0,
            temperature=req.temperature if req.temperature > 0 else None,
        )
        holder: dict = {}
        t0 = time.time()

        def _run():
            with _LOCK:
                with torch.inference_mode():
                    try:
                        holder["out"] = model.generate(**gen_kwargs)
                    except Exception as exc:  # surfaced after stream ends
                        holder["error"] = exc

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        def sse():
            full = []
            try:
                first = True
                for piece in streamer:
                    full.append(piece)
                    delta = {"content": piece}
                    if first:
                        delta["role"] = "assistant"
                        first = False
                    payload = {
                        "id": cmpl_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {"index": 0, "delta": delta, "finish_reason": None}
                        ],
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
            finally:
                thread.join(timeout=300)
            text_full = "".join(full)
            # Exact token counts when generate returned; else estimate.
            try:
                out = holder.get("out")
                if out is not None:
                    new_tokens = out.shape[-1] - prompt_len
                else:
                    new_tokens = len(tok.encode(text_full)) if text_full else 0
            except Exception:
                new_tokens = 0
            dt = time.time() - t0
            print(
                f"[stream] ~{new_tokens} tok in {dt:.1f}s = {new_tokens/dt:.2f} tok/s"
                if dt > 0 and new_tokens
                else "[stream] done"
            )
            if "error" in holder:
                print(f"[stream] generate error: {holder['error']}")
            done = {
                "id": cmpl_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "length" if new_tokens >= limit else "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_len,
                    "completion_tokens": new_tokens,
                    "total_tokens": prompt_len + new_tokens,
                },
            }
            yield f"data: {json.dumps(done)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    text, prompt_len, new_tokens, limit = _generate(req, messages)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "length" if new_tokens >= limit else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_len,
            "completion_tokens": new_tokens,
            "total_tokens": prompt_len + new_tokens,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)