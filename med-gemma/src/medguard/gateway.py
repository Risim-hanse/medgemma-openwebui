"""Optional OpenWebUI-gateway backend (OpenAI-compatible chat API).

The deterministic engine NEVER uses this — only MedGemma's two jobs
(transcription/extraction prose and findings-grounded explanation) can be
routed here when local GPU inference is unavailable.

Env:
  MEDGUARD_BACKEND       local | openwebui   (default: local)
  OPENWEBUI_BASE_URL     e.g. http://gpu-box:3000 (default: http://localhost:3000)
  OPENWEBUI_API_KEY      optional bearer token
  MEDGUARD_GATEWAY_MODEL model name as served by the gateway (default: medgemma-4b-it)

No extra dependencies: plain requests + manual SSE parsing.
"""

from __future__ import annotations

import base64
import io
import json
import os

import requests

BACKEND = os.environ.get("MEDGUARD_BACKEND", "local").lower()
BASE_URL = os.environ.get("OPENWEBUI_BASE_URL", "http://localhost:3000").rstrip("/")
API_KEY = os.environ.get("OPENWEBUI_API_KEY", "")
GATEWAY_MODEL = os.environ.get("MEDGUARD_GATEWAY_MODEL", "medgemma-4b-it")


def _live(key: str, default: str = "") -> str:
    """Read env at call time so process-lifetime changes (tests, restarts) apply."""
    return os.environ.get(key, default)


def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    if _live("OPENWEBUI_API_KEY", API_KEY):
        h["Authorization"] = f"Bearer {_live('OPENWEBUI_API_KEY', API_KEY)}"
    return h


def _base_url() -> str:
    return _live("OPENWEBUI_BASE_URL", "http://localhost:3000").rstrip("/") or BASE_URL


def _model(override: str | None) -> str:
    return override or _live("MEDGUARD_GATEWAY_MODEL", GATEWAY_MODEL)


def pil_to_data_url(image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def to_openai_messages(typed_messages: list[dict]) -> list[dict]:
    """Convert transformers-style typed parts to OpenAI chat format."""
    out = []
    for m in typed_messages:
        content = m.get("content")
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        parts = []
        for c in content:
            if not isinstance(c, dict):
                parts.append({"type": "text", "text": str(c)})
            elif c.get("type") == "image":
                parts.append({"type": "image_url",
                              "image_url": {"url": pil_to_data_url(c["image"])}})
            else:
                parts.append({"type": "text", "text": c.get("text", "")})
        out.append({"role": m["role"], "content": parts})
    return out


def ping(timeout: int = 5) -> tuple[bool, str]:
    try:
        r = requests.get(f"{_base_url()}/api/models", headers=_headers(), timeout=timeout)
        if r.status_code == 200:
            models = [m.get("name", m.get("id")) for m in r.json().get("data", [])]
            return True, f"{len(models)} models: {', '.join(map(str, models[:5]))}"
        return False, f"HTTP {r.status_code}"
    except requests.RequestException as e:
        return False, str(e)


def complete(messages: list[dict], max_tokens: int = 1200,
             model: str | None = None) -> str:
    r = requests.post(f"{_base_url()}/api/chat/completions",
                      headers=_headers(), timeout=600,
                      json={"model": _model(model), "messages": to_openai_messages(messages),
                            "max_tokens": max_tokens, "stream": False})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def complete_stream(messages: list[dict], max_tokens: int = 1200,
                    model: str | None = None):
    """Yield text deltas from the SSE stream."""
    with requests.post(f"{_base_url()}/api/chat/completions",
                       headers=_headers(), timeout=600, stream=True,
                       json={"model": _model(model), "messages": to_openai_messages(messages),
                             "max_tokens": max_tokens, "stream": True}) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                delta = json.loads(payload)["choices"][0]["delta"]
                if delta.get("content"):
                    yield delta["content"]
            except (KeyError, ValueError):
                continue
