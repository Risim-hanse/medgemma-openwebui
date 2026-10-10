"""MedGuard on Chainlit — prescription inspector with a deterministic safety core.

Pipeline per message:  transcribe/extract (MedGemma, local GPU or OpenWebUI
gateway) → brand→salt resolve (SQLite) → rules engine (DDInter, duplicates,
CDSCO banned FDC, Schedule H1) → findings-grounded explanation (streamed).

Run:  uv run chainlit run chainlit_app.py --host 0.0.0.0 --port 8002

Env:
  MEDGUARD_BACKEND=local|openwebui   (default local)
  MEDGEMMA_MODEL_ID=google/medgemma-4b-it (default; cached)
  OPENWEBUI_BASE_URL / OPENWEBUI_API_KEY / MEDGUARD_GATEWAY_MODEL (gateway mode)
"""

from __future__ import annotations

import os
import tempfile
import time

import chainlit as cl
from PIL import Image

from medguard.config import DISCLAIMER, MODEL_ID
from medguard.db import MedGuardDB
from medguard.models import Medication
from medguard.report import render_markdown
from medguard.rules.engine import run_review

STATE = {"db": None, "model": False, "backend": "local"}

INTRO = """## 💊 MedGuard — Prescription & Interaction Inspector

**How this works:** I transcribe your prescription, resolve Indian brand names
to active ingredients, and check them against reference data
(DDInter, CDSCO banned-FDC list, Schedule H1) with a deterministic engine.
The AI explains the findings in plain language — it never invents safety claims.

⚠️ {disclaimer}

**Send a prescription photo, paste medicine lines, or tap a starter below.**
Manual lines work even without the model — one medicine per line, e.g.
`Tab. Dolo 650 TDS`."""


def parse_manual_lines(text: str) -> list[Medication]:
    """No-model fallback: one medicine per line; the resolver strips doses."""
    meds = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*•0123456789. ")
        if len(line) >= 3:
            meds.append(Medication(name=line))
    return meds


def backend_status() -> str:
    from medguard import gateway
    be = os.environ.get("MEDGUARD_BACKEND", "local").lower()
    if be == "openwebui":
        ok, info = gateway.ping()
        return f"gateway ({gateway.BASE_URL}): {'✅ ' + info if ok else '❌ ' + info}"
    return "local GPU"


@cl.on_chat_start
async def start():
    STATE["backend"] = os.environ.get("MEDGUARD_BACKEND", "local").lower()
    await cl.Message(content=INTRO.format(disclaimer=DISCLAIMER)).send()
    await cl.Message(content=(f"Reference DB: connecting… | Model backend: "
                              f"{backend_status()}")).send()
    try:
        STATE["db"] = MedGuardDB()
        n = STATE["db"].con.execute("SELECT COUNT(*) FROM brands").fetchone()[0]
        await cl.Message(content=f"✅ Reference DB ready ({n:,} brands).").send()
    except Exception as e:
        await cl.Message(content=f"❌ Reference DB failed: {e}\n"
                                 "Run `uv run python data/build_db.py`.").send()
        return

    if STATE["backend"] == "local":
        await cl.Message(content=f"Loading {MODEL_ID} onto GPU (~1 min)…").send()
        try:
            from medguard.extract import load_model, device_info
            await cl.make_async(load_model)()
            STATE["model"] = True
            await cl.Message(content=f"✅ Model ready ({device_info()}).").send()
        except Exception as e:
            STATE["model"] = False
            await cl.Message(content=("⚠️ Model could not load "
                                      f"(`{type(e).__name__}: {str(e)[:200]}`).\n"
                                      "Extraction + explanation are off, but you can still "
                                      "type medicine lines and get the full rules-engine "
                                      "review.")).send()
    await cl.Message(content="Send a prescription photo or medicine list.").send()
    await cl.ChatSettings(
        [cl.input_widget.Select(id="language", label="Explanation language",
                                values=["en", "hi"],
                                initial_index=0,
                                description="हिंदी for Hindi explanation"),
         cl.input_widget.Slider(id="max_tokens", label="Max explanation tokens",
                                initial=900, min=200, max=2000, step=100)]
    ).send()


@cl.set_starters
async def starters():
    return [
        cl.Starter(label="📷 I'll upload a prescription photo",
                   message="Please upload a clear photo of the prescription."),
        cl.Starter(label="💊 Demo: Ecosprin 75 + Warf 5",
                   message="Ecosprin 75 OD\nWarf 5 OD"),
        cl.Starter(label="📋 Demo: Telma-AM + Dolo 650",
                   message="Telma-AM OD\nDolo 650 TDS"),
    ]


@cl.on_settings_update
async def on_settings_update(settings):
    cl.user_session.set("settings", settings)


async def _extract(images: list, text: str):
    """Returns (raw_text, meds, method). Falls back to manual parsing."""
    settings = cl.user_session.get("settings") or {}
    _ = settings
    if images:
        if STATE["model"] or STATE["backend"] == "openwebui":
            with cl.Step(name="📷 Transcribe & extract (MedGemma)") as step:
                from medguard.extract import extract_from_image
                raw, meds = await cl.make_async(extract_from_image)(images[0])
                step.output = raw[:2000]
                return raw, meds, "vision"
        return "", parse_manual_lines(text), "manual(no-model)"
    if STATE["model"] or STATE["backend"] == "openwebui":
        with cl.Step(name="📝 Parse medicine list (MedGemma)") as step:
            from medguard.extract import extract_from_text
            try:
                raw, meds = await cl.make_async(extract_from_text)(text)
                step.output = raw[:2000]
                return raw, meds, "text-llm"
            except Exception:
                pass  # fall through to manual parse
    return text, parse_manual_lines(text), "manual"


@cl.on_message
async def on_message(message: cl.Message):
    db: MedGuardDB | None = STATE["db"]
    if db is None:
        await cl.Message(content="Reference DB is not ready — see above.").send()
        return

    images = []
    for el in message.elements:
        if isinstance(el, cl.Image):
            images.append(Image.open(el.path).convert("RGB"))
        elif isinstance(el, Image):
            images.append(el.convert("RGB"))

    if not images and not message.content.strip():
        await cl.Message(content="Please attach a photo or type the medicines.").send()
        return

    settings = cl.user_session.get("settings") or {}
    language = settings.get("language", "en")
    max_tokens = int(settings.get("max_tokens", 900))

    # ---- 1. extract ----
    raw_text, meds, method = await _extract(images, message.content or "")
    if not meds:
        await cl.Message(content="I couldn't find any medicines in that. "
                                 "Try one per line, e.g. `Telma-AM OD`.").send()
        return
    med_lines = "\n".join(
        f"- **{m.name}**" + "".join(f" {x}" for x in
                                    [m.dosage, m.frequency, m.duration] if x)
        for m in meds)
    await cl.Message(content=f"**Medicines found** (`{method}`):\n{med_lines}\n"
                             + (f"\n<details><summary>Raw transcription</summary>\n\n"
                                f"{raw_text[:3000]}\n</details>" if raw_text else "")).send()

    # ---- 2. deterministic review ----
    with cl.Step(name="🔬 Safety review (rules engine)") as step:
        review = await cl.make_async(run_review)(meds, db)
        step.output = (f"{len(review.interactions)} interactions, "
                       f"{len(review.flags)} flags, worst={review.max_severity()}")

    worst = review.max_severity()
    icon = {"HIGH_CONCERN": "🔴", "MODERATE_CONCERN": "🟠",
            "LOW_CONCERN": "🟡", "UNVERIFIED": "⚪"}.get(worst, "⚪")
    parts = [f"## {icon} Overall: {worst}",
             f"_Coverage: {review.pairs_with_record}/{review.pairs_checked} pairs "
             "have a DDInter record. No warning ≠ proof of safety._", ""]
    for rm in review.meds:
        badge = {"resolved": "✅", "fuzzy": "🔍",
                 "generic": "🧪", "unresolved": "❓"}.get(rm.status, "")
        parts.append(f"{badge} **{rm.brand_display or rm.med.name}** `{rm.status}`")
        if rm.ingredients and rm.status in ("resolved", "fuzzy"):
            parts.append("  Active: " + ", ".join(
                f"{i.name}{f' {i.strength}' if i.strength else ''}" for i in rm.ingredients))
        parts.extend("  ⚠️ " + n for n in rm.notes)
    if review.interactions:
        parts.append("\n### ⚡ Interactions")
        for ix in review.interactions:
            parts.append(f"- **{ix.level}**: {ix.drug_a} + {ix.drug_b}")
            parts.append(f"  _{ix.evidence}_")
    else:
        parts.append("\n_No interaction identified from available information. "
                     "This does not prove safety._")
    for f in review.flags:
        parts.append(f"\n🚩 **[{f.severity}] {f.title}**\n{f.detail}")
    if review.missing_info:
        parts.append("\n### ❔ Missing info")
        parts.extend("- " + m for m in review.missing_info)
    parts.append(f"\n> ⚠️ {DISCLAIMER}")
    await cl.Message(content="\n".join(parts)).send()

    md = render_markdown(review)
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False,
                                     prefix="medguard_review_") as fh:
        fh.write(md)
        report_path = fh.name
    await cl.Message(content="📄 Full deterministic report:",
                     elements=[cl.File(name="medguard_review.md",
                                       path=report_path)]).send()

    # ---- 3. grounded explanation (streamed) ----
    if STATE["model"] or STATE["backend"] == "openwebui":
        from medguard.explain import explain_stream
        msg = cl.Message(content="")
        with cl.Step(name="💬 Plain-language explanation (grounded)") as step:
            try:
                for chunk in explain_stream(review, language,
                                            max_new_tokens=max_tokens):
                    if chunk:
                        await msg.stream_token(chunk)
                await msg.send()
                step.output = f"{len(msg.content)} chars"
            except Exception as e:
                step.output = f"failed: {e}"
                await cl.Message(content=f"⚠️ Explanation failed: `{e}`\n"
                                         "The deterministic report above stands.").send()
    else:
        await cl.Message(content="_Plain-language explanation unavailable "
                                 "(no model). The report above is complete._").send()
