# MedGuard Architecture

One page that explains the whole system: what runs where, what talks to what,
what breaks when, and why each load-bearing decision was made.
Companion docs: `README.md` (setup/run), `data/SOURCES.md` (data provenance).

## 1. The one idea

**The LLM never adjudicates safety.** MedGemma does two jobs — reading
(transcription) and explaining (prose). Every safety claim (interactions,
duplicates, bans, schedules) comes from a deterministic rules engine over
checked-in reference data, and every claim carries its source. Uncertainty is
a first-class output (`UNVERIFIED`, `unresolved`, coverage counters), never
smoothed over.

```
                    ┌──────────────────────────────────────────────┐
                    │              CHAINLIT UI (:8002)              │
                    │  photo/text in → streamed answer + report.md  │
                    └──────────────┬───────────────────────────────┘
                                   │ ① extract        ③ explain
                    ┌──────────────▼───────────────────────────────┐
                    │  MedGemma 4B multimodal (ONE backend/tagged)  │
                    │  local T4 bf16/sdpa  ──or──  OpenWebUI gw     │
                    └──────────────┬───────────────────────────────┘
                                   │ ② Medication[] (JSON, confidence-tagged)
                    ┌──────────────▼───────────────────────────────┐
                    │  RESOLVE  brand→salt   (SQLite, offline)      │
                    │  exact → generic → fuzzy(≥85) → unresolved   │
                    └──────────────┬───────────────────────────────┘
                                   │ ResolvedMed[]
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
       DDI pairs            duplicates           CDSCO banned/      Schedule H1
       (DDInter)            (same salt,          restricted FDC     (indicative)
                             2 brands)          (subset match)
              └────────────────────┼────────────────────┘
                                   │ Review {interactions, flags,
                                   ▼         missing_info, coverage}
                    ┌──────────────────────────────┐
                    │ REPORT (deterministic .md)   │◄── download, no LLM
                    └──────────────────────────────┘
```

## 2. Box topology (shared machine)

```
Tesla T4 15 GB (14.6 usable) · 15 GB RAM · ~187 GB disk
│
├─ med-gemma/.venv ──► Chainlit :8002 (THIS APP, running)
│     └─ loads ONE 4B model ≈ 8.2 GB VRAM when a chat starts
│
├─ studio venv + ui/chainlit_app.py ──► Chainlit :8001 (sibling project,
│     currently stopped; conversational-only, no rules engine — NOT ours,
│     do not edit. Shares GPU + HF cache with us.)
│
├─ HF hub cache (SHARED, 9.7 GB): medgemma-4b-it ✅ complete (8.1 GB),
│     medgemma-1.5-4b-it ❌ README only. HF_HUB_OFFLINE=1 works for 4b-it.
│
├─ LuxTTS/.venv (exists ✅) — bundled TTS; `tts.py` bridges via subprocess
│     (verified: 48 kHz wav out; clarity tuning is LuxTTS-side)
│
└─ No OpenWebUI / Ollama on this box. Gateway mode targets an EXTERNAL host.
```

**Hard constraint: exactly one 4B model fits in VRAM.** Our loader and the
sibling's loader OOM each other if run simultaneously — coordinate before demos.

## 3. Request lifecycle (one chat message)

| Step | Code | Input → Output | Model? |
|---|---|---|---|
| 1. Collect | `chainlit_app.on_message` | photo (`cl.Image`) and/or text | no |
| 2. Extract | `extract.extract_from_image/text` → `_parse_extraction` | image/text → `(raw_text, Medication[])` | YES (or manual line-parse fallback) |
| 3. Show & confirm | `on_message` | med list + raw transcription | no |
| 4. Resolve | `resolve.resolve_all` | `Medication[]` → `ResolvedMed[]` | no |
| 5. Rules | `rules/engine.run_review` | `ResolvedMed[]` → `Review` | no |
| 6. Report | `report.render_markdown` | `Review` → Markdown + `cl.File` | no |
| 7. Explain | `explain.explain_stream` | `Review` JSON → streamed prose | YES |

Steps 2 and 7 are the ONLY model calls. Step 7 receives the Review as JSON
plus a grounding preamble that forbids adding any claim; the app appends the
disclaimer if the model omits it.

Measured on T4 (2026-10-10, vision e2e): load ~30 s, extraction ~84 s
(900×500 photo → 3 meds JSON), explanation ~45 s / ~400 tokens. Streaming
(`TextIteratorStreamer` + worker thread) is what makes this tolerable.

## 4. Backend switch (one seam, not two systems)

`MEDGUARD_BACKEND=local|openwebui` (default `local`), read dynamically in
`extract._use_gateway()`. Both backends speak the same internal message format
(transformers-style typed parts); `gateway.to_openai_messages()` converts to
OpenAI schema (PIL → base64 JPEG data URL) at the boundary.

| | local | openwebui |
|---|---|---|
| Endpoint | in-process transformers | `{OPENWEBUI_BASE_URL}/api/chat/completions` |
| Auth | HF token / offline cache | `OPENWEBUI_API_KEY` bearer (optional) |
| Model | `MEDGEMMA_MODEL_ID` (default `google/medgemma-4b-it`) | `MEDGUARD_GATEWAY_MODEL` |
| Streaming | `TextIteratorStreamer` + thread | SSE `data:` parsing |
| Proven against | ✅ live vision e2e | ❌ code only — no gateway exists yet |

`gateway.ping()` (`GET /api/models`) reports status in the UI sidebar flow.

## 5. Module contracts (`src/medguard/`)

| Module | Owns | Must never |
|---|---|---|
| `config.py` | paths, `MODEL_ID`, severity maps, disclaimer | import torch/transformers |
| `models.py` | `Medication/ResolvedMed/Interaction/SafetyFlag/Review` (all `extra="forbid"`) | contain logic |
| `normalize.py` | ONE normalization truth: `normalize/base_molecule/lookup_keys/compact` + salt list + aliases | hit the network |
| `db.py` | ALL SQL (`MedGuardDB` helpers) | contain matching logic |
| `resolve.py` | brand→salt strategy incl. OCR hardening (glued doses, misspellings) | guess (`unresolved` is a valid answer) |
| `rules/ddi.py` | pairwise DDInter lookup w/ alias+base expansion; severity max-win | emit findings for pairs with no record |
| `rules/duplicates.py` | same base molecule via ≥2 meds | — |
| `rules/banned_fdc.py` | subset match on compacted sets; **prohibited=violation, restricted=advisory, quashed+=silent** | fire on non-prohibited rows |
| `rules/schedule.py` | H1 screen, always `needs_verification` | present as legal advice |
| `rules/labels.py` | optional openFDA excerpts attached to EXISTING findings | discover interactions from prose |
| `rules/engine.py` | orchestration → `Review` + `missing_info` | call the LLM |
| `extract.py` | model load (bf16/sdpa recipe) + JSON extraction | adjudicate |
| `explain.py` | grounded explanation + streaming | add claims (enforced by prompt) |
| `report.py` | deterministic Markdown | call the LLM |
| `prompts.py` | verbatim MedGuard safety prompt + extraction/grounding prompts | — |
| `gateway.py` | OpenWebUI transport only | touch the engine |
| `tts.py` | LuxTTS subprocess bridge (own venv) | import LuxTTS here |

## 6. Data layer (`data/`)

`fetch_data.py` (download + `manifest.json` with sha256) → `build_db.py`
(normalize + dedupe + banned-part cleaning + H1 seed) → `medguard.db` (170 MB).

| Table | Rows | Key design |
|---|---|---|
| `brands` | 253,973 | `brand_norm` exact + stem-prefix match |
| `brand_ingredients` | 366,202 | `ingredient_norm` index (generic path) |
| `ddi` | 454,280 directed | `UNIQUE(norm_a,norm_b,level)`; `norm` + `base` indexes |
| `banned_fdc` | 1,080 (683 prohibited) | `ingredients_json` compacted sets; gazette URL per row |
| `schedule_h1` | 44 | `confidence='indicative'` on every row |
| `meta` | 7 | licenses, build time, manifest |

Rebuild ≈ 100 s. DB is gitignored; `manifest.json` is tracked (provenance pin).

## 7. Safety invariant & enforcement

1. **Separation**: `rules/` imports nothing from `extract/explain/prompts`.
   `explain` receives only `Review` JSON. (Structurally true — verify by import graph.)
2. **Grounding preamble** + verbatim MedGuard system prompt (`prompts.py`).
3. **Coverage counters** (`pairs_with_record/pairs_checked`) — absence of a
   warning is displayed as a coverage gap, never as safety.
4. **`extra="forbid"`** on all schemas — silent field drops fail loudly
   (regression-tested after a real incident).
5. **Contracts over test suite** — the dedicated test suite was removed
   2026-10-10 per scope (last green that day: 21 unit + 8 golden, incl.
   quashed-must-not-fire and unresolved-must-surface). Enforcement now rests
   on module contracts (§5 table), `extra="forbid"` schemas, coverage counters,
   and the README sanity-check command.

## 8. Failure ladder (each layer degrades, nothing crashes the chat)

- GPU busy (sibling model loaded) → model load fails → extraction/explanation
  off, **manual-line review still fully works** (resolver strips doses itself).
- Gateway unreachable → `ping()` reports ❌ at chat start; same fallback.
- OCR garbage → `unresolved` + confirm-spelling notes, never a guessed salt.
- No DDInter record → no finding + coverage line (not a clean bill of health).
- openFDA down → label excerpts silently absent (opt-in anyway).
- DB missing → chat start tells the user the exact rebuild command.

## 9. Glossary

- Concern: `HIGH_CONCERN` (DDInter Major) · `MODERATE` · `LOW` (Minor) ·
  `UNVERIFIED` (Unknown-level record, or non-authoritative basis).
- Resolution: `resolved` (exact) · `generic` (stated salt) · `fuzzy` (≥85,
  user must confirm) · `unresolved` (ask, don't guess).
- Banned kinds: `banned_fdc` (prohibited=violation) · `restricted_fdc`
  (advisory) · quashed/halted/stayed/under-examination/revoked (**never fire**).

## 10. Decision log (measured, not assumed)

1. **bf16, not fp16** — fp16 gives NaN logits on Gemma 3/T4 (exponent overflow);
   bf16 emulated but correct. `attn=sdpa` (flash-attn needs SM 8.0+).
2. **4B, not 27B** — 27B doesn't fit 15 GB VRAM at any usable precision.
3. **DDInter CSVs, not API** — the API doesn't exist (404s verified); MyChem has
   no interaction fields; NLM's interaction API died Jan 2024.
4. **Chainlit, not Streamlit** — token streaming over websocket vs full reruns;
   T4 generation is seconds-per-answer slow, streaming is a UX requirement.
5. **Local-first, gateway-optional** — no OpenWebUI/Ollama fits this box's RAM
   alongside the model; gateway is an external-host option behind one env var.
6. **Subset matching for bans** — gazette names carry formulation tails
   (`Dispersible Tablets`) and salt variants; exact matching misses real bans.
7. **Twin-row dedupe** — identical cleaned sets from multiple gazette waves
   collapse to one finding (distinct sets stay distinct).
8. **Glued-split as extra variant, never destructive** — OCR glues doses
   (`Warf5`); splitting inside `strip_sigils` broke correct `450MG` spellings
   (self-resolution audit: 82 fuzzy + 5 unresolved → **318/318 resolved**).
   Split forms are additional exact candidates; fuzzy/ prefix hits with
   differing dose tokens carry a strength warning instead of silently
   substituting (e.g. Auxidol 100/6mg vs stocked 100/4mg, 100/8mg).
9. **One vocab home** — `FORM/FREQ/FORM_TAIL` sets live in `normalize.py`;
   `resolve.py` and `data/build_db.py` import them. Aliases are only added
   with a verified DDInter target; 774 absent norms (aceclofenac, domperidone,
   …) stay honest UNVERIFIED-by-absence gaps (see `SOURCES.md`).
