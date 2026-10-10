# MedGuard Reference — MedGemma & Project Key Details

Single compilation of the key facts about the model and the project.
Deep dives: `ARCHITECTURE.md` (system design) · `data/SOURCES.md` (provenance).
Compiled 2026-10-10; measurements taken live on this box unless marked (doc).

## 1. MedGemma — the model

Google's open-**weight** (not open-source) Gemma-3 variants for medical text+image,
under the Health AI Developer Foundations (HAI-DEF) program.

### Variants

| Repo | Type | Params | Notes |
|---|---|---|---|
| `google/medgemma-4b-it` ⭐ in use | multimodal, instruction-tuned | 4B | SigLIP vision; cached locally, loads offline |
| `google/medgemma-4b-pt` | multimodal, pre-trained | 4B | for fine-tune experiments, not chat |
| `google/medgemma-1.5-4b-it` | multimodal, instruction-tuned | 4B | newer (Apr 2026), better reasoning; NOT cached (README only) |
| `google/medgemma-27b-text-it` | text-only, instruction-tuned | 27B | best text scores; does NOT fit our 15 GB T4 |
| `google/medgemma-27b-it` | multimodal, instruction-tuned | 27B | same; needs Colab-Enterprise-class GPU |
| MedSigLIP (separate) | image/text encoder, 400M | — | classification/retrieval; not generation |

### Architecture (read from OUR cached `config.json`, not docs)

- Class `Gemma3ForConditionalGeneration`, `model_type: gemma3`
- Text tower: hidden 2560 · **34 layers** · 8 attn heads / 4 KV heads (GQA) ·
  head_dim 256 · vocab **262,208** · `gelu_pytorch_tanh` · sliding window 1024 ·
  rope_theta 1,000,000
- Vision tower: SigLIP, hidden 1152 · 27 layers · **896×896** input · patch 14
- Context **128K** tokens (doc) · max output 8192 (doc) · 2× safetensors shards
- Benchmarks (doc): MedQA 64.4 (4B) · 87.7–89.8 (27B text); 81% of 4B chest-X-ray
  reports judged management-equivalent (unblinded, US radiologist)

### License — HAI-DEF Terms of Use (weights), Apache 2.0 (code repos)

Obligations that bind THIS project (full text:
`https://developers.google.com/health-ai-developer-foundations/terms`):

1. **Propagate §3.2 use restrictions** as enforceable terms to anyone we
   distribute to (demo users count — our disclaimer + review headers do this).
2. **Provide recipients a copy of the Agreement.**
3. **Seek health-regulatory authorization where applicable.**
4. **Mark modified files; ship a NOTICE file** with any distribution:
   *"HAI-DEF is provided under and subject to the Health AI Developer
   Foundations Terms of Use…"*
5. **Prohibited Use Policy applies** (incorporated by reference); no use that
   makes Google a "manufacturer" of a medical device; Google may remotely
   restrict violating usage.
6. **Model-card constraint**: outputs are preliminary, must not directly inform
   diagnosis/management without validation + adaptation — our deterministic
   engine + grounding design exists largely to satisfy this.
7. Commercial use is allowed **subject to the above** (FAQ confirms).

## 2. Measured on our Tesla T4 (sm_75, 14.6 GB usable)

| Fact | Value |
|---|---|
| Working recipe | `dtype=bf16`, `attn=sdpa`, `device_map=cuda:0`, pixel_values→bf16 |
| **fp16 is broken** | NaN logits (Gemma-3 attention overflows fp16 range) — verified |
| VRAM, model resident | ~8.2–8.4 GB (room for KV cache of short chats; NOT 128K context) |
| Load time | ~30–60 s from local cache (`HF_HUB_OFFLINE=1` works) |
| Vision extraction | ~84 s for a 900×500 photo → 3-meds JSON |
| Text explanation | ~45 s / ~400 tokens (streamed, so UI stays live) |
| Gated access | `medgemma-4b-it` fully cached (8.1 GB) — no HF auth needed to run |

## 3. MedGuard — project snapshot

**Goal:** privacy-conscious prescription inspector for Indian healthcare:
photo/text in → brand→salt resolve → deterministic safety checks → plain-language
explanation with citations. **The LLM never adjudicates safety** (reads + explains only).

| Component | Path | Status |
|---|---|---|
| Chat UI | `chainlit_app.py` (:8002, live) | ✅ serving |
| Extraction (vision/text → JSON) | `src/medguard/extract.py` | ✅ live vision e2e |
| Brand→salt resolver | `src/medguard/resolve.py` | ✅ 318/318 self-resolve |
| Rules: DDI / dup / banned / H1 / labels | `src/medguard/rules/` | ✅ verified 2026-10-10 (suite since removed per scope) |
| Deterministic report | `src/medguard/report.py` | ✅ |
| Grounded explanation + stream | `src/medguard/explain.py` | ✅ live e2e |
| OpenWebUI gateway backend | `src/medguard/gateway.py` | ⚠️ code-only, no gateway to test against |
| LuxTTS voice bridge | `src/medguard/tts.py` | ✅ valid 48 kHz wav verified (clarity tuning LuxTTS-side) |
| Data pipeline | `data/fetch_data.py` → `build_db.py` → `medguard.db` (163 MB) | ✅ rebuild ≈100 s |

Self-resolution audit (2026-10-10, n=318): **318 resolved, 0 fuzzy, 0 unresolved**
(was 231/82/5 before the glued-split fix).

## 4. Data & licenses

| Source | Rows | License | Role |
|---|---|---|---|
| DDInter v1 CSVs (15 ATC files) | 454k directed pairs | **CC BY-NC-SA 4.0** | DDI severity (Major/Moderate/Minor/Unknown) |
| blacklisted.co.in `banned-drugs.csv` | 1,080 (683 prohibited) | **CC BY 4.0** (attribute CDSCO/FSSAI via blacklisted.co.in) | banned=violation, restricted=advisory, quashed+=silent |
| `revooda/indian-pharma-data` (1mg dump) | 254k brands / 366k links | **none stated** (demo OK; swap to MIT source before commercializing) | brand→salt+strength |
| openFDA `drug/label.json` | on-demand, cached | free (1k/day keyless) | label excerpts on existing findings |
| Schedule H1 seed (44, in `build_db.py`) | 44 | indicative, gazette-cited | screen + verify warning |
| DailyMed / RxNorm | — | free | evaluated, NOT wired (see `SOURCES.md`) |

**Known gap (measured):** 774/1,721 ingredient norms have zero DDInter presence —
aceclofenac, domperidone, nimesulide, ambroxol, ornidazole, etoricoxib, sulbactam…
(~102k brand links). These pairs can only be UNVERIFIED-by-absence. Verified
bridges that DO exist: `tazobactum`→`tazobactam`, `methylcobalamin`→`mecobalamin`,
`vitamin b12`→`cyanocobalamin`, proxetil/pivoxil ester stripping.

## 5. Environment & operations

- **Box:** Tesla T4 15 GB · 15 GB RAM · 187 GB free disk · Python 3.12.13 ·
  torch 2.14.1+cu130 · transformers 5.19.0 · chainlit 2.12.0
- **`.venv` is a symlink** to shared uv cache (`/system/conda/miniconda3/uv/venvs/…`) —
  works with `uv run`; don't assume it's self-contained for copying.
- **Ports:** :8002 MedGuard (live) · :8001 sibling conversational app (stopped, not ours).
  Only ONE 4B model fits in VRAM — coordinate GPU use.
- **Env vars:** `MEDGEMMA_MODEL_ID` (default `google/medgemma-4b-it`) ·
  `MEDGUARD_BACKEND=local|openwebui` · `OPENWEBUI_BASE_URL/_API_KEY/_MODEL` ·
  `LUXTTS_REPO/_PYTHON` · `HF_HUB_OFFLINE=1` (works — weights cached).
- **Commands:** `uv sync` · `uv run python data/fetch_data.py && uv run python data/build_db.py` ·
  `uv run chainlit run chainlit_app.py --host 0.0.0.0 --port 8002` ·
  `uv run python -m medguard.tts "text" -o out.wav`.

## 6. Key decisions & limits

1. bf16/sdpa (fp16 NaNs; flash-attn needs SM 8.0+). 2. 4B not 27B (VRAM).
3. DDInter CSVs (no API exists; MyChem/NLM alternatives verified dead).
4. Chainlit over Streamlit (token streaming for slow T4 gen).
5. Subset + compact matching for bans; quashed/restricted never violations; twin-row dedupe.
6. Resolver never guesses (`unresolved` + confirm prompts; strength-mismatch warnings).
7. `extra="forbid"` schemas after a real silent-drop incident.
8. **Limits:** 45%-of-salts DDI gap (above); H1 list indicative; labels US-centric;
   ~1–2 min per full review on T4; gateway/TTS unproven; 1mg data license unclear.
