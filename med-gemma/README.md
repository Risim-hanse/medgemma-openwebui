# MedGuard — Prescription & Drug Interaction Inspector

On-device medication safety aid for Indian healthcare. **MedGemma 4B** reads
prescriptions (photo or text); a **deterministic rules engine** checks
interactions (DDInter), duplicate salts, CDSCO-banned FDCs, and Schedule H1.
The LLM explains findings in plain language — it never decides. Uncertainty is
always stated; a pharmacist's review is always required.

## Run

```bash
cd med-gemma && uv sync
uv run python data/fetch_data.py && uv run python data/build_db.py  # ~100MB → SQLite
uv run chainlit run chainlit_app.py --host 0.0.0.0 --port 8002
uv run python -m medguard.tts "text to speak" -o out.wav            # LuxTTS voice
```

## Layout

- `chainlit_app.py` — chat UI (extract → review → grounded explanation)
- `src/medguard/` — `extract`, `resolve`, `rules/` (ddi, duplicates, banned_fdc,
  schedule, labels, engine), `explain`, `report`, `gateway` (OpenWebUI backend),
  `tts` (LuxTTS bridge, separate venv)
- `data/` — `fetch_data.py`, `build_db.py`, `SOURCES.md`, `manifest.json`
  (raw CSVs + built DB excluded from git — rebuild with the two scripts)
- `ARCHITECTURE.md` / `REFERENCE.md` — design decisions and key facts

## Notes

- T4 recipe: **bf16 + sdpa** (fp16 produces NaN logits). 27B variants don't fit 15 GB.
- MedGemma weights: Health AI Developer Foundations Terms of Use
  (open-weight, not open-source; outputs must not directly guide care).
- Data: DDInter (CC BY-NC-SA 4.0), CDSCO list via blacklisted.co.in (CC BY 4.0),
  1mg brand dump (no license stated — demo only).
