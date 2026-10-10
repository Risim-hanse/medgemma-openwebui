"""ZipVoice zero-shot TTS bridge (Phase 6).

Runs the sibling ZipVoice checkout as a SUBPROCESS — never import it into the
MedGemma venv (k2/lhotse/piper_phonemize conflict with this env).

Needs:
  - a working ZipVoice env:  ../ZipVoice/.venv  (or set ZIPVOICE_PYTHON)
  - a voice prompt wav + its transcript (zero-shot cloning)
  - a ZipVoice checkpoint (auto-downloads from HF on first use)

Long texts are chunked into sentences, synthesized via one --test-list run,
then concatenated.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZIPVOICE_REPO = Path(os.environ.get("ZIPVOICE_REPO",
                                    PROJECT_ROOT.parent / "ZipVoice"))
ZIPVOICE_PYTHON = os.environ.get(
    "ZIPVOICE_PYTHON", str(ZIPVOICE_REPO / ".venv" / "bin" / "python"))


def _sentences(text: str, max_chars: int = 220) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text).strip())
    chunks, cur = [], ""
    for p in parts:
        if len(cur) + len(p) + 1 > max_chars and cur:
            chunks.append(cur)
            cur = p
        else:
            cur = (cur + " " + p).strip()
    if cur:
        chunks.append(cur)
    return [c for c in chunks if c]


def speak(text: str, prompt_wav, prompt_text: str = "",
          model_name: str = "zipvoice", out_path: str | Path | None = None) -> Path:
    """Synthesize `text`; returns path to wav. `prompt_wav` is bytes or path."""
    if not Path(ZIPVOICE_PYTHON).exists():
        raise RuntimeError(
            f"ZipVoice python not found at {ZIPVOICE_PYTHON}. "
            "Set ZIPVOICE_PYTHON or create ../ZipVoice/.venv (see ZipVoice README).")
    if not (ZIPVOICE_REPO / "zipvoice" / "bin" / "infer_zipvoice.py").exists():
        raise RuntimeError(f"ZipVoice repo not found at {ZIPVOICE_REPO}.")

    chunks = _sentences(text)
    if not chunks:
        raise ValueError("Nothing to synthesize.")
    tmp = Path(tempfile.mkdtemp(prefix="medguard_tts_"))
    if hasattr(prompt_wav, "read"):  # file-like upload object
        prompt_path = tmp / "prompt.wav"
        prompt_path.write_bytes(prompt_wav.getvalue())
    else:
        prompt_path = Path(prompt_wav)
    if not prompt_text:
        raise ValueError("ZipVoice is zero-shot: provide the transcript of the voice sample.")
    tsv = tmp / "test.tsv"
    with open(tsv, "w") as f:
        for i, ch in enumerate(chunks):
            f.write(f"seg{i}\t{prompt_text}\t{prompt_path}\t{ch}\n")
    res_dir = tmp / "out"
    cmd = [ZIPVOICE_PYTHON, "-m", "zipvoice.bin.infer_zipvoice",
           "--model-name", model_name, "--test-list", str(tsv),
           "--res-dir", str(res_dir)]
    r = subprocess.run(cmd, cwd=ZIPVOICE_REPO, capture_output=True, text=True, timeout=1200)
    if r.returncode != 0:
        raise RuntimeError(f"ZipVoice failed:\n{r.stderr[-2000:]}")
    wavs = sorted(res_dir.glob("*.wav"))
    if not wavs:
        raise RuntimeError("ZipVoice produced no audio.")
    out = Path(out_path) if out_path else tmp / "medguard_explanation.wav"
    if len(wavs) == 1:
        out.write_bytes(wavs[0].read_bytes())
        return out
    # concatenate via stdlib wave (all chunks share sample params)
    import wave
    params = None
    frames = []
    for w in wavs:
        with wave.open(str(w), "rb") as wh:
            if params is None:
                params = wh.getparams()
            frames.append(wh.readframes(wh.getnframes()))
    with wave.open(str(out), "wb") as wh:
        wh.setparams(params)
        for fr in frames:
            wh.writeframes(fr)
    return out
