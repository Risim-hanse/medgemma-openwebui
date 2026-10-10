"""LuxTTS voice bridge — read MedGuard explanations aloud.

LuxTTS is the bundled sibling TTS (lightweight ZipVoice-based cloner, 48 kHz,
~1 GB VRAM, voices in ../LuxTTS/voices). It pins transformers<=4.57.6, which
conflicts with this venv's transformers 5.x — so it ALWAYS runs as a
subprocess under its own venv. Never import it here.

Env:
  LUXTTS_REPO    default: <med-gemma>/../LuxTTS
  LUXTTS_PYTHON  default: <repo>/.venv/bin/python
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path

from .config import PROJECT_ROOT

LUXTTS_REPO = Path(os.environ.get("LUXTTS_REPO", PROJECT_ROOT.parent / "LuxTTS"))
LUXTTS_PYTHON = os.environ.get("LUXTTS_PYTHON", str(LUXTTS_REPO / ".venv" / "bin" / "python"))
SAMPLE_RATE = 48000


def _check() -> Path:
    if not Path(LUXTTS_PYTHON).exists():
        raise RuntimeError(
            f"LuxTTS python not found at {LUXTTS_PYTHON}. "
            "Set LUXTTS_PYTHON or create the LuxTTS venv (see LuxTTS README).")
    cli = LUXTTS_REPO / "infer.py"
    if not cli.exists():
        raise RuntimeError(f"LuxTTS repo not found at {LUXTTS_REPO} (set LUXTTS_REPO).")
    return cli


def list_voices() -> list[str]:
    """Reference voices available for cloning."""
    cli = _check()
    r = subprocess.run([LUXTTS_PYTHON, str(cli), "--list-voices"],
                       cwd=LUXTTS_REPO, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"LuxTTS --list-voices failed:\n{r.stderr[-1500:]}")
    voices = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if line[:2].strip().rstrip(".").isdigit():
            voices.append(line.split(".", 1)[1].strip())
    return voices


def speak(text: str, voice: str | None = None, out_path: str | Path | None = None,
          device: str = "auto", chunk_chars: int = 300, ref_text: str | None = None,
          steps: int = 4, speed: float = 1.0, guidance: float = 3.0,
          timeout: int = 1200) -> Path:
    """Synthesize `text` with a cloned voice; returns the .wav path.

    `voice` is a name fragment, index, or audio path (None = single default).
    Long texts are chunked by LuxTTS itself (--chunk-chars).
    """
    cli = _check()
    text = " ".join(text.split())
    if not text:
        raise ValueError("Nothing to synthesize.")
    if ref_text is not None:
        raise ValueError("ref_text passthrough is not supported by LuxTTS infer.py; "
                         "omit it (whisper transcribes the prompt automatically).")
    out = Path(out_path) if out_path else (
        Path(tempfile.mkdtemp(prefix="medguard_tts_")) / f"medguard_{int(time.time())}.wav")
    cmd = [LUXTTS_PYTHON, str(cli), "--device", device, "--steps", str(steps),
           "--speed", str(speed), "--guidance", str(guidance),
           "--chunk-chars", str(chunk_chars), "-o", str(out)]
    if voice:
        cmd += ["--voice", voice]
    cmd.append(text)
    r = subprocess.run(cmd, cwd=LUXTTS_REPO, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"LuxTTS failed:\n{(r.stderr or r.stdout)[-2000:]}")
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="MedGuard LuxTTS bridge")
    p.add_argument("text", nargs="?", default=None, help="text to speak")
    p.add_argument("--voice", default=None)
    p.add_argument("-o", "--out", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--list-voices", action="store_true")
    a = p.parse_args()
    if a.list_voices:
        print("\n".join(list_voices()))
    elif not a.text:
        p.error("no text given")
    else:
        print(speak(a.text, voice=a.voice, out_path=a.out, device=a.device))
