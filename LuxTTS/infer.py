#!/usr/bin/env python3
"""LuxTTS inference CLI.

Clones a reference voice found in ./voices (or any audio file) and speaks text.

Examples:
    python infer.py "Hello there, how's it going?"
    python infer.py --list-voices
    python infer.py --voice Nobara -o out.wav "First line"
    python infer.py --voice 1 --text-file script.txt -o outputs/
    python infer.py --chunk-chars 200 --steps 6 "A very long paragraph..."
    echo "Piped text works too" | python infer.py -o out.wav
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import soundfile as sf

AUDIO_EXTS = {".aac", ".aiff", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma"}
SAMPLE_RATE = 48000
CHUNK_GAP = 0.25  # seconds of silence between generated chunks
ROOT = Path(__file__).resolve().parent
DEFAULT_VOICES_DIR = ROOT / "voices"
DEFAULT_MODEL = "YatharthS/LuxTTS"


# ------------------------------------------------------------------------ backends
class TorchTTS:
    """PyTorch backend.

    The bundled ONNX/CPU path (`--backend onnx`) scales the whole sequence length
    by `speed`, so the prompt strip in `generate_cpu` can end up negative and short
    texts crash the vocoder. The PyTorch model keeps the prompt length intact, so it
    is the default on every device.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL, device: str = "cpu", threads: int = 4):
        import torch
        from huggingface_hub import snapshot_download
        from linacodec.vocoder.vocos import Vocos
        from torch.nn.utils import parametrize
        from transformers import pipeline

        from zipvoice.models.zipvoice_distill import ZipVoiceDistill
        from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
        from zipvoice.utils.checkpoint import load_checkpoint
        from zipvoice.utils.feature import VocosFbank

        model_dir = _resolve_model_dir(model_path, snapshot_download)
        if device == "cpu" and threads > 0:
            torch.set_num_threads(threads)

        self.device = device
        self.transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device)
        self.tokenizer = EmiliaTokenizer(token_file=f"{model_dir}/tokens.txt")
        config = json.loads(Path(model_dir, "config.json").read_text(encoding="utf-8"))
        self.model = ZipVoiceDistill(
            **config["model"],
            vocab_size=self.tokenizer.vocab_size,
            pad_id=self.tokenizer.pad_id,
        )
        load_checkpoint(filename=f"{model_dir}/model.pt", model=self.model, strict=True)
        self.model = self.model.to(device).eval()

        self.feature_extractor = VocosFbank()
        self.vocos = Vocos.from_hparams(f"{model_dir}/vocoder/config.yaml").to(device).eval()
        parametrize.remove_parametrizations(self.vocos.upsampler.upsample_layers[0], "weight")
        parametrize.remove_parametrizations(self.vocos.upsampler.upsample_layers[1], "weight")
        self.vocos.load_state_dict(torch.load(f"{model_dir}/vocoder/vocos.bin", map_location=device))
        self.vocos.freq_range = 12000
        self.vocos.return_48k = True

    def encode_prompt(self, prompt_audio, duration: float = 5, rms: float = 0.001, prompt_text: str | None = None) -> dict:
        from zipvoice.modeling_utils import process_audio

        prompt_tokens, prompt_features_lens, prompt_features, prompt_rms = process_audio(
            prompt_audio,
            self.transcriber,
            self.tokenizer,
            self.feature_extractor,
            self.device,
            target_rms=rms,
            duration=duration,
            prompt_text=prompt_text,
        )
        return {
            "prompt_tokens": prompt_tokens,
            "prompt_features_lens": prompt_features_lens,
            "prompt_features": prompt_features,
            "prompt_rms": prompt_rms,
        }

    def generate_speech(
        self,
        text: str,
        encode_dict: dict,
        num_steps: int = 4,
        guidance_scale: float = 3.0,
        t_shift: float = 0.5,
        speed: float = 1.0,
        return_smooth: bool = False,
    ):
        from zipvoice.modeling_utils import generate

        self.vocos.return_48k = not return_smooth
        prompt_tokens, prompt_features_lens, prompt_features, prompt_rms = encode_dict.values()
        wav = generate(
            prompt_tokens,
            prompt_features_lens,
            prompt_features,
            prompt_rms,
            text,
            self.model,
            self.vocos,
            self.tokenizer,
            num_step=num_steps,
            guidance_scale=guidance_scale,
            t_shift=t_shift,
            speed=speed,
        )
        return wav.cpu()


def _resolve_model_dir(model_path: str, snapshot_download) -> str:
    if Path(model_path).is_dir():
        return str(model_path)
    if model_path != DEFAULT_MODEL:
        print(f"warning: {model_path!r} is not a local directory, downloading it", file=sys.stderr)
    return snapshot_download(model_path)


def build_backend(args: argparse.Namespace, device: str):
    if args.backend == "onnx":
        if device != "cpu":
            print(f"note: the ONNX backend is CPU-only, using cpu instead of {device}", file=sys.stderr)
        print(
            "warning: ONNX backend truncates audio (and can crash) because the exported "
            "text encoder divides the prompt length by speed; use --backend torch for correct output",
            file=sys.stderr,
        )
        from zipvoice.luxvoice import LuxTTS

        return LuxTTS(args.model, device="cpu", threads=args.threads)
    return TorchTTS(args.model, device=device, threads=args.threads)


# --------------------------------------------------------------------------- voices
def _dedupe(paths: list[Path]) -> list[Path]:
    seen: set = set()
    out: list[Path] = []
    for p in paths:
        try:
            key = (p.stat().st_size, p.name.lower())
        except OSError:
            key = (str(p),)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _extract_archives(voices_dir: Path) -> list[Path]:
    """No loose audio next to the archives? Pull the samples out of the zips."""
    cache = Path(tempfile.gettempdir()) / "luxtts_voices"
    found: list[Path] = []
    for archive in sorted(voices_dir.glob("*.zip")):
        try:
            with zipfile.ZipFile(archive) as zf:
                for info in zf.infolist():
                    if info.is_dir() or Path(info.filename).suffix.lower() not in AUDIO_EXTS:
                        continue
                    target = cache / f"{archive.stem}__{Path(info.filename).name}"
                    if not target.exists():
                        cache.mkdir(parents=True, exist_ok=True)
                        with zf.open(info) as src, open(target, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                    found.append(target)
        except zipfile.BadZipFile:
            print(f"warning: skipping unreadable archive {archive.name}", file=sys.stderr)
    return found


def discover_voices(voices_dir: Path) -> list[Path]:
    if not voices_dir.is_dir():
        return []
    files = [
        p
        for p in sorted(voices_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS and ".cache" not in p.parts
    ]
    return _dedupe(files or _extract_archives(voices_dir))


def _print_voices(voices: list[Path], voices_dir: Path) -> None:
    print(f"voices in {voices_dir}:")
    for i, v in enumerate(voices, 1):
        print(f"  {i:2d}. {v.name}")


def select_voice(voices: list[Path], spec: str | None, voices_dir: Path) -> Path:
    if not voices:
        sys.exit(f"error: no reference audio found in {voices_dir}")
    if spec is None:
        if len(voices) == 1:
            return voices[0]
        _print_voices(voices, voices_dir)
        sys.exit("error: several voices available, choose one with --voice N or --voice NAME")

    direct = Path(spec).expanduser()
    if direct.is_file():
        return direct

    if spec.isdigit():
        idx = int(spec)
        if 1 <= idx <= len(voices):
            return voices[idx - 1]
        sys.exit(f"error: --voice {idx} out of range (1-{len(voices)})")

    lowered = spec.lower()
    exact = [v for v in voices if v.stem.lower() == lowered]
    partial = [v for v in voices if lowered in v.name.lower()]
    matches = exact or partial
    if len(matches) == 1:
        return matches[0]
    if not matches:
        _print_voices(voices, voices_dir)
        sys.exit(f"error: no voice matching {spec!r}")
    print(f"voices matching {spec!r}:")
    for i, v in enumerate(matches, 1):
        print(f"  {i:2d}. {v.name}")
    sys.exit(f"error: {len(matches)} matches for {spec!r}, be more specific")


def make_prompt(path: Path, start: float, duration: float) -> str:
    """Optionally cut a window out of the reference so the clone uses the best part."""
    if start <= 0:
        return str(path)
    import librosa

    wav, sr = librosa.load(str(path), sr=None, offset=start, duration=duration or None)
    if wav.size == 0:
        sys.exit(f"error: --ref-start {start}s is past the end of {path.name}")
    tmp = Path(tempfile.gettempdir()) / f"luxtts_prompt_{int(time.time())}.wav"
    sf.write(tmp, wav, sr)
    return str(tmp)


# ----------------------------------------------------------------------------- text
def read_texts(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[str]:
    if args.text_file:
        raw = sys.stdin.read() if args.text_file == "-" else Path(args.text_file).read_text(encoding="utf-8")
        items = [line.strip() for line in raw.splitlines() if line.strip()]
    elif args.text:
        items = [" ".join(args.text).strip()]
    elif not sys.stdin.isatty():
        items = [sys.stdin.read().strip()]
    else:
        parser.error("no text given (pass it as an argument, via --text-file, or on stdin)")

    items = [item for item in items if item]
    if not items:
        parser.error("no text given")
    return items


def split_chunks(text: str, max_chars: int) -> list[str]:
    """Split long text at sentence boundaries so each synthesis pass stays short."""
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]

    pieces = [s for s in re.split(r"(?<=[.!?…。！？])\s+", text.strip()) if s] or [text]
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        while len(piece) > max_chars:  # a single oversized sentence
            if current:
                chunks.append(current)
                current = ""
            cut = piece.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            chunks.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        if current and len(current) + 1 + len(piece) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = f"{current} {piece}".strip()
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------- output
def slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "voice"


def output_paths(args: argparse.Namespace, count: int, voice: Path) -> list[Path]:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    slug = slugify(voice.stem)

    if args.output is None:
        base = ROOT / "outputs"
        if count == 1:
            return [base / f"{stamp}_{slug}.wav"]
        return [base / f"{stamp}_{slug}_{i:03d}.wav" for i in range(1, count + 1)]

    out = Path(args.output).expanduser()
    if out.suffix.lower() == ".wav":
        if count > 1:
            sys.exit(f"error: {count} inputs need a directory, not a single .wav file ({out})")
        return [out]
    return [out / (f"{slug}_{i:03d}.wav" if count > 1 else f"{stamp}_{slug}.wav") for i in range(1, count + 1)]


# ------------------------------------------------------------------------ generation
def pick_device(spec: str) -> str:
    if spec != "auto":
        return spec
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synthesize(lux, text: str, args: argparse.Namespace, encode_dict: dict, label: str):
    chunks = split_chunks(text, args.chunk_chars)
    parts = []
    started = time.perf_counter()
    for i, chunk in enumerate(chunks, 1):
        note = f" ({i}/{len(chunks)})" if len(chunks) > 1 else ""
        print(f"  {label}{note}: {len(chunk)} chars ...", flush=True)
        wav = lux.generate_speech(
            chunk,
            encode_dict,
            num_steps=args.steps,
            guidance_scale=args.guidance,
            t_shift=args.t_shift,
            speed=args.speed,
            return_smooth=args.smooth,
        )
        parts.append(wav.squeeze(0))

    elapsed = time.perf_counter() - started
    wav = parts[0] if len(parts) == 1 else _join(parts)
    audio_secs = wav.numel() / SAMPLE_RATE
    rtf = audio_secs / elapsed if elapsed > 0 else 0.0
    print(f"  {audio_secs:.1f}s audio in {elapsed:.1f}s ({rtf:.1f}x realtime)")
    return wav


def _join(parts):
    import torch

    gap = torch.zeros(int(SAMPLE_RATE * CHUNK_GAP))
    out = []
    for i, part in enumerate(parts):
        if i:
            out.append(gap)
        out.append(part)
    return torch.cat(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clone a reference voice from ./voices and generate speech.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("text", nargs="*", help="text to speak (or use --text-file / stdin)")
    parser.add_argument("-o", "--output", help="output .wav file, or a directory for one/batch outputs")
    parser.add_argument(
        "--text-file", metavar="FILE", help="read text from file, one utterance per line ('-' for stdin)"
    )
    parser.add_argument("--list-voices", action="store_true", help="list the reference voices and exit")

    voice = parser.add_argument_group("voice")
    voice.add_argument(
        "--voice", metavar="N|NAME|PATH", help="reference voice: index from --list-voices, name fragment, or audio path"
    )
    voice.add_argument(
        "--voices-dir", type=Path, default=DEFAULT_VOICES_DIR, help="directory scanned for reference voices"
    )
    voice.add_argument(
        "--ref-start", type=float, default=0.0, metavar="SEC", help="start offset inside the reference audio"
    )
    voice.add_argument(
        "--ref-duration", type=float, default=5.0, metavar="SEC", help="seconds of reference audio used for the prompt"
    )
    voice.add_argument(
        "--rms", type=float, default=0.01, help="target prompt loudness, higher is louder (~0.01 recommended)"
    )
    voice.add_argument(
        "--ref-text",
        metavar="FILE",
        help="ground-truth transcript of the reference audio (skips whisper)",
    )

    gen = parser.add_argument_group("generation")
    gen.add_argument("--steps", type=int, default=4, help="flow-matching steps, 3-4 is the sweet spot")
    gen.add_argument("--guidance", type=float, default=3.0, help="classifier-free guidance scale")
    gen.add_argument(
        "--t-shift",
        type=float,
        default=0.5,
        help="sampling time shift, higher = better quality but more mispronunciations",
    )
    gen.add_argument("--speed", type=float, default=1.0, help="speaking rate multiplier")
    gen.add_argument(
        "--smooth", action="store_true", help="smoother (but less crisp) output, helps with metallic artifacts"
    )
    gen.add_argument(
        "--chunk-chars",
        type=int,
        default=0,
        metavar="N",
        help="split text into <=N char pieces before synthesis (0 = whole text at once)",
    )
    gen.add_argument("--seed", type=int, help="random seed for reproducible output")

    sys_run = parser.add_argument_group("runtime")
    sys_run.add_argument(
        "--backend",
        default="torch",
        choices=["torch", "onnx"],
        help="torch = correct output everywhere; onnx = faster int8 CPU path with a known duration bug",
    )
    sys_run.add_argument("--model", default=DEFAULT_MODEL, help="HF repo id or local checkpoint directory")
    sys_run.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"], help="inference device")
    sys_run.add_argument("--threads", type=int, default=4, help="intra-op threads on CPU")

    args = parser.parse_args()

    voices = discover_voices(args.voices_dir)
    if args.list_voices:
        if not voices:
            sys.exit(f"no audio files found in {args.voices_dir}")
        _print_voices(voices, args.voices_dir)
        return

    voice_path = select_voice(voices, args.voice, args.voices_dir)
    items = read_texts(args, parser)
    targets = output_paths(args, len(items), voice_path)  # validate before the model loads

    if args.seed is not None:
        import torch

        torch.manual_seed(args.seed)

    device = pick_device(args.device)
    if args.device not in ("auto", "cpu") and device == "cpu":
        print(f"warning: {args.device} unavailable, falling back to CPU", file=sys.stderr)
    print(f"device: {device} | backend: {args.backend}")

    load_started = time.perf_counter()
    lux = build_backend(args, device)
    print(f"model loaded in {time.perf_counter() - load_started:.1f}s")

    prompt_path = make_prompt(voice_path, args.ref_start, args.ref_duration)
    print(f"voice: {voice_path.name} (prompt {args.ref_duration:g}s from {args.ref_start:g}s)")
    ref_text = None
    if args.ref_text:
        ref_text = Path(args.ref_text).read_text(encoding="utf-8").strip()
        print(f"ref transcript from {args.ref_text}: {ref_text[:80]}")
    encode_started = time.perf_counter()
    encode_dict = lux.encode_prompt(prompt_path, duration=args.ref_duration, rms=args.rms, prompt_text=ref_text)
    print(f"prompt encoded in {time.perf_counter() - encode_started:.1f}s")

    for i, (text, target) in enumerate(zip(items, targets), 1):
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{i}/{len(items)}] -> {target}")
        wav = synthesize(lux, text, args, encode_dict, label=text[:60])
        sf.write(target, wav.numpy(), SAMPLE_RATE)
        print(f"  saved {target}")


if __name__ == "__main__":
    main()
