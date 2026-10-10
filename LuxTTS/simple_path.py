"""Minimal README path: LuxTTS class -> encode_prompt -> generate_speech. CPU."""
import time
import soundfile as sf
from zipvoice.luxvoice import LuxTTS

t0 = time.perf_counter()
lux = LuxTTS('YatharthS/LuxTTS', device='cpu', threads=4)
print(f"LOAD: {time.perf_counter()-t0:.1f}s | backend model type: {type(lux.model).__name__}")

t0 = time.perf_counter()
enc = lux.encode_prompt('voices/Data base-nobara [vocals].mp3', duration=5, rms=0.01)
print(f"ENCODE: {time.perf_counter()-t0:.1f}s")

t0 = time.perf_counter()
wav = lux.generate_speech("The battle begins now. Stay close to me.", enc, num_steps=4)
dt = time.perf_counter()-t0
secs = wav.numel() / 48000
print(f"GENERATE: {dt:.1f}s for {secs:.2f}s audio ({secs/dt:.2f}x realtime)")
sf.write('/teamspace/studios/this_studio/output/simple_path.wav', wav.numpy(), 48000)
print("saved output/simple_path.wav")
