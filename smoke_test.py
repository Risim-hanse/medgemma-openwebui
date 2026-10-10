"""Smoke test for MedGemma on the T4.

Run:  ./venv/bin/python smoke_test.py
"""

import sys

import requests
from PIL import Image

from medgemma_common import MODEL_ID, device_note, generate, load_model

def main():
    print(f"Loading {MODEL_ID}")
    print(f"Device: {device_note()}")
    model, processor = load_model()

    import torch

    free, total = torch.cuda.mem_get_info()
    print(f"After load -- free {free/2**30:.2f} GiB / {total/2**30:.2f} GiB")

    failures = []

    # --- Test 1: text-only medical reasoning -------------------------------
    print("\n=== Test 1: text-only ===")
    text_out = generate(
        model,
        processor,
        [
            {"role": "system", "content": "You are an expert physician."},
            {
                "role": "user",
                "content": "A 45-year-old man has sudden crushing chest pain "
                "radiating to the left arm. What is the most likely diagnosis?",
            },
        ],
        max_new_tokens=128,
    )
    print(text_out)
    if len(text_out.strip()) < 20:
        failures.append("text-only produced suspiciously short output")

    # --- Test 2: synthetic medical-ish image -------------------------------
    print("\n=== Test 2: image / vision tower (multimodal) ===")
    # We avoid hotlinking Wikimedia here: it rate-limits (HTTP 429) from CI.
    # The point of this test is to prove the vision tower + chat template +
    # fp32->GPU image token path works end-to-end, so we synthesize a simple
    # grayscale "radiograph-like" image locally.
    import numpy as np

    rng = np.random.default_rng(0)
    arr = rng.integers(30, 60, size=(448, 448), dtype=np.uint8)
    arr[180:270, 150:300] = 140  # a bright central region
    arr[:, 224:230] = 30# a dark midline
    image = Image.fromarray(arr).convert("RGB")
    print(f"Synthetic image mode={image.mode} size={image.size}")

    img_out = generate(
        model,
        processor,
        [
            {"role": "system", "content": "You are an expert radiologist."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What imaging modality and findings do you see?"},
                    {"type": "image", "image": image},
                ],
            },
        ],
        max_new_tokens=150,
    )
    print(img_out)
    if len(img_out.strip()) < 20:
        failures.append("multimodal produced suspiciously short output")

    # --- VRAM report ------------------------------------------------------
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"\n=== Peak VRAM allocated: {peak:.2f} GiB ===")
    if peak > 14.0:
        failures.append(f"peak VRAM {peak:.2f} GiB exceeds 14 GiB safety budget")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()