"""Download all reference datasets into data/raw/ (idempotent, skips existing).

Sources (see data/SOURCES.md for licenses & provenance):
  1. DDInter v1 DDI pairs, split by ATC chapter  (CC BY-NC-SA 4.0)
  2. blacklisted.co.in banned-drugs.csv          (CC BY 4.0)
  3. revooda/indian-pharma-data 1mg brand dump  (no license stated)

Usage:
    uv run python data/fetch_data.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"

DDI_LETTERS = list("ABCDGHJLMNPR SUV".replace(" ", ""))
DDI_URL = "https://ddinter.scbdd.com/static/media/download/ddinter_downloads_code_{letter}.csv"
BANNED_URL = "https://blacklisted.co.in/data/banned-drugs.csv"
BRAND_URL = (
    "https://huggingface.co/datasets/revooda/indian-pharma-data"
    "/resolve/main/indian_pharmaceutical_products_clean.csv"
)


def _download(url: str, dest: Path) -> dict:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip (exists): {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"  GET {url}")
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        print(f"  saved {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    h = hashlib.sha256()
    with open(dest, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {"url": url, "file": str(dest.relative_to(ROOT)),
            "bytes": dest.stat().st_size, "sha256": h.hexdigest()}


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"fetched_at": datetime.now(timezone.utc).isoformat(),
                      "files": {}}
    print("== DDInter DDI pairs ==")
    for letter in DDI_LETTERS:
        dest = RAW / f"ddinter_code_{letter}.csv"
        try:
            manifest["files"][dest.name] = _download(DDI_URL.format(letter=letter), dest)
        except requests.HTTPError as e:
            print(f"  !! {letter}: {e} (skipping)")
    print("== CDSCO banned FDC (blacklisted.co.in, CC BY 4.0) ==")
    manifest["files"]["banned-drugs.csv"] = _download(BANNED_URL, RAW / "banned-drugs.csv")
    print("== Indian brand -> salt dump (revooda/indian-pharma-data) ==")
    manifest["files"]["indian_brands.csv"] = _download(BRAND_URL, RAW / "indian_brands.csv")
    with open(ROOT / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {ROOT / 'manifest.json'}")


if __name__ == "__main__":
    sys.exit(main())
