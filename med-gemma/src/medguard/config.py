"""Project-wide constants: paths, model id, severity maps, disclaimer."""

from __future__ import annotations

import os
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent  # src/
PROJECT_ROOT = SRC_ROOT.parent  # med-gemma/
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "medguard.db"
CACHE_DIR = DATA_DIR / "cache"

# Default: fully cached locally. Override for 1.5 when downloaded + access granted.
MODEL_ID = os.environ.get("MEDGEMMA_MODEL_ID", "google/medgemma-4b-it")

# DDInter level -> MedGuard concern level
DDI_SEVERITY = {
    "Major": "HIGH_CONCERN",
    "Moderate": "MODERATE_CONCERN",
    "Minor": "LOW_CONCERN",
    "Unknown": "UNVERIFIED",
}
SEVERITY_RANK = {"HIGH_CONCERN": 3, "MODERATE_CONCERN": 2, "LOW_CONCERN": 1, "UNVERIFIED": 0}

DISCLAIMER = (
    "This AI-generated review is an informational safety aid, not a medical "
    "diagnosis or a substitute for a pharmacist's or doctor's review. "
    "It cannot guarantee that a prescription is safe. Do not change or stop "
    "prescribed medication solely on the basis of this assessment."
)

OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"
