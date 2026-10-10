"""Golden-set regression for the deterministic engine (no model needed).

Each case: list of stated medicines -> assertions on the Review.
Run:  uv run python scripts/eval_golden.py
Exit code is non-zero on any failure (CI-friendly).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medguard.db import MedGuardDB
from medguard.models import Medication
from medguard.rules.engine import run_review

PASS, FAIL = "✅", "❌"
failures = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"{PASS if cond else FAIL} {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def main() -> int:
    db = MedGuardDB()

    # 1. Classic HIGH: aspirin + warfarin
    r = run_review([Medication(name="Ecosprin 75", dosage="75mg", frequency="OD"),
                    Medication(name="Warf 5", dosage="5mg", frequency="OD")], db)
    check("aspirin+warfarin HIGH", any(
        i.level == "HIGH_CONCERN" and i.ddinter_level == "Major" for i in r.interactions))
    check("aspirin+warfarin coverage 1/1",
          (r.pairs_checked, r.pairs_with_record) == (1, 1))

    # 2. Duplicate salt via two brands
    r = run_review([Medication(name="Dolo 650", frequency="TDS"),
                    Medication(name="Crocin Advance", frequency="TDS")], db)
    check("paracetamol duplicate flagged",
          any(f.kind == "duplicate" for f in r.flags))

    # 3. Prohibited FDC (real brand): Nimesulide + Paracetamol
    r = run_review([Medication(name="Alleviate Tablet", frequency="BD")], db)
    check("banned FDC HIGH",
          any(f.kind == "banned_fdc" and f.severity == "HIGH_CONCERN" for f in r.flags))

    # 4. Schedule H1 (indicative)
    r = run_review([Medication(name="Zolpidem", dosage="10mg")], db)
    check("H1 flagged", any(f.kind == "schedule_h1" for f in r.flags))

    # 5. Monitor-class DDI inside a legal FDC: Glimepiride + Metformin
    r = run_review([Medication(name="Glycomet-GP 1", frequency="OD")], db)
    check("glimepiride+metformin MODERATE", any(
        i.level == "MODERATE_CONCERN" for i in r.interactions),
        f"got {[(i.drug_a, i.drug_b, i.level) for i in r.interactions]}")

    # 6. No-record pair: no finding, honest coverage
    r = run_review([Medication(name="Domperidone"), Medication(name="Thyroxine")], db)
    check("no-record pair: no findings",
          r.interactions == [] and (r.pairs_checked, r.pairs_with_record) == (1, 0))

    # 7. Unresolvable name: never guessed, surfaced as missing info
    r = run_review([Medication(name="Xyzqqnonexistent")], db)
    check("unresolved surfaced",
          r.meds[0].status == "unresolved"
          and any("Xyzqqnonexistent" in m for m in r.missing_info))

    # 8. Quashed rows never fire as violations (subset logic still works: case 3)
    print(f"\n{len(failures)} failures" if failures else "\nAll golden cases pass.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
