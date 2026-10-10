"""Deterministic review pipeline: resolve -> rules -> Review (no LLM inside)."""

from __future__ import annotations

from ..db import MedGuardDB
from ..models import Medication, Review
from ..resolve import resolve_all
from .banned_fdc import check_banned_fdc
from .ddi import check_ddi
from .duplicates import check_duplicates
from .labels import fetch_label_snippets
from .schedule import check_schedule


def run_review(meds: list[Medication], db: MedGuardDB,
               fetch_labels: bool = False) -> Review:
    resolved = resolve_all(meds, db)
    interactions, pairs_checked, pairs_with_record = check_ddi(resolved, db)
    flags = []
    flags += check_duplicates(resolved)
    flags += check_banned_fdc(resolved, db)
    flags += check_schedule(resolved, db)

    if fetch_labels:
        for ix in interactions:
            terms = [ix.drug_b, *(s.split(" (from")[0] for s in ix.sources)]
            for cand in [ix.drug_a, ix.drug_a.lower()]:
                snips = fetch_label_snippets(cand, terms)
                if snips:
                    ix.label_evidence = snips[0]
                    break

    missing: list[str] = []
    for rm in resolved:
        m = rm.med
        if rm.status == "unresolved":
            missing.append(f"Could not identify '{m.name}' — please confirm the spelling.")
        elif rm.status == "fuzzy":
            missing.append(f"Confirm brand: interpreted '{m.name}' as '{rm.brand_display}'.")
        if not m.dosage:
            missing.append(f"Dosage missing for '{m.name}'.")
        if not m.frequency:
            missing.append(f"Frequency/timing missing for '{m.name}'.")
    if pairs_checked and not pairs_with_record:
        missing.append("No DDInter record covers any pair here — this says nothing about safety.")

    return Review(meds=resolved, interactions=interactions, flags=flags,
                  pairs_checked=pairs_checked,
                  pairs_with_record=pairs_with_record,
                  missing_info=missing)
