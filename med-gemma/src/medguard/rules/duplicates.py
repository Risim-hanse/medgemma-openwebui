"""Duplicate active-ingredient detection (same salt via two brands / FDC overlap)."""

from __future__ import annotations

from ..models import ResolvedMed, SafetyFlag
from ..normalize import base_molecule


def check_duplicates(meds: list[ResolvedMed]) -> list[SafetyFlag]:
    owners: dict[str, list[str]] = {}
    display: dict[str, str] = {}
    for rm in meds:
        if rm.status == "unresolved":
            continue
        for ing in rm.ingredients:
            key = base_molecule(ing.norm)
            owners.setdefault(key, [])
            if rm.brand_display not in owners[key]:
                owners[key].append(rm.brand_display)
            display.setdefault(key, ing.name)
    flags = []
    for key, srcs in owners.items():
        if len(srcs) > 1:
            flags.append(SafetyFlag(
                kind="duplicate", title=f"Possible duplicate ingredient: {display[key]}",
                detail=(f"'{display[key]}' appears in {len(srcs)} medicines "
                        f"({', '.join(srcs)}). This may be intentional, but combined "
                        "doses can exceed safe limits — ask the pharmacist to confirm "
                        "the total daily dose."),
                severity="MODERATE_CONCERN",
                source="MedGuard duplicate-ingredient check (local brand database)",
                needs_verification=True,
            ))
    return flags
