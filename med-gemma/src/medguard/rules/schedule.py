"""Schedule H1 screen (indicative list — always served with a verify warning)."""

from __future__ import annotations

from ..db import MedGuardDB
from ..models import ResolvedMed, SafetyFlag
from ..normalize import lookup_keys


def check_schedule(meds: list[ResolvedMed], db: MedGuardDB) -> list[SafetyFlag]:
    flags: list[SafetyFlag] = []
    seen: set[str] = set()
    for rm in meds:
        if rm.status == "unresolved":
            continue
        for ing in rm.ingredients:
            for key in lookup_keys(ing.norm):
                for row in db.schedule_h1(key):
                    if row["name_norm"] in seen:
                        continue
                    seen.add(row["name_norm"])
                    flags.append(SafetyFlag(
                        kind="schedule_h1",
                        title=f"Schedule H1 medicine (indicative): {ing.name}",
                        detail=(f"'{ing.name}' (in {rm.brand_display}) may fall under Schedule H1 "
                                f"of the Drugs & Cosmetics Rules ({row['note']}). H1 medicines need "
                                "a valid prescription on record and carry a red-box label warning. "
                                "This list is indicative — confirm against the current gazette/rules."),
                        severity="MODERATE_CONCERN",
                        source="MedGuard indicative H1 list (see data/SOURCES.md)",
                        needs_verification=True,
                    ))
    return flags
