"""CDSCO banned / restricted FDC check (blacklisted.co.in, CC BY 4.0).

Matching is on compacted (space/hyphen-insensitive) ingredient sets with
salt-suffix and alias expansion, because gazette names ("Acetyl Salicylic
Acid") rarely equal database names ("Aspirin" / "Acetylsalicylic acid").

CRITICAL: only rows with action == 'prohibited' are violations. 'quashed',
'halted', 'under-examination', 'revoked', 'stayed' rows must NOT fire.
'restricted' rows fire as low-severity advisories, not violations.
"""

from __future__ import annotations

import json

from ..db import MedGuardDB
from ..models import ResolvedMed, SafetyFlag
from ..normalize import base_molecule, compact, lookup_keys


def _key_set(names: list[str]) -> set[str]:
    keys: set[str] = set()
    for n in names:
        for k in lookup_keys(n):
            keys.add(compact(k))
            keys.add(compact(base_molecule(k)))
    return keys


def check_banned_fdc(meds: list[ResolvedMed], db: MedGuardDB) -> list[SafetyFlag]:
    flags: list[SafetyFlag] = []
    seen: set[tuple] = set()  # dedupe twin gazette rows firing on one med
    for rm in meds:
        if rm.status == "unresolved" or not rm.ingredients:
            continue
        med_keys = _key_set([i.norm for i in rm.ingredients])
        med_names = ", ".join(i.name for i in rm.ingredients)

        for row in db.prohibited_rows():
            banned = json.loads(row["ingredients_json"])
            banned_keys = _key_set(banned)
            if banned_keys and banned_keys <= med_keys:
                sig = ("banned_fdc", rm.brand_display, frozenset(banned_keys))
                if sig in seen:
                    continue
                seen.add(sig)
                flags.append(SafetyFlag(
                    kind="banned_fdc",
                    title=f"Prohibited formulation suspected: {rm.brand_display}",
                    detail=(f"Its ingredients ({med_names}) contain a combination prohibited "
                            f"by CDSCO: '{row['name_raw']}' ({row['notification']}, {row['date']}). "
                            f"{row['status_note'] or ''} Do not use — consult the prescriber "
                            "immediately."),
                    severity="HIGH_CONCERN",
                    source="CDSCO via blacklisted.co.in (CC BY 4.0)",
                    source_url=row["source_url"] or "",
                    needs_verification=True,
                ))
        for row in db.restricted_rows():
            banned = json.loads(row["ingredients_json"])
            banned_keys = _key_set(banned)
            if banned_keys and banned_keys <= med_keys:
                sig = ("restricted_fdc", rm.brand_display, frozenset(banned_keys))
                if sig in seen:
                    continue
                seen.add(sig)
                flags.append(SafetyFlag(
                    kind="restricted_fdc",
                    title=f"Restricted-use combination: {rm.brand_display}",
                    detail=(f"CDSCO restricts '{row['name_raw']}' ({row['notification']}, "
                            f"{row['date']}). {row['status_note'] or ''} Follow the label "
                            "restriction and confirm with the pharmacist."),
                    severity="LOW_CONCERN",
                    source="CDSCO via blacklisted.co.in (CC BY 4.0)",
                    source_url=row["source_url"] or "",
                    needs_verification=True,
                ))
    return flags
