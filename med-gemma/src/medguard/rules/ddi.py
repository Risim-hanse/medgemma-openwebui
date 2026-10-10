"""Pairwise drug-drug interaction check against local DDInter data.

Severity mapping: Major->HIGH, Moderate->MODERATE, Minor->LOW, Unknown->UNVERIFIED.
Pairs with NO DDInter record produce no finding (absence is not evidence),
but the counts are reported so the UI can state coverage honestly.
"""

from __future__ import annotations

from itertools import combinations

from ..config import DDI_SEVERITY, SEVERITY_RANK
from ..db import MedGuardDB
from ..models import Interaction, ResolvedMed
from ..normalize import base_molecule, lookup_keys


def _best_record(db: MedGuardDB, ka: str, kb: str):
    """Strongest DDInter record for a key pair, or None."""
    rows = db.ddi_lookup(ka, kb)
    if not rows:
        rows = db.ddi_lookup_base(base_molecule(ka), base_molecule(kb))
    if not rows:
        return None
    return max(rows, key=lambda r: SEVERITY_RANK.get(DDI_SEVERITY.get(r["level"], "UNVERIFIED"), 0))


def check_ddi(meds: list[ResolvedMed], db: MedGuardDB) -> tuple[list[Interaction], int, int]:
    # unique ingredients across resolved meds: norm -> (display, [source meds])
    seen: dict[str, tuple[str, list[str]]] = {}
    for rm in meds:
        if rm.status == "unresolved":
            continue
        for ing in rm.ingredients:
            disp, srcs = seen.get(ing.norm, (ing.name, []))
            if rm.brand_display not in srcs:
                srcs.append(rm.brand_display)
            seen[ing.norm] = (disp, srcs)

    norms = sorted(seen)
    interactions: list[Interaction] = []
    pairs_checked = 0
    pairs_with_record = 0
    for na, nb in combinations(norms, 2):
        pairs_checked += 1
        record = None
        for ka in lookup_keys(na):
            for kb in lookup_keys(nb):
                record = _best_record(db, ka, kb)
                if record:
                    break
            if record:
                break
        if not record:
            continue
        pairs_with_record += 1
        level = DDI_SEVERITY.get(record["level"], "UNVERIFIED")
        da, srcs_a = seen[na]
        dbb, srcs_b = seen[nb]
        interactions.append(Interaction(
            drug_a=da, drug_b=dbb, level=level,
            ddinter_level=record["level"],
            evidence=f"DDInter: {record['name_a']} + {record['name_b']} -> {record['level']}",
            sources=[f"{da} (from {', '.join(srcs_a)})", f"{dbb} (from {', '.join(srcs_b)})"],
            needs_verification=True,
        ))
    interactions.sort(key=lambda i: SEVERITY_RANK.get(i.level, 0), reverse=True)
    return interactions, pairs_checked, pairs_with_record
