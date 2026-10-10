"""Brand -> active-ingredient resolution against the local SQLite DB.

Strategy (in order):
  1. exact match on normalized brand string (plus glued-split and
     dose/form-stripped variants — "Warf5" and "450MG" spellings both hit)
  2. generic path: the name itself is a known active ingredient
     (checked BEFORE fuzzy so salts never fuzzy-match brands)
  3. fuzzy match (max(token_set, partial) >= 85) over SQL-prefiltered
     candidates, with a strength-mismatch warning when dose tokens differ
  4. unresolved (never guessed — returned with status for the user to confirm)
"""

from __future__ import annotations

import re

from rapidfuzz import fuzz, process

from .db import MedGuardDB
from .models import Ingredient, Medication, ResolvedMed
from .normalize import FORM_WORDS, FREQ_WORDS, lookup_keys, normalize

# Leading sigils seen on Indian prescriptions: "Tab.", "Cap.", "Syp.", "Inj."
_LEAD_RE = re.compile(r"^(tab|caps?|syp|syrup|inj|tabs?|injections?)\.?\s+", re.IGNORECASE)
# Dosage tokens: "650", "500mg", "40/5", "12.5", "625mg", "2ml", "100mcg"
_DOSE_RE = re.compile(r"\b\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?\s*(?:mg|g|mcg|µg|ml|l|iu|units?)?\b",
                      re.IGNORECASE)

FUZZY_CUTOFF = 85


def _fuzzy_score(a: str, b: str, **_) -> float:
    """token_set for reordered/extra tokens, partial for OCR misspellings."""
    return max(fuzz.token_set_ratio(a, b), fuzz.partial_ratio(a, b))


def strip_sigils(name: str) -> str:
    return _LEAD_RE.sub("", name.strip()).strip()


def split_glued(name: str) -> str:
    """Split glued letter-digit runs from OCR ("Warf5" -> "Warf 5").

    Applied as an ADDITIONAL exact-match variant, never destructively:
    correct "450MG" spellings must keep hitting.
    """
    s = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", name)
    return re.sub(r"(\d)([a-zA-Z])", r"\1 \2", s).strip()


def strip_dose_and_form(name: str) -> str:
    s = strip_sigils(name)
    s = _DOSE_RE.sub(" ", s)
    toks = [t for t in re.split(r"[\s\-]+", s)
            if t and t.lower() not in FORM_WORDS and t.lower() not in FREQ_WORDS]
    return " ".join(toks)


def _dose_tokens(text: str) -> set[str]:
    return {t.replace(" ", "").lower() for t in _DOSE_RE.findall(text) if t.strip()}


def _warn_strength_mismatch(stated_name: str, matched_norm: str, notes: list[str]) -> None:
    stated, matched = _dose_tokens(stated_name), _dose_tokens(matched_norm)
    if stated and stated != matched:
        notes.append(f"Stated strength ({', '.join(sorted(stated))}) differs from "
                     f"matched product ({', '.join(sorted(matched)) or 'unstated'}) — "
                     "confirm the exact pack.")


def _ingredients_of(db: MedGuardDB, product_id: str) -> list[Ingredient]:
    return [Ingredient(name=r["ingredient_raw"] or r["ingredient_norm"],
                       norm=r["ingredient_norm"],
                       strength=r["strength"] or "")
            for r in db.brand_ingredients(product_id)]


def _pick_product(db: MedGuardDB, rows: list, notes: list[str]) -> tuple[str, list[Ingredient]]:
    """Multiple pack-sizes share one brand_norm: use the majority composition."""
    groups: dict[tuple, list] = {}
    for r in rows:
        key = tuple((i["ingredient_norm"], i["strength"])
                    for i in db.brand_ingredients(r["product_id"]))
        groups.setdefault(key, []).append(r)
    best = max(groups.values(), key=len)
    if len(groups) > 1:
        notes.append(f"{len(rows)} pack variants found; showing most common composition.")
    win = best[0]
    return win["brand_raw"], _ingredients_of(db, win["product_id"])


def _fuzzy_candidates(db: MedGuardDB, cleaned: str) -> list:
    first = re.split(r"[\s\-]+", cleaned)[0] if cleaned else ""
    cands = db.prefix_brands(first, limit=300) if first else []
    if not cands and first:
        cands = db.contains_brands(first, limit=300)
    if not cands and len(first) >= 2:
        # OCR misspellings ("Ecospirin"): 2-letter prefix net, then fuzzy rank
        cands = db.prefix_brands(first[:2], limit=2000)
    return cands


def resolve_medication(med: Medication, db: MedGuardDB) -> ResolvedMed:
    notes: list[str] = []
    no_sigil = strip_sigils(med.name)
    cleaned_full = normalize(no_sigil)
    stripped = normalize(strip_dose_and_form(med.name))
    # Glued-split variants are ADDITIONAL candidates ("Warf5" hits; "450MG" still hits).
    variants = list(dict.fromkeys([
        cleaned_full, normalize(split_glued(no_sigil)),
        stripped, normalize(split_glued(strip_dose_and_form(med.name)))]))
    variants = [v for v in variants if v]
    tried = []

    # 1. exact matches (incl. brand-stem prefix)
    for norm in variants:
        tried.append(norm)
        rows = db.exact_brand(norm)
        if rows:
            brand_display, ing = _pick_product(db, rows, notes)
            if not any(r["brand_norm"] == norm for r in rows):
                _warn_strength_mismatch(med.name, rows[0]["brand_norm"], notes)
            return ResolvedMed(med=med, status="resolved", brand_display=brand_display,
                               ingredients=ing, notes=notes)

    # 2. generic path BEFORE fuzzy: a stated salt name must not fuzzy-match a brand
    for key in lookup_keys(no_sigil):
        if db.ingredient_exists(key):
            notes.append("Treated as generic (salt) name — no brand match needed.")
            return ResolvedMed(med=med, status="generic",
                               brand_display=med.name.strip(),
                               ingredients=[Ingredient(name=key, norm=key)],
                               notes=notes)

    # 3. fuzzy over prefiltered candidates (best across query variants)
    best_hits: list = []
    for query in dict.fromkeys([stripped or cleaned_full,
                                normalize(split_glued(strip_dose_and_form(med.name)))]):
        cands = _fuzzy_candidates(db, query)
        if not cands:
            continue
        names = [r["brand_norm"] for r in cands]
        hits = process.extract(query, names, scorer=_fuzzy_score,
                               score_cutoff=FUZZY_CUTOFF, limit=3)
        best_hits = sorted(hits + best_hits, key=lambda h: h[1], reverse=True)[:3]
    if best_hits:
        best_norm = best_hits[0][0]
        rows = db.exact_brand(best_norm)
        brand_display, ing = _pick_product(db, rows, notes)
        alts = sorted({db.exact_brand(h[0])[0]["brand_raw"] for h in best_hits[1:]})
        notes.append(f"Spelling not exact — matched '{brand_display}'. Please confirm.")
        _warn_strength_mismatch(med.name, best_norm, notes)
        return ResolvedMed(med=med, status="fuzzy", brand_display=brand_display,
                           ingredients=ing, candidates=alts, notes=notes)

    # 4. unresolved — never guess
    notes.append(f"Could not resolve '{med.name}' (tried: {', '.join(tried)}). "
                 "Please confirm the spelling.")
    return ResolvedMed(med=med, status="unresolved", brand_display=med.name.strip(),
                       notes=notes)


def resolve_all(meds: list[Medication], db: MedGuardDB) -> list[ResolvedMed]:
    return [resolve_medication(m, db) for m in meds]
