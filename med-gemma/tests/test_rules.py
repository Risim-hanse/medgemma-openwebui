"""CPU-only tests for the deterministic MedGuard pipeline (no model needed)."""

from __future__ import annotations

import pytest

from medguard.config import DB_PATH
from medguard.db import MedGuardDB
from medguard.models import Ingredient, Medication, ResolvedMed
from medguard.normalize import base_molecule, compact, lookup_keys, normalize
from medguard.resolve import resolve_medication
from medguard.rules.banned_fdc import check_banned_fdc
from medguard.rules.ddi import check_ddi
from medguard.rules.duplicates import check_duplicates
from medguard.rules.engine import run_review
from medguard.rules.schedule import check_schedule


@pytest.fixture(scope="module")
def db() -> MedGuardDB:
    assert DB_PATH.exists(), "run data/build_db.py first"
    return MedGuardDB()


def M(name: str, **kw) -> Medication:
    return Medication(name=name, **kw)


# ---------- normalize ----------
def test_normalize_basic():
    assert normalize("  Tab. Dolo 650 ") == "tab dolo 650"
    assert base_molecule("warfarin sodium") == "warfarin"
    assert base_molecule("metformin hydrochloride") == "metformin"
    assert "acetylsalicylic acid" in lookup_keys("Aspirin")
    assert "aspirin" in lookup_keys("acetylsalicylic acid")
    assert compact("Acetyl Salicylic Acid") == compact("acetylsalicylic-acid")


# ---------- resolve ----------
def test_resolve_telma_am(db):
    rm = resolve_medication(M("Telma-AM", dosage="1 tab", frequency="OD"), db)
    assert rm.status == "resolved"
    salts = {i.norm for i in rm.ingredients}
    assert salts == {"telmisartan", "amlodipine"}


def test_resolve_with_sigil_and_dose(db):
    rm = resolve_medication(M("Tab. Dolo 650 TDS"), db)
    assert rm.status == "resolved"
    assert [i.norm for i in rm.ingredients] == ["paracetamol"]


def test_resolve_augmentin_duo(db):
    rm = resolve_medication(M("Augmentin 625 Duo", frequency="BD"), db)
    assert rm.status == "resolved"
    # NOTE: the 1mg dump uses the Indian spelling "Amoxycillin" — kept verbatim;
    # DDI matching expands it via aliases.
    assert {i.norm for i in rm.ingredients} == {"amoxycillin", "clavulanic acid"}


def test_resolve_generic_path(db):
    rm = resolve_medication(M("Metformin", dosage="500mg", frequency="BD"), db)
    assert rm.status == "generic"
    assert rm.ingredients[0].norm == "metformin"


def test_resolve_unresolved_never_guesses(db):
    rm = resolve_medication(M("Xyzqqnonexistent"), db)
    assert rm.status == "unresolved"
    assert rm.ingredients == []


def test_resolve_ocr_glued_dose(db):
    # Vision models often glue strength to the name ("Warf5", "Dolo650").
    rm = resolve_medication(M("Warf5", frequency="OD"), db)
    assert rm.status == "resolved"
    assert [i.norm for i in rm.ingredients] == ["warfarin"]
    rm = resolve_medication(M("Dolo650", frequency="TDS"), db)
    assert rm.status == "resolved"
    assert [i.norm for i in rm.ingredients] == ["paracetamol"]


def test_resolve_ocr_misspelling(db):
    # "Ecospirin" (misread Ecosprin) must fuzzy-match, never silently resolve wrong.
    rm = resolve_medication(M("Ecospirin", frequency="OD"), db)
    assert rm.status == "fuzzy"
    assert [i.norm for i in rm.ingredients] == ["aspirin"]


def test_resolve_correct_spelling_stays_exact(db):
    # Glued-split is an ADDITIONAL variant: correct "450MG"/"1gm" spellings
    # must still hit exact, not fuzzy.
    rm = resolve_medication(M("Citafine 1gm Injection"), db)
    assert rm.status == "resolved"
    rm = resolve_medication(M("ARZADE 450MG/300MG CAPSULE"), db)
    assert rm.status == "resolved"


def test_resolve_strength_mismatch_warns(db):
    # 100mg/6mg exists in neither Auxidol variant (4mg, 8mg): fuzzy hit MUST
    # carry a strength warning, not silently substitute.
    rm = resolve_medication(M("Auxidol TH 100mg/6mg Tablet"), db)
    assert rm.status == "fuzzy"
    assert any("differs" in n for n in rm.notes), rm.notes


def test_alias_tazobactum(db):
    from medguard.normalize import lookup_keys
    assert "tazobactam" in lookup_keys("Tazobactum")
    rm = resolve_medication(M("Tazobactum"), db)
    assert rm.status == "generic"


# ---------- DDI ----------
def test_ddi_aspirin_warfarin_is_major(db):
    meds = [resolve_medication(M("Ecosprin 75", frequency="OD"), db),
            resolve_medication(M("Warf 5", frequency="OD"), db)]
    assert all(m.status != "unresolved" for m in meds)
    ixs, checked, with_record = check_ddi(meds, db)
    assert checked == 1 and with_record == 1
    assert len(ixs) == 1
    assert ixs[0].ddinter_level == "Major"
    assert ixs[0].level == "HIGH_CONCERN"


def test_ddi_no_record_no_finding(db):
    # Domperidone & thyroxine are both absent from DDInter: absence must
    # produce NO finding (absence is not evidence of safety either).
    meds = [resolve_medication(M("Domperidone"), db),
            resolve_medication(M("Thyroxine"), db)]
    assert all(m.status != "unresolved" for m in meds)
    ixs, checked, with_record = check_ddi(meds, db)
    assert checked == 1 and with_record == 0 and ixs == []


# ---------- duplicates ----------
def test_duplicate_paracetamol(db):
    meds = [resolve_medication(M("Dolo 650"), db),
            resolve_medication(M("Crocin Advance"), db)]
    flags = check_duplicates(meds)
    assert len(flags) == 1 and flags[0].kind == "duplicate"
    assert "paracetamol" in flags[0].title.lower()


# ---------- banned FDC ----------
def _synthetic(brand: str, *salts: str) -> ResolvedMed:
    return ResolvedMed(med=M(brand), status="resolved", brand_display=brand,
                       ingredients=[Ingredient(name=s, norm=normalize(s)) for s in salts])


def test_banned_fdc_fires(db):
    meds = [_synthetic("Test", "Acetyl Salicylic Acid", "Ethoheptazine")]
    flags = check_banned_fdc(meds, db)
    assert len(flags) == 1 and flags[0].kind == "banned_fdc"
    assert flags[0].severity == "HIGH_CONCERN"


def test_quashed_row_must_not_fire(db):
    # 'Chlorpheniramine + Phenylephrine + Caffiene' was quashed (court-reversed),
    # and no prohibited subset of it exists (verified) — so no violation may fire.
    # (Restricted-use advisories for CPM+PE are a separate kind.)
    meds = [_synthetic("Test", "Chlorpheniramine", "Phenylephrine", "Caffiene")]
    flags = [f for f in check_banned_fdc(meds, db) if f.kind == "banned_fdc"]
    assert flags == []


def test_restricted_is_advisory_not_violation(db):
    meds = [_synthetic("Test", "Chlorpheniramine Maleate", "Phenylephrine Hydrochloride")]
    flags = check_banned_fdc(meds, db)
    assert len(flags) == 1 and flags[0].kind == "restricted_fdc"
    assert flags[0].severity == "LOW_CONCERN"


# ---------- schedule ----------
def test_schedule_h1_flags_zolpidem(db):
    meds = [resolve_medication(M("Zolpidem", dosage="10mg"), db)]
    flags = check_schedule(meds, db)
    assert len(flags) == 1 and flags[0].kind == "schedule_h1"


def test_schedule_h1_clean_for_paracetamol(db):
    meds = [resolve_medication(M("Dolo 650"), db)]
    assert check_schedule(meds, db) == []


# ---------- engine ----------
def test_engine_end_to_end(db):
    review = run_review([
        M("Ecosprin 75", dosage="75mg", frequency="OD"),
        M("Warf 5", dosage="5mg", frequency="OD"),
        M("Dolo 650", dosage="650mg", frequency="TDS"),
    ], db)
    assert len(review.meds) == 3
    assert any(i.level == "HIGH_CONCERN" for i in review.interactions)
    assert review.pairs_checked == 3  # 3 ingredients -> 3 pairs
    assert review.max_severity() == "HIGH_CONCERN"


def test_coverage_counter_propagates(db):
    # Regression: engine must propagate pairs_with_record (no silent drops).
    review = run_review([M("Ecosprin 75"), M("Warf 5")], db)
    assert (review.pairs_checked, review.pairs_with_record) == (1, 1)
    from medguard.report import render_markdown
    assert "1/1" in render_markdown(review)
