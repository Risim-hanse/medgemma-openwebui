"""Pydantic schemas for the MedGuard pipeline.

Medication  = one row as stated on the prescription (extracted).
ResolvedMed = that row mapped to active ingredient(s) via the local DB.
Review      = the deterministic output of the rules engine (facts only).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .config import SEVERITY_RANK


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Medication(StrictModel):
    raw_text: str = ""
    name: str
    dosage: str = ""
    frequency: str = ""
    route: str = ""
    duration: str = ""
    confidence: str = "medium"  # high | medium | low (extraction confidence)


class Ingredient(StrictModel):
    name: str       # display name, e.g. "Telmisartan"
    norm: str       # normalized key, e.g. "telmisartan"
    strength: str = ""


class ResolvedMed(StrictModel):
    med: Medication
    status: str = "unresolved"  # resolved | fuzzy | generic | unresolved
    brand_display: str = ""
    ingredients: list[Ingredient] = Field(default_factory=list)
    candidates: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class Interaction(StrictModel):
    drug_a: str
    drug_b: str
    level: str  # HIGH_CONCERN | MODERATE_CONCERN | LOW_CONCERN | UNVERIFIED
    ddinter_level: str = "none"  # Major | Moderate | Minor | Unknown | none
    evidence: str = ""           # which record this came from
    sources: list[str] = Field(default_factory=list)
    label_evidence: str = ""     # openFDA/DailyMed excerpt, if fetched
    needs_verification: bool = True


class SafetyFlag(StrictModel):
    kind: str  # duplicate | banned_fdc | restricted_fdc | schedule_h1 | unclear | info
    title: str
    detail: str
    severity: str = "UNVERIFIED"
    source: str = ""
    source_url: str = ""
    needs_verification: bool = True


class Review(StrictModel):
    meds: list[ResolvedMed] = Field(default_factory=list)
    interactions: list[Interaction] = Field(default_factory=list)
    flags: list[SafetyFlag] = Field(default_factory=list)
    pairs_checked: int = 0
    pairs_with_record: int = 0
    missing_info: list[str] = Field(default_factory=list)

    def max_severity(self) -> str:
        levels = [i.level for i in self.interactions] + [f.severity for f in self.flags]
        if not levels:
            return "UNVERIFIED"
        return max(levels, key=lambda l: SEVERITY_RANK.get(l, 0))
