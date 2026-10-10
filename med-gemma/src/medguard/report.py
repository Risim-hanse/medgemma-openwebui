"""Deterministic Markdown rendering of a Review (no LLM needed)."""

from __future__ import annotations

from .config import DISCLAIMER
from .models import Review

_SEV_ICON = {"HIGH_CONCERN": "🔴", "MODERATE_CONCERN": "🟠",
             "LOW_CONCERN": "🟡", "UNVERIFIED": "⚪"}


def render_markdown(review: Review) -> str:
    L: list[str] = []
    A = L.append
    A("# MEDGUARD PRESCRIPTION REVIEW\n")
    A("## 1. MEDICATIONS IDENTIFIED\n")
    for rm in review.meds:
        m = rm.med
        bits = [b for b in [m.dosage, m.frequency, m.route, m.duration] if b]
        A(f"- **{rm.brand_display or m.name}**"
          + (f" — {', '.join(bits)}" if bits else " — instructions missing")
          + f" `[{rm.status}]`")
        if rm.ingredients and rm.status in ("resolved", "fuzzy"):
            A("  - Active: " + ", ".join(
                f"{i.name}{f' {i.strength}' if i.strength else ''}" for i in rm.ingredients))
        for n in rm.notes:
            A(f"  - ⚠️ {n}")
    A("\n## 2. POTENTIAL INTERACTIONS\n")
    if review.interactions:
        for ix in review.interactions:
            A(f"### {_SEV_ICON.get(ix.level, '')} {ix.drug_a} + {ix.drug_b} — {ix.level}")
            A(f"- Evidence: {ix.evidence}")
            if ix.label_evidence:
                A(f"- Label excerpt: {ix.label_evidence}")
            A(f"- Sources: {'; '.join(ix.sources)}")
            A("- Requires pharmacist/clinician verification: yes")
    else:
        A("No interaction was identified from the available information. "
          "This does not prove that the combination is safe.")
    if review.pairs_checked:
        A(f"\n_Coverage: {review.pairs_with_record}/{review.pairs_checked} "
          "ingredient pairs have a DDInter record._")
    A("\n## 3. OTHER SAFETY CONCERNS\n")
    if review.flags:
        for f in review.flags:
            A(f"- {_SEV_ICON.get(f.severity, '')} **[{f.kind}] {f.title}**")
            A(f"  {f.detail}")
            if f.source:
                A(f"  Source: {f.source}" + (f" ({f.source_url})" if f.source_url else ""))
    else:
        A("None flagged.")
    A("\n## 4. MISSING INFORMATION\n")
    A("\n".join(f"- {m}" for m in review.missing_info) if review.missing_info else "None.")
    A("\n## 5. RECOMMENDED NEXT STEPS\n")
    A("- Show this review to the prescribing clinician or a qualified pharmacist.")
    if review.max_severity() == "HIGH_CONCERN":
        A("- A HIGH concern finding is present — seek pharmacist review promptly; "
          "do not wait for the next routine visit.")
    A("\n## 6. IMPORTANT DISCLAIMER\n")
    A('"' + DISCLAIMER + '"')
    return "\n".join(L) + "\n"
