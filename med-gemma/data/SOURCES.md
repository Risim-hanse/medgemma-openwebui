# Reference data provenance

Built 2026-10-10. Rebuild anytime: `uv run python data/fetch_data.py && uv run python data/build_db.py`.
`data/manifest.json` pins exact URLs, byte sizes, and sha256 per file.

## 1. Drug–drug interactions — DDInter v1 (local CSV, offline)

- Files: `ddinter_code_{A,B,C,D,G,H,J,L,M,N,P,R,S,U,V}.csv` from
  `https://ddinter.scbdd.com/static/media/download/`
- ~542k unique pairs → 1.08M directed rows. Levels: Major / Moderate / Minor / Unknown.
- **License: CC BY-NC-SA 4.0** (https://ddinter.scbdd.com/terms/) — non-commercial, share-alike.
- Names are INN/systematic (`Acetylsalicylic acid`, `Acetaminophen`); resolved via
  alias + salt-suffix + base-molecule expansion in `src/medguard/normalize.py`.
  Verified bridges: `tazobactum`→`tazobactam` (dump typo), `methylcobalamin`→`mecobalamin`,
  `vitamin b12`→`cyanocobalamin`, prodrug esters (`proxetil`, `pivoxil` stripped).
- **Known coverage gap (measured 2026-10-10):** 774 of 1,721 ingredient norms have
  no DDInter presence at all — including very common Indian salts: aceclofenac,
  domperidone, nimesulide, ambroxol, ornidazole, etoricoxib, sulbactam,
  guaifenesin, thiocolchicoside, levosulpiride, serratiopeptidase (~102k brand
  links). Pairs involving these can only ever be UNVERIFIED-by-absence — the UI
  must keep saying so, and this list is the reason.
- `Unknown`-level rows are kept and surface as UNVERIFIED (record exists, severity unstated).
- ⚠️ There is **no DDInter REST API** (online "API docs" are hallucinated; endpoints 404).
  MyChem.info also has **no interaction data** (DrugBank fields removed Aug 2021). Both
  dead-ends were verified live — do not build on them.

## 2. Indian brand → salt — 1mg dump (local CSV, offline)

- `indian_pharmaceutical_products_clean.csv` from
  `https://huggingface.co/datasets/revooda/indian-pharma-data` (not gated).
- 253,973 products → 366,202 ingredient links, salt + strength pre-parsed.
- **License: none stated** — all-rights-reserved by default. Fine for a hackathon demo.
  For anything commercial, swap to MIT `junioralive/Indian-Medicine-Dataset` or
  CC BY 4.0 Kuinbee data (same schema shape; only `build_db.py` brand parsing changes).

## 3. CDSCO banned / restricted FDC — blacklisted.co.in (local CSV, offline)

- `banned-drugs.csv` → 1,080 rows, of which **683 `action='prohibited'`**.
- **License: CC BY 4.0** — attribute `"Source: CDSCO / FSSAI, compiled by blacklisted.co.in"`.
- **Critical:** 335 rows are `quashed` (court-reversed) plus halted/stayed/under-examination.
  The engine matches **only** `action='prohibited'` as violations; `restricted` rows are
  low-severity advisories. Every row carries its gazette `source_url`.
- Matching is subset-on-compacted-ingredient-sets with formulation-tail stripping
  (`Dispersible Tablets`, `Suspension`, doses…), salt expansion, and aliases.

## 4. Schedule H1 — hardcoded seed (indicative, NOT authoritative)

- 44 substances with gazette notes (G.S.R. 588(E) 2013 base + Oxytocin 2018, Tapentadol 2021,
  Oseltamivir/Zanamivir 2024, Pregabalin 2026) in `data/build_db.py::H1_SEED`.
- Served to users as *"indicative — confirm against Drugs Rules 1945 / gazette"*.
- No machine-readable official release exists (Schedule H consolidated list and H1/X
  amendments are PDFs/gazettes). Parsing them is roadmap, not done.

## 5. Online evidence (optional, cached, never authoritative)

- **openFDA** `drug/label.json` — free (240/min; 1k/day keyless, 120k/day with free key).
  Section-7 `drug_interactions` prose attached as excerpts to existing findings.
- **DailyMed** `spls/{setid}.xml` section `34073-7` — same role, richer tables.
- **RxNorm** `/REST/rxcui.json`, `approximateTerm`, `RxClass` — evaluated, free,
  no key. Name normalization is currently done locally (alias/base expansion)
  instead, so RxNorm is NOT wired into the pipeline; revisit if fuzzy
  mis-resolution needs an authoritative tiebreaker.
  The old RxNav **interaction API is discontinued (Jan 2024)** and returns 404.
- All three are **US-centric** (Indian brands resolve to nothing there); India coverage
  comes entirely from §1–§3 above.

## Deliberately excluded

- **DrugBank** (commercial license), **EkaCare/DrugSetu** (key-gated), **ABDM APIs**
  (consent-based record exchange — no drug/interaction endpoints; wrong tool here),
  **NLEM/NFI** (PDF only, no machine-readable release).
