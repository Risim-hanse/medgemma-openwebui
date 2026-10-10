"""Build data/medguard.db from data/raw/ + data/manifest.json (re-runnable).

Tables:
  meta(key, value)                       provenance / licenses / build info
  brands(product_id, brand_raw, brand_norm, manufacturer, dosage_form,
         pack_size, price_inr, is_discontinued, therapeutic_class, num_ingredients)
  brand_ingredients(product_id, seq, ingredient_raw, ingredient_norm, strength)
  ddi(name_a, norm_a, base_a, name_b, norm_b, base_b, level)   (both directions)
  banned_fdc(name_raw, kind, category, action, notification, date, wave,
             status_note, source_url, ingredients_json)
  schedule_h1(name_raw, name_norm, note, confidence)

Usage:
    uv run python data/build_db.py
"""

from __future__ import annotations

import ast
import csv
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from medguard.normalize import FORM_TAIL, base_molecule, normalize  # noqa: E402

# Trailing formulation/dose tokens stripped from gazette ingredient parts:
# "Paracetamol Dispersible Tablets" -> "paracetamol" (FORM_TAIL is shared).
_DOSE_TOK = re.compile(r"^\d+(?:\.\d+)?(?:mg|g|mcg|µg|ml|l|iu|units?|sp)?$", re.IGNORECASE)


def clean_banned_part(part: str) -> str:
    toks = normalize(part).split()
    toks = [t for t in toks if not _DOSE_TOK.match(t)]
    while toks and toks[-1] in FORM_TAIL:
        toks.pop()
    return " ".join(toks)

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
DB = ROOT / "medguard.db"

SCHEMA = """
PRAGMA journal_mode=WAL;
DROP TABLE IF EXISTS brands;
DROP TABLE IF EXISTS brand_ingredients;
DROP TABLE IF EXISTS ddi;
DROP TABLE IF EXISTS banned_fdc;
DROP TABLE IF EXISTS schedule_h1;
DROP TABLE IF EXISTS meta;
CREATE TABLE brands(
  product_id TEXT PRIMARY KEY, brand_raw TEXT, brand_norm TEXT,
  manufacturer TEXT, dosage_form TEXT, pack_size TEXT, price_inr REAL,
  is_discontinued INTEGER, therapeutic_class TEXT, num_ingredients INTEGER);
CREATE INDEX idx_brand_norm ON brands(brand_norm);
CREATE TABLE brand_ingredients(
  product_id TEXT, seq INTEGER, ingredient_raw TEXT,
  ingredient_norm TEXT, strength TEXT);
CREATE INDEX idx_ing_norm ON brand_ingredients(ingredient_norm);
CREATE INDEX idx_ing_pid ON brand_ingredients(product_id);
CREATE TABLE ddi(
  name_a TEXT, norm_a TEXT, base_a TEXT,
  name_b TEXT, norm_b TEXT, base_b TEXT, level TEXT);
CREATE INDEX idx_ddi_ab ON ddi(norm_a, norm_b);
CREATE INDEX idx_ddi_base ON ddi(base_a, base_b);
CREATE UNIQUE INDEX idx_ddi_unique ON ddi(norm_a, norm_b, level);
CREATE TABLE banned_fdc(
  name_raw TEXT, kind TEXT, category TEXT, action TEXT, notification TEXT,
  date TEXT, wave TEXT, status_note TEXT, source_url TEXT, ingredients_json TEXT);
CREATE INDEX idx_banned_action ON banned_fdc(action);
CREATE TABLE schedule_h1(name_raw TEXT, name_norm TEXT, note TEXT, confidence TEXT);
CREATE INDEX idx_h1_norm ON schedule_h1(name_norm);
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
"""

# Indicative Schedule-H1 substance list with gazette provenance.
# Authoritative text lives in the Drugs & Cosmetics Rules, 1945 as amended;
# every row is served to users as "indicative — verify", never as legal advice.
H1_SEED: list[tuple[str, str]] = [
    *[(d, "G.S.R. 588(E), 30 Aug 2013 (original H1 schedule)") for d in [
        "Alprazolam", "Balofloxacin", "Buprenorphine", "Capreomycin",
        "Cefdinir", "Cefditoren", "Cefepime", "Cefetamet", "Cefixime",
        "Cefoperazone", "Cefotaxime", "Cefpirome", "Cefpodoxime",
        "Ceftazidime", "Ceftizoxime", "Ceftriaxone", "Cefuroxime",
        "Chlordiazepoxide", "Clonazepam", "Codeine", "Cycloserine",
        "Diazepam", "Ethionamide", "Gatifloxacin", "Gemifloxacin",
        "Kanamycin", "Lorazepam", "Midazolam", "Moxifloxacin",
        "Nitrazepam", "Ofloxacin", "Para-aminosalicylic acid",
        "Pazufloxacin", "Pentazocine", "Prulifloxacin", "Rifabutin",
        "Sparfloxacin", "Tramadol", "Zolpidem",
    ]],
    ("Oxytocin", "G.S.R. 795(E), 2018 (added to H1)"),
    ("Tapentadol", "G.S.R. 258(E), 2021 (added to H1)"),
    ("Oseltamivir", "G.S.R. 95(E), Feb 2024 (added to H1)"),
    ("Zanamivir", "G.S.R. 95(E), Feb 2024 (added to H1)"),
    ("Pregabalin", "G.S.R. 377(E), May 2026 (added to H1)"),
]


def _parse_ingredients(cell: str) -> list[tuple[str, str]]:
    """Parse "[{'name': ..., 'strength': ...}, ...]" reprs from the 1mg dump."""
    if not cell or not cell.strip():
        return []
    try:
        items = ast.literal_eval(cell)
    except (ValueError, SyntaxError):
        return []
    out = []
    for it in items if isinstance(items, list) else []:
        if isinstance(it, dict) and it.get("name"):
            out.append((str(it["name"]), str(it.get("strength") or "")))
    return out


def build() -> None:
    t0 = time.time()
    if DB.exists():
        DB.unlink()
    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)

    # ---- brands ----
    brand_csv = RAW / "indian_brands.csv"
    n_brands = n_ing = 0
    with open(brand_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        brand_rows, ing_rows = [], []
        for r in reader:
            raw = (r.get("brand_name") or "").strip()
            if not raw:
                continue
            pid = (r.get("product_id") or raw).strip()
            try:
                price = float(r.get("price_inr") or 0) or None
            except ValueError:
                price = None
            brand_rows.append((
                pid, raw, normalize(raw), (r.get("manufacturer") or "").strip(),
                (r.get("dosage_form") or "").strip(), (r.get("pack_size") or "").strip(),
                price, 1 if str(r.get("is_discontinued")).lower() == "true" else 0,
                (r.get("therapeutic_class") or "").strip(),
                int(r.get("num_active_ingredients") or 0)))
            for seq, (iname, strength) in enumerate(_parse_ingredients(r.get("active_ingredients") or "")):
                ing_rows.append((pid, seq, iname, normalize(iname), strength))
            n_brands += 1
            if len(brand_rows) >= 5000:
                con.executemany(
                    "INSERT OR REPLACE INTO brands VALUES (?,?,?,?,?,?,?,?,?,?)", brand_rows)
                con.executemany(
                    "INSERT INTO brand_ingredients VALUES (?,?,?,?,?)", ing_rows)
                n_ing += len(ing_rows)
                brand_rows, ing_rows = [], []
                if n_brands % 50000 == 0:
                    con.commit()
                    print(f"  ... {n_brands:,} brands", flush=True)
        if brand_rows:
            con.executemany("INSERT OR REPLACE INTO brands VALUES (?,?,?,?,?,?,?,?,?,?)", brand_rows)
            con.executemany("INSERT INTO brand_ingredients VALUES (?,?,?,?,?)", ing_rows)
            n_ing += len(ing_rows)
    con.commit()
    print(f"brands: {n_brands:,} products, {n_ing:,} ingredient links", flush=True)

    # ---- DDInter pairs (both directions) ----
    n_ddi = 0
    ddi_files = sorted(RAW.glob("ddinter_code_*.csv"))
    for path in ddi_files:
        rows = []
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                na, nb, lvl = (r.get("Drug_A") or "").strip(), (r.get("Drug_B") or "").strip(), (r.get("Level") or "").strip()
                if not na or not nb or not lvl:
                    continue
                n_a, n_b = normalize(na), normalize(nb)
                rows.append((na, n_a, base_molecule(n_a), nb, n_b, base_molecule(n_b), lvl))
                rows.append((nb, n_b, base_molecule(n_b), na, n_a, base_molecule(n_a), lvl))
        con.executemany("INSERT OR IGNORE INTO ddi VALUES (?,?,?,?,?,?,?)", rows)
        con.commit()
        n_ddi += len(rows)
        print(f"  ... {path.name}: {len(rows) // 2:,} pairs", flush=True)
    print(f"ddi: {n_ddi:,} directed pairs from {len(ddi_files)} files", flush=True)

    # ---- banned FDC / molecules (utf-8-sig: source file has a BOM) ----
    n_ban = 0
    with open(RAW / "banned-drugs.csv", newline="", encoding="utf-8-sig") as f:
        rows = []
        for r in csv.DictReader(f):
            name = (r.get("name") or "").strip()
            if r.get("kind") == "fdc":
                parts = [clean_banned_part(p) for p in name.split("+")]
                parts = [p for p in parts if p]
            else:
                parts = [normalize(name)]
            rows.append((name, r.get("kind"), r.get("category"), r.get("action"),
                         r.get("notification"), r.get("date"), r.get("wave"),
                         r.get("status_note"), r.get("source_url"), json.dumps(parts)))
        con.executemany("INSERT INTO banned_fdc VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        n_ban = len(rows)
    con.commit()
    n_proh = con.execute(
        "SELECT COUNT(*) FROM banned_fdc WHERE action='prohibited'").fetchone()[0]
    print(f"banned: {n_ban:,} rows ({n_proh} prohibited)")

    # ---- schedule H1 seed ----
    con.executemany(
        "INSERT INTO schedule_h1 VALUES (?,?,?,?)",
        [(d, normalize(d), note, "indicative") for d, note in H1_SEED])
    print(f"schedule_h1: {len(H1_SEED)} indicative entries")

    # ---- meta ----
    manifest = json.loads((ROOT / "manifest.json").read_text())
    con.executemany("INSERT INTO meta VALUES (?,?)", [
        ("built_at_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        ("build_seconds", f"{time.time() - t0:.1f}"),
        ("ddinter_license", "CC BY-NC-SA 4.0 (https://ddinter.scbdd.com/terms/)"),
        ("banned_fdc_license", "CC BY 4.0, Source: CDSCO/FSSAI compiled by blacklisted.co.in"),
        ("brand_dump_source", "huggingface.co/datasets/revooda/indian-pharma-data (no license stated)"),
        ("schedule_h1_confidence", "indicative only — verify against Drugs Rules 1945 / gazettes"),
        ("manifest", json.dumps(manifest)),
    ])
    con.commit()
    size_mb = DB.stat().st_size / 1e6
    print(f"OK {DB} ({size_mb:.1f} MB) in {time.time() - t0:.1f}s")
    con.close()


if __name__ == "__main__":
    sys.exit(build())
