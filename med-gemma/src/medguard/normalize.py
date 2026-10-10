"""Shared drug-name normalization used by both the DB builder and the runtime.

One function, one truth: every brand / salt / ingredient string that enters
the SQLite DB is normalized here, and every runtime query normalizes with the
same function before lookup. The dosage-form / frequency / formulation-tail
vocabularies also live here so resolve.py and data/build_db.py cannot drift.
"""

from __future__ import annotations

import re

# Common salt / ester / hydrate suffixes stripped to reach the base molecule.
# Applied ONLY to derive an additional `base` lookup key — the full normalized
# form is always kept as well.
_SALT_WORDS = {
    "sodium", "potassium", "calcium", "magnesium", "zinc", "ferrous", "ferric",
    "hydrochloride", "hydrobromide", "sulphate", "sulfate", "phosphate",
    "maleate", "fumarate", "tartrate", "citrate", "mesylate", "mesilate",
    "besylate", "besilate", "tosylate", "camsylate", "embonate", "pamoate",
    "succinate", "acetate", "lactate", "gluconate", "orotate", "meglumine",
    "tromethamine", "trometamol", "dihydrochloride", "sesquihydrate",
    "monohydrate", "dihydrate", "trihydrate", "anhydrous", "hydrochlorid",
    "maleat", "sulfat",  # tolerate source typos
    "hcl", "hbr",  # abbreviated salt suffixes used in gazette names
    "proxetil", "pivoxil",  # prodrug esters: cefpodoxime proxetil -> cefpodoxime
}

# INN <-> USAN / spelling variants. Both directions are expanded at query time.
ALIASES: dict[str, str] = {
    "acetaminophen": "paracetamol",
    "paracetamol": "acetaminophen",
    "epinephrine": "adrenaline",
    "adrenaline": "epinephrine",
    "norepinephrine": "noradrenaline",
    "noradrenaline": "norepinephrine",
    "lidocaine": "lignocaine",
    "lignocaine": "lidocaine",
    "amoxycillin": "amoxicillin",
    "sulphate": "sulfate",
    "aluminium": "aluminum",
    "oestrogen": "estrogen",
    "sulphur": "sulfur",
    "glyceryl trinitrate": "nitroglycerin",
    "nitroglycerin": "glyceryl trinitrate",
    "vitamin c": "ascorbic acid",
    "ascorbic acid": "vitamin c",
    # DDInter uses systematic names for a few very common drugs
    "aspirin": "acetylsalicylic acid",
    "acetylsalicylic acid": "aspirin",
    "vitamin b9": "folic acid",
    "folic acid": "vitamin b9",
    # Verified against DDInter contents (target must exist there):
    "tazobactum": "tazobactam",  # 1mg-dump typo; DDInter uses tazobactam
    "methylcobalamin": "mecobalamin",  # DDInter's name for methyl-B12
    "vitamin b12": "cyanocobalamin",  # generic B12 reference name
}

_PAREN_RE = re.compile(r"\([^)]*\)")
_WS_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s+\-/]")

# Dosage-form tokens. resolve.py strips these ANYWHERE in a name (fallback
# path only — exact match is always tried first).
FORM_WORDS = {
    "tablet", "tablets", "tab", "tabs", "capsule", "capsules", "cap", "caps",
    "syrup", "syp", "sy", "injection", "inj", "suspension", "susp",
    "drops", "drop", "ointment", "oint", "cream", "gel", "lotion", "spray",
    "powder", "sachet", "solution", "tabletform",
}

# Latin frequency shorthands patients copy from prescriptions. Same use as above.
FREQ_WORDS = {
    "od", "bd", "tds", "bid", "tid", "qid", "qd", "hs", "sos", "stat",
    "ac", "pc", "bbf", "om", "on", "eod",
}

# Trailing-only formulation qualifiers for gazette ingredient parts
# (data/build_db.py): "Paracetamol Dispersible Tablets" -> "paracetamol".
FORM_TAIL = FORM_WORDS | {
    "dispersible", "oral", "liquid", "retard", "sustained", "release",
    "extended", "immediate", "enteric", "coated", "granules", "pr", "sr",
    "er", "xr", "xl", "cr", "mr", "ds", "forte",
}


def normalize(name: str) -> str:
    """Canonical form: lowercase, de-parenthesised, punctuation-light."""
    s = name.lower().strip()
    s = _PAREN_RE.sub(" ", s)
    s = _NON_ALNUM_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def base_molecule(normalized: str) -> str:
    """Strip trailing salt words, e.g. 'warfarin sodium' -> 'warfarin'."""
    parts = normalized.split()
    while len(parts) > 1 and parts[-1] in _SALT_WORDS:
        parts.pop()
    return " ".join(parts)


def lookup_keys(name: str) -> list[str]:
    """All DB keys to try for a raw ingredient string, in priority order."""
    norm = normalize(name)
    keys = [norm]
    b = base_molecule(norm)
    if b and b != norm:
        keys.append(b)
    alias = ALIASES.get(norm) or ALIASES.get(b)
    if alias and alias not in keys:
        keys.append(alias)
    return keys


def split_combination(name: str) -> list[str]:
    """Split 'A + B + C' style combination strings into raw parts."""
    return [p.strip() for p in name.split("+") if p.strip()]


def compact(s: str) -> str:
    """Space/hyphen-insensitive key: 'acetyl salicylic acid' == 'acetylsalicylic-acid'."""
    return re.sub(r"[\s\-]+", "", s.lower())
