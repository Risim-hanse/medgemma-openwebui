"""Optional online evidence: pull FDA label 'drug interactions' text (openFDA).

Used ONLY to attach citable label excerpts to findings the local engine
already produced. Never used to discover interactions (labels are prose).
All calls are cached under data/cache/ and degrade gracefully offline.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import requests

from ..config import CACHE_DIR, OPENFDA_LABEL_URL


def _cache_path(kind: str, query: str) -> Path:
    h = hashlib.sha256(query.encode()).hexdigest()[:16]
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in query)[:40]
    return CACHE_DIR / kind / f"{safe}_{h}.json"


def fetch_label_snippets(generic: str, other_terms: list[str], limit: int = 2) -> list[str]:
    """Return up to `limit` label excerpts whose interactions section names a term."""
    q = f'openfda.generic_name:"{generic}" AND drug_interactions:*'
    cp = _cache_path("openfda", q)
    if cp.exists():
        data = json.loads(cp.read_text())
    else:
        try:
            r = requests.get(OPENFDA_LABEL_URL,
                             params={"search": q, "limit": 3}, timeout=20)
            if r.status_code != 200:
                return []
            data = r.json()
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_text(json.dumps(data)[:200_000])
        except requests.RequestException:
            return []
    out: list[str] = []
    for res in data.get("results", []):
        for section in res.get("drug_interactions", []):
            low = section.lower()
            for term in other_terms:
                if term.lower() in low and len(term) > 3:
                    idx = low.find(term.lower())
                    out.append("... " + section[max(0, idx - 300): idx + 500].strip() + " ...")
                    break
            if len(out) >= limit:
                return out
    return out
