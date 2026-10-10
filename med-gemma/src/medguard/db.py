"""SQLite access layer. All SQL lives here; rules modules call these helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .config import DB_PATH


class MedGuardDB:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else DB_PATH
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found. Run `uv run python data/build_db.py` first.")
        self.con = sqlite3.connect(self.path)
        self.con.row_factory = sqlite3.Row
        self._prohibited: list[sqlite3.Row] | None = None
        self._restricted: list[sqlite3.Row] | None = None

    # ---- brands ----
    def exact_brand(self, norm: str, limit: int = 10) -> list[sqlite3.Row]:
        # Exact hit OR the query is the brand stem ("telma-am" -> "telma-am tablet").
        # Form words live in the DB norm, so a trailing-word match is still exact.
        like = norm.replace("%", "\\%").replace("_", "\\_") + " %"
        return self.con.execute(
            "SELECT * FROM brands WHERE brand_norm=? OR brand_norm LIKE ? ESCAPE '\\' "
            "ORDER BY is_discontinued LIMIT ?",
            (norm, like, limit)).fetchall()

    def prefix_brands(self, prefix: str, limit: int = 300) -> list[sqlite3.Row]:
        return self.con.execute(
            "SELECT * FROM brands WHERE brand_norm LIKE ? ESCAPE '\\' ORDER BY is_discontinued LIMIT ?",
            (prefix.replace("%", "\\%").replace("_", "\\_") + "%", limit)).fetchall()

    def contains_brands(self, token: str, limit: int = 300) -> list[sqlite3.Row]:
        like = "%" + token.replace("%", "\\%").replace("_", "\\_") + "%"
        return self.con.execute(
            "SELECT * FROM brands WHERE brand_norm LIKE ? ESCAPE '\\' ORDER BY is_discontinued LIMIT ?",
            (like, limit)).fetchall()

    def brand_ingredients(self, product_id: str) -> list[sqlite3.Row]:
        return self.con.execute(
            "SELECT * FROM brand_ingredients WHERE product_id=? ORDER BY seq",
            (product_id,)).fetchall()

    # ---- generic-ingredient path ----
    def ingredient_exists(self, norm: str) -> bool:
        return self.con.execute(
            "SELECT 1 FROM brand_ingredients WHERE ingredient_norm=? LIMIT 1",
            (norm,)).fetchone() is not None

    # ---- DDI ----
    def ddi_lookup(self, norm_a: str, norm_b: str) -> list[sqlite3.Row]:
        """Exact normalized pair lookup (directed rows stored both ways)."""
        return self.con.execute(
            "SELECT name_a, name_b, level FROM ddi WHERE norm_a=? AND norm_b=?",
            (norm_a, norm_b)).fetchall()

    def ddi_lookup_base(self, base_a: str, base_b: str) -> list[sqlite3.Row]:
        return self.con.execute(
            "SELECT name_a, name_b, level FROM ddi WHERE base_a=? AND base_b=?",
            (base_a, base_b)).fetchall()

    # ---- banned / restricted ----
    def prohibited_rows(self) -> list[sqlite3.Row]:
        if self._prohibited is None:
            self._prohibited = self.con.execute(
                "SELECT * FROM banned_fdc WHERE action='prohibited'").fetchall()
        return self._prohibited

    def restricted_rows(self) -> list[sqlite3.Row]:
        if self._restricted is None:
            self._restricted = self.con.execute(
                "SELECT * FROM banned_fdc WHERE action='restricted'").fetchall()
        return self._restricted

    # ---- schedule ----
    def schedule_h1(self, norm: str) -> list[sqlite3.Row]:
        return self.con.execute(
            "SELECT * FROM schedule_h1 WHERE name_norm=?", (norm,)).fetchall()

    def close(self) -> None:
        self.con.close()
