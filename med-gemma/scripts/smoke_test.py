"""Offline smoke test: DB stats + unit tests. Run: uv run python scripts/smoke_test.py."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from medguard.config import DB_PATH


def main() -> int:
    con = sqlite3.connect(DB_PATH)
    print("== medguard.db ==")
    for t in ["brands", "brand_ingredients", "ddi", "banned_fdc", "schedule_h1"]:
        print(f"  {t:20s} {con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]:>10,}")
    print("  prohibited rows:  ",
          con.execute("SELECT COUNT(*) FROM banned_fdc WHERE action='prohibited'").fetchone()[0])
    print("== meta ==")
    for k, v in con.execute("SELECT key, value FROM meta WHERE key != 'manifest'"):
        print(f"  {k}: {v}")
    con.close()
    print("== pytest ==")
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"],
                       cwd=Path(__file__).resolve().parent.parent)
    print("== eval_golden ==")
    r2 = subprocess.run([sys.executable, "scripts/eval_golden.py"],
                        cwd=Path(__file__).resolve().parent.parent)
    return 0 if r.returncode == 0 and r2.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
