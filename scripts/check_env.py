#!/usr/bin/env python
"""Quick sanity check before the first real run: verifies the heavy ML
imports actually work (the Python-3.14/torch-wheel risk called out in the
README) and that Postgres is reachable. Run this before `python cli.py run`
if you just set up the venv.

    python scripts/check_env.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Run as `python scripts/check_env.py`, so sys.path[0] is scripts/, not the
# project root -- add the root so `import modelscout` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    ok = True

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import sentence_transformers  # noqa: F401

        print("[OK] torch / transformers / sentence-transformers import correctly")
    except ImportError as exc:
        ok = False
        print(f"[FAIL] ML imports failed: {exc}")
        print("       If this is a 'no matching distribution' error for torch, your Python")
        print("       version is likely too new for available wheels -- see README.md's")
        print("       Python 3.12 fallback instructions.")

    try:
        from modelscout.db import get_connection

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        print("[OK] Postgres connection succeeded")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"[FAIL] Could not connect to Postgres: {exc}")
        print("       Is Docker Desktop running? Did you run `docker compose up -d`?")

    try:
        from modelscout.config import settings

        if settings.anthropic_api_key and not settings.anthropic_api_key.startswith("sk-ant-..."):
            print("[OK] ANTHROPIC_API_KEY is set in .env")
        else:
            ok = False
            print("[FAIL] ANTHROPIC_API_KEY looks unset or still the placeholder value in .env")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"[FAIL] Could not load settings: {exc}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
