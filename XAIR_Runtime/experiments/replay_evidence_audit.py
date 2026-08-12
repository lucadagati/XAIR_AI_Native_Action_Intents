#!/usr/bin/env python3
"""Replay evidence audit trail from ai_intents.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
AUDIT_JSONL = RESULTS_DIR / "audit" / "ai_intents.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intent-id", type=str, default=None)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    if not AUDIT_JSONL.is_file():
        print(f"No audit log at {AUDIT_JSONL}")
        return 1

    rows = []
    for line in AUDIT_JSONL.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if args.intent_id and rec.get("intent_id") != args.intent_id:
            continue
        rows.append(rec)
        if len(rows) >= args.limit:
            break

    for r in rows:
        print(json.dumps(r, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
