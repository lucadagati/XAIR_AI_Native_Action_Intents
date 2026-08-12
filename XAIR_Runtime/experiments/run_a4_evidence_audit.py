#!/usr/bin/env python3
"""A4 — Evidence audit replay (Paper 2, RQ-A4)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.a1_common import OUT_DIR, RESULTS_DIR  # noqa: E402

AUDIT_JSONL = RESULTS_DIR / "audit" / "ai_intents.jsonl"


def load_revoked_a1_runs() -> list[dict]:
    rows = []
    for p in sorted(OUT_DIR.glob("a1_run_*_A1c.json")):
        data = json.loads(p.read_text())
        if data.get("outcome") == "REVOKE" or data.get("response", {}).get("outcome") == "REVOKE":
            rows.append(data)
    return rows


def traceability(entry: dict) -> bool:
    resp = entry.get("response", {})
    ext_evidence = entry.get("frame_id")
    has_frame = bool(ext_evidence)
    has_reason = bool(resp.get("reason"))
    return has_frame and has_reason


def main() -> int:
    parser = argparse.ArgumentParser(description="A4 evidence audit")
    parser.add_argument("--sample", type=int, default=20)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "a4_evidence_audit.json")
    args = parser.parse_args()

    revoked = load_revoked_a1_runs()
    audit_lines = []
    if AUDIT_JSONL.is_file():
        audit_lines = [json.loads(l) for l in AUDIT_JSONL.read_text().splitlines() if l.strip()]

    sample = revoked[: args.sample]
    traced = sum(1 for r in sample if traceability(r))
    report = {
        "revoked_total": len(revoked),
        "sample_n": len(sample),
        "traceability_rate": traced / len(sample) if sample else 0.0,
        "audit_log_entries": len(audit_lines),
        "samples": [
            {
                "run": r.get("run"),
                "frame_id": r.get("frame_id"),
                "reason": r.get("response", {}).get("reason"),
                "traceable": traceability(r),
            }
            for r in sample
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
