#!/usr/bin/env python3
"""A2 — Inference latency sweep vs SER (Paper 2, RQ-A2). Requires live Ollama."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_a1_vlm_ais import run_single  # noqa: E402
from experiments.a1_common import DATASET_ROOT, RESULTS_DIR, load_manifest  # noqa: E402
from xair.ai.ollama_client import OllamaClient  # noqa: E402
from xair.ai.structured_intent import StructuredIntentProducer  # noqa: E402

DELAYS_MS = [0, 50, 100, 200, 350, 500]
MODELS = ["qwen2.5-coder:7b", "qwen3-coder:30b", "qwen2.5vl:7b"]


def main() -> int:
    parser = argparse.ArgumentParser(description="A2 latency sweep")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delays", nargs="+", type=float, default=DELAYS_MS)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--out-csv", type=Path, default=RESULTS_DIR / "a2_latency_sweep.csv")
    args = parser.parse_args()

    episodes = load_manifest(DATASET_ROOT / "manifest.jsonl")
    if not episodes:
        print(f"No manifest at {DATASET_ROOT / 'manifest.jsonl'}", file=sys.stderr)
        return 1

    client = OllamaClient()
    if not client.health():
        print(f"Ollama unreachable at {client.host}", file=sys.stderr)
        return 1

    producer = StructuredIntentProducer(client)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for model in args.models:
        for delay in args.delays:
            for mode_label, arm in (("xair", "A1c"), ("direct", "A1b")):
                stale = 0
                lats = []
                for i in range(args.runs):
                    ep = episodes[i % len(episodes)]
                    row = run_single(
                        i, arm, ep, 400, 500, args.seed, producer,
                        model=model, post_delay_ms=delay,
                    )
                    if row.get("stale_executed"):
                        stale += 1
                    lats.append(row.get("inference_latency_ms", 0) + delay)
                rows.append({
                    "model": model,
                    "post_delay_ms": delay,
                    "gate": mode_label,
                    "arm": arm,
                    "runs": args.runs,
                    "SER": stale / args.runs,
                    "delta_v_ms_p50": sorted(lats)[len(lats) // 2] if lats else delay,
                })

    with args.out_csv.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(json.dumps({"out": str(args.out_csv), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
