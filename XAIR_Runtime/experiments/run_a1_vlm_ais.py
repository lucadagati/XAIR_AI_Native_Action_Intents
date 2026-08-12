#!/usr/bin/env python3
"""
A1 — legacy saturated ablation (superseded by suites B1-B6).

Kept runnable for continuity with Paper 1, but this driver does NOT support any
headline claim: drift is applied unconditionally, the ground truth is asserted rather
than measured, the prompt leaks the target action, and validity is anchored to emission
so inference latency never reaches the freshness check. SER can therefore only take the
values 0 or 1. Use run_b2_validity_frontier.py for the real measurements.

Arms: A1a script replay, A1b raw/direct, A1c AIS+xair, A1d AIS/naive.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.a1_common import (  # noqa: E402
    ARM_MODES,
    DATASET_ROOT,
    OUT_DIR,
    RESULTS_DIR,
    apply_drift_pause,
    audit_count,
    load_manifest,
    post_adapter,
    score_row,
    seed_context,
)
from xair.ai.ollama_client import OllamaClient  # noqa: E402
from xair.ai.structured_intent import StructuredIntentProducer  # noqa: E402


def build_script_resume_intent(seed: int, run_idx: int, freshness_ms: int) -> dict:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    return {
        "id": str(uuid.uuid4()),
        "source": "ai",
        "timestamp_decision": ts,
        "freshness_window_ms": freshness_ms,
        "deadline_ms": 800,
        "preconditions": [
            {"expr": "line.state == 'RUN'"},
            {"expr": "gripper.state == 'OPEN'"},
        ],
        "payload": {
            "action_type": "RESUME",
            "target_entity": "robot_3",
            "parameters": {"reason": "e1_replay", "seed": seed, "run": run_idx},
        },
        "priority": 5,
    }


def run_single(
    run_idx: int,
    arm: str,
    episode: dict,
    pause_ms: float,
    freshness_ms: int,
    seed: int,
    producer: StructuredIntentProducer,
    *,
    model: str | None = None,
    post_delay_ms: float = 0,
) -> dict:
    seed_context(episode)
    time.sleep(pause_ms / 1000.0)
    apply_drift_pause()
    time.sleep(0.05)

    inference_latency_ms = 0.0
    schema_valid = True
    mode = ARM_MODES[arm]

    if arm == "A1a":
        intent = build_script_resume_intent(seed, run_idx, freshness_ms)
    else:
        img = DATASET_ROOT / episode.get("path", "")
        out, oresp, schema_valid = producer.produce_ais_legacy(
            episode, arm=arm, image_path=img, model=model
        )
        inference_latency_ms = oresp.latency_ms
        if arm == "A1b":
            intent = producer.raw_to_legacy_intent(out, episode)
        else:
            intent = out
            intent["freshness_window_ms"] = freshness_ms

    if post_delay_ms > 0:
        time.sleep(post_delay_ms / 1000.0)

    submit_mode = "ai_proposed" if arm == "A1c" else mode
    audit_before = audit_count()
    resp = post_adapter("intent", intent, mode=submit_mode)
    time.sleep(0.05)
    audit_after = audit_count()

    extra = {
        "frame_id": episode.get("frame_id"),
        "source_dataset": episode.get("source_dataset"),
        "defect_present": episode.get("defect_present"),
        "ground_truth_action": episode.get("ground_truth_action"),
        "schema_valid": schema_valid,
        "inference_latency_ms": inference_latency_ms,
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    return score_row(
        run_idx, arm, submit_mode if arm == "A1c" else mode,
        seed, pause_ms, freshness_ms, resp, audit_before, audit_after, extra,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="A1 VLM/AIS grounding experiment")
    parser.add_argument("--arm", choices=list(ARM_MODES.keys()), default="A1c")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pause-ms", type=float, default=400)
    parser.add_argument("--freshness-ms", type=int, default=500)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Single live run, print JSON, no CSV append")
    parser.add_argument("--manifest", type=Path, default=DATASET_ROOT / "manifest.jsonl")
    parser.add_argument("--out-csv", type=Path, default=RESULTS_DIR / "a1_baselines.csv")
    args = parser.parse_args()

    episodes = load_manifest(args.manifest)
    if not episodes:
        print(f"No manifest at {args.manifest}; run build_manifest.py first", file=sys.stderr)
        return 1

    if args.arm != "A1a":
        client = OllamaClient()
        if not client.health():
            print(
                f"Ollama unreachable at {client.host}. Set OLLAMA_HOST and ensure the L40 node is up.",
                file=sys.stderr,
            )
            return 1

    producer = StructuredIntentProducer()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    n = 1 if args.dry_run else args.runs
    for i in range(n):
        ep = episodes[i % len(episodes)]
        row = run_single(
            i, args.arm, ep, args.pause_ms, args.freshness_ms, args.seed,
            producer, model=args.model,
        )
        if not args.dry_run:
            path = OUT_DIR / f"a1_run_{i:03d}_{args.arm}.json"
            path.write_text(json.dumps(row, indent=2))
        rows.append({k: row[k] for k in (
            "scenario", "arm", "run", "baseline", "seed", "outcome", "ros_published",
            "stale_executed", "obsolete_intent", "correct_revoke", "schema_valid",
            "inference_latency_ms", "validation_latency_ms", "e2e_latency_ms",
        ) if k in row})

    if args.dry_run:
        print(json.dumps(row, indent=2))
        return 0

    with args.out_csv.open("a", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        if fp.tell() == 0:
            w.writeheader()
        w.writerows(rows)

    attempted = len(rows)
    stale = sum(1 for r in rows if r.get("stale_executed"))
    valid = sum(1 for r in rows if r.get("schema_valid"))
    summary = {
        "arm": args.arm,
        "attempted": attempted,
        "SER": stale / attempted if attempted else 0,
        "schema_validity_rate": valid / attempted if attempted else 0,
        "inference_p50_ms": sorted(r["inference_latency_ms"] for r in rows)[len(rows) // 2] if rows else 0,
    }
    summary_path = RESULTS_DIR / f"a1_{args.arm}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"summary": summary, "csv": str(args.out_csv)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
