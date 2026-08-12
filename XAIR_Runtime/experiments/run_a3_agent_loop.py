#!/usr/bin/env python3
"""
A3 — legacy degenerate agent arm (superseded by run_b5_agent_policy.py).

This driver submits two hardcoded RESUME intents with a pause in between. It has no
tool loop and no decision point, so it cannot discriminate agent behaviour from
single-shot behaviour. Retained only for provenance.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.a1_common import DATASET_ROOT, RESULTS_DIR, load_manifest, post_adapter, seed_context  # noqa: E402
from experiments.run_a1_vlm_ais import run_single  # noqa: E402
from xair.ai.ollama_client import OllamaClient  # noqa: E402
from xair.ai.structured_intent import StructuredIntentProducer  # noqa: E402


def agent_episode(episode: dict, producer: StructuredIntentProducer, *, model: str | None) -> dict:
    """Multi-step: observe → propose RESUME → pause line → propose again."""
    seed_context(episode)
    steps = []
    img = DATASET_ROOT / episode.get("path", "")

    intent1, _, _ = producer.produce_ais_legacy(episode, arm="A1c", image_path=img, model=model)
    intent1["freshness_window_ms"] = 500
    r1 = post_adapter("intent", intent1, mode="ai_proposed")
    steps.append({"step": 1, "outcome": r1.get("outcome"), "ros_published": r1.get("ros_published")})

    time.sleep(0.1)
    post_adapter("context", {"line": {"state": "PAUSED"}, "gripper": {"state": "CLOSED"}})
    time.sleep(0.05)

    intent2, _, _ = producer.produce_ais_legacy(episode, arm="A1c", image_path=img, model=model)
    intent2["freshness_window_ms"] = 500
    r2 = post_adapter("intent", intent2, mode="ai_proposed")
    steps.append({"step": 2, "outcome": r2.get("outcome"), "ros_published": r2.get("ros_published")})

    stale_attempts = sum(1 for s in steps if s.get("ros_published"))
    first_revoke = next((s["step"] for s in steps if s.get("outcome") == "REVOKE"), None)
    return {
        "mode": "agent_multi_step",
        "frame_id": episode.get("frame_id"),
        "steps": steps,
        "stale_attempts": stale_attempts,
        "steps_until_revoke": first_revoke,
        "SER": stale_attempts / len(steps),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="A3 agent multi-step experiment")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--out-csv", type=Path, default=RESULTS_DIR / "a3_agent_loop.csv")
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

    for i in range(args.runs):
        ep = episodes[i % len(episodes)]
        agent_row = agent_episode(ep, producer, model=args.model)
        single = run_single(i, "A1c", ep, 400, 500, 42, producer, model=args.model)
        rows.append({
            "run": i,
            "frame_id": ep.get("frame_id"),
            "agent_stale_attempts": agent_row["stale_attempts"],
            "agent_SER": agent_row["SER"],
            "single_shot_SER": 1 if single.get("stale_executed") else 0,
            "steps_until_revoke": agent_row.get("steps_until_revoke"),
        })

    with args.out_csv.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(json.dumps({
        "out": str(args.out_csv),
        "agent_mean_SER": sum(r["agent_SER"] for r in rows) / len(rows),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
