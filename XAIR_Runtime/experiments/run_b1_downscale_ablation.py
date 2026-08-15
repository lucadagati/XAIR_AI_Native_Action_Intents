#!/usr/bin/env python3
"""
B1 downscale ablation: grounding vs longest-side encode on a stratified subsample.

Usage:
    export OLLAMA_HOST=http://...
    python3 experiments/run_b1_downscale_ablation.py --tag phase_p --n 200
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper2_common import DATASET_ROOT, RESULTS_DIR, load_manifest  # noqa: E402
from xair.ai.ollama_client import OllamaClient  # noqa: E402
from xair.ai.structured_intent import StructuredIntentProducer  # noqa: E402

SIDES = (512, 1024, 2048)
MODEL = "qwen2.5vl:7b"


def stratified_sample(episodes: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_sev: dict[str, list[dict]] = defaultdict(list)
    for e in episodes:
        by_sev[str(e.get("severity") or "none")].append(e)
    # Round-robin across severity buckets.
    for v in by_sev.values():
        rng.shuffle(v)
    keys = sorted(by_sev)
    out: list[dict] = []
    i = 0
    while len(out) < n and any(by_sev[k] for k in keys):
        k = keys[i % len(keys)]
        if by_sev[k]:
            out.append(by_sev[k].pop())
        i += 1
    return out[:n]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--sides", nargs="+", type=int, default=list(SIDES))
    args = parser.parse_args()

    os.environ.setdefault("OLLAMA_HOST", os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    client = OllamaClient(timeout_s=120)
    if not client.health():
        print(f"Ollama unreachable at {client.host}", file=sys.stderr)
        return 1
    producer = StructuredIntentProducer(client)
    episodes = load_manifest()
    sample = stratified_sample(episodes, args.n, args.seed)
    print(f"[downscale] n={len(sample)} model={args.model} sides={args.sides} host={client.host}", flush=True)

    rows = []
    for side in args.sides:
        os.environ["OLLAMA_IMAGE_MAX_SIDE"] = str(side)
        correct = 0
        lats = []
        errors = 0
        for i, ep in enumerate(sample):
            img = DATASET_ROOT / ep.get("path", "")
            t0 = time.perf_counter()
            try:
                result = producer.produce(
                    ep,
                    image_path=img,
                    model=args.model,
                    variant="blind",
                    use_case=ep.get("use_case", "uc1_triage"),
                )
                latency = (time.perf_counter() - t0) * 1000.0
                ok = bool(result.action) and result.action == ep.get("ground_truth_action")
                correct += int(ok)
                lats.append(latency)
                rows.append(
                    {
                        "frame_id": ep["frame_id"],
                        "max_side": side,
                        "grounding_correct": ok,
                        "action": result.action,
                        "gt_action": ep.get("ground_truth_action"),
                        "latency_ms": latency,
                        "error": None,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                errors += 1
                rows.append(
                    {
                        "frame_id": ep["frame_id"],
                        "max_side": side,
                        "grounding_correct": False,
                        "action": None,
                        "gt_action": ep.get("ground_truth_action"),
                        "latency_ms": (time.perf_counter() - t0) * 1000.0,
                        "error": str(exc),
                    }
                )
            if (i + 1) % 25 == 0:
                print(f"[downscale] side={side} {i+1}/{len(sample)}", flush=True)
        lats_ok = sorted(lats)
        p50 = lats_ok[len(lats_ok) // 2] if lats_ok else 0.0
        print(
            f"[downscale] side={side} grounding={correct}/{len(sample)} "
            f"p50_lat={p50:.0f}ms errors={errors}",
            flush=True,
        )
        try:
            client.unload(args.model)
        except Exception:
            pass

    # Aggregate
    summary = {"n_frames": len(sample), "model": args.model, "by_side": {}}
    for side in args.sides:
        sub = [r for r in rows if r["max_side"] == side and not r.get("error")]
        all_side = [r for r in rows if r["max_side"] == side]
        g = sum(1 for r in sub if r["grounding_correct"])
        lats = sorted(r["latency_ms"] for r in sub)
        summary["by_side"][str(side)] = {
            "n_ok": len(sub),
            "n_total": len(all_side),
            "grounding": g / len(sub) if sub else 0.0,
            "grounding_k": g,
            "latency_p50_ms": lats[len(lats) // 2] if lats else 0.0,
            "errors": sum(1 for r in all_side if r.get("error")),
        }

    out = RESULTS_DIR / "b1_downscale_ablation.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps({"out": str(out), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
