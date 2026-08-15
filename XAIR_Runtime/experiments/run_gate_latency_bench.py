#!/usr/bin/env python3
"""Bench HTTP gate latency for AIS intents through the adapter → XAIR path."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper2_common import ADAPTER, RESULTS_DIR, XAIR, _get, _post, stack_health  # noqa: E402


def intent_body() -> dict:
    return {
        "id": str(uuid.uuid4()),
        "source": "ai",
        "timestamp_decision": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "freshness_window_ms": 5000,
        "preconditions": [{"expr": "line.state == 'RUN'"}],
        "safety_constraints": [{"expr": "gripper.state == 'OPEN'"}],
        "payload": {"action_type": "RESUME", "target_entity": "line_1"},
        "extensions": {
            "model_id": "bench",
            "inference_latency_ms": 0,
            "evidence": {"frame_id": "gate-latency-bench"},
        },
    }


def pct(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate latency bench")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--mode", default="ai_proposed")
    args = parser.parse_args()

    ok, msg = stack_health()
    if not ok:
        print(f"[gate-latency] stack unavailable: {msg}", file=sys.stderr)
        return 1

    _post(f"{ADAPTER}/context", {"line": {"state": "RUN"}, "gripper": {"state": "OPEN"}})
    try:
        _post(f"{XAIR}/v1/context/snapshot", {"line": {"state": "RUN"}, "gripper": {"state": "OPEN"}})
    except Exception:
        pass

    url = f"{ADAPTER}/intent?mode={args.mode}"
    for _ in range(args.warmup):
        _post(url, intent_body())

    samples: list[float] = []
    for _ in range(args.n):
        body = intent_body()
        t0 = time.perf_counter()
        _post(url, body)
        samples.append((time.perf_counter() - t0) * 1000.0)

    payload = {
        "n": len(samples),
        "warmup": args.warmup,
        "mode": args.mode,
        "adapter": ADAPTER,
        "xair": XAIR,
        "host": platform.node(),
        "platform": platform.platform(),
        "latency_ms": {
            "mean": statistics.fmean(samples),
            "p50": pct(samples, 0.50),
            "p95": pct(samples, 0.95),
            "min": min(samples),
            "max": max(samples),
        },
        "note": "HTTP round-trip adapter→XAIR on development host (not L40).",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "gate_latency.json"
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"[gate-latency] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
