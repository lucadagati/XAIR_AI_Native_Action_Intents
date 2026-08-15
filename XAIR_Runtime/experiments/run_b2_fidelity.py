#!/usr/bin/env python3
"""
B2 fidelity check: offline gate model vs live adapter/XAIR.

Replays N cached decisions through both paths with synthetic timestamps so that
wall-clock freshness matches the offline elapsed model (no waiting for Δ_inf).

Usage:
    python3 experiments/run_b2_fidelity.py --tag phase_p --n 500
    # or via: python3 experiments/run_b2_validity_frontier.py --tag phase_p --fidelity 500
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper2_common import (  # noqa: E402
    RESULTS_DIR,
    Notifier,
    apply_drift,
    load_manifest,
    seed_nominal,
    stack_health,
    submit_intent,
)
from experiments.perception_cache import cache_path  # noqa: E402
from experiments.run_b2_validity_frontier import (  # noqa: E402
    ReplayRecord,
    load_replay_records,
    replay_one,
)
from xair.ai.structured_intent import PerceptionResult, build_submission, precondition_syntax_ok  # noqa: E402


PRIMARY = "qwen2.5vl:7b"
GATES = ("direct", "freshness_only", "xair")
ANCHORS = ("capture", "emission")
# Informative operating point (not w=500 where temporal gates vacuous).
DEFAULT_W_MS = 4000
DEFAULT_P_DRIFT = 0.5
DEFAULT_OFFSET_MS = 250


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def load_raw_records(tag: str, frame_ids: set[str], model: str) -> dict[str, dict]:
    path = cache_path(tag)
    latest: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("error"):
            continue
        if rec.get("model") != model or rec.get("prompt_variant") != "blind":
            continue
        fid = rec.get("frame_id")
        if fid in frame_ids:
            latest[fid] = rec
    return latest


def live_published(response: dict) -> bool:
    """Gate admit decision; do not require ROS transport success."""
    if response.get("outcome") == "EXECUTE":
        return True
    if response.get("outcome") == "REVOKE":
        return False
    # Fallback for odd payloads.
    return bool(response.get("ros_published"))


def run_one_live(
    raw: dict,
    episode: dict,
    *,
    gate: str,
    anchor: str,
    freshness_ms: int,
    invalid_at_submit: bool,
) -> dict:
    seed_nominal(episode)
    if invalid_at_submit:
        apply_drift(episode)
    else:
        # Ensure clean nominal after previous drifted trials.
        seed_nominal(episode)

    latency = float(raw.get("latency_ms") or 0.0)
    now = datetime.now(timezone.utc)
    # Align wall-clock elapsed with offline model without sleeping Δ_inf.
    if anchor == "capture":
        capture = now - timedelta(milliseconds=latency)
        emitted = now
    else:
        capture = now - timedelta(milliseconds=latency)
        emitted = now  # decision ts = emission → elapsed ≈ 0

    patched = dict(raw)
    patched["capture_ts"] = _iso(capture)
    patched["emitted_ts"] = _iso(emitted)
    result = PerceptionResult.from_json(patched)
    # Match Phase-G offline ReplayRecord: only syntax-ok exprs enter the gate model.
    filtered = [e for e in (result.preconditions or []) if precondition_syntax_ok(e)]
    intent = build_submission(
        result, anchor=anchor, freshness_ms=freshness_ms, preconditions=filtered
    )
    t0 = time.perf_counter()
    resp = submit_intent(intent, gate)
    dt = (time.perf_counter() - t0) * 1000.0
    published = live_published(resp)
    return {
        "published": published,
        "outcome": resp.get("outcome"),
        "reason": resp.get("reason") or resp.get("validation_reason") or resp.get("detail"),
        "e2e_ms": dt,
        "raw": {k: resp.get(k) for k in ("ok", "outcome", "ros_published", "state", "error")},
    }


def run_fidelity(
    *,
    tag: str,
    n: int,
    freshness_ms: int,
    p_drift: float,
    drift_offset_ms: float,
    seed: int,
    model: str = PRIMARY,
    gates: tuple[str, ...] = GATES,
    anchors: tuple[str, ...] = ANCHORS,
) -> dict:
    ok, msg = stack_health()
    if not ok:
        raise SystemExit(f"stack not healthy: {msg}")

    episodes = {e["frame_id"]: e for e in load_manifest()}
    records = load_replay_records(tag, models={model}, variants={"blind"}, use_repaired=False)
    records = [r for r in records if r.frame_id in episodes]
    if len(records) < n:
        raise SystemExit(f"only {len(records)} {model} blind records; need {n}")

    rng = random.Random(seed)
    sample = rng.sample(records, n)
    raw_by = load_raw_records(tag, {r.frame_id for r in sample}, model)

    rows: list[dict] = []
    disagree = Counter()
    total = Counter()

    for i, rec in enumerate(sample):
        raw = raw_by.get(rec.frame_id)
        if raw is None:
            continue
        episode = episodes[rec.frame_id]
        drift_fires = rng.random() < p_drift
        # Capture-elapsed model: invalid when drift lands at/before submission.
        # For emission offline elapsed=0, so invalid only if offset==0 and fires —
        # but we keep the same plant flag for both anchors (drift applied or not)
        # and let each path's timing semantics differ as in B2.
        invalid_capture = bool(drift_fires and drift_offset_ms <= rec.latency_ms)
        invalid_emission = bool(drift_fires and drift_offset_ms <= 0.0)

        for gate in gates:
            for anchor in anchors:
                invalid = invalid_capture if anchor == "capture" else invalid_emission
                offline = replay_one(
                    rec,
                    gate=gate,
                    anchor=anchor,
                    freshness_ms=freshness_ms,
                    invalid_at_submit=invalid,
                )
                try:
                    live = run_one_live(
                        raw,
                        episode,
                        gate=gate,
                        anchor=anchor,
                        freshness_ms=freshness_ms,
                        invalid_at_submit=invalid,
                    )
                    live_err = None
                except Exception as exc:  # noqa: BLE001 — record and continue
                    live = {"published": None, "outcome": None, "reason": str(exc), "e2e_ms": 0.0}
                    live_err = str(exc)

                agree = live["published"] is not None and bool(live["published"]) == bool(
                    offline["published"]
                )
                key = f"{gate}|{anchor}"
                total[key] += 1
                total["all"] += 1
                if not agree:
                    disagree[key] += 1
                    disagree["all"] += 1

                rows.append(
                    {
                        "i": i,
                        "frame_id": rec.frame_id,
                        "gate": gate,
                        "anchor": anchor,
                        "freshness_ms": freshness_ms,
                        "invalid_at_submit": invalid,
                        "offline_published": offline["published"],
                        "live_published": live["published"],
                        "agree": agree,
                        "offline_reason": offline.get("reason"),
                        "live_outcome": live.get("outcome"),
                        "live_reason": live.get("reason"),
                        "live_error": live_err,
                        "latency_ms": rec.latency_ms,
                    }
                )
        if (i + 1) % 50 == 0:
            rate = disagree["all"] / max(1, total["all"])
            print(f"[fidelity] {i+1}/{n}  disagreement={rate:.3%}", flush=True)

    by_cell = {}
    for key in list(total.keys()):
        if key == "all":
            continue
        d = disagree[key]
        t = total[key]
        by_cell[key] = {
            "n": t,
            "disagreements": d,
            "disagreement_rate": d / t if t else 0.0,
        }

    summary = {
        "n_frames": n,
        "model": model,
        "freshness_ms": freshness_ms,
        "p_drift": p_drift,
        "drift_offset_ms": drift_offset_ms,
        "seed": seed,
        "n_comparisons": total["all"],
        "disagreements": disagree["all"],
        "disagreement_rate": disagree["all"] / total["all"] if total["all"] else 0.0,
        "acceptance_threshold": 0.05,
        "accepted": (disagree["all"] / total["all"] if total["all"] else 1.0) < 0.05,
        "by_gate_anchor": by_cell,
    }
    return {"summary": summary, "rows": rows}


def write_outputs(result: dict, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "b2_fidelity.json"
    csv_path = out_dir / "b2_fidelity.csv"
    json_path.write_text(json.dumps({"summary": result["summary"]}, indent=2))
    rows = result["rows"]
    if rows:
        with csv_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    # Also keep full summary+counts without huge raw dumps
    return {"json": json_path, "csv": csv_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="B2 offline vs live fidelity check")
    parser.add_argument("--tag", default="phase_p")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--freshness-ms", type=int, default=DEFAULT_W_MS)
    parser.add_argument("--p-drift", type=float, default=DEFAULT_P_DRIFT)
    parser.add_argument("--drift-offset-ms", type=float, default=DEFAULT_OFFSET_MS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=PRIMARY)
    parser.add_argument("--no-notify", action="store_true")
    args = parser.parse_args(argv)

    notifier = Notifier.from_env()
    if args.no_notify:
        notifier.enabled = False

    result = run_fidelity(
        tag=args.tag,
        n=args.n,
        freshness_ms=args.freshness_ms,
        p_drift=args.p_drift,
        drift_offset_ms=args.drift_offset_ms,
        seed=args.seed,
        model=args.model,
    )
    paths = write_outputs(result, RESULTS_DIR)
    s = result["summary"]
    print(json.dumps(s, indent=2))
    for k, p in paths.items():
        print(f"[fidelity] {k}: {p}")
    notifier.send(
        f"B2 fidelity: disagreement={s['disagreement_rate']:.2%} "
        f"({s['disagreements']}/{s['n_comparisons']}) accepted={s['accepted']}"
    )
    return 0 if s["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
