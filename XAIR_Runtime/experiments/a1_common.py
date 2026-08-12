"""
Legacy harness for the A1-A4 ablation. Superseded by paper2_common.py.

Two properties make this harness unsuitable for headline results and are preserved
only so the ablation stays reproducible: apply_drift_pause() is called unconditionally
by its callers, and score_row() asserts ``context_invalid_at_te = True`` instead of
measuring it. Together they force every trial to be obsolete, so SER is confined to
0 or 1 regardless of what the producer does.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

ADAPTER = "http://127.0.0.1:9092"
XAIR = "http://127.0.0.1:8080"
DATASET_ROOT = Path(__file__).resolve().parent / "datasets" / "manufacturing-a1"
OUT_DIR = Path(__file__).resolve().parents[1].parent / "AdaptiX-Quest" / "TestResults"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
AUDIT_FILE = RESULTS_DIR / "ros_audit_state.json"

ARM_MODES = {
    "A1a": "xair",
    "A1b": "direct",
    "A1c": "xair",
    "A1d": "naive",
}


def load_manifest(path: Path | None = None) -> list[dict]:
    manifest = path or (DATASET_ROOT / "manifest.jsonl")
    if not manifest.is_file():
        return []
    rows = []
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def get_xair_context() -> dict:
    with urllib.request.urlopen(f"{XAIR}/v1/context/snapshot", timeout=5) as r:
        return json.loads(r.read().decode()).get("context", {})


def post_adapter(path: str, body: dict, mode: str | None = None) -> dict:
    url = f"{ADAPTER}/{path}"
    if mode and path in ("intent", "command"):
        url = f"{url}?mode={mode}"
    t0 = time.perf_counter()
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        out = json.loads(r.read().decode())
    out["e2e_latency_ms"] = (time.perf_counter() - t0) * 1000.0
    return out


def audit_count() -> int | None:
    if not AUDIT_FILE.exists():
        return None
    try:
        data = json.loads(AUDIT_FILE.read_text())
        return int(data.get("pose_count", 0))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def seed_context(episode: dict) -> None:
    ctx = episode.get("context") or {
        "line": {"state": "RUN"},
        "robot": {"speed": 0.05},
        "gripper": {"state": "OPEN"},
        "defect": {"absent": not episode.get("defect_present", False)},
    }
    post_adapter("context", ctx)


def apply_drift_pause() -> None:
    post_adapter("context", {"line": {"state": "PAUSED"}, "gripper": {"state": "CLOSED"}})
    for _ in range(5):
        ctx = get_xair_context()
        if ctx.get("line", {}).get("state") == "PAUSED":
            break
        time.sleep(0.02)
        post_adapter("context", {"line": {"state": "PAUSED"}, "gripper": {"state": "CLOSED"}})


def score_row(
    run_idx: int,
    arm: str,
    baseline: str,
    seed: int,
    pause_ms: float,
    freshness_ms: int,
    resp: dict,
    audit_before: int | None,
    audit_after: int | None,
    extra: dict | None = None,
) -> dict:
    # Asserted, not measured: the reason this harness saturates. See paper2_common.
    context_invalid_at_te = True
    outcome = resp.get("outcome", "UNKNOWN")
    ros_published = bool(resp.get("ros_published"))
    ros_observed = (
        audit_after - audit_before > 0
        if audit_before is not None and audit_after is not None
        else None
    )
    stale_executed = ros_published and context_invalid_at_te
    stale_observed = bool(ros_observed) and context_invalid_at_te and outcome != "UNKNOWN"
    obsolete = context_invalid_at_te and outcome != "UNKNOWN"
    correct_revoke = obsolete and outcome == "REVOKE" and not resp.get("ros_published")
    row = {
        "scenario": "a1",
        "arm": arm,
        "run": run_idx,
        "baseline": baseline,
        "seed": seed,
        "pause_ms": pause_ms,
        "freshness_ms": freshness_ms,
        "pause_order": "before_intent",
        "response": resp,
        "context_invalid_at_te": context_invalid_at_te,
        "ros_published": ros_published,
        "ros_observed": ros_observed,
        "stale_executed": stale_executed,
        "stale_observed": stale_observed,
        "witness_agreement": (ros_observed is None) or (ros_observed == ros_published),
        "obsolete_intent": obsolete,
        "correct_revoke": correct_revoke,
        "outcome": outcome,
        "validation_latency_ms": float(resp.get("validation_latency_ms") or 0),
        "e2e_latency_ms": float(resp.get("e2e_latency_ms") or 0),
    }
    if extra:
        row.update(extra)
    return row
