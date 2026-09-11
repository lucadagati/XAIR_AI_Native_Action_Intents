#!/usr/bin/env python3
"""
B2 precondition-repair sensitivity check.

Compares mean capture-anchored XAIR utility between the as-emitted blind
decisions and their syntax-repaired counterparts (ReplayRecord's
use_repaired=True path, i.e. preconditions_repaired / schema_valid_repaired
from the Phase-P cache), across the full 168-cell freshness x p_drift x
drift_offset grid (7 x 4 x 6), all five VLMs, five drift seeds, xair gate,
capture anchoring -- the same grid Phase G / B2 uses elsewhere.

This replaces an earlier one-off analysis whose generating script was lost;
this version is permanent and re-runnable via:
    python3 experiments/run_b2_repaired_sensitivity.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper2_common import RESULTS_DIR, trial_reward  # noqa: E402
from experiments.run_b2_validity_frontier import (  # noqa: E402
    DRIFT_OFFSET_GRID_MS,
    FRESHNESS_GRID_MS,
    P_DRIFT_GRID,
    SEEDS,
    load_replay_records,
    replay_one,
)


def mean_utility(records, *, freshness_ms: int, p_drift: float, drift_offset_ms: float) -> float:
    vals: list[float] = []
    for seed in SEEDS:
        rng = random.Random(f"{seed}:{p_drift}:{drift_offset_ms}")
        for rec in records:
            invalid = rng.random() < p_drift and drift_offset_ms <= rec.latency_ms
            scored = replay_one(
                rec,
                gate="xair",
                anchor="capture",
                freshness_ms=freshness_ms,
                invalid_at_submit=invalid,
            )
            if scored.get("unknown"):
                continue
            vals.append(trial_reward(scored))
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> int:
    unrepaired = load_replay_records("phase_p", models=None, variants={"blind"}, use_repaired=False)
    repaired = load_replay_records("phase_p", models=None, variants={"blind"}, use_repaired=True)
    assert len(unrepaired) == len(repaired), "repaired/unrepaired record counts must match"

    deltas: list[float] = []
    for w in FRESHNESS_GRID_MS:
        for p in P_DRIFT_GRID:
            for off in DRIFT_OFFSET_GRID_MS:
                u_un = mean_utility(unrepaired, freshness_ms=w, p_drift=p, drift_offset_ms=off)
                u_re = mean_utility(repaired, freshness_ms=w, p_drift=p, drift_offset_ms=off)
                deltas.append(u_re - u_un)

    n_diff_schema = sum(
        1 for a, b in zip(unrepaired, repaired) if a.gate_schema_valid != b.gate_schema_valid
    )

    payload = {
        "n_records": len(unrepaired),
        "n_cells": len(deltas),
        "n_records_with_changed_gate_schema_valid": n_diff_schema,
        "mean_delta_U_repaired_minus_unrepaired": sum(deltas) / len(deltas),
        "min_delta_U": min(deltas),
        "max_delta_U": max(deltas),
        "note": (
            "delta = mean_reward(repaired) - mean_reward(as-emitted), averaged over the "
            "168-cell freshness x p_drift x drift_offset grid, xair gate, capture anchor, "
            "all five VLMs, five drift seeds, known (non-unknown) trials only."
        ),
    }
    out = RESULTS_DIR / "b2_repaired_comparison.json"
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"[b2-repaired-sensitivity] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
