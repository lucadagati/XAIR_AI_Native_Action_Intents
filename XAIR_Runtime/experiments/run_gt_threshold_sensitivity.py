#!/usr/bin/env python3
"""
CPU sensitivity of mask-derived industrial GT: shift the within-category area
cut from the median to the 40th/60th percentiles and report how many frames
flip their assigned ground-truth action.

Usage:
    python3 experiments/run_gt_threshold_sensitivity.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper2_common import RESULTS_DIR, load_manifest  # noqa: E402

GT_ACTIONS: dict[str, dict[str, str]] = {
    "uc1_triage": {"none": "ACCEPT", "minor": "HOLD_FOR_OPERATOR", "major": "REJECT_TO_BIN"},
    "uc2_restart": {"none": "RESUME", "minor": "STOP", "major": "STOP"},
    "uc3_speed": {"none": "ACCEPT", "minor": "SLOW_DOWN", "major": "STOP"},
    "uc4_conflict": {"none": "RESUME", "minor": "STOP", "major": "STOP"},
    "uc5_safety": {"none": "ACCEPT", "minor": "HOLD_FOR_OPERATOR", "major": "E_STOP"},
}


def percentile(xs: list[float], q: float) -> float:
    ys = sorted(xs)
    if not ys:
        raise ValueError("empty")
    if len(ys) == 1:
        return ys[0]
    idx = (len(ys) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(ys) - 1)
    frac = idx - lo
    return ys[lo] * (1.0 - frac) + ys[hi] * frac


def grade(area: float, cut: float | None, defect_present: bool) -> str:
    if not defect_present:
        return "none"
    if area <= 0 or cut is None:
        return "minor"
    return "major" if area >= cut else "minor"


def main() -> int:
    rows = load_manifest()
    if not rows:
        print("[gt-sens] no manifest", file=sys.stderr)
        return 1

    by_cat: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("defect_present") and float(r.get("defect_area") or 0) > 0:
            by_cat[str(r["category"])].append(float(r["defect_area"]))

    cuts = {}
    for cat, areas in by_cat.items():
        cuts[cat] = {
            "p40": percentile(areas, 0.40),
            "p50": percentile(areas, 0.50),
            "p60": percentile(areas, 0.60),
            "n": len(areas),
        }

    n = len(rows)
    flips = {"p40": 0, "p60": 0}
    sev_flips = {"p40": 0, "p60": 0}
    by_uc: dict[str, dict[str, int]] = defaultdict(lambda: {"p40": 0, "p60": 0, "n": 0})

    for r in rows:
        uc = str(r.get("use_case") or "uc1_triage")
        area = float(r.get("defect_area") or 0)
        present = bool(r.get("defect_present"))
        cat = str(r.get("category"))
        cutset = cuts.get(cat)
        base_gt = r.get("ground_truth_action") or GT_ACTIONS[uc][
            grade(area, cutset["p50"] if cutset else None, present)
        ]
        by_uc[uc]["n"] += 1
        for q in ("p40", "p60"):
            alt_sev = grade(area, cutset[q] if cutset else None, present)
            alt_gt = GT_ACTIONS[uc][alt_sev]
            base_sev = grade(area, cutset["p50"] if cutset else None, present)
            if alt_sev != base_sev:
                sev_flips[q] += 1
            if alt_gt != base_gt:
                flips[q] += 1
                by_uc[uc][q] += 1

    out = {
        "n_frames": n,
        "n_defective_with_area": sum(c["n"] for c in cuts.values()),
        "action_flip_fraction": {q: flips[q] / n for q in ("p40", "p60")},
        "action_flip_count": flips,
        "severity_flip_fraction": {q: sev_flips[q] / n for q in ("p40", "p60")},
        "severity_flip_count": sev_flips,
        "by_use_case": {
            uc: {
                "n": blob["n"],
                "p40_flip_fraction": blob["p40"] / blob["n"] if blob["n"] else 0.0,
                "p60_flip_fraction": blob["p60"] / blob["n"] if blob["n"] else 0.0,
            }
            for uc, blob in sorted(by_uc.items())
        },
        "note": (
            "Synthetic protocol: minor/major split at the within-category defect-area "
            "percentile. Shifting the cut from the median to the 40th/60th percentile "
            "changes the assigned GT action on the reported fraction of frames."
        ),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "gt_threshold_sensitivity.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({"out": str(path), "action_flip_fraction": out["action_flip_fraction"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
