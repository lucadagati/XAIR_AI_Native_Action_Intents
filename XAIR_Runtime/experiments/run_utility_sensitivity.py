#!/usr/bin/env python3
"""
Utility sensitivity: recompute U under alternate (λ, μ) on cached suite summaries.

Usage:
    python3 experiments/run_utility_sensitivity.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper2_common import RESULTS_DIR  # noqa: E402

LAMBDAS = (1.0, 5.0, 10.0)
MUS = (0.5, 1.0, 2.0)


def u(sar: float, hazard: float, wrr: float, lam: float, mu: float) -> float:
    return sar - lam * hazard - mu * wrr


def load_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    del args

    sources = {
        "b3": (RESULTS_DIR / "b3_headline_table.csv", "policy", "headline"),
        "b4": (RESULTS_DIR / "b4_routing_table.csv", "router", "p50"),
        "b5": (RESULTS_DIR / "b5_headline_table.csv", "policy", None),
    }
    out: dict = {"grid": {"lambda": list(LAMBDAS), "mu": list(MUS)}, "suites": {}}

    for suite, (path, name_key, setting) in sources.items():
        rows = load_rows(path)
        if setting:
            rows = [r for r in rows if r.get("setting") == setting]
        suite_out = []
        for r in rows:
            sar = float(r.get("SAR") or r.get("sar") or 0)
            hazard = float(r.get("hazard") or r.get("hazardous_publish_rate") or 0)
            wrr = float(r.get("WRR") or r.get("wrr") or 0)
            name = r[name_key]
            grid = []
            for lam in LAMBDAS:
                for mu in MUS:
                    grid.append({"lambda": lam, "mu": mu, "U": u(sar, hazard, wrr, lam, mu)})
            # Ranking stability vs default λ=5, μ=1
            default_rank = sorted(
                [(r2[name_key], float(r2.get("SAR") or 0), float(r2.get("hazard") or 0), float(r2.get("WRR") or 0)) for r2 in rows],
                key=lambda t: -u(t[1], t[2], t[3], 5.0, 1.0),
            )
            suite_out.append({"name": name, "SAR": sar, "hazard": hazard, "WRR": wrr, "grid": grid})
        # ranking matrices
        rankings = {}
        for lam in LAMBDAS:
            for mu in MUS:
                ordered = sorted(suite_out, key=lambda x: -u(x["SAR"], x["hazard"], x["WRR"], lam, mu))
                rankings[f"l{lam}_m{mu}"] = [x["name"] for x in ordered]
        baseline = rankings.get("l5.0_m1.0", [])
        kendallish = {}
        for key, order in rankings.items():
            # Spearman footrule / pairwise agreement with baseline
            if not baseline:
                continue
            pos_b = {n: i for i, n in enumerate(baseline)}
            agree = 0
            total = 0
            for i, a in enumerate(order):
                for b in order[i + 1 :]:
                    total += 1
                    if pos_b[a] < pos_b[b]:
                        agree += 1
            kendallish[key] = agree / total if total else 1.0
        out["suites"][suite] = {"policies": suite_out, "rankings": rankings, "rank_agreement_vs_default": kendallish}

    path = RESULTS_DIR / "utility_sensitivity.json"
    path.write_text(json.dumps(out, indent=2))
    # CSV flat
    csv_path = RESULTS_DIR / "utility_sensitivity.csv"
    flat = []
    for suite, blob in out["suites"].items():
        for pol in blob["policies"]:
            for g in pol["grid"]:
                flat.append(
                    {
                        "suite": suite,
                        "name": pol["name"],
                        "lambda": g["lambda"],
                        "mu": g["mu"],
                        "U": g["U"],
                        "SAR": pol["SAR"],
                        "hazard": pol["hazard"],
                        "WRR": pol["WRR"],
                    }
                )
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(flat[0].keys()))
        w.writeheader()
        w.writerows(flat)
    print(json.dumps({"out": str(path), "rank_agreement_b5": out["suites"].get("b5", {}).get("rank_agreement_vs_default")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
