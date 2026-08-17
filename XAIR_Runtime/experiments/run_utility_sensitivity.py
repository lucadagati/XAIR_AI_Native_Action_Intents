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

from experiments.paper2_common import LAMBDA_HAZARD, MU_WRONGFUL_REVOKE, RESULTS_DIR  # noqa: E402

LAMBDAS = (1.0, 5.0, 10.0)
MUS = (0.5, 1.0, 2.0)


def u(sar: float, hazard: float, wrr: float, lam: float, mu: float) -> float:
    return sar - lam * hazard - mu * wrr


def trial_wrr(sar: float, hazard: float, reported_wrr: float, mean_reward: float | None) -> float:
    """Prefer trial-level wrongful-revoke rate inverted from mean reward at default (λ,μ)."""
    if mean_reward is None:
        return reported_wrr
    return sar - LAMBDA_HAZARD * hazard - mean_reward


def rank_suite(suite_out: list[dict]) -> dict:
    rankings = {}
    for lam in LAMBDAS:
        for mu in MUS:
            ordered = sorted(suite_out, key=lambda x: -u(x["SAR"], x["hazard"], x["WRR"], lam, mu))
            rankings[f"l{lam}_m{mu}"] = [x["name"] for x in ordered]
    baseline = rankings.get("l5.0_m1.0", [])
    kendallish = {}
    for key, order in rankings.items():
        if not baseline:
            continue
        pos_b = {n: i for i, n in enumerate(baseline)}
        agree = total = 0
        for i, a in enumerate(order):
            for b in order[i + 1 :]:
                total += 1
                if pos_b[a] < pos_b[b]:
                    agree += 1
        kendallish[key] = agree / total if total else 1.0
    return {"policies": suite_out, "rankings": rankings, "rank_agreement_vs_default": kendallish}


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
        "b3": (RESULTS_DIR / "b3_headline_table.csv", "policy", None),
        "b4": (RESULTS_DIR / "b4_routing_table.csv", "router", "p50"),
        "b5": (RESULTS_DIR / "b5_headline_table.csv", "policy", None),
    }
    out: dict = {"grid": {"lambda": list(LAMBDAS), "mu": list(MUS)}, "suites": {}}

    stats_path = RESULTS_DIR / "paper2_stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text())
        rates = stats.get("b2_headline_rates") or {}
        if rates:
            suite_out = []
            for name, r in rates.items():
                sar, hazard, wrr = float(r["SAR"]), float(r["hazard"]), float(r["WRR"])
                grid = [
                    {"lambda": lam, "mu": mu, "U": u(sar, hazard, wrr, lam, mu)}
                    for lam in LAMBDAS
                    for mu in MUS
                ]
                suite_out.append({"name": name, "SAR": sar, "hazard": hazard, "WRR": wrr, "grid": grid})
            blob = rank_suite(suite_out)
            blob["source"] = "paper2_stats.b2_headline_rates"
            out["suites"]["b2"] = blob

    for suite, (path, name_key, setting) in sources.items():
        rows = load_rows(path)
        if setting:
            rows = [r for r in rows if r.get("setting") == setting]
        suite_out = []
        for r in rows:
            sar = float(r.get("SAR") or r.get("sar") or 0)
            hazard = float(r.get("hazard") or r.get("hazardous_publish_rate") or 0)
            reported_wrr = float(r.get("WRR") or r.get("wrr") or 0)
            mean_r = r.get("mean_reward") or r.get("utility")
            mean_r_f = float(mean_r) if mean_r not in (None, "") else None
            wrr = trial_wrr(sar, hazard, reported_wrr, mean_r_f)
            name = r[name_key]
            grid = [
                {"lambda": lam, "mu": mu, "U": u(sar, hazard, wrr, lam, mu)}
                for lam in LAMBDAS
                for mu in MUS
            ]
            suite_out.append({"name": name, "SAR": sar, "hazard": hazard, "WRR": wrr, "grid": grid})
        out["suites"][suite] = rank_suite(suite_out)

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
