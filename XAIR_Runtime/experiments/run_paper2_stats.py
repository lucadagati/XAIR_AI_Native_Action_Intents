#!/usr/bin/env python3
"""
Publication stats package: frame-cluster bootstrap CI on mean-reward U, McNemar at
informative freshness windows, and McNemar power from discordant pairs.

Usage:
    python3 experiments/run_paper2_stats.py
    python3 experiments/run_paper2_stats.py --tag phase_p --n-boot 2000
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper2_common import (  # noqa: E402
    LAMBDA_HAZARD,
    MU_WRONGFUL_REVOKE,
    RESULTS_DIR,
    trial_reward,
    utility_mean_reward,
)
from experiments.paper2_splits import load_split  # noqa: E402
from experiments.run_b2_validity_frontier import (  # noqa: E402
    HEADLINE as B2_HEADLINE_GRID,
    ReplayRecord,
    load_replay_records,
    mcnemar,
    mcnemar_frame_majority,
    replay_one,
)

# Pre-declared headline configs for publication claims (not the full B2 grid).
HEADLINE = {
    "freshness_ms": 4000,
    "p_drift": 0.5,
    "drift_offset_ms": 250,
    "anchor": "capture",
    "model": "qwen2.5vl:7b",
    "prompt_variant": "blind",
}
OPS_W = (2000, 4000, 8000)
SEEDS = (1, 2, 3, 4, 5)
GATES = ("direct", "freshness_only", "xair")


def format_p_value(p: float) -> str:
    """Never emit p≪0.001; use scientific notation for tiny values."""
    if p >= 0.001:
        return f"{p:.6f}".rstrip("0").rstrip(".")
    if p == 0.0:
        return "0"
    return f"{p:.3e}"


def mcnemar_reported(
    rows_a: list[dict],
    rows_b: list[dict],
    field: str,
    *,
    clustered: bool = True,
) -> dict:
    raw = (
        mcnemar_frame_majority(rows_a, rows_b, field)
        if clustered
        else mcnemar(rows_a, rows_b, field)
    )
    p = float(raw["p_value"])
    n_pairs = int(raw["n_pairs"])
    b, c = int(raw["b"]), int(raw["c"])
    n_disc = b + c
    psi = n_disc / n_pairs if n_pairs else 0.0
    return {
        "n_pairs": n_pairs,
        "b": b,
        "c": c,
        "n_discordant": n_disc,
        "discordant_rate_psi": psi,
        "chi2": float(raw["chi2"]),
        "p_value": p,
        "p_value_fmt": format_p_value(p),
        "unit": raw.get("unit", "frame_majority" if clustered else "frame_seed"),
    }


def mcnemar_power(n_disc: int, p_alt: float, *, alpha: float = 0.05) -> float:
    """
    Approximate two-sided McNemar power given discordant pairs ``n_disc`` and
    P(first arm wins | discordant) = ``p_alt`` under the alternative.
    """
    if n_disc <= 0:
        return 0.0
    p_alt = min(max(p_alt, 1e-9), 1 - 1e-9)
    z_alpha = 1.959963984540054
    # Normal approximation to binomial on discordant pairs.
    se_null = math.sqrt(0.25 / n_disc)
    z = abs(p_alt - 0.5) / se_null - z_alpha
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def mcnemar_required_discordant(
    p_alt: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Discordant pairs needed for target power at alternative ``p_alt``."""
    p_alt = min(max(abs(p_alt - 0.5) + 0.5, 0.51), 0.999)
    z_alpha = 1.959963984540054
    z_beta = 0.8416212335729143
    delta = abs(p_alt - 0.5)
    if delta <= 0:
        return 0
    # Solve n from (delta / sqrt(p(1-p)/n) - z_alpha) >= z_beta with p≈0.5 at planning.
    n = ((z_alpha + z_beta) ** 2) * 0.25 / (delta**2)
    return int(math.ceil(n))


def generate_trials(
    records: list[ReplayRecord],
    *,
    gate: str,
    freshness_ms: int,
    seeds: tuple[int, ...] = SEEDS,
    p_drift: float = HEADLINE["p_drift"],
    drift_offset_ms: float = HEADLINE["drift_offset_ms"],
    anchor: str = HEADLINE["anchor"],
) -> list[dict]:
    trials: list[dict] = []
    for seed in seeds:
        rng = random.Random(seed)
        for rec in records:
            invalid = rng.random() < p_drift and drift_offset_ms <= rec.latency_ms
            scored = replay_one(
                rec,
                gate=gate,
                anchor=anchor,
                freshness_ms=freshness_ms,
                invalid_at_submit=invalid,
            )
            scored["seed"] = seed
            scored["reward"] = trial_reward(scored)
            trials.append(scored)
    return trials


def trials_by_frame(trials: list[dict]) -> dict[str, list[dict]]:
    by_frame: dict[str, list[dict]] = defaultdict(list)
    for t in trials:
        by_frame[t["frame_id"]].append(t)
    return dict(by_frame)


def bootstrap_utility_frames(
    trials: list[dict],
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """
    Cluster bootstrap: resample frame_id with replacement, include all seed trials
    per selected frame, compute mean per-trial reward U each replicate.
    """
    by_frame = trials_by_frame(trials)
    frame_ids = sorted(by_frame)
    if not frame_ids:
        return {"n_frames": 0, "n_trials": 0}

    rng = random.Random(seed)
    n_frames = len(frame_ids)
    boots: list[float] = []
    for _ in range(n_boot):
        sample_frames = [frame_ids[rng.randrange(n_frames)] for _ in range(n_frames)]
        sample_trials: list[dict] = []
        for fid in sample_frames:
            sample_trials.extend(by_frame[fid])
        boots.append(utility_mean_reward(sample_trials, reward_key="reward"))

    boots.sort()
    point = utility_mean_reward(trials, reward_key="reward")
    lo = boots[int(0.025 * (n_boot - 1))]
    hi = boots[int(0.975 * (n_boot - 1))]
    known = [t for t in trials if not t.get("unknown")]
    return {
        "n_frames": n_frames,
        "n_trials": len(trials),
        "n_known_trials": len(known),
        "U": point,
        "U_ci95": [lo, hi],
        "n_boot": n_boot,
        "bootstrap_unit": "frame_id_cluster",
        "reward_definition": "mean_trial_reward",
    }


def mcnemar_at_ops(records: list[ReplayRecord], *, seeds: tuple[int, ...] = SEEDS) -> dict:
    out: dict = {}
    for w in OPS_W:
        by_gate: dict[str, list[dict]] = defaultdict(list)
        for seed in seeds:
            rng = random.Random(seed)
            for rec in records:
                invalid = rng.random() < HEADLINE["p_drift"] and HEADLINE["drift_offset_ms"] <= rec.latency_ms
                for gate in GATES:
                    scored = replay_one(
                        rec,
                        gate=gate,
                        anchor=HEADLINE["anchor"],
                        freshness_ms=w,
                        invalid_at_submit=invalid,
                    )
                    scored["seed"] = seed
                    by_gate[gate].append(scored)
        cell: dict = {}
        for i, a in enumerate(GATES):
            for b in GATES[i + 1 :]:
                for field in ("hazardous_publish", "successful_actuation"):
                    cell[f"{a}_vs_{b}:{field}"] = mcnemar_reported(
                        by_gate[a], by_gate[b], field, clustered=True
                    )
                    cell[f"{a}_vs_{b}:{field}:frame_seed_exploratory"] = mcnemar_reported(
                        by_gate[a], by_gate[b], field, clustered=False
                    )
        out[f"w{w}"] = cell
    return out


def power_from_mcnemar(mcnemar_ops: dict) -> dict:
    """McNemar power / required discordant pairs from observed tests at w=4000."""
    ref = mcnemar_ops.get("w4000", {})
    key = "direct_vs_xair:successful_actuation"
    obs = ref.get(key, {})
    n_disc = int(obs.get("n_discordant", 0))
    b, c = int(obs.get("b", 0)), int(obs.get("c", 0))
    p_alt = b / n_disc if n_disc else 0.5
    psi = float(obs.get("discordant_rate_psi", 0.0))
    req_5pp = mcnemar_required_discordant(0.55)
    req_obs_effect = mcnemar_required_discordant(p_alt) if n_disc else None
    return {
        "test_reference": f"w4000/{key}",
        "unit": obs.get("unit", "frame_majority"),
        "observed_discordant_pairs": n_disc,
        "observed_b": b,
        "observed_c": c,
        "discordant_rate_psi": psi,
        "p_alt_b_over_discordant": p_alt,
        "observed_power_at_discordant": mcnemar_power(n_disc, p_alt) if n_disc else None,
        "required_discordant_pairs_80_power_5pp_effect": req_5pp,
        "required_discordant_pairs_80_power_observed_effect": req_obs_effect,
        "alpha": 0.05,
        "power_target": 0.8,
        "note": (
            "Primary McNemar units are frames (majority over seeds). "
            "Power uses discordant pairs, not independent proportions. "
            "Planning example uses |p_alt-0.5|=0.05 on discordant wins."
        ),
    }


def paired_seed_contrast(
    by_policy: dict[str, list[dict]],
    a: str,
    b: str,
    fields: tuple[str, ...] = ("utility", "SAR", "hazard"),
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Paired differences across aligned seeds for two policies."""
    rows_a = {int(r["seed"]): r for r in by_policy.get(a, [])}
    rows_b = {int(r["seed"]): r for r in by_policy.get(b, [])}
    shared = sorted(set(rows_a) & set(rows_b))
    if not shared:
        return {"n_seeds": 0, "a": a, "b": b}
    rng = random.Random(seed)
    out: dict = {"n_seeds": len(shared), "a": a, "b": b, "seeds": shared, "fields": {}}
    for field in fields:
        diffs = [float(rows_a[s][field]) - float(rows_b[s][field]) for s in shared]
        mean_d = sum(diffs) / len(diffs)
        boots = []
        for _ in range(n_boot):
            sample = [diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))]
            boots.append(sum(sample) / len(sample))
        boots.sort()
        wins_a = sum(1 for d in diffs if d > 0)
        wins_b = sum(1 for d in diffs if d < 0)
        out["fields"][field] = {
            "mean_delta_a_minus_b": mean_d,
            "ci95": [boots[int(0.025 * (n_boot - 1))], boots[int(0.975 * (n_boot - 1))]],
            "seed_wins_a": wins_a,
            "seed_wins_b": wins_b,
            "seed_ties": len(diffs) - wins_a - wins_b,
            "ci_excludes_zero": not (boots[int(0.025 * (n_boot - 1))] <= 0 <= boots[int(0.975 * (n_boot - 1))]),
        }
    return out


def load_csv_trials(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open() as fh:
        for r in csv.DictReader(fh):
            for k, v in list(r.items()):
                if v in ("True", "true", "1"):
                    r[k] = True
                elif v in ("False", "false", "0"):
                    r[k] = False
                else:
                    try:
                        if v is not None and v != "" and "." in v:
                            r[k] = float(v)
                        elif v is not None and v.isdigit():
                            r[k] = int(v)
                    except ValueError:
                        pass
            rows.append(r)
    return rows


def bootstrap_from_seed_utilities(seed_utils: list[float], *, n_boot: int = 1000, seed: int = 0) -> dict:
    if not seed_utils:
        return {"n": 0}
    rng = random.Random(seed)
    n = len(seed_utils)
    point = sum(seed_utils) / n
    boots = []
    for _ in range(n_boot):
        sample = [seed_utils[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    return {
        "n_seeds": n,
        "U": point,
        "U_ci95": [boots[int(0.025 * (n_boot - 1))], boots[int(0.975 * (n_boot - 1))]],
        "n_boot": n_boot,
        "note": "bootstrap over seed-level utilities (coarse; B3–B5 only)",
    }


def filter_test_frames(records: list[ReplayRecord], assignment: dict[str, str]) -> list[ReplayRecord]:
    return [r for r in records if assignment.get(r.frame_id) == "test"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="phase_p")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap RNG seed")
    args = parser.parse_args()

    split = load_split()
    records_all = load_replay_records(
        args.tag,
        models={HEADLINE["model"]},
        variants={HEADLINE["prompt_variant"]},
        use_repaired=False,
    )
    records_test = filter_test_frames(records_all, split)

    out: dict = {
        "lambda": LAMBDA_HAZARD,
        "mu": MU_WRONGFUL_REVOKE,
        "headline_config": HEADLINE,
        "b2_grid_note": (
            "The full B2 freshness × p_drift × drift_offset grid is exploratory. "
            f"Headline claims use pre-declared config; vacuous w={B2_HEADLINE_GRID['freshness_ms']}ms "
            "headline noted separately."
        ),
        "frame_split": {
            "n_train": sum(1 for v in split.values() if v == "train"),
            "n_test": sum(1 for v in split.values() if v == "test"),
            "bootstrap_frames": "test",
        },
    }

    print("[stats] McNemar at informative w (all seeds)...", flush=True)
    out["mcnemar_ops"] = mcnemar_at_ops(records_test if records_test else records_all)
    out["mcnemar_frame_set"] = "test" if records_test else "all"

    print("[stats] Frame-cluster bootstrap U at headline w...", flush=True)
    w = HEADLINE["freshness_ms"]
    boot_records = records_test if records_test else records_all
    b2_boot = {}
    for gate in GATES:
        trials = generate_trials(boot_records, gate=gate, freshness_ms=w)
        b2_boot[gate] = bootstrap_utility_frames(trials, n_boot=args.n_boot, seed=args.seed)
    out["b2_bootstrap"] = {
        "freshness_ms": w,
        "frame_set": out["mcnemar_frame_set"],
        "gates": b2_boot,
    }

    b2_sum = RESULTS_DIR / "b2_validity_frontier.json"
    if b2_sum.is_file():
        prev = json.loads(b2_sum.read_text())
        out["b2_headline_w500_paired_tests"] = prev.get("paired_tests")
        out["b2_headline_w500_note"] = (
            f"At w={B2_HEADLINE_GRID['freshness_ms']}ms both temporal gates revoke all intents; "
            "McNemar freshness_only vs xair is vacuous."
        )

    for label, path in (
        ("b3", RESULTS_DIR / "b3_eval.csv"),
        ("b4", RESULTS_DIR / "b4_routing_table.csv"),
        ("b5", RESULTS_DIR / "b5_policy_by_seed.csv"),
    ):
        rows = load_csv_trials(path)
        if not rows:
            continue
        grouped: dict[str, list[float]] = defaultdict(list)
        by_pol: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            setting = r.get("setting") or r.get("label") or "headline"
            if setting not in ("headline", "p50"):
                continue
            key = str(r.get("policy") or r.get("router") or "unknown")
            u = r.get("utility")
            if u is None:
                u = r.get("utility_aggregate")
            if u is not None:
                grouped[key].append(float(u))
            # Normalize metric aliases for paired contrasts.
            rr = dict(r)
            if "utility" not in rr and "utility_aggregate" in rr:
                rr["utility"] = rr["utility_aggregate"]
            if "SAR" not in rr and "sum_SAR" in rr:
                rr["SAR"] = rr["sum_SAR"]
            if "hazard" not in rr and "sum_hazardous_publish_rate" in rr:
                rr["hazard"] = rr["sum_hazardous_publish_rate"]
            if "seed" in rr and all(k in rr for k in ("utility", "SAR", "hazard")):
                by_pol[key].append(rr)
        out[f"{label}_seed_bootstrap"] = {
            k: bootstrap_from_seed_utilities(vs, n_boot=args.n_boot) for k, vs in grouped.items()
        }
        if by_pol:
            out[f"{label}_metric_seed_bootstrap"] = {
                k: {
                    "utility": bootstrap_from_seed_utilities(
                        [float(x["utility"]) for x in xs], n_boot=args.n_boot
                    ),
                    "SAR": bootstrap_from_seed_utilities(
                        [float(x["SAR"]) for x in xs], n_boot=args.n_boot
                    ),
                    "hazard": bootstrap_from_seed_utilities(
                        [float(x["hazard"]) for x in xs], n_boot=args.n_boot
                    ),
                }
                for k, xs in by_pol.items()
            }

    # Paired seed contrasts (n=5): report Δ with seed-bootstrap CI; do not overclaim.
    b3_by: dict[str, list[dict]] = defaultdict(list)
    for r in load_csv_trials(RESULTS_DIR / "b3_eval.csv"):
        if r.get("setting") != "headline" or "seed" not in r:
            continue
        rr = dict(r)
        if "utility" not in rr and "utility_aggregate" in rr:
            rr["utility"] = rr["utility_aggregate"]
        if "SAR" not in rr and "sum_SAR" in rr:
            rr["SAR"] = rr["sum_SAR"]
        if "hazard" not in rr and "sum_hazardous_publish_rate" in rr:
            rr["hazard"] = rr["sum_hazardous_publish_rate"]
        if all(k in rr for k in ("utility", "SAR", "hazard")):
            b3_by[str(rr["policy"])].append(rr)
    b5_by: dict[str, list[dict]] = defaultdict(list)
    for r in load_csv_trials(RESULTS_DIR / "b5_policy_by_seed.csv"):
        if r.get("setting") == "headline":
            b5_by[str(r["policy"])].append(r)
    out["paired_seed_contrasts"] = {
        "b3_linucb_vs_best_fixed": paired_seed_contrast(
            b3_by, "linucb:a1.0", "fixed:w500_strict", n_boot=args.n_boot
        ),
        "b5_qlearn_vs_single_shot": paired_seed_contrast(
            b5_by, "qlearn:e0.1", "single_shot", n_boot=args.n_boot
        ),
        "b5_reobserve_vs_single_shot": paired_seed_contrast(
            b5_by, "always_reobserve", "single_shot", n_boot=args.n_boot
        ),
        "note": (
            "Paired differences across five seeds; CI from seed-level bootstrap. "
            "With n_seeds=5, intervals are coarse—overlapping hazard rates do not establish superiority."
        ),
    }

    out["power_analysis"] = power_from_mcnemar(out["mcnemar_ops"])

    path = RESULTS_DIR / "paper2_stats.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(
        json.dumps(
            {
                "out": str(path),
                "frame_set": out["mcnemar_frame_set"],
                "b2_U": {g: b2_boot[g]["U"] for g in b2_boot},
                "power": out["power_analysis"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
