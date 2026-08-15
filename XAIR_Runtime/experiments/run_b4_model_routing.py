#!/usr/bin/env python3
"""
Suite B4 — latency-aware model routing.

For frames where multiple VLMs produced blind decisions, compare static model
choices against routers that trade grounding accuracy against inference latency
under capture-anchored XAIR validation and increasing plant volatility.

Usage:
    python3 experiments/run_b4_model_routing.py --tag phase_p
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper2_common import (  # noqa: E402
    LAMBDA_HAZARD,
    MU_WRONGFUL_REVOKE,
    RESULTS_DIR,
    Notifier,
    summarize,
)
from experiments.paper2_splits import SPLIT_SEED, load_split  # noqa: E402
from experiments.rl.budget_env import instantaneous_reward  # noqa: E402
from experiments.run_b2_validity_frontier import (  # noqa: E402
    ReplayRecord,
    load_replay_records,
    replay_one,
)

MODELS = (
    "qwen2.5vl:3b",
    "qwen2.5vl:7b",
    "llama3.2-vision:11b",
    "gemma3:12b",
    "qwen2.5vl:32b",
)

# Prefer faster models when several achieve the same instantaneous reward.
MODEL_SPEED_RANK = {
    "qwen2.5vl:3b": 0,
    "qwen2.5vl:7b": 1,
    "gemma3:12b": 2,
    "llama3.2-vision:11b": 3,
    "qwen2.5vl:32b": 4,
}

VOLATILITY = (
    {"label": "p0", "p_drift": 0.0, "drift_offset_ms": 250},
    {"label": "p25", "p_drift": 0.25, "drift_offset_ms": 250},
    {"label": "p50", "p_drift": 0.5, "drift_offset_ms": 250},
    {"label": "p75", "p_drift": 0.75, "drift_offset_ms": 250},
    {"label": "p75_early", "p_drift": 0.75, "drift_offset_ms": 0},
)

FRESHNESS_POLICY = "adaptive"  # w = max(2000, ceil(latency))
FIXED_W_MS = 4000


def build_frame_index(records: list[ReplayRecord]) -> dict[str, dict[str, ReplayRecord]]:
    by_frame: dict[str, dict[str, ReplayRecord]] = defaultdict(dict)
    for rec in records:
        by_frame[rec.frame_id][rec.model] = rec
    # Keep frames with at least 4 models (preferably all five).
    return {
        fid: models
        for fid, models in by_frame.items()
        if len(models) >= 4 and "qwen2.5vl:7b" in models
    }


def freshness_for(rec: ReplayRecord, policy: str = FRESHNESS_POLICY) -> int:
    if policy == "fixed":
        return FIXED_W_MS
    # Adaptive: window must cover inference under capture anchoring, with a floor.
    return max(2000, int(rec.latency_ms + 250))


def score_choice(
    rec: ReplayRecord,
    *,
    invalid: bool,
    gate: str = "xair",
) -> dict:
    scored = replay_one(
        rec,
        gate=gate,
        anchor="capture",
        freshness_ms=freshness_for(rec),
        invalid_at_submit=invalid,
    )
    scored["reward"] = instantaneous_reward(scored)
    scored["chosen_model"] = rec.model
    scored["chosen_latency_ms"] = rec.latency_ms
    return scored


def sample_invalid(rec: ReplayRecord, rng: random.Random, p_drift: float, offset_ms: int) -> bool:
    if rng.random() >= p_drift:
        return False
    return offset_ms <= rec.latency_ms


def static_router(model: str):
    def choose(cands: dict[str, ReplayRecord], _ctx: dict) -> ReplayRecord:
        if model in cands:
            return cands[model]
        # fallback: fastest available
        return min(cands.values(), key=lambda r: r.latency_ms)

    choose.name = f"static:{model}"  # type: ignore[attr-defined]
    return choose


def oracle_router(cands: dict[str, ReplayRecord], ctx: dict) -> ReplayRecord:
    """Retrospective best model for this trial's drift draw."""
    best = None
    best_key = None
    for rec in cands.values():
        invalid = ctx["offset_ms"] <= rec.latency_ms and ctx["drift_fires"]
        scored = score_choice(rec, invalid=invalid)
        key = (
            scored["reward"],
            -int(scored["hazardous_publish"]),
            -MODEL_SPEED_RANK.get(rec.model, 9),
            -rec.latency_ms,
        )
        if best_key is None or key > best_key:
            best, best_key = rec, key
    assert best is not None
    return best


oracle_router.name = "oracle"  # type: ignore[attr-defined]


def latency_cap_router(cap_ms: float):
    """Among models with latency <= cap (or the fastest if none), pick highest historical grounding proxy."""

    def choose(cands: dict[str, ReplayRecord], _ctx: dict) -> ReplayRecord:
        eligible = [r for r in cands.values() if r.latency_ms <= cap_ms]
        if not eligible:
            return min(cands.values(), key=lambda r: r.latency_ms)
        # Prefer grounding_correct on this frame (observable only retrospectively for oracle-ish;
        # for online we use schema_valid + confidence proxy via n_preconditions as weak signal).
        # Online-honest: prefer schema_valid then lower latency.
        eligible.sort(key=lambda r: (-int(r.schema_valid), -r.n_preconditions, r.latency_ms))
        return eligible[0]

    choose.name = f"cap:{int(cap_ms)}ms"  # type: ignore[attr-defined]
    return choose


def accuracy_first_router(cands: dict[str, ReplayRecord], _ctx: dict) -> ReplayRecord:
    """Prefer 7b then 32b then others — static accuracy prior from B1."""
    for m in ("qwen2.5vl:7b", "qwen2.5vl:32b", "qwen2.5vl:3b", "gemma3:12b", "llama3.2-vision:11b"):
        if m in cands:
            return cands[m]
    return next(iter(cands.values()))


accuracy_first_router.name = "prior:accuracy"  # type: ignore[attr-defined]


def speed_first_router(cands: dict[str, ReplayRecord], _ctx: dict) -> ReplayRecord:
    return min(cands.values(), key=lambda r: (MODEL_SPEED_RANK.get(r.model, 9), r.latency_ms))


speed_first_router.name = "prior:speed"  # type: ignore[attr-defined]


def cascade_router(cands: dict[str, ReplayRecord], ctx: dict) -> ReplayRecord:
    """
    Simulated cascade: accept the fastest model if it is schema-valid and plant is calm;
    otherwise escalate toward higher-accuracy models as volatility rises.
    """
    ordered = sorted(cands.values(), key=lambda r: (MODEL_SPEED_RANK.get(r.model, 9), r.latency_ms))
    p = ctx["p_drift"]
    if p <= 0.25:
        # prefer speed among schema-valid
        for r in ordered:
            if r.schema_valid:
                return r
        return ordered[0]
    if p <= 0.5:
        for m in ("qwen2.5vl:7b", "qwen2.5vl:3b", "gemma3:12b"):
            if m in cands and cands[m].schema_valid:
                return cands[m]
        return ordered[0]
    # high volatility: prefer accuracy
    return accuracy_first_router(cands, ctx)


cascade_router.name = "cascade:vol"  # type: ignore[attr-defined]


def evaluate_router(
    frames: dict[str, dict[str, ReplayRecord]],
    router,
    *,
    setting: dict,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    scored_rows: list[dict] = []
    model_counts: Counter[str] = Counter()
    latencies: list[float] = []
    grounding = 0

    for fid, cands in frames.items():
        drift_fires = rng.random() < setting["p_drift"]
        ctx = {
            "p_drift": setting["p_drift"],
            "offset_ms": setting["drift_offset_ms"],
            "drift_fires": drift_fires,
        }
        chosen = router(cands, ctx)
        invalid = drift_fires and setting["drift_offset_ms"] <= chosen.latency_ms
        scored = score_choice(chosen, invalid=invalid)
        scored["frame_id"] = fid
        scored_rows.append(scored)
        model_counts[chosen.model] += 1
        latencies.append(chosen.latency_ms)
        if chosen.grounding_correct:
            grounding += 1

    n = len(scored_rows)
    agg = summarize(scored_rows)
    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else 0.0
    return {
        "router": getattr(router, "name", getattr(router, "__name__", "router")),
        "setting": setting["label"],
        "p_drift": setting["p_drift"],
        "drift_offset_ms": setting["drift_offset_ms"],
        "n": n,
        "grounding": grounding / n if n else 0.0,
        "latency_p50_ms": p50,
        "latency_mean_ms": sum(latencies) / n if n else 0.0,
        "mean_reward": sum(r["reward"] for r in scored_rows) / n if n else 0.0,
        "utility": agg.get("utility", 0.0),
        "utility_legacy_rate_combo": agg.get("utility_legacy_rate_combo"),
        "summary": agg,
        "model_share": dict(model_counts),
    }


def static_pareto(frames: dict[str, dict[str, ReplayRecord]]) -> list[dict]:
    """Model-level grounding vs latency on the multi-model frame set."""
    out = []
    for model in MODELS:
        lats = []
        g = 0
        n = 0
        for cands in frames.values():
            if model not in cands:
                continue
            rec = cands[model]
            n += 1
            lats.append(rec.latency_ms)
            if rec.grounding_correct:
                g += 1
        if n == 0:
            continue
        lats.sort()
        out.append(
            {
                "model": model,
                "n": n,
                "grounding": g / n,
                "latency_p50_ms": lats[len(lats) // 2],
                "latency_mean_ms": sum(lats) / n,
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="B4 latency-aware model routing")
    parser.add_argument("--tag", default="phase_p")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--no-notify", action="store_true")
    args = parser.parse_args()

    notifier = Notifier.from_env()
    if args.no_notify:
        notifier.enabled = False

    records = load_replay_records(args.tag, models=None, variants={"blind"}, use_repaired=False)
    frames = build_frame_index(records)
    split = load_split()
    eval_frames = {fid: cands for fid, cands in frames.items() if split.get(fid) == "test"}
    if not eval_frames:
        print("[b4] empty test split", file=sys.stderr)
        return 1
    print(
        f"[b4] multi-model frames={len(frames)} (eval test={len(eval_frames)})",
        flush=True,
    )
    notifier.send(f"B4 routing started\n{len(eval_frames)} test frames for router eval")

    pareto = static_pareto(frames)

    routers = (
        [static_router(m) for m in MODELS]
        + [speed_first_router, accuracy_first_router, cascade_router]
        + [latency_cap_router(c) for c in (3000, 4000, 5000, 8000)]
        + [oracle_router]
    )

    rows = []
    for seed in args.seeds:
        for setting in VOLATILITY:
            for router in routers:
                rows.append(evaluate_router(eval_frames, router, setting=setting, seed=seed))

    # Aggregate across seeds for each (router, setting)
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        buckets[(r["router"], r["setting"])].append(r)

    table = []
    for (router, setting), group in sorted(buckets.items()):
        table.append(
            {
                "router": router,
                "setting": setting,
                "grounding": sum(g["grounding"] for g in group) / len(group),
                "latency_p50_ms": sum(g["latency_p50_ms"] for g in group) / len(group),
                "mean_reward": sum(g["mean_reward"] for g in group) / len(group),
                "utility": sum(g["utility"] for g in group) / len(group),
                "SAR": sum(g["summary"]["SAR"] for g in group) / len(group),
                "hazard": sum(g["summary"]["hazardous_publish_rate"] for g in group) / len(group),
                "WRR": sum(g["summary"]["WRR"] for g in group) / len(group),
                "n_seeds": len(group),
            }
        )

    # Headline = p50
    headline = [r for r in table if r["setting"] == "p50"]
    headline.sort(key=lambda r: r["utility"], reverse=True)

    out = {
        "n_frames": len(frames),
        "split": {
            "seed": SPLIT_SEED,
            "n_train": sum(1 for fid in frames if split.get(fid) == "train"),
            "n_test": len(eval_frames),
        },
        "seeds": args.seeds,
        "freshness_policy": FRESHNESS_POLICY,
        "static_pareto": pareto,
        "headline_p50": headline,
        "table": table,
        "cost_weights": {"lambda_hazard": LAMBDA_HAZARD, "mu_wrongful_revoke": MU_WRONGFUL_REVOKE},
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "b4_model_routing.json"
    table_path = RESULTS_DIR / "b4_routing_table.csv"
    pareto_path = RESULTS_DIR / "b4_static_pareto.csv"
    summary_path.write_text(json.dumps(out, indent=2))

    with table_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    with pareto_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pareto[0].keys()))
        w.writeheader()
        w.writerows(pareto)

    print(json.dumps({"headline_top5": headline[:5], "pareto": pareto}, indent=2))
    best = headline[0]
    notifier.send(
        f"OK B4 routing\nbest={best['router']} U={best['utility']:.3f}\n"
        f"ground={best['grounding']:.3f} p50_lat={best['latency_p50_ms']/1000:.2f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
