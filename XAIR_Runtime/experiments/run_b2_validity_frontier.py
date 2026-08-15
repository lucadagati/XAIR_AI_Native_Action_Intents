#!/usr/bin/env python3
"""
Suite B2 - the validity frontier (Phase G).

Replays every cached VLM decision through each publication gate over a grid of freshness
windows, plant volatilities and drift offsets, under both validity anchors. Because the
perception output is fixed, every gate sees byte-identical decisions, so differences
between gates are attributable to the gate alone and the comparison is properly paired.

The replay is a deterministic model of the publication boundary rather than a live run:

  * a decision is submitted ``inference_latency_ms`` after the evidence was acquired;
  * drift, when scheduled, lands ``drift_offset_ms`` after acquisition, so the context is
    invalid at evaluation exactly when the drift beat the submission;
  * ``direct`` publishes unconditionally, ``freshness_only`` applies the temporal window,
    and ``xair`` applies the temporal window and the model's own preconditions.

A separate fidelity check (``--fidelity N``) replays a random sample against the live
adapter and XAIR runtime and reports the disagreement rate, which is what licenses using
the offline model for the full grid.

Usage:
    python3 experiments/run_b2_validity_frontier.py --tag phase_p
    python3 experiments/run_b2_validity_frontier.py --tag phase_p --fidelity 500
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
    GATES,
    LAMBDA_HAZARD,
    MU_WRONGFUL_REVOKE,
    RESULTS_DIR,
    ContextValidator,
    Measurement,
    Notifier,
    classify,
    drift_patch,
    hazard_predicate,
    load_manifest,
    nominal_context,
    summarize,
    utility,
    wilson_ci,
)
from experiments.perception_cache import cache_path  # noqa: E402
from xair.ai.structured_intent import precondition_syntax_ok  # noqa: E402

# Grid. Freshness spans two orders of magnitude around the observed inference latency,
# because the interesting regime is precisely where the window and the latency are
# comparable; a window far above latency can never bind.
FRESHNESS_GRID_MS = (100, 250, 500, 1000, 2000, 4000, 8000)
P_DRIFT_GRID = (0.0, 0.25, 0.5, 0.75)
DRIFT_OFFSET_GRID_MS = (0, 250, 500, 1000, 2000, 4000)
ANCHORS = ("capture", "emission")
SEEDS = (1, 2, 3, 4, 5)

# The configuration whose per-trial rows are kept for paired significance testing.
HEADLINE = {"freshness_ms": 500, "p_drift": 0.5, "drift_offset_ms": 250, "anchor": "capture"}


def merge_context(base: dict, patch: dict) -> dict:
    out = json.loads(json.dumps(base))
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key].update(value)
        else:
            out[key] = value
    return out


def all_hold(context: dict, exprs: list[str]) -> bool:
    validator = ContextValidator(context)
    return all(validator._check(e)[0] for e in exprs)


class ReplayRecord:
    """
    A cached decision with everything the grid needs precomputed.

    Precondition evaluation is the only costly step, and it depends solely on whether the
    drift has landed, so both outcomes are resolved once here instead of once per grid
    point.
    """

    __slots__ = (
        "frame_id", "use_case", "model", "prompt_variant", "latency_ms", "gt_action",
        "model_action", "grounding_correct", "schema_valid", "n_preconditions",
        "preconds_hold_nominal", "preconds_hold_drifted", "severity", "defect_present",
        "category",
    )

    def __init__(self, record: dict, episode: dict, *, use_repaired: bool):
        self.frame_id = record["frame_id"]
        self.use_case = record.get("use_case", episode.get("use_case", "uc1_triage"))
        self.model = record.get("model", "")
        self.prompt_variant = record.get("prompt_variant", "blind")
        self.latency_ms = float(record.get("latency_ms") or 0.0)
        self.gt_action = record.get("gt_action") or episode.get("ground_truth_action", "")
        self.model_action = record.get("action", "")
        self.grounding_correct = bool(self.model_action) and self.model_action == self.gt_action
        self.severity = record.get("severity") or episode.get("severity")
        self.defect_present = bool(episode.get("defect_present"))
        self.category = episode.get("category", "")

        key = "preconditions_repaired" if use_repaired else "preconditions"
        exprs = [e for e in (record.get(key) or []) if precondition_syntax_ok(e)]
        emitted = record.get(key) or []
        self.n_preconditions = len(emitted)
        self.schema_valid = bool(
            record.get("schema_valid_repaired" if use_repaired else "schema_valid")
        )

        nominal = nominal_context(episode)
        drifted = merge_context(nominal, drift_patch(episode))
        self.preconds_hold_nominal = all_hold(nominal, exprs) if exprs else True
        self.preconds_hold_drifted = all_hold(drifted, exprs) if exprs else True


def load_replay_records(
    tag: str, *, models: set[str] | None, variants: set[str] | None, use_repaired: bool
) -> list[ReplayRecord]:
    path = cache_path(tag)
    if not path.is_file():
        raise SystemExit(f"No perception cache at {path}. Run perception_cache.py first.")

    episodes = {e["frame_id"]: e for e in load_manifest()}
    # The cache is an append-only log that may be written by a resumed run, so the same
    # decision can appear twice. Keying by call identity keeps the last write and stops a
    # re-run from silently double-weighting those frames in every aggregate.
    latest: dict[tuple, dict] = {}
    skipped = Counter()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            skipped["truncated"] += 1
            continue
        if models and rec.get("model") not in models:
            continue
        if variants and rec.get("prompt_variant") not in variants:
            continue
        if rec.get("error"):
            skipped["inference_error"] += 1
            continue
        key = (
            rec.get("frame_id"),
            rec.get("model"),
            rec.get("prompt_variant"),
            rec.get("use_case"),
        )
        if key in latest:
            skipped["duplicate"] += 1
        latest[key] = rec

    out: list[ReplayRecord] = []
    for rec in latest.values():
        episode = episodes.get(rec.get("frame_id"))
        if episode is None:
            skipped["no_episode"] += 1
            continue
        out.append(ReplayRecord(rec, episode, use_repaired=use_repaired))

    out.sort(key=lambda r: (r.frame_id, r.model, r.prompt_variant))
    if skipped:
        print(f"[b2] skipped: {dict(skipped)}", flush=True)
    return out


def gate_publishes(
    gate: str, *, elapsed_ms: float, freshness_ms: int, preconds_hold: bool
) -> tuple[bool, str]:
    """Model one gate's publication decision, and why."""
    fresh = elapsed_ms <= freshness_ms
    if gate == "direct":
        return True, "no_validation"
    if gate == "freshness_only":
        return (True, "fresh") if fresh else (False, "stale_window")
    if gate == "xair":
        if not fresh:
            return False, "stale_window"
        return (True, "validated") if preconds_hold else (False, "precondition_failed")
    raise ValueError(f"unknown gate: {gate}")


def replay_one(
    rec: ReplayRecord,
    *,
    gate: str,
    anchor: str,
    freshness_ms: int,
    invalid_at_submit: bool,
) -> dict:
    elapsed_ms = rec.latency_ms if anchor == "capture" else 0.0
    preconds_hold = rec.preconds_hold_drifted if invalid_at_submit else rec.preconds_hold_nominal
    published, reason = gate_publishes(
        gate, elapsed_ms=elapsed_ms, freshness_ms=freshness_ms, preconds_hold=preconds_hold
    )
    # The offline context is deterministic, so the trial is never ambiguous.
    measurement = Measurement(
        context_valid_before=not invalid_at_submit,
        context_valid_after=not invalid_at_submit,
        version_before=1,
        version_after=1,
    )
    scored = classify(
        gt_action=rec.gt_action,
        model_action=rec.model_action,
        published=published,
        measurement=measurement,
    )
    scored.update(
        {
            "frame_id": rec.frame_id,
            "use_case": rec.use_case,
            "gate": gate,
            "anchor": anchor,
            "model": rec.model,
            "prompt_variant": rec.prompt_variant,
            "freshness_ms": freshness_ms,
            "inference_latency_ms": rec.latency_ms,
            "elapsed_at_submit_ms": elapsed_ms,
            "published": published,
            "reason": reason,
            "schema_valid": rec.schema_valid,
            "n_preconditions": rec.n_preconditions,
            "severity": rec.severity,
            "defect_present": rec.defect_present,
            "gt_action": rec.gt_action,
            "model_action": rec.model_action,
        }
    )
    return scored


def _accumulate(acc: Counter, scored: dict) -> None:
    acc["n"] += 1
    for key in (
        "successful_actuation",
        "stale_publish",
        "unsafe_publish",
        "hazardous_publish",
        "correct_revoke",
        "blocked_grounding_error",
        "grounding_correct",
    ):
        if scored[key]:
            acc[key] += 1
    if scored["context_valid_at_eval"] is True:
        acc["ctx_valid"] += 1
        if scored["grounding_correct"]:
            acc["revocable"] += 1
            if scored["wrongful_revoke"]:
                acc["wrongful_revoke"] += 1
    else:
        acc["ctx_invalid"] += 1


def _finalize(acc: Counter) -> dict:
    n = acc["n"]
    if n == 0:
        return {}
    sar = acc["successful_actuation"] / n
    ser = acc["stale_publish"] / n
    hazard = acc["hazardous_publish"] / n
    revocable = acc["revocable"]
    wrr = acc["wrongful_revoke"] / revocable if revocable else 0.0
    # Mean per-trial reward (shared denominator n); WRR remains a secondary conditional rate.
    mean_r = (
        acc["successful_actuation"]
        - LAMBDA_HAZARD * acc["hazardous_publish"]
        - MU_WRONGFUL_REVOKE * acc["wrongful_revoke"]
    ) / n
    return {
        "n": n,
        "SAR": sar,
        "SAR_k": acc["successful_actuation"],
        "SAR_ci95": list(wilson_ci(acc["successful_actuation"], n)),
        "SER": ser,
        "SER_k": acc["stale_publish"],
        "SER_ci95": list(wilson_ci(acc["stale_publish"], n)),
        "unsafe_publish_rate": acc["unsafe_publish"] / n,
        "unsafe_publish_k": acc["unsafe_publish"],
        "hazardous_publish_rate": hazard,
        "hazardous_publish_k": acc["hazardous_publish"],
        "hazardous_publish_ci95": list(wilson_ci(acc["hazardous_publish"], n)),
        "WRR": wrr,
        "WRR_k": acc["wrongful_revoke"],
        "WRR_n": revocable,
        "WRR_ci95": list(wilson_ci(acc["wrongful_revoke"], revocable)) if revocable else [0.0, 1.0],
        "blocked_grounding_error_k": acc["blocked_grounding_error"],
        "grounding_accuracy": acc["grounding_correct"] / n,
        "context_invalid_rate": acc["ctx_invalid"] / n,
        "utility": mean_r,
        "mean_reward": mean_r,
        "utility_legacy_rate_combo": utility(sar, hazard, wrr),
    }


def run_grid(
    records: list[ReplayRecord],
    *,
    gates: list[str],
    anchors: list[str],
    freshness_grid: tuple[int, ...],
    p_drift_grid: tuple[float, ...],
    offset_grid: tuple[int, ...],
    seeds: tuple[int, ...],
    notifier: Notifier | None = None,
) -> tuple[list[dict], list[dict]]:
    """Sweep the grid, returning per-config aggregates and the headline per-trial rows."""
    aggregates: dict[tuple, Counter] = defaultdict(Counter)
    headline_rows: list[dict] = []

    total_cells = (
        len(p_drift_grid) * len(offset_grid) * len(seeds) * len(anchors) * len(freshness_grid)
    )
    cell = 0
    trials = 0

    for p_drift in p_drift_grid:
        for offset_ms in offset_grid:
            for seed in seeds:
                # One drift draw per (record, volatility, offset, seed), shared across gates,
                # anchors and windows: the same plant history is presented to every policy,
                # which is what makes the gate comparison paired rather than independent.
                rng = random.Random(f"{seed}:{p_drift}:{offset_ms}")
                draws = [rng.random() < p_drift for _ in records]

                for anchor in anchors:
                    for freshness_ms in freshness_grid:
                        cell += 1
                        acc_by_gate = {g: aggregates[(g, anchor, freshness_ms, p_drift, offset_ms)]
                                       for g in gates}
                        keep_headline = (
                            freshness_ms == HEADLINE["freshness_ms"]
                            and p_drift == HEADLINE["p_drift"]
                            and offset_ms == HEADLINE["drift_offset_ms"]
                            and anchor == HEADLINE["anchor"]
                        )
                        for rec, scheduled in zip(records, draws):
                            invalid = scheduled and offset_ms <= rec.latency_ms
                            for gate in gates:
                                scored = replay_one(
                                    rec,
                                    gate=gate,
                                    anchor=anchor,
                                    freshness_ms=freshness_ms,
                                    invalid_at_submit=invalid,
                                )
                                _accumulate(acc_by_gate[gate], scored)
                                trials += 1
                                if keep_headline:
                                    row = dict(scored)
                                    row["seed"] = seed
                                    row["p_drift"] = p_drift
                                    row["drift_offset_ms"] = offset_ms
                                    headline_rows.append(row)

                        if notifier and cell % max(1, total_cells // 10) == 0:
                            notifier.send(
                                f"B2 replay {100 * cell / total_cells:.0f}% "
                                f"({trials:,} trials)",
                                silent=True,
                            )

    out: list[dict] = []
    for (gate, anchor, freshness_ms, p_drift, offset_ms), acc in aggregates.items():
        row = {
            "gate": gate,
            "anchor": anchor,
            "freshness_ms": freshness_ms,
            "p_drift": p_drift,
            "drift_offset_ms": offset_ms,
        }
        row.update(_finalize(acc))
        out.append(row)
    out.sort(key=lambda r: (r["gate"], r["anchor"], r["p_drift"], r["drift_offset_ms"],
                            r["freshness_ms"]))
    print(f"[b2] {trials:,} trials over {len(out)} configurations", flush=True)
    return out, headline_rows


def best_per_gate(rows: list[dict]) -> dict:
    """The utility-maximising freshness window for each gate and anchor."""
    best: dict[str, dict] = {}
    for row in rows:
        key = f"{row['gate']}|{row['anchor']}"
        if key not in best or row["utility"] > best[key]["utility"]:
            best[key] = row
    return {
        k: {
            "freshness_ms": v["freshness_ms"],
            "p_drift": v["p_drift"],
            "drift_offset_ms": v["drift_offset_ms"],
            "utility": v["utility"],
            "SAR": v["SAR"],
            "SER": v["SER"],
            "hazardous_publish_rate": v["hazardous_publish_rate"],
            "WRR": v["WRR"],
        }
        for k, v in best.items()
    }


def anchor_effect(rows: list[dict]) -> dict:
    """
    How much of the hazard the emission anchor hides.

    Under emission anchoring the freshness window is measured from the moment the model
    finished, so inference latency cannot consume the window and staleness caused by slow
    perception is invisible to the validator. The gap between anchors at equal settings is
    the size of that blind spot.
    """
    by_key: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_key[(row["gate"], row["freshness_ms"], row["p_drift"], row["drift_offset_ms"])][
            row["anchor"]
        ] = row

    deltas = []
    for key, pair in by_key.items():
        if "capture" in pair and "emission" in pair:
            deltas.append(
                {
                    "gate": key[0],
                    "freshness_ms": key[1],
                    "p_drift": key[2],
                    "drift_offset_ms": key[3],
                    "delta_SER": pair["emission"]["SER"] - pair["capture"]["SER"],
                    "delta_SAR": pair["emission"]["SAR"] - pair["capture"]["SAR"],
                    "delta_utility": pair["emission"]["utility"] - pair["capture"]["utility"],
                }
            )
    worst = max(deltas, key=lambda d: d["delta_SER"], default=None)
    return {
        "configurations_compared": len(deltas),
        "mean_delta_SER": (
            sum(d["delta_SER"] for d in deltas) / len(deltas) if deltas else 0.0
        ),
        "max_delta_SER": worst,
    }


def hazard_curves(rows: list[dict], *, gate: str, anchor: str, freshness_ms: int) -> list[dict]:
    """Staleness as a function of how soon after acquisition the plant moves."""
    picked = [
        r
        for r in rows
        if r["gate"] == gate and r["anchor"] == anchor and r["freshness_ms"] == freshness_ms
    ]
    picked.sort(key=lambda r: (r["p_drift"], r["drift_offset_ms"]))
    return [
        {
            "p_drift": r["p_drift"],
            "drift_offset_ms": r["drift_offset_ms"],
            "SER": r["SER"],
            "SER_ci95": r["SER_ci95"],
            "SAR": r["SAR"],
            "utility": r["utility"],
        }
        for r in picked
    ]


def mcnemar(rows_a: list[dict], rows_b: list[dict], field: str) -> dict:
    """
    Exact-ish paired test between two gates on identical perception output.

    Only discordant pairs carry information, which is the whole point of pairing: the
    variance from perception is differenced away.

    Pairing key defaults to ``(frame_id, seed)``. Prefer :func:`mcnemar_frame_majority`
    for publication claims so multiple drift seeds on the same frame are not treated
    as independent units.
    """
    index_a = {(r["frame_id"], r.get("seed", 0)): bool(r[field]) for r in rows_a}
    index_b = {(r["frame_id"], r.get("seed", 0)): bool(r[field]) for r in rows_b}
    shared = set(index_a) & set(index_b)
    b = sum(1 for k in shared if index_a[k] and not index_b[k])
    c = sum(1 for k in shared if index_b[k] and not index_a[k])
    n_disc = b + c
    if n_disc == 0:
        return {
            "n_pairs": len(shared),
            "b": 0,
            "c": 0,
            "n_discordant": 0,
            "chi2": 0.0,
            "p_value": 1.0,
            "unit": "frame_seed",
        }
    chi2 = (abs(b - c) - 1) ** 2 / n_disc  # Yates-corrected
    from math import erfc, sqrt

    p = erfc(sqrt(max(chi2, 0.0) / 2.0))
    return {
        "n_pairs": len(shared),
        "b": b,
        "c": c,
        "n_discordant": n_disc,
        "chi2": chi2,
        "p_value": p,
        "unit": "frame_seed",
    }


def mcnemar_frame_majority(rows_a: list[dict], rows_b: list[dict], field: str) -> dict:
    """
    Frame-clustered McNemar: majority-vote the binary field across seeds per frame,
    then pair on ``frame_id`` only (one unit per image).
    """
    def majority_by_frame(rows: list[dict]) -> dict[str, bool]:
        buckets: dict[str, list[bool]] = defaultdict(list)
        for r in rows:
            buckets[str(r["frame_id"])].append(bool(r[field]))
        out: dict[str, bool] = {}
        for fid, vals in buckets.items():
            out[fid] = sum(vals) * 2 >= len(vals)  # True wins ties
        return out

    a = majority_by_frame(rows_a)
    bmap = majority_by_frame(rows_b)
    shared = set(a) & set(bmap)
    b = sum(1 for k in shared if a[k] and not bmap[k])
    c = sum(1 for k in shared if bmap[k] and not a[k])
    n_disc = b + c
    if n_disc == 0:
        return {
            "n_pairs": len(shared),
            "b": 0,
            "c": 0,
            "n_discordant": 0,
            "chi2": 0.0,
            "p_value": 1.0,
            "unit": "frame_majority",
        }
    chi2 = (abs(b - c) - 1) ** 2 / n_disc
    from math import erfc, sqrt

    p = erfc(sqrt(max(chi2, 0.0) / 2.0))
    return {
        "n_pairs": len(shared),
        "b": b,
        "c": c,
        "n_discordant": n_disc,
        "chi2": chi2,
        "p_value": p,
        "unit": "frame_majority",
    }


def paired_tests(headline_rows: list[dict], gates: list[str]) -> dict:
    by_gate: dict[str, list[dict]] = defaultdict(list)
    for row in headline_rows:
        by_gate[row["gate"]].append(row)

    out = {}
    for i, a in enumerate(gates):
        for b in gates[i + 1 :]:
            for field in ("hazardous_publish", "successful_actuation"):
                out[f"{a}_vs_{b}:{field}"] = mcnemar(by_gate[a], by_gate[b], field)
    return out


def write_outputs(
    rows: list[dict], headline_rows: list[dict], summary: dict, out_dir: Path
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "grid_csv": out_dir / "b2_validity_frontier.csv",
        "summary_json": out_dir / "b2_validity_frontier.json",
        "headline_csv": out_dir / "b2_headline_trials.csv",
    }
    if rows:
        with paths["grid_csv"].open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    if headline_rows:
        fields = sorted({k for r in headline_rows for k in r})
        with paths["headline_csv"].open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(headline_rows)
    paths["summary_json"].write_text(json.dumps(summary, indent=2, default=str))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="B2 validity frontier replay")
    parser.add_argument("--tag", default="phase_p", help="perception cache tag")
    parser.add_argument("--models", nargs="*", default=None, help="restrict to these models")
    parser.add_argument("--variants", nargs="*", default=["blind"],
                        help="prompt variants to replay; blind only by default")
    parser.add_argument("--gates", nargs="+", default=list(GATES))
    parser.add_argument("--anchors", nargs="+", default=list(ANCHORS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--use-repaired", action="store_true",
                        help="submit repaired preconditions instead of exactly what was emitted")
    parser.add_argument("--fidelity", type=int, default=0, metavar="N",
                        help="if N>0, run offline-vs-live fidelity check on N frames and exit")
    parser.add_argument("--no-notify", action="store_true")
    args = parser.parse_args()

    if args.fidelity and args.fidelity > 0:
        from experiments.run_b2_fidelity import main as fidelity_main

        return fidelity_main(
            [
                "--tag",
                args.tag,
                "--n",
                str(args.fidelity),
                *(["--no-notify"] if args.no_notify else []),
            ]
        )

    notifier = Notifier.from_env()
    if args.no_notify:
        notifier.enabled = False

    records = load_replay_records(
        args.tag,
        models=set(args.models) if args.models else None,
        variants=set(args.variants) if args.variants else None,
        use_repaired=args.use_repaired,
    )
    if not records:
        print("[b2] no usable cached decisions", file=sys.stderr)
        return 1
    print(f"[b2] replaying {len(records)} cached decisions", flush=True)

    rows, headline_rows = run_grid(
        records,
        gates=list(args.gates),
        anchors=list(args.anchors),
        freshness_grid=FRESHNESS_GRID_MS,
        p_drift_grid=P_DRIFT_GRID,
        offset_grid=DRIFT_OFFSET_GRID_MS,
        seeds=tuple(args.seeds),
        notifier=notifier,
    )

    summary = {
        "cache_tag": args.tag,
        "decisions_replayed": len(records),
        "preconditions": "repaired" if args.use_repaired else "as_emitted",
        "cost_weights": {"lambda_hazard": LAMBDA_HAZARD, "mu_wrongful_revoke": MU_WRONGFUL_REVOKE},
        "grid": {
            "freshness_ms": list(FRESHNESS_GRID_MS),
            "p_drift": list(P_DRIFT_GRID),
            "drift_offset_ms": list(DRIFT_OFFSET_GRID_MS),
            "anchors": list(args.anchors),
            "seeds": list(args.seeds),
            "gates": list(args.gates),
        },
        "headline_configuration": HEADLINE,
        "best_per_gate": best_per_gate(rows),
        "anchor_effect": anchor_effect(rows),
        "headline_summary": {
            gate: summarize([r for r in headline_rows if r["gate"] == gate])
            for gate in args.gates
        },
        "paired_tests": paired_tests(headline_rows, list(args.gates)),
        "hazard_curves": {
            f"{gate}|capture|500ms": hazard_curves(
                rows, gate=gate, anchor="capture", freshness_ms=500
            )
            for gate in args.gates
        },
    }

    out_dir = RESULTS_DIR
    if args.use_repaired:
        # Write to *_repaired paths so as-emitted artifacts stay intact.
        class _Redirect:
            def __init__(self, base: Path):
                self.base = base

            def __truediv__(self, name: str) -> Path:
                stem = Path(name).stem
                suffix = Path(name).suffix
                if stem.startswith("b2_"):
                    return self.base / f"{stem}_repaired{suffix}"
                return self.base / name

            def mkdir(self, *a, **k):
                return self.base.mkdir(*a, **k)

        # Monkey-patch via custom write
        paths = {
            "grid_csv": RESULTS_DIR / "b2_validity_frontier_repaired.csv",
            "summary_json": RESULTS_DIR / "b2_validity_frontier_repaired.json",
            "headline_csv": RESULTS_DIR / "b2_headline_trials_repaired.csv",
        }
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        if rows:
            with paths["grid_csv"].open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        if headline_rows:
            fields = sorted({k for r in headline_rows for k in r})
            with paths["headline_csv"].open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                writer.writerows(headline_rows)
        paths["summary_json"].write_text(json.dumps(summary, indent=2, default=str))
    else:
        paths = write_outputs(rows, headline_rows, summary, RESULTS_DIR)
    print(json.dumps(summary["best_per_gate"], indent=2))
    print(json.dumps(summary["anchor_effect"], indent=2, default=str))
    for name, path in paths.items():
        print(f"[b2] {name}: {path}")

    head = summary["headline_summary"]
    notifier.send(
        "OK B2 validity frontier done\n"
        + "\n".join(
            f"{g}: SAR {head[g].get('SAR', 0):.3f} SER {head[g].get('SER', 0):.3f} "
            f"U {head[g].get('utility', 0):.3f}"
            for g in args.gates
            if head.get(g)
        )
        + f"\nanchor blind spot: mean dSER {summary['anchor_effect']['mean_delta_SER']:+.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
