#!/usr/bin/env python3
"""
B1: what the VLM actually perceives, and whether its own preconditions protect it.

All of this is computed offline from the Phase P cache, because none of it needs a
running gate: a precondition set can be evaluated against any context we like.

Four rates are reported per model and prompt variant.

``grounding_accuracy``
    The action matches the declared ground truth. Under blind prompting this is a real
    perception measurement; under the ``leaky`` control it mostly measures transcription,
    and the gap between the two quantifies the leakage.

``protective_rate``
    At least one emitted precondition fails once the declared drift patch is applied.
    A precondition set that survives the drift cannot protect against stale actuation
    no matter how good the gate is.

``self_catch_rate``
    Among decisions where the model chose the *wrong* action, at least one of its own
    preconditions already fails against the true nominal context. This is the case where
    the model's structured output blocks the model's own misjudgement: it says ACCEPT but
    asserts ``defect.absent == true`` on a part that is in fact defective, and the gate
    revokes on that assertion.

``false_block_rate``
    Among decisions where the model chose the *right* action, at least one precondition
    nonetheless fails against the true nominal context. These become wrongful revocations
    in the gating phase, so this is the cost side of asking for stricter preconditions.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper2_common import (  # noqa: E402
    DATASET_ROOT,
    RESULTS_DIR,
    load_manifest,
    wilson_ci,
)
from experiments.perception_cache import cache_path, load_cache  # noqa: E402
from xair.ai.structured_intent import precondition_syntax_ok  # noqa: E402
from xair.core.context_validator import ContextValidator  # noqa: E402
from xair.core.deep_merge import deep_merge  # noqa: E402


def _holds(context: dict, expr: str) -> bool:
    return ContextValidator(context)._check(expr)[0]


def all_hold(context: dict, exprs: list[str]) -> bool:
    return all(_holds(context, e) for e in exprs)


def evaluate_record(record: dict, episode: dict) -> dict:
    """Score one cached decision against the episode's declared context and drift."""
    uc = record.get("use_case") or episode["use_case"]
    spec = episode["use_cases"][uc]
    nominal = episode["context"]
    drifted = deep_merge(json.loads(json.dumps(nominal)), spec["drift_patch"])

    preconds = [p for p in (record.get("preconditions") or []) if isinstance(p, str)]
    syntax_ok = all(precondition_syntax_ok(p) for p in preconds)
    gt = record.get("gt_action") or spec["ground_truth_action"]
    action = record.get("action") or ""
    correct = bool(action) and action == gt

    holds_nominal = all_hold(nominal, preconds) if preconds else True
    holds_drifted = all_hold(drifted, preconds) if preconds else True

    return {
        "frame_id": record.get("frame_id"),
        "model": record.get("model"),
        "prompt_variant": record.get("prompt_variant"),
        "use_case": uc,
        "category": record.get("category") or episode.get("category"),
        "source_dataset": record.get("source_dataset") or episode.get("source_dataset"),
        "severity": record.get("severity") or episode.get("severity"),
        "defect_present": record.get("defect_present", episode.get("defect_present")),
        "gt_action": gt,
        "action": action,
        "grounding_correct": correct,
        "defect_judgement": record.get("defect_judgement"),
        "defect_judgement_correct": (
            None
            if record.get("defect_judgement") is None
            else bool(record["defect_judgement"]) == bool(episode.get("defect_present"))
        ),
        "confidence": record.get("confidence"),
        "latency_ms": record.get("latency_ms") or 0.0,
        "n_preconditions": len(preconds),
        "has_preconditions": bool(preconds),
        "precondition_syntax_ok": syntax_ok,
        "schema_valid": bool(record.get("schema_valid")),
        "parse_ok": bool(record.get("parse_ok")),
        # Protective: the set notices the drift.
        "protective": bool(preconds) and not holds_drifted,
        # Self-catch: a wrong action already blocked by the model's own assertions.
        "self_catch": bool(preconds) and (not correct) and (not holds_nominal),
        # False block: a right action that its own assertions would veto anyway.
        "false_block": bool(preconds) and correct and (not holds_nominal),
        "error": record.get("error"),
    }


def _rate(rows: list[dict], field: str, subset=None) -> dict:
    pool = [r for r in rows if subset(r)] if subset else rows
    n = len(pool)
    k = sum(1 for r in pool if r.get(field))
    if n == 0:
        return {"k": 0, "n": 0, "rate": None, "ci95": None}
    lo, hi = wilson_ci(k, n)
    return {"k": k, "n": n, "rate": k / n, "ci95": [lo, hi]}


def summarize_group(rows: list[dict]) -> dict:
    lats = sorted(r["latency_ms"] for r in rows if r["latency_ms"] > 0)

    def pct(p: float) -> float:
        if not lats:
            return 0.0
        return lats[min(len(lats) - 1, max(0, int(round(p * (len(lats) - 1)))))]

    judged = [r for r in rows if r["defect_judgement_correct"] is not None]
    npre = [r["n_preconditions"] for r in rows]

    return {
        "n": len(rows),
        "grounding_accuracy": _rate(rows, "grounding_correct"),
        "defect_judgement_accuracy": _rate(judged, "defect_judgement_correct"),
        "schema_valid": _rate(rows, "schema_valid"),
        "parse_ok": _rate(rows, "parse_ok"),
        "precondition_syntax_ok": _rate(rows, "precondition_syntax_ok"),
        "has_preconditions": _rate(rows, "has_preconditions"),
        "protective_rate": _rate(rows, "protective"),
        "self_catch_rate": _rate(rows, "self_catch", subset=lambda r: not r["grounding_correct"]),
        "false_block_rate": _rate(rows, "false_block", subset=lambda r: r["grounding_correct"]),
        "errors": sum(1 for r in rows if r.get("error")),
        "n_preconditions_mean": statistics.mean(npre) if npre else 0.0,
        "latency_p50_ms": pct(0.50),
        "latency_p95_ms": pct(0.95),
    }


def confusion(rows: list[dict]) -> dict:
    table: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        table[r["gt_action"]][r["action"] or "<none>"] += 1
    return {gt: dict(sorted(preds.items())) for gt, preds in sorted(table.items())}


def leakage_effect(rows: list[dict], primary_model: str) -> dict:
    """How much apparent accuracy comes from the leaked label rather than the image."""
    blind = [r for r in rows if r["model"] == primary_model and r["prompt_variant"] == "blind"]
    leaky = [r for r in rows if r["model"] == primary_model and r["prompt_variant"] == "leaky"]
    if not blind or not leaky:
        return {}
    b = _rate(blind, "grounding_correct")
    lk = _rate(leaky, "grounding_correct")
    paired = {r["frame_id"]: r["grounding_correct"] for r in blind}
    both = [
        (paired[r["frame_id"]], r["grounding_correct"])
        for r in leaky
        if r["frame_id"] in paired
    ]
    only_leaky = sum(1 for a, c in both if c and not a)
    only_blind = sum(1 for a, c in both if a and not c)
    return {
        "model": primary_model,
        "blind": b,
        "leaky": lk,
        "delta": (lk["rate"] - b["rate"]) if (b["rate"] is not None and lk["rate"] is not None) else None,
        "paired_n": len(both),
        "leaky_only_correct": only_leaky,
        "blind_only_correct": only_blind,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="B1 blind grounding analysis")
    parser.add_argument("--tag", default="phase_p")
    parser.add_argument("--primary-model", default="qwen2.5vl:7b")
    parser.add_argument("--out-json", type=Path, default=RESULTS_DIR / "b1_grounding.json")
    parser.add_argument("--out-csv", type=Path, default=RESULTS_DIR / "b1_grounding.csv")
    args = parser.parse_args()

    episodes = {e["frame_id"]: e for e in load_manifest()}
    if not episodes:
        print(f"No manifest at {DATASET_ROOT / 'manifest.jsonl'}", file=sys.stderr)
        return 1

    records, _ = load_cache(cache_path(args.tag))
    if not records:
        print(f"Empty perception cache: {cache_path(args.tag)}", file=sys.stderr)
        return 1

    rows = [
        evaluate_record(r, episodes[r["frame_id"]])
        for r in records
        if r.get("frame_id") in episodes
    ]

    by_model: dict[str, list[dict]] = defaultdict(list)
    by_model_variant: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_severity: dict[str, list[dict]] = defaultdict(list)
    by_use_case: dict[str, list[dict]] = defaultdict(list)
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
        by_model_variant[(r["model"], r["prompt_variant"])].append(r)
        by_severity[r["severity"] or "unknown"].append(r)
        by_use_case[r["use_case"]].append(r)
        by_source[r["source_dataset"] or "unknown"].append(r)

    blind_only = [r for r in rows if r["prompt_variant"] != "leaky"]

    summary = {
        "cache": str(cache_path(args.tag)),
        "decisions": len(rows),
        "overall_blind": summarize_group(blind_only),
        "by_model": {m: summarize_group(v) for m, v in sorted(by_model.items())},
        "by_model_variant": {
            f"{m}|{pv}": summarize_group(v) for (m, pv), v in sorted(by_model_variant.items())
        },
        "by_severity": {k: summarize_group(v) for k, v in sorted(by_severity.items())},
        "by_use_case": {k: summarize_group(v) for k, v in sorted(by_use_case.items())},
        "by_source": {k: summarize_group(v) for k, v in sorted(by_source.items())},
        "confusion_blind": confusion(blind_only),
        "leakage_effect": leakage_effect(rows, args.primary_model),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2))

    fields = list(rows[0].keys())
    with args.out_csv.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    headline = {
        "decisions": len(rows),
        "blind_grounding_accuracy": summary["overall_blind"]["grounding_accuracy"],
        "blind_protective_rate": summary["overall_blind"]["protective_rate"],
        "blind_self_catch_rate": summary["overall_blind"]["self_catch_rate"],
        "blind_false_block_rate": summary["overall_blind"]["false_block_rate"],
        "leakage_delta": summary["leakage_effect"].get("delta"),
        "out": str(args.out_json),
    }
    print(json.dumps(headline, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
