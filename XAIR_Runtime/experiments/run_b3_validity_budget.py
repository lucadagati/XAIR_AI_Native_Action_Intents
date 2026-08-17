#!/usr/bin/env python3
"""
Suite B3 — learned validity budget (Phase RL).

Online policies select a freshness window and precondition strictness for each cached
perception decision. The environment is the same offline publication-boundary model as
B2, so learning does not re-invoke the VLM. Policies are compared against fixed budgets
and a retrospective oracle on instantaneous utility
``r = 1[SAR] - λ 1[Hazard] - μ 1[WRR]``.

Usage:
    python3 experiments/run_b3_validity_budget.py --tag phase_p
"""

from __future__ import annotations

import argparse
import csv
import json
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
    Notifier,
    summarize,
)
from experiments.paper2_splits import SPLIT_SEED, filter_by_split, load_split  # noqa: E402
from experiments.perception_cache import cache_path  # noqa: E402
from experiments.rl.budget_env import (  # noqa: E402
    ACTIONS,
    BudgetAction,
    apply_action,
    discrete_state,
    feature_vector,
    oracle_action,
)
from experiments.rl.policies import FixedBudget, LinUCB, QLearning  # noqa: E402
from experiments.run_b2_validity_frontier import (  # noqa: E402
    ReplayRecord,
    load_replay_records,
)

P_DRIFT_TRAIN = (0.25, 0.5, 0.75)
OFFSET_TRAIN_MS = (0, 250, 500, 1000, 2000, 4000)
EVAL_SETTINGS = (
    {"p_drift": 0.5, "drift_offset_ms": 250, "label": "headline"},
    {"p_drift": 0.75, "drift_offset_ms": 0, "label": "high_vol_early"},
    {"p_drift": 0.25, "drift_offset_ms": 2000, "label": "low_vol_late"},
)


def load_raw_by_key(tag: str) -> dict[tuple, dict]:
    path = cache_path(tag)
    latest: dict[tuple, dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("error"):
            continue
        if rec.get("prompt_variant") != "blind":
            continue
        key = (rec.get("frame_id"), rec.get("model"), rec.get("prompt_variant"), rec.get("use_case"))
        latest[key] = rec
    return latest


def paired_records(tag: str) -> list[tuple[ReplayRecord, dict]]:
    replay = load_replay_records(tag, models=None, variants={"blind"}, use_repaired=False)
    raw = load_raw_by_key(tag)
    out: list[tuple[ReplayRecord, dict]] = []
    for rec in replay:
        key = (rec.frame_id, rec.model, rec.prompt_variant, rec.use_case)
        if key not in raw:
            continue
        out.append((rec, raw[key]))
    return out


def sample_invalid(rec: ReplayRecord, rng: random.Random, p_drift: float, offset_ms: int) -> bool:
    if rng.random() >= p_drift:
        return False
    return offset_ms <= rec.latency_ms


def run_online(
    pairs: list[tuple[ReplayRecord, dict]],
    policy,
    *,
    epochs: int,
    seed: int,
    learn: bool,
    privileged_severity: bool = False,
) -> dict:
    rng = random.Random(seed)
    order = list(range(len(pairs)))
    rewards: list[float] = []
    scored_rows: list[dict] = []
    action_counts = defaultdict(int)

    for epoch in range(epochs):
        rng.shuffle(order)
        for idx in order:
            rec, raw = pairs[idx]
            p_drift = rng.choice(P_DRIFT_TRAIN)
            offset = rng.choice(OFFSET_TRAIN_MS)
            invalid = sample_invalid(rec, rng, p_drift, offset)
            feats = feature_vector(raw, rec, privileged_severity=privileged_severity)
            state = discrete_state(raw, rec, privileged_severity=privileged_severity)
            if isinstance(policy, FixedBudget):
                action = policy.select()
            elif isinstance(policy, LinUCB):
                action = policy.select(feats)
            else:
                action = policy.select(feats, state)
            scored = apply_action(rec, action, invalid_at_submit=invalid, anchor="capture")
            r = scored["reward"]
            rewards.append(r)
            action_counts[action.name] += 1
            scored_rows.append(scored)
            if learn:
                if isinstance(policy, LinUCB):
                    policy.update(feats, action, r)
                elif isinstance(policy, QLearning):
                    policy.update(feats, action, r, state)

    curve = []
    window = max(200, len(rewards) // 50)
    cum = 0.0
    for i, r in enumerate(rewards, 1):
        cum += r
        if i % window == 0 or i == len(rewards):
            curve.append({"step": i, "mean_reward": cum / i})

    agg = summarize(scored_rows)
    return {
        "policy": getattr(policy, "name", type(policy).__name__),
        "n_steps": len(rewards),
        "mean_reward": cum / len(rewards) if rewards else 0.0,
        "utility_aggregate": agg.get("utility", 0.0),
        "utility_legacy_rate_combo": agg.get("utility_legacy_rate_combo"),
        "summary": agg,
        "action_counts": dict(action_counts),
        "learning_curve": curve,
    }


def eval_frozen(
    pairs: list[tuple[ReplayRecord, dict]],
    policy,
    *,
    setting: dict,
    seed: int,
    privileged_severity: bool = False,
    keep_trials: bool = False,
) -> dict:
    rng = random.Random(seed + 17)
    scored_rows: list[dict] = []
    action_counts = defaultdict(int)
    old_eps = getattr(policy, "epsilon", None)
    if old_eps is not None:
        policy.epsilon = 0.0

    for rec, raw in pairs:
        invalid = sample_invalid(rec, rng, setting["p_drift"], setting["drift_offset_ms"])
        feats = feature_vector(raw, rec, privileged_severity=privileged_severity)
        state = discrete_state(raw, rec, privileged_severity=privileged_severity)
        if isinstance(policy, FixedBudget):
            action = policy.select()
        elif isinstance(policy, LinUCB):
            action = policy.select(feats)
        else:
            action = policy.select(feats, state)
        scored = apply_action(rec, action, invalid_at_submit=invalid, anchor="capture")
        scored["frame_id"] = rec.frame_id
        scored_rows.append(scored)
        action_counts[action.name] += 1

    if old_eps is not None:
        policy.epsilon = old_eps

    agg = summarize(scored_rows)
    out = {
        "setting": setting["label"],
        "p_drift": setting["p_drift"],
        "drift_offset_ms": setting["drift_offset_ms"],
        "mean_reward": sum(r["reward"] for r in scored_rows) / max(1, len(scored_rows)),
        "utility_aggregate": agg.get("utility", 0.0),
        "utility_legacy_rate_combo": agg.get("utility_legacy_rate_combo"),
        "summary": agg,
        "action_counts": dict(action_counts),
    }
    if keep_trials:
        out["trials"] = [
            {
                "frame_id": r["frame_id"],
                "reward": r["reward"],
                "successful_actuation": bool(r.get("successful_actuation")),
                "hazardous_publish": bool(r.get("hazardous_publish")),
                "wrongful_revoke": bool(r.get("wrongful_revoke")),
            }
            for r in scored_rows
        ]
    return out


def run_oracle(
    pairs: list[tuple[ReplayRecord, dict]],
    *,
    setting: dict,
    seed: int,
    keep_trials: bool = False,
) -> dict:
    rng = random.Random(seed + 99)
    scored_rows: list[dict] = []
    action_counts = defaultdict(int)
    for rec, raw in pairs:
        invalid = sample_invalid(rec, rng, setting["p_drift"], setting["drift_offset_ms"])
        action, scored = oracle_action(rec, invalid_at_submit=invalid, anchor="capture")
        scored["frame_id"] = rec.frame_id
        scored_rows.append(scored)
        action_counts[action.name] += 1
    agg = summarize(scored_rows)
    out = {
        "setting": setting["label"],
        "policy": "oracle",
        "mean_reward": sum(r["reward"] for r in scored_rows) / max(1, len(scored_rows)),
        "utility_aggregate": agg.get("utility", 0.0),
        "utility_legacy_rate_combo": agg.get("utility_legacy_rate_combo"),
        "summary": agg,
        "action_counts": dict(action_counts),
    }
    if keep_trials:
        out["trials"] = [
            {
                "frame_id": r["frame_id"],
                "reward": r["reward"],
                "successful_actuation": bool(r.get("successful_actuation")),
                "hazardous_publish": bool(r.get("hazardous_publish")),
                "wrongful_revoke": bool(r.get("wrongful_revoke")),
            }
            for r in scored_rows
        ]
    return out


def _strip_trials(row: dict) -> dict:
    return {k: v for k, v in row.items() if k != "trials"}


def _aggregate_headline(eval_results: list[dict]) -> list[dict]:
    headline = [r for r in eval_results if r.get("setting") == "headline"]
    summary_table = []
    for name in sorted({r["policy"] for r in headline}):
        rows = [r for r in headline if r["policy"] == name]
        summary_table.append(
            {
                "policy": name,
                "mean_reward": sum(r["mean_reward"] for r in rows) / len(rows),
                "utility": sum(r["utility_aggregate"] for r in rows) / len(rows),
                "SAR": sum(r["summary"]["SAR"] for r in rows) / len(rows),
                "hazard": sum(r["summary"]["hazardous_publish_rate"] for r in rows) / len(rows),
                "WRR": sum(r["summary"]["WRR"] for r in rows) / len(rows),
                "n_seeds": len(rows),
                "privileged": bool(rows[0].get("privileged")),
                "role": rows[0].get("role", "other"),
            }
        )
    summary_table.sort(key=lambda r: r["utility"], reverse=True)
    return summary_table


def _eval_policy_all_settings(
    policy,
    *,
    test_pairs,
    seed: int,
    policy_name: str,
    privileged: bool,
    role: str,
) -> list[dict]:
    rows = []
    for setting in EVAL_SETTINGS:
        keep = setting["label"] == "headline"
        ev = eval_frozen(
            test_pairs,
            policy,
            setting=setting,
            seed=seed,
            privileged_severity=privileged,
            keep_trials=keep,
        )
        ev["policy"] = policy_name
        ev["seed"] = seed
        ev["split"] = "test"
        ev["privileged"] = privileged
        ev["role"] = role
        rows.append(ev)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="B3 learned validity budget")
    parser.add_argument("--tag", default="phase_p")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--limit", type=int, default=None, help="cap records for smoke tests")
    parser.add_argument(
        "--privileged-severity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also train LinUCB/Q with GT severity one-hots (ablation only; default on)",
    )
    parser.add_argument("--no-notify", action="store_true")
    args = parser.parse_args()

    notifier = Notifier.from_env()
    if args.no_notify:
        notifier.enabled = False

    pairs = paired_records(args.tag)
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        print("[b3] no records", file=sys.stderr)
        return 1

    split = load_split()
    frame_id_fn = lambda p: p[0].frame_id  # noqa: E731
    train_pairs = filter_by_split(pairs, split="train", frame_id_fn=frame_id_fn, assignment=split)
    test_pairs = filter_by_split(pairs, split="test", frame_id_fn=frame_id_fn, assignment=split)
    if not train_pairs or not test_pairs:
        print("[b3] empty train or test split", file=sys.stderr)
        return 1

    n_feat = len(feature_vector(train_pairs[0][1], train_pairs[0][0], privileged_severity=False))
    n_feat_priv = len(feature_vector(train_pairs[0][1], train_pairs[0][0], privileged_severity=True))
    print(
        f"[b3] {len(pairs)} decisions (train={len(train_pairs)} test={len(test_pairs)}), "
        f"feature_dim={n_feat} priv={n_feat_priv}, actions={len(ACTIONS)}",
        flush=True,
    )
    notifier.send(
        f"Phase RL / B3 started\n"
        f"train={len(train_pairs)} test={len(test_pairs)} × {args.epochs} epochs"
    )

    train_results = []
    eval_results = []
    curves = []
    best_fixed_by_seed: dict[int, dict] = {}

    for seed in args.seeds:
        print(f"[b3] seed={seed}", flush=True)

        # All 10 fixed actions: score on train, freeze best, report all on test.
        train_fixed_u: list[tuple[float, BudgetAction, FixedBudget]] = []
        for action in ACTIONS:
            policy = FixedBudget(action)
            train_ev = eval_frozen(
                train_pairs,
                policy,
                setting=EVAL_SETTINGS[0],
                seed=seed,
                privileged_severity=False,
                keep_trials=False,
            )
            train_fixed_u.append((train_ev["mean_reward"], action, policy))
            train_ev["policy"] = policy.name
            train_ev["seed"] = seed
            train_ev["split"] = "train"
            train_ev["privileged"] = False
            train_ev["role"] = "fixed"
            train_results.append(train_ev)
            eval_results.extend(
                _eval_policy_all_settings(
                    policy,
                    test_pairs=test_pairs,
                    seed=seed,
                    policy_name=policy.name,
                    privileged=False,
                    role="fixed",
                )
            )

        train_fixed_u.sort(key=lambda t: t[0], reverse=True)
        best_u, best_action, best_policy = train_fixed_u[0]
        best_fixed_by_seed[seed] = {
            "action": best_action.name,
            "train_mean_reward": best_u,
        }
        eval_results.extend(
            _eval_policy_all_settings(
                best_policy,
                test_pairs=test_pairs,
                seed=seed,
                policy_name="best_fixed:train_selected",
                privileged=False,
                role="best_fixed",
            )
        )

        learners = [
            (LinUCB(n_feat, alpha=1.0), False, "learned"),
            (QLearning(epsilon=0.15, alpha=0.25, gamma=0.0, seed=seed), False, "learned"),
        ]
        if args.privileged_severity:
            lin_priv = LinUCB(n_feat_priv, alpha=1.0)
            lin_priv.name = "linucb:priv"
            q_priv = QLearning(epsilon=0.15, alpha=0.25, gamma=0.0, seed=seed)
            q_priv.name = "qlearn:priv"
            learners.extend(
                [
                    (lin_priv, True, "privileged"),
                    (q_priv, True, "privileged"),
                ]
            )

        for policy, priv, role in learners:
            online = run_online(
                train_pairs,
                policy,
                epochs=args.epochs,
                seed=seed,
                learn=True,
                privileged_severity=priv,
            )
            online["seed"] = seed
            online["split"] = "train"
            online["privileged"] = priv
            online["role"] = role
            train_results.append(online)
            policy_name = online["policy"]
            for pt in online["learning_curve"]:
                curves.append(
                    {
                        "seed": seed,
                        "policy": policy_name,
                        "step": pt["step"],
                        "mean_reward": pt["mean_reward"],
                        "privileged": priv,
                    }
                )
            eval_results.extend(
                _eval_policy_all_settings(
                    policy,
                    test_pairs=test_pairs,
                    seed=seed,
                    policy_name=policy_name,
                    privileged=priv,
                    role=role,
                )
            )

        for setting in EVAL_SETTINGS:
            keep = setting["label"] == "headline"
            ora = run_oracle(test_pairs, setting=setting, seed=seed, keep_trials=keep)
            ora["seed"] = seed
            ora["split"] = "test"
            ora["privileged"] = False
            ora["role"] = "oracle"
            eval_results.append(ora)

    summary_table = _aggregate_headline(eval_results)
    headline_compact = [
        r
        for r in summary_table
        if r["role"] in {"oracle", "learned", "best_fixed"} and not r["privileged"]
    ]
    all_fixed_table = [r for r in summary_table if r["role"] == "fixed"]
    privileged_table = [r for r in summary_table if r["privileged"]]

    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": out_dir / "b3_validity_budget.json",
        "eval_csv": out_dir / "b3_eval.csv",
        "curve_csv": out_dir / "b3_learning_curves.csv",
        "table_csv": out_dir / "b3_headline_table.csv",
        "fixed_csv": out_dir / "b3_all_fixed_table.csv",
        "trials": out_dir / "b3_eval_trials.jsonl",
    }
    trial_count = 0
    with paths["trials"].open("w") as fh:
        for r in eval_results:
            for t in r.get("trials") or []:
                fh.write(
                    json.dumps(
                        {
                            "policy": r["policy"],
                            "seed": r["seed"],
                            "setting": r.get("setting"),
                            "role": r.get("role"),
                            "privileged": r.get("privileged", False),
                            **t,
                        }
                    )
                    + "\n"
                )
                trial_count += 1

    eval_slim = [_strip_trials(r) for r in eval_results]
    payload = {
        "cache_tag": args.tag,
        "n_decisions": len(pairs),
        "split": {
            "seed": SPLIT_SEED,
            "n_train": len(train_pairs),
            "n_test": len(test_pairs),
        },
        "epochs": args.epochs,
        "seeds": args.seeds,
        "actions": [a.name for a in ACTIONS],
        "feature_dim": n_feat,
        "feature_dim_privileged": n_feat_priv,
        "privileged_severity_run": bool(args.privileged_severity),
        "cost_weights": {"lambda_hazard": LAMBDA_HAZARD, "mu_wrongful_revoke": MU_WRONGFUL_REVOKE},
        "best_fixed_by_seed": best_fixed_by_seed,
        "headline_table": headline_compact,
        "all_fixed_table": all_fixed_table,
        "privileged_table": privileged_table,
        "all_policies_table": summary_table,
        "eval": eval_slim,
        "train": [
            {k: v for k, v in t.items() if k != "learning_curve"}
            for t in train_results
        ],
        "n_eval_trials": trial_count,
    }
    paths["summary"].write_text(json.dumps(payload, indent=2, default=str))

    if eval_slim:
        flat = []
        for r in eval_slim:
            row = {k: v for k, v in r.items() if k != "summary" and k != "action_counts"}
            for mk, mv in r.get("summary", {}).items():
                if isinstance(mv, (int, float)):
                    row[f"sum_{mk}"] = mv
            flat.append(row)
        fields = sorted({k for row in flat for k in row})
        with paths["eval_csv"].open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(flat)

    if curves:
        with paths["curve_csv"].open("w", newline="") as fh:
            w = csv.DictWriter(
                fh, fieldnames=["seed", "policy", "step", "mean_reward", "privileged"]
            )
            w.writeheader()
            w.writerows(curves)

    with paths["table_csv"].open("w", newline="") as fh:
        fields = list(headline_compact[0].keys()) if headline_compact else list(summary_table[0].keys())
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(headline_compact or summary_table)

    if all_fixed_table:
        with paths["fixed_csv"].open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_fixed_table[0].keys()))
            w.writeheader()
            w.writerows(all_fixed_table)

    print(
        json.dumps(
            {
                "headline_table": headline_compact,
                "all_fixed_table": all_fixed_table,
                "privileged_table": privileged_table,
                "best_fixed_by_seed": best_fixed_by_seed,
                "paths": {k: str(v) for k, v in paths.items()},
            },
            indent=2,
            default=str,
        )
    )
    ranked = headline_compact or summary_table
    best = ranked[0]
    notifier.send(
        f"OK B3 validity budget\n"
        f"best={best['policy']} U={best['utility']:.3f} "
        f"reward={best['mean_reward']:.3f}\n"
        f"SAR={best['SAR']:.3f} hazard={best['hazard']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
