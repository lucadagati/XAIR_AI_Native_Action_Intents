#!/usr/bin/env python3
"""
Suite B5 — agent re-observation after revocation.

Offline multi-step episodes over the Phase-P multi-model cache. After an XAIR
revoke, policies choose among abstain / retry-stale / re-observe / escalate.
No live VLM calls: perception outputs are fixed from cache; only the
publication-boundary and plant-recovery dynamics evolve.

This suite does **not** re-infer the VLM. Perceptual errors are perfectly
correlated within a single decision (all models on a frame share the same
ground-truth action); independence holds only across models/frames.

Headline metrics are reported on held-out test frames (70/30 frame split);
learned policies train on train frames only, then evaluate frozen on test.

Usage:
    python3 experiments/run_b5_agent_policy.py --tag phase_p
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
    utility,
)
from experiments.rl.agent_env import (  # noqa: E402
    POST_REVOKE_ACTIONS,
    AgentLinUCB,
    AgentQLearning,
    PlantState,
    next_escalate_model,
    oracle_post_action,
    run_episode,
)
from experiments.run_b2_validity_frontier import ReplayRecord, load_replay_records  # noqa: E402
from experiments.paper2_splits import SPLIT_SEED, filter_by_split, load_split  # noqa: E402
from experiments.run_b4_model_routing import build_frame_index  # noqa: E402

PRIMARY = "qwen2.5vl:7b"
SEEDS = (1, 2, 3, 4, 5)
EVAL_SETTINGS = (
    {"label": "headline", "p_drift": 0.5, "drift_offset_ms": 250},
    {"label": "high_vol", "p_drift": 0.75, "drift_offset_ms": 0},
    {"label": "low_vol", "p_drift": 0.25, "drift_offset_ms": 2000},
)


def fixed_chooser(action: str):
    def _choose(feats, state, available, models, current, plant, wall, rng):
        if action in available:
            return action
        return "abstain"

    return _choose


def single_shot_chooser():
    """Never take a second step (episode ends after first revoke via abstain)."""
    return fixed_chooser("abstain")


def oracle_chooser():
    def _choose(feats, state, available, models, current, plant, wall, rng):
        a = oracle_post_action(models, current, plant=plant, wall_ms=wall, rng=rng)
        return a if a in available else "abstain"

    return _choose


def make_q_chooser(learner: AgentQLearning):
    pending: dict = {}

    def _choose(feats, state, available, models, current, plant, wall, rng):
        a = learner.select(state, available)
        pending["state"] = state
        pending["action"] = a
        return a

    def _update(episode_reward: float):
        if "state" in pending and "action" in pending:
            learner.update(pending["state"], pending["action"], episode_reward)
            pending.clear()

    _choose.update = _update  # type: ignore[attr-defined]
    return _choose


def make_linucb_chooser(learner: AgentLinUCB):
    pending: dict = {}

    def _choose(feats, state, available, models, current, plant, wall, rng):
        a = learner.select(feats, available)
        pending["feats"] = feats
        pending["action"] = a
        return a

    def _update(episode_reward: float):
        if "feats" in pending and "action" in pending:
            learner.update(pending["feats"], pending["action"], episode_reward)
            pending.clear()

    _choose.update = _update  # type: ignore[attr-defined]
    return _choose


def frame_id_from_frame(frame: dict[str, ReplayRecord]) -> str:
    return next(iter(frame.values())).frame_id


def compact_trials(rows: list[dict], *, policy: str) -> list[dict]:
    return [
        {
            "policy": policy,
            "seed": r["seed"],
            "setting": r.get("setting"),
            "frame_id": r["frame_id"],
            "reward": r.get("reward", r.get("total_reward")),
            "successful_actuation": bool(r.get("successful_actuation")),
            "hazardous_publish": bool(r.get("hazardous_publish")),
            "wrongful_revoke": bool(r.get("wrongful_revoke")),
        }
        for r in rows
    ]


def aggregate_episodes(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {}
    sar = sum(1 for r in rows if r["successful_actuation"]) / n
    hazard = sum(1 for r in rows if r["hazardous_publish"]) / n
    # WRR: episodes that ended with wrongful revoke and no success
    wrr_num = sum(1 for r in rows if r["wrongful_revoke"])
    wrr_den = sum(
        1
        for r in rows
        if r.get("context_valid_at_eval") is True and r.get("grounding_correct")
    )
    wrr = wrr_num / wrr_den if wrr_den else 0.0
    mean_total_reward = sum(r["total_reward"] for r in rows) / n
    return {
        "n": n,
        "utility": mean_total_reward,
        "SAR": sar,
        "hazard": hazard,
        "WRR": wrr,
        "mean_total_reward": mean_total_reward,
        "utility_legacy_rate_combo": utility(sar, hazard, wrr),
        "mean_steps": sum(r["steps"] for r in rows) / n,
        "publish_rate": sum(1 for r in rows if r["final_published"]) / n,
        "recovered_rate": sum(1 for r in rows if r["plant_recovered"]) / n,
    }


def run_policy_on_frames(
    frames: list[dict[str, ReplayRecord]],
    *,
    policy_name: str,
    choose_action,
    setting: dict,
    seed: int,
    train_updates: bool = False,
) -> list[dict]:
    rng = random.Random(seed)
    order = list(range(len(frames)))
    rng.shuffle(order)
    rows = []
    for idx in order:
        models = frames[idx]
        ep = run_episode(
            models,
            primary=PRIMARY,
            policy_name=policy_name,
            choose_action=choose_action,
            p_drift=setting["p_drift"],
            drift_offset_ms=setting["drift_offset_ms"],
            rng=rng,
        )
        ep["setting"] = setting["label"]
        ep["seed"] = seed
        ep["frame_id"] = frame_id_from_frame(models)
        ep["reward"] = ep["total_reward"]
        rows.append(ep)
        if train_updates and hasattr(choose_action, "update"):
            choose_action.update(ep["total_reward"])
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="B5 agent re-observation suite")
    parser.add_argument("--tag", default="phase_p")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = all multi-model frames")
    parser.add_argument("--notify", action="store_true")
    args = parser.parse_args()

    notifier = Notifier.from_env()
    records = load_replay_records(args.tag, models=None, variants={"blind"}, use_repaired=False)
    by_frame = build_frame_index(records)
    frames = list(by_frame.values())
    if args.max_frames > 0:
        frames = frames[: args.max_frames]

    split = load_split()
    train_frames = filter_by_split(
        frames, split="train", frame_id_fn=frame_id_from_frame, assignment=split
    )
    test_frames = filter_by_split(
        frames, split="test", frame_id_fn=frame_id_from_frame, assignment=split
    )
    if not train_frames or not test_frames:
        print("[b5] empty train or test split", file=sys.stderr)
        return 1

    print(
        f"[b5] multi-model frames: {len(frames)} "
        f"(train={len(train_frames)} test={len(test_frames)})",
        flush=True,
    )

    # Warm-up feature dim for LinUCB
    dummy_feats = [0.0] * 9
    policies_spec = [
        ("single_shot", lambda: single_shot_chooser(), False),
        ("retry_stale", lambda: fixed_chooser("retry_stale"), False),
        ("always_reobserve", lambda: fixed_chooser("reobserve"), False),
        ("always_escalate", lambda: fixed_chooser("escalate"), False),
        ("oracle", lambda: oracle_chooser(), False),
    ]

    table_rows: list[dict] = []
    learning_curves: list[dict] = []
    headline_detail: dict[str, list] = {}
    headline_trials: list[dict] = []

    for setting in EVAL_SETTINGS:
        for seed in SEEDS:
            # Fixed / oracle policies: evaluate on held-out test frames only
            for name, factory, _ in policies_spec:
                chooser = factory()
                rows = run_policy_on_frames(
                    test_frames,
                    policy_name=name,
                    choose_action=chooser,
                    setting=setting,
                    seed=seed,
                )
                agg = aggregate_episodes(rows)
                agg.update({"policy": name, "setting": setting["label"], "seed": seed})
                table_rows.append(agg)
                if setting["label"] == "headline":
                    headline_trials.extend(compact_trials(rows, policy=name))
                    if seed == SEEDS[0]:
                        headline_detail[name] = rows

            # Q-learning: train on train frames, evaluate frozen on test
            q = AgentQLearning(epsilon=0.1, alpha=0.25, seed=seed)
            q_choose = make_q_chooser(q)
            train_rows = run_policy_on_frames(
                train_frames,
                policy_name=q.name,
                choose_action=q_choose,
                setting=setting,
                seed=seed,
                train_updates=True,
            )
            window = 100
            for i in range(window, len(train_rows) + 1, window):
                chunk = train_rows[i - window : i]
                learning_curves.append(
                    {
                        "policy": q.name,
                        "setting": setting["label"],
                        "seed": seed,
                        "episode": i,
                        "mean_total_reward": sum(r["total_reward"] for r in chunk) / len(chunk),
                        "SAR": sum(1 for r in chunk if r["successful_actuation"]) / len(chunk),
                    }
                )
            old_eps = q.epsilon
            q.epsilon = 0.0
            eval_rows = run_policy_on_frames(
                test_frames,
                policy_name=q.name,
                choose_action=q_choose,
                setting=setting,
                seed=seed + 1000,
                train_updates=False,
            )
            q.epsilon = old_eps
            agg = aggregate_episodes(eval_rows)
            agg.update({"policy": q.name, "setting": setting["label"], "seed": seed})
            table_rows.append(agg)
            if setting["label"] == "headline":
                headline_trials.extend(compact_trials(eval_rows, policy=q.name))
                if seed == SEEDS[0]:
                    headline_detail[q.name] = eval_rows

            # LinUCB: train on train, evaluate frozen on test
            lin = AgentLinUCB(len(dummy_feats), alpha=0.75)
            lin_choose = make_linucb_chooser(lin)
            lin_train_rows = run_policy_on_frames(
                train_frames,
                policy_name=lin.name,
                choose_action=lin_choose,
                setting=setting,
                seed=seed,
                train_updates=True,
            )
            for i in range(window, len(lin_train_rows) + 1, window):
                chunk = lin_train_rows[i - window : i]
                learning_curves.append(
                    {
                        "policy": lin.name,
                        "setting": setting["label"],
                        "seed": seed,
                        "episode": i,
                        "mean_total_reward": sum(r["total_reward"] for r in chunk) / len(chunk),
                        "SAR": sum(1 for r in chunk if r["successful_actuation"]) / len(chunk),
                    }
                )
            eval_rows = run_policy_on_frames(
                test_frames,
                policy_name=lin.name,
                choose_action=lin_choose,
                setting=setting,
                seed=seed + 1000,
                train_updates=False,
            )
            agg = aggregate_episodes(eval_rows)
            agg.update({"policy": lin.name, "setting": setting["label"], "seed": seed})
            table_rows.append(agg)
            if setting["label"] == "headline":
                headline_trials.extend(compact_trials(eval_rows, policy=lin.name))
                if seed == SEEDS[0]:
                    headline_detail[lin.name] = eval_rows

    # Mean over seeds for each policy × setting
    grouped: dict[tuple, list] = defaultdict(list)
    for r in table_rows:
        grouped[(r["policy"], r["setting"])].append(r)

    summary_rows = []
    for (policy, setting), xs in sorted(grouped.items()):
        summary_rows.append(
            {
                "policy": policy,
                "setting": setting,
                "utility": sum(x["utility"] for x in xs) / len(xs),
                "SAR": sum(x["SAR"] for x in xs) / len(xs),
                "hazard": sum(x["hazard"] for x in xs) / len(xs),
                "WRR": sum(x["WRR"] for x in xs) / len(xs),
                "mean_total_reward": sum(x["mean_total_reward"] for x in xs) / len(xs),
                "mean_steps": sum(x["mean_steps"] for x in xs) / len(xs),
                "publish_rate": sum(x["publish_rate"] for x in xs) / len(xs),
                "n_seeds": len(xs),
            }
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table_csv = RESULTS_DIR / "b5_policy_table.csv"
    with table_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    raw_csv = RESULTS_DIR / "b5_policy_by_seed.csv"
    with raw_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table_rows[0].keys()))
        w.writeheader()
        w.writerows(table_rows)

    curve_csv = RESULTS_DIR / "b5_learning_curves.csv"
    if learning_curves:
        with curve_csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(learning_curves[0].keys()))
            w.writeheader()
            w.writerows(learning_curves)

    headline = [r for r in summary_rows if r["setting"] == "headline"]
    headline.sort(key=lambda r: -r["utility"])
    headline_csv = RESULTS_DIR / "b5_headline_table.csv"
    with headline_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(headline[0].keys()))
        w.writeheader()
        w.writerows(headline)

    out = {
        "n_frames": len(frames),
        "split": {
            "seed": SPLIT_SEED,
            "n_train": len(train_frames),
            "n_test": len(test_frames),
        },
        "seeds": list(SEEDS),
        "p_recover": 0.35,
        "headline": headline,
        "summary": summary_rows,
        "cost_weights": {"lambda_hazard": LAMBDA_HAZARD, "mu_wrr": MU_WRONGFUL_REVOKE},
    }
    json_path = RESULTS_DIR / "b5_agent_policy.json"
    json_path.write_text(json.dumps(out, indent=2))

    trials_path = RESULTS_DIR / "b5_eval_trials.jsonl"
    with trials_path.open("w") as fh:
        for t in headline_trials:
            fh.write(json.dumps(t) + "\n")

    print(json.dumps({"out": str(json_path), "headline_top": headline[:6]}, indent=2))
    notifier.send(f"B5 done: {len(frames)} frames, best={headline[0]['policy']} U={headline[0]['utility']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
