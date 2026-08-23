"""Multi-step post-revocation agent environment for Suite B5.

Episodes reuse cached VLM decisions (no live inference). After a revoke the agent
may abstain, re-submit the same observation (stale carry-over), re-observe with a
fresh capture clock, or escalate to another cached model.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Hashable

from experiments.paper2_common import LAMBDA_HAZARD, MU_WRONGFUL_REVOKE, Measurement, classify
from experiments.rl.budget_env import instantaneous_reward
from experiments.run_b2_validity_frontier import ReplayRecord, gate_publishes

POST_REVOKE_ACTIONS = ("abstain", "retry_stale", "reobserve", "escalate")
ACTION_INDEX = {a: i for i, a in enumerate(POST_REVOKE_ACTIONS)}

# Traditional size ladder starting from the primary 7B model.
ESCALATE_ORDER = (
    "qwen2.5vl:7b",
    "qwen2.5vl:32b",
    "gemma3:12b",
    "llama3.2-vision:11b",
    "qwen2.5vl:3b",
)

STEP_COST = 0.05  # small penalty per extra tool call after the first shot
REOBSERVE_THINK_MS = 200.0
P_RECOVER_DEFAULT = 0.35  # chance the plant returns to nominal when re-observing after drift
P_RECOVER = P_RECOVER_DEFAULT  # mutable for sweeps


def set_p_recover(value: float) -> None:
    global P_RECOVER
    P_RECOVER = float(value)


def adaptive_freshness_ms(latency_ms: float) -> int:
    return max(2000, int(math.ceil(latency_ms)) + 250)


@dataclass
class PlantState:
    drift_fires: bool
    drift_time_ms: float
    recovered: bool = False

    def invalid_at(self, submit_ms: float) -> bool:
        if self.recovered:
            return False
        return bool(self.drift_fires and submit_ms >= self.drift_time_ms)


@dataclass
class AttemptResult:
    scored: dict
    reward: float
    published: bool
    wall_ms: float
    model: str


def submit_attempt(
    rec: ReplayRecord,
    *,
    capture_ms: float,
    wall_start_ms: float,
    plant: PlantState,
    gate: str = "xair",
    freshness_ms: int | None = None,
) -> AttemptResult:
    """Submit one cached decision under capture anchoring."""
    latency = float(rec.latency_ms)
    submit_ms = capture_ms + latency
    elapsed = submit_ms - capture_ms  # == latency under capture
    w = freshness_ms if freshness_ms is not None else adaptive_freshness_ms(latency)
    invalid = plant.invalid_at(submit_ms)
    preconds = rec.preconds_hold_drifted if invalid else rec.preconds_hold_nominal
    published, reason = gate_publishes(
        gate,
        elapsed_ms=elapsed,
        freshness_ms=w,
        preconds_hold=preconds,
        schema_valid=getattr(rec, "gate_schema_valid", rec.schema_valid),
    )
    measurement = Measurement(
        context_valid_before=not invalid,
        context_valid_after=not invalid,
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
            "model": rec.model,
            "published": published,
            "reason": reason,
            "inference_latency_ms": latency,
            "capture_ms": capture_ms,
            "submit_ms": submit_ms,
            "freshness_ms": w,
            "invalid_at_submit": invalid,
            "grounding_correct": rec.grounding_correct,
        }
    )
    return AttemptResult(
        scored=scored,
        reward=instantaneous_reward(scored),
        published=published,
        wall_ms=submit_ms,
        model=rec.model,
    )


def next_escalate_model(current: str, available: dict[str, ReplayRecord]) -> str | None:
    try:
        start = ESCALATE_ORDER.index(current)
    except ValueError:
        start = -1
    for name in ESCALATE_ORDER[start + 1 :]:
        if name in available and name != current:
            return name
    return None


REASON_FEATURE_KEYS = ("stale_window", "precondition_failed", "schema_invalid")


def episode_features(
    first: AttemptResult,
    rec: ReplayRecord,
    *,
    n_steps: int,
    can_escalate: bool,
) -> list[float]:
    """Observable features after the first revoke (no GT-derived labels)."""
    lat = float(rec.latency_ms)
    reason = str(first.scored.get("reason") or "")
    return [
        lat / 1000.0,
        math.log1p(lat),
        float(rec.n_preconditions) / 10.0,
        1.0 if rec.schema_valid else 0.0,
        *[1.0 if reason == key else 0.0 for key in REASON_FEATURE_KEYS],
        float(n_steps),
        1.0 if can_escalate else 0.0,
    ]


def episode_state(first: AttemptResult, rec: ReplayRecord, *, can_escalate: bool) -> Hashable:
    reason = str(first.scored.get("reason") or "none")
    lat = float(rec.latency_ms)
    if lat < 2000:
        bucket = "lt2"
    elif lat < 4000:
        bucket = "2to4"
    else:
        bucket = "ge4"
    return (bucket, reason, rec.model, can_escalate)


def oracle_post_action(
    models: dict[str, ReplayRecord],
    current: str,
    *,
    plant: PlantState,
    wall_ms: float,
    rng: random.Random,
) -> str:
    """Retrospective best post-revoke action under one-step lookahead."""
    best_a = "abstain"
    best_r = 0.0  # abstain yields 0 instantaneous reward
    for action in ("retry_stale", "reobserve", "escalate"):
        if action == "escalate" and next_escalate_model(current, models) is None:
            continue
        r = _lookahead_reward(action, models, current, plant=plant, wall_ms=wall_ms, rng=rng)
        if r > best_r:
            best_r = r
            best_a = action
    return best_a


def _lookahead_reward(
    action: str,
    models: dict[str, ReplayRecord],
    current: str,
    *,
    plant: PlantState,
    wall_ms: float,
    rng: random.Random,
) -> float:
    if action == "abstain":
        return 0.0
    if action == "retry_stale":
        rec = models[current]
        latency = float(rec.latency_ms)
        submit_ms = wall_ms
        elapsed = wall_ms
        w = adaptive_freshness_ms(latency)
        invalid = plant.invalid_at(submit_ms)
        preconds = rec.preconds_hold_drifted if invalid else rec.preconds_hold_nominal
        published, _ = gate_publishes(
            "xair",
            elapsed_ms=elapsed,
            freshness_ms=w,
            preconds_hold=preconds,
            schema_valid=getattr(rec, "gate_schema_valid", rec.schema_valid),
        )
        measurement = Measurement(
            context_valid_before=not invalid,
            context_valid_after=not invalid,
            version_before=1,
            version_after=1,
        )
        scored = classify(
            gt_action=rec.gt_action,
            model_action=rec.model_action,
            published=published,
            measurement=measurement,
        )
        return instantaneous_reward(scored) - STEP_COST
    if action == "reobserve":
        # Expected reward under stochastic plant recovery.
        def _once(recovered: bool) -> float:
            plant2 = PlantState(plant.drift_fires, plant.drift_time_ms, plant.recovered or recovered)
            capture = wall_ms + REOBSERVE_THINK_MS
            att = submit_attempt(models[current], capture_ms=capture, wall_start_ms=capture, plant=plant2)
            return att.reward - STEP_COST

        if plant.drift_fires and not plant.recovered:
            return P_RECOVER * _once(True) + (1.0 - P_RECOVER) * _once(False)
        return _once(False)
    if action == "escalate":
        nxt = next_escalate_model(current, models)
        if nxt is None:
            return -STEP_COST
        plant2 = PlantState(plant.drift_fires, plant.drift_time_ms, plant.recovered)
        capture = wall_ms + REOBSERVE_THINK_MS
        att = submit_attempt(models[nxt], capture_ms=capture, wall_start_ms=capture, plant=plant2)
        return att.reward - STEP_COST
    return 0.0


def run_episode(
    models: dict[str, ReplayRecord],
    *,
    primary: str,
    policy_name: str,
    choose_action,
    p_drift: float,
    drift_offset_ms: float,
    rng: random.Random,
    max_extra_steps: int = 2,
) -> dict:
    """
    Run one multi-step episode.

    ``choose_action(feats, state, available_actions) -> str`` is invoked after each revoke.
    """
    if primary not in models:
        raise KeyError(primary)
    plant = PlantState(
        drift_fires=rng.random() < p_drift,
        drift_time_ms=float(drift_offset_ms),
    )
    current = primary
    capture_ms = 0.0
    attempts: list[AttemptResult] = []
    total_reward = 0.0
    steps = 0

    first = submit_attempt(models[current], capture_ms=capture_ms, wall_start_ms=0.0, plant=plant)
    attempts.append(first)
    total_reward += first.reward
    steps += 1
    wall = first.wall_ms

    if first.published:
        return _finalize_episode(
            attempts, total_reward, policy_name, plant, steps, terminated="published"
        )

    extra = 0
    while extra < max_extra_steps:
        can_esc = next_escalate_model(current, models) is not None
        feats = episode_features(attempts[-1], models[current], n_steps=steps, can_escalate=can_esc)
        state = episode_state(attempts[-1], models[current], can_escalate=can_esc)
        available = list(POST_REVOKE_ACTIONS)
        if not can_esc:
            available = [a for a in available if a != "escalate"]
        action = choose_action(feats, state, available, models, current, plant, wall, rng)

        if action == "abstain":
            return _finalize_episode(
                attempts, total_reward, policy_name, plant, steps, terminated="abstain"
            )

        if action == "retry_stale":
            # Same capture clock; evidence ages with wall time.
            rec = models[current]
            latency = float(rec.latency_ms)
            submit_ms = wall + 50.0
            elapsed = submit_ms - capture_ms
            w = adaptive_freshness_ms(latency)
            invalid = plant.invalid_at(submit_ms)
            preconds = rec.preconds_hold_drifted if invalid else rec.preconds_hold_nominal
            published, reason = gate_publishes(
                "xair",
                elapsed_ms=elapsed,
                freshness_ms=w,
                preconds_hold=preconds,
                schema_valid=getattr(rec, "gate_schema_valid", rec.schema_valid),
            )
            measurement = Measurement(
                context_valid_before=not invalid,
                context_valid_after=not invalid,
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
                    "model": rec.model,
                    "published": published,
                    "reason": reason,
                    "inference_latency_ms": latency,
                    "freshness_ms": w,
                    "invalid_at_submit": invalid,
                    "grounding_correct": rec.grounding_correct,
                    "post_action": action,
                }
            )
            att = AttemptResult(scored, instantaneous_reward(scored) - STEP_COST, published, submit_ms, rec.model)
        elif action == "reobserve":
            if plant.drift_fires and not plant.recovered and rng.random() < P_RECOVER:
                plant.recovered = True
            capture_ms = wall + REOBSERVE_THINK_MS
            att = submit_attempt(
                models[current], capture_ms=capture_ms, wall_start_ms=capture_ms, plant=plant
            )
            att.scored["post_action"] = action
            att = AttemptResult(att.scored, att.reward - STEP_COST, att.published, att.wall_ms, att.model)
        elif action == "escalate":
            nxt = next_escalate_model(current, models)
            if nxt is None:
                return _finalize_episode(
                    attempts, total_reward, policy_name, plant, steps, terminated="abstain"
                )
            current = nxt
            capture_ms = wall + REOBSERVE_THINK_MS
            att = submit_attempt(
                models[current], capture_ms=capture_ms, wall_start_ms=capture_ms, plant=plant
            )
            att.scored["post_action"] = action
            att = AttemptResult(att.scored, att.reward - STEP_COST, att.published, att.wall_ms, att.model)
        else:
            return _finalize_episode(
                attempts, total_reward, policy_name, plant, steps, terminated="abstain"
            )

        attempts.append(att)
        total_reward += att.reward
        steps += 1
        extra += 1
        wall = att.wall_ms
        if att.published:
            return _finalize_episode(
                attempts, total_reward, policy_name, plant, steps, terminated="published"
            )

    return _finalize_episode(
        attempts, total_reward, policy_name, plant, steps, terminated="budget_exhausted"
    )


def _finalize_episode(
    attempts: list[AttemptResult],
    total_reward: float,
    policy_name: str,
    plant: PlantState,
    steps: int,
    *,
    terminated: str,
) -> dict:
    last = attempts[-1].scored
    any_hazard = any(a.scored.get("hazardous_publish") for a in attempts)
    any_success = any(a.scored.get("successful_actuation") for a in attempts)
    any_wrr = any(a.scored.get("wrongful_revoke") for a in attempts)
    # Episode-level flags for aggregate utility (terminal semantics).
    return {
        "policy": policy_name,
        "terminated": terminated,
        "steps": steps,
        "n_attempts": len(attempts),
        "total_reward": total_reward,
        "successful_actuation": bool(any_success and not any_hazard),
        "hazardous_publish": bool(any_hazard),
        "wrongful_revoke": bool(any_wrr and not any_success and not any_hazard),
        "correct_revoke": bool(last.get("correct_revoke") and terminated != "published"),
        "grounding_correct": bool(last.get("grounding_correct")),
        "context_valid_at_eval": last.get("context_valid_at_eval"),
        "final_model": attempts[-1].model,
        "final_published": bool(attempts[-1].published),
        "plant_drifted": plant.drift_fires,
        "plant_recovered": plant.recovered,
        "mean_reward": total_reward / max(1, len(attempts)),
        "inference_latency_ms": sum(float(a.scored.get("inference_latency_ms") or 0) for a in attempts),
    }


class AgentQLearning:
    """ε-greedy Q-learning over post-revoke actions."""

    def __init__(self, *, epsilon: float = 0.1, alpha: float = 0.2, seed: int = 0):
        self.epsilon = epsilon
        self.alpha = alpha
        self.rng = random.Random(seed)
        self.name = f"qlearn:e{epsilon}"
        self.Q: dict[Hashable, list[float]] = {}

    def _row(self, state: Hashable) -> list[float]:
        if state not in self.Q:
            self.Q[state] = [0.0] * len(POST_REVOKE_ACTIONS)
        return self.Q[state]

    def select(self, state: Hashable, available: list[str]) -> str:
        if self.rng.random() < self.epsilon:
            return self.rng.choice(available)
        values = self._row(state)
        best = max(available, key=lambda a: values[ACTION_INDEX[a]])
        return best

    def update(self, state: Hashable, action: str, reward: float) -> None:
        i = ACTION_INDEX[action]
        q = self._row(state)
        q[i] += self.alpha * (reward - q[i])


class AgentLinUCB:
    """Disjoint LinUCB over post-revoke actions."""

    def __init__(self, n_features: int, *, alpha: float = 1.0, l2: float = 1.0):
        import numpy as np

        self.np = np
        self.n_features = n_features
        self.alpha = alpha
        self.name = f"linucb:a{alpha}"
        eye = np.eye(n_features, dtype=np.float64)
        n = len(POST_REVOKE_ACTIONS)
        self.Ainv = {i: (1.0 / l2) * eye.copy() for i in range(n)}
        self.b = {i: np.zeros(n_features, dtype=np.float64) for i in range(n)}
        self.theta = {i: np.zeros(n_features, dtype=np.float64) for i in range(n)}

    def select(self, feats: list[float], available: list[str]) -> str:
        x = self.np.asarray(feats, dtype=self.np.float64)
        best_a = available[0]
        best_ucb = -1e18
        for a in available:
            i = ACTION_INDEX[a]
            mean = float(self.theta[i] @ x)
            bonus = self.alpha * float(self.np.sqrt(max(0.0, x @ self.Ainv[i] @ x)))
            ucb = mean + bonus
            if ucb > best_ucb:
                best_ucb = ucb
                best_a = a
        return best_a

    def update(self, feats: list[float], action: str, reward: float) -> None:
        i = ACTION_INDEX[action]
        x = self.np.asarray(feats, dtype=self.np.float64)
        Ainv = self.Ainv[i]
        Ax = Ainv @ x
        denom = 1.0 + float(x @ Ax)
        self.Ainv[i] = Ainv - self.np.outer(Ax, Ax) / denom
        self.b[i] = self.b[i] + reward * x
        self.theta[i] = self.Ainv[i] @ self.b[i]
