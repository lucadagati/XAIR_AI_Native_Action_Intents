"""Unit tests for B5 agent re-observation helpers (no GPU)."""

from __future__ import annotations

import random

from experiments.rl.agent_env import (
    POST_REVOKE_ACTIONS,
    AgentLinUCB,
    AgentQLearning,
    PlantState,
    adaptive_freshness_ms,
    next_escalate_model,
    run_episode,
    submit_attempt,
)


class _FakeRec:
    def __init__(self, model="qwen2.5vl:7b", latency=3000.0, grounding=True, action="STOP"):
        self.frame_id = "f"
        self.use_case = "uc1_triage"
        self.model = model
        self.prompt_variant = "blind"
        self.latency_ms = latency
        self.gt_action = "STOP"
        self.model_action = action
        self.grounding_correct = grounding
        self.n_preconditions = 2
        self.schema_valid = True
        self.gate_schema_valid = True
        self.severity = "minor"
        self.defect_present = True
        self.category = "cable"
        self.preconds_hold_nominal = True
        self.preconds_hold_drifted = False


def test_adaptive_freshness():
    assert adaptive_freshness_ms(1000) == 2000
    assert adaptive_freshness_ms(3500) == 3750


def test_plant_invalidity():
    p = PlantState(drift_fires=True, drift_time_ms=250)
    assert not p.invalid_at(100)
    assert p.invalid_at(300)
    p.recovered = True
    assert not p.invalid_at(300)


def test_escalate_order():
    models = {
        "qwen2.5vl:7b": _FakeRec("qwen2.5vl:7b"),
        "qwen2.5vl:32b": _FakeRec("qwen2.5vl:32b"),
    }
    assert next_escalate_model("qwen2.5vl:7b", models) == "qwen2.5vl:32b"
    assert next_escalate_model("qwen2.5vl:32b", models) is None


def test_submit_and_single_shot_episode():
    models = {"qwen2.5vl:7b": _FakeRec()}
    plant = PlantState(True, 250)
    att = submit_attempt(models["qwen2.5vl:7b"], capture_ms=0.0, wall_start_ms=0.0, plant=plant)
    assert att.scored["invalid_at_submit"] is True
    assert att.published is False

    def abstain(*_a, **_k):
        return "abstain"

    ep = run_episode(
        models,
        primary="qwen2.5vl:7b",
        policy_name="single_shot",
        choose_action=abstain,
        p_drift=1.0,
        drift_offset_ms=0,
        rng=random.Random(0),
    )
    assert ep["terminated"] == "abstain"
    assert ep["steps"] == 1


def test_reobserve_can_recover():
    models = {"qwen2.5vl:7b": _FakeRec()}

    def always_reobs(*_a, **_k):
        return "reobserve"

    # Force recovery path with many trials; at least some should recover.
    recovered = 0
    for seed in range(40):
        ep = run_episode(
            models,
            primary="qwen2.5vl:7b",
            policy_name="reobserve",
            choose_action=always_reobs,
            p_drift=1.0,
            drift_offset_ms=0,
            rng=random.Random(seed),
            max_extra_steps=1,
        )
        if ep["plant_recovered"]:
            recovered += 1
    assert recovered > 0


def test_learners_select():
    q = AgentQLearning(epsilon=0.0, seed=1)
    a = q.select(("lt2", "precondition_failed", "qwen2.5vl:7b", True), list(POST_REVOKE_ACTIONS))
    assert a in POST_REVOKE_ACTIONS
    q.update(("lt2", "precondition_failed", "qwen2.5vl:7b", True), a, 1.0)

    lin = AgentLinUCB(9, alpha=0.5)
    feats = [0.1] * 9
    a2 = lin.select(feats, list(POST_REVOKE_ACTIONS))
    assert a2 in POST_REVOKE_ACTIONS
    lin.update(feats, a2, 0.5)
