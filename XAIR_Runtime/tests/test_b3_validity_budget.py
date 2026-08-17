"""Unit tests for B3 validity-budget RL helpers (no GPU)."""

from __future__ import annotations

from experiments.rl.budget_env import (
    ACTIONS,
    BudgetAction,
    instantaneous_reward,
    feature_vector,
    discrete_state,
)
from experiments.rl.policies import FixedBudget, LinUCB, QLearning
from experiments.run_b2_validity_frontier import ReplayRecord


class _FakeRec:
    def __init__(self):
        self.frame_id = "f"
        self.use_case = "uc1_triage"
        self.model = "qwen2.5vl:7b"
        self.prompt_variant = "blind"
        self.latency_ms = 3500.0
        self.gt_action = "STOP"
        self.model_action = "STOP"
        self.grounding_correct = True
        self.n_preconditions = 2
        self.schema_valid = True
        self.severity = "minor"
        self.defect_present = True
        self.category = "cable"
        self.preconds_hold_nominal = True
        self.preconds_hold_drifted = False


def test_actions_cover_grid():
    assert len(ACTIONS) == 10
    assert all(isinstance(a, BudgetAction) for a in ACTIONS)


def test_instantaneous_reward_weights():
    assert instantaneous_reward({"successful_actuation": True}) == 1.0
    assert instantaneous_reward({"hazardous_publish": True}) == -5.0
    assert instantaneous_reward({"wrongful_revoke": True}) == -1.0
    assert instantaneous_reward(
        {"successful_actuation": True, "hazardous_publish": True}
    ) == 1.0 - 5.0


def test_features_and_state_shapes():
    rec = _FakeRec()
    raw = {"confidence": 0.8, "defect_judgement": True}
    feats = feature_vector(raw, rec)  # type: ignore[arg-type]
    # latency, log lat, conf, n_pre, schema, defect_hat + 5 use-cases + 5 models
    assert len(feats) == 6 + 5 + 5
    priv = feature_vector(raw, rec, privileged_severity=True)  # type: ignore[arg-type]
    assert len(priv) == len(feats) + 3
    state = discrete_state(raw, rec)  # type: ignore[arg-type]
    assert state[0] == "2to4"
    assert len(state) == 3
    assert "minor" not in state
    priv_state = discrete_state(raw, rec, privileged_severity=True)  # type: ignore[arg-type]
    assert priv_state[1] == "minor"


def test_linucb_selects_and_updates():
    rec = _FakeRec()
    raw = {"confidence": 0.5, "defect_judgement": False}
    feats = feature_vector(raw, rec)  # type: ignore[arg-type]
    pol = LinUCB(len(feats), alpha=0.5)
    a = pol.select(feats)
    assert a in ACTIONS
    pol.update(feats, a, reward=0.5)


def test_qlearning_epsilon_and_update():
    pol = QLearning(epsilon=0.0, seed=1)
    state = ("lt2", "none", "qwen2.5vl:7b", "ok")
    a = pol.select(None, state)
    assert a in ACTIONS
    pol.update(None, a, 1.0, state)


def test_fixed_budget():
    a = BudgetAction(2000, "strict")
    pol = FixedBudget(a)
    assert pol.select().name == "w2000_strict"
