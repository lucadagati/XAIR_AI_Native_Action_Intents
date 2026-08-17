"""Action space, features, and instantaneous rewards for B3 validity-budget learning."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from experiments.paper2_common import LAMBDA_HAZARD, MU_WRONGFUL_REVOKE, trial_reward
from experiments.run_b2_validity_frontier import ReplayRecord, replay_one

# Freshness levels on the latency scale of open VLMs (sub-second windows are vacuous).
FRESHNESS_ACTIONS_MS: tuple[int, ...] = (500, 1000, 2000, 4000, 8000)
STRICTNESS = ("lenient", "strict")  # freshness_only vs xair
GATE_FOR_STRICTNESS = {"lenient": "freshness_only", "strict": "xair"}

SEVERITY_LEVELS = ("none", "minor", "major")
USE_CASES = (
    "uc1_triage",
    "uc2_restart",
    "uc3_speed",
    "uc4_conflict",
    "uc5_safety",
)
MODELS = (
    "qwen2.5vl:3b",
    "qwen2.5vl:7b",
    "llama3.2-vision:11b",
    "gemma3:12b",
    "qwen2.5vl:32b",
)


@dataclass(frozen=True)
class BudgetAction:
    freshness_ms: int
    strictness: str  # lenient | strict

    @property
    def gate(self) -> str:
        return GATE_FOR_STRICTNESS[self.strictness]

    @property
    def name(self) -> str:
        return f"w{self.freshness_ms}_{self.strictness}"


ACTIONS: tuple[BudgetAction, ...] = tuple(
    BudgetAction(w, s) for w in FRESHNESS_ACTIONS_MS for s in STRICTNESS
)
ACTION_INDEX = {a.name: i for i, a in enumerate(ACTIONS)}


def instantaneous_reward(scored: dict) -> float:
    """Alias for :func:`trial_reward` (learners maximise mean trial reward)."""
    return trial_reward(scored)


def feature_vector(
    raw: dict[str, Any],
    rec: ReplayRecord,
    *,
    privileged_severity: bool = False,
) -> list[float]:
    """
    Features available at emission.

    Headline features exclude mask-derived ground-truth severity. Use-case is the
    known cell identity. ``defect_hat`` is the VLM's own defect judgement.
    ``privileged_severity`` re-adds GT severity one-hots for a leakage ablation.
    """
    lat = max(0.0, float(rec.latency_ms))
    conf = float(raw.get("confidence") or 0.0)
    n_pre = float(rec.n_preconditions)
    uc = str(rec.use_case or "uc1_triage")
    model = str(rec.model or "")
    defect_hat = 1.0 if raw.get("defect_judgement") else 0.0
    schema = 1.0 if rec.schema_valid else 0.0

    feats: list[float] = [
        lat / 1000.0,
        math.log1p(lat),
        conf,
        n_pre / 10.0,
        schema,
        defect_hat,
    ]
    for use in USE_CASES:
        feats.append(1.0 if uc == use else 0.0)
    for m in MODELS:
        feats.append(1.0 if model == m else 0.0)
    if privileged_severity:
        sev = str(rec.severity or "none")
        for level in SEVERITY_LEVELS:
            feats.append(1.0 if sev == level else 0.0)
    return feats


def discrete_state(
    raw: dict[str, Any],
    rec: ReplayRecord,
    *,
    privileged_severity: bool = False,
) -> tuple:
    """Coarse state for tabular Q-learning (headline: no GT severity)."""
    lat = float(rec.latency_ms)
    if lat < 2000:
        bucket = "lt2"
    elif lat < 4000:
        bucket = "2to4"
    elif lat < 8000:
        bucket = "4to8"
    else:
        bucket = "ge8"
    model = str(rec.model or "unknown")
    defect_hat = "def" if raw.get("defect_judgement") else "ok"
    if privileged_severity:
        return (bucket, str(rec.severity or "none"), model, defect_hat)
    return (bucket, model, defect_hat)


def apply_action(
    rec: ReplayRecord,
    action: BudgetAction,
    *,
    invalid_at_submit: bool,
    anchor: str = "capture",
) -> dict:
    scored = replay_one(
        rec,
        gate=action.gate,
        anchor=anchor,
        freshness_ms=action.freshness_ms,
        invalid_at_submit=invalid_at_submit,
    )
    scored["reward"] = instantaneous_reward(scored)
    scored["action"] = action.name
    return scored


def oracle_action(
    rec: ReplayRecord,
    *,
    invalid_at_submit: bool,
    anchor: str = "capture",
) -> tuple[BudgetAction, dict]:
    """Retrospective best action on this trial (breaks ties toward safer / tighter)."""
    best: BudgetAction | None = None
    best_scored: dict | None = None
    best_key: tuple | None = None
    for action in ACTIONS:
        scored = apply_action(
            rec, action, invalid_at_submit=invalid_at_submit, anchor=anchor
        )
        key = (
            scored["reward"],
            -int(scored["hazardous_publish"]),
            -action.freshness_ms,
            int(action.strictness == "strict"),
        )
        if best_key is None or key > best_key:
            best, best_scored, best_key = action, scored, key
    assert best is not None and best_scored is not None
    return best, best_scored
