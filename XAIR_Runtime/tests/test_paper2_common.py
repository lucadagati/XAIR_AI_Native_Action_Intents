"""Tests for the Paper 2 harness: measured admissibility and outcome classification."""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper2_common import (  # noqa: E402
    Measurement,
    classify,
    drift_fires_before_submit,
    replay_elapsed_ms,
    summarize,
    utility,
    wilson_ci,
)

STABLE_VALID = Measurement(True, True, 7, 7)
STABLE_INVALID = Measurement(False, False, 7, 7)
MOVED = Measurement(True, False, 7, 8)


def test_measurement_stability():
    assert STABLE_VALID.stable and STABLE_VALID.context_valid_at_eval is True
    assert STABLE_INVALID.stable and STABLE_INVALID.context_valid_at_eval is False
    assert not MOVED.stable and MOVED.context_valid_at_eval is None


def test_stale_publication_requires_invalid_context():
    c = classify(gt_action="RESUME", model_action="RESUME", published=True, measurement=STABLE_INVALID)
    assert c["stale_publish"] and c["hazardous_publish"]
    assert not c["successful_actuation"] and not c["unsafe_publish"]


def test_successful_actuation_needs_valid_context_and_right_action():
    c = classify(gt_action="RESUME", model_action="RESUME", published=True, measurement=STABLE_VALID)
    assert c["successful_actuation"] and not c["hazardous_publish"]


def test_unsafe_publication_is_a_grounding_error_reaching_the_plant():
    c = classify(gt_action="STOP", model_action="RESUME", published=True, measurement=STABLE_VALID)
    assert c["unsafe_publish"] and c["hazardous_publish"]
    assert not c["stale_publish"] and not c["successful_actuation"]


def test_gate_blocking_a_grounding_error_is_scored_separately():
    c = classify(gt_action="STOP", model_action="RESUME", published=False, measurement=STABLE_VALID)
    assert c["blocked_grounding_error"]
    assert not c["wrongful_revoke"] and not c["correct_revoke"]


def test_wrongful_revocation_only_when_action_was_right():
    c = classify(gt_action="RESUME", model_action="RESUME", published=False, measurement=STABLE_VALID)
    assert c["wrongful_revoke"] and not c["blocked_grounding_error"]


def test_correct_revocation_under_drift():
    c = classify(gt_action="RESUME", model_action="RESUME", published=False, measurement=STABLE_INVALID)
    assert c["correct_revoke"] and not c["wrongful_revoke"]


def test_ambiguous_trials_score_nothing():
    c = classify(gt_action="RESUME", model_action="RESUME", published=True, measurement=MOVED)
    assert c["unknown"]
    assert not any(
        c[k]
        for k in (
            "stale_publish",
            "unsafe_publish",
            "successful_actuation",
            "wrongful_revoke",
            "correct_revoke",
            "blocked_grounding_error",
        )
    )


def test_capture_anchor_exposes_inference_latency():
    assert replay_elapsed_ms(6000.0, "capture") == 6000.0
    assert replay_elapsed_ms(6000.0, "emission") == 0.0


def test_drift_before_submit_depends_on_offset_vs_latency():
    rng = random.Random(0)
    _, invalid = drift_fires_before_submit(
        p_drift=1.0, drift_offset_ms=100, inference_latency_ms=6000, rng=rng
    )
    assert invalid
    _, invalid_late = drift_fires_before_submit(
        p_drift=1.0, drift_offset_ms=9000, inference_latency_ms=6000, rng=rng
    )
    assert not invalid_late


def test_no_drift_leaves_context_valid():
    rng = random.Random(0)
    scheduled, invalid = drift_fires_before_submit(
        p_drift=0.0, drift_offset_ms=0, inference_latency_ms=6000, rng=rng
    )
    assert not scheduled and not invalid


def test_summarize_separates_hazard_sources():
    rows = [
        classify(gt_action="RESUME", model_action="RESUME", published=True, measurement=STABLE_VALID),
        classify(gt_action="RESUME", model_action="RESUME", published=True, measurement=STABLE_INVALID),
        classify(gt_action="STOP", model_action="RESUME", published=True, measurement=STABLE_VALID),
        classify(gt_action="RESUME", model_action="RESUME", published=False, measurement=STABLE_VALID),
    ]
    s = summarize(rows)
    assert s["known"] == 4
    assert s["SAR_k"] == 1
    assert s["SER_k"] == 1
    assert s["unsafe_publish_k"] == 1
    assert s["hazardous_publish_k"] == 2
    # Wrongful revocation is conditioned on valid context and a correct action: 2 such
    # trials here, one of which was revoked.
    assert s["WRR_n"] == 2 and s["WRR_k"] == 1


def test_summarize_reports_unknown_rate():
    rows = [
        classify(gt_action="RESUME", model_action="RESUME", published=True, measurement=MOVED),
        classify(gt_action="RESUME", model_action="RESUME", published=True, measurement=STABLE_VALID),
    ]
    s = summarize(rows)
    assert s["known"] == 1 and abs(s["unknown_rate"] - 0.5) < 1e-9


def test_summarize_empty():
    assert summarize([])["known"] == 0


def test_utility_penalises_hazard_more_than_lost_cycles():
    assert utility(1.0, 0.0, 0.0) > utility(1.0, 0.0, 0.5) > utility(1.0, 0.1, 0.5)


def test_wilson_ci_bounds():
    lo, hi = wilson_ci(0, 100)
    assert lo == 0.0 and 0.0 < hi < 0.05
    lo, hi = wilson_ci(50, 100)
    assert lo < 0.5 < hi
