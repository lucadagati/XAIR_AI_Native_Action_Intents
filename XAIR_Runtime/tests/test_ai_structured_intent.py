"""Tests for the intent producer: no leakage, no injection, explicit anchoring."""

from __future__ import annotations

from xair.ai.structured_intent import (
    PROMPT_VARIANTS,
    PerceptionResult,
    StructuredIntentProducer,
    _ensure_ais_envelope,
    _extract_json,
    build_submission,
    precondition_syntax_ok,
    repair_precondition,
)

EPISODE = {
    "frame_id": "visa_cashew_000",
    "source_dataset": "visa",
    "category": "cashew",
    "defect_present": True,
    "ground_truth_action": "REJECT_TO_BIN",
    "context": {"line": {"state": "RUN"}, "gripper": {"state": "OPEN"}, "robot": {"speed": 0.05}},
}


def _producer() -> StructuredIntentProducer:
    return StructuredIntentProducer.__new__(StructuredIntentProducer)


def _prompt(variant: str, use_case: str = "uc1_triage") -> str:
    return StructuredIntentProducer.build_user_prompt(
        _producer(), EPISODE, variant=variant, use_case=use_case
    )


def test_extract_json_plain():
    assert _extract_json('{"id": "abc", "source": "ai"}')["source"] == "ai"


def test_extract_json_embedded():
    assert _extract_json('prefix {"action":"RESUME","x":0} suffix')["action"] == "RESUME"


def test_blind_prompts_are_invariant_to_labels():
    """
    The real leakage invariant: a blind prompt must be byte-identical whatever the
    ground-truth action and defect label are. Enumerating the action space is task
    specification; singling out the answer is not.
    """
    other = dict(EPISODE, defect_present=False, ground_truth_action="ACCEPT")
    for variant in ("blind", "blind_cot", "blind_noctx"):
        a = StructuredIntentProducer.build_user_prompt(
            _producer(), EPISODE, variant=variant, use_case="uc1_triage"
        )
        b = StructuredIntentProducer.build_user_prompt(
            _producer(), other, variant=variant, use_case="uc1_triage"
        )
        assert a == b, variant
        assert "defect_present" not in a, variant
        assert "Target action" not in a, variant


def test_leaky_control_is_not_invariant_to_labels():
    other = dict(EPISODE, defect_present=False, ground_truth_action="ACCEPT")
    a = StructuredIntentProducer.build_user_prompt(
        _producer(), EPISODE, variant="leaky", use_case="uc1_triage"
    )
    b = StructuredIntentProducer.build_user_prompt(
        _producer(), other, variant="leaky", use_case="uc1_triage"
    )
    assert a != b


def test_leaky_control_does_leak():
    text = _prompt("leaky")
    assert "defect_present=True" in text
    assert "REJECT_TO_BIN" in text


def test_blind_noctx_omits_plant_state():
    assert "line.state" not in _prompt("blind_noctx")
    assert "line.state" in _prompt("blind")


def test_unknown_variant_rejected():
    try:
        _prompt("nonsense")
    except ValueError:
        return
    raise AssertionError("unknown variant should raise")


def test_all_variants_declared():
    assert set(PROMPT_VARIANTS) == {
        "blind",
        "blind_cot",
        "blind_noctx",
        "blind_ref",
        "leaky",
    }


def test_precondition_syntax():
    assert precondition_syntax_ok("line.state == 'RUN'")
    assert precondition_syntax_ok("defect.absent == true")
    assert precondition_syntax_ok("robot.speed <= 0.05")
    assert not precondition_syntax_ok("line.state == 'RUN' and gripper.state == 'OPEN'")
    assert not precondition_syntax_ok("abs(robot.speed) < 1")


def test_repair_only_quotes_bare_string_literals():
    """
    The repair pass exists to separate a punctuation slip from a substantive error, so it
    must never turn an expression into a different claim than the model made.
    """
    assert repair_precondition("line.state != RUN") == "line.state != 'RUN'"
    assert repair_precondition("gripper.state == CLOSED;") == "gripper.state == 'CLOSED'"
    # Already valid forms pass through untouched, including booleans and numbers.
    for expr in ("line.state == 'RUN'", "robot.speed < 0.1", "defect.absent == true"):
        assert repair_precondition(expr) == expr


def test_repair_reads_a_bare_path_as_a_truthiness_test():
    """The only reading a lone boolean path admits, and a common reference-variant slip."""
    assert repair_precondition("defect.absent") == "defect.absent == true"
    assert not precondition_syntax_ok("defect")  # a single token is not a context path


def test_repair_refuses_substantive_errors():
    """Compound expressions and copied placeholders stay invalid rather than being guessed."""
    for expr in ("...", "line.state == RUN and gripper.state == OPEN", "abs(robot.speed) < 1"):
        assert not precondition_syntax_ok(repair_precondition(expr)), expr


def test_repair_does_not_invent_a_boolean_string():
    """`true` is a boolean literal, not a bare word to be quoted into the string 'true'."""
    assert repair_precondition("defect.absent == True") == "defect.absent == True"


def _result() -> PerceptionResult:
    return PerceptionResult(
        frame_id="f0",
        use_case="uc1_triage",
        prompt_variant="blind",
        model="qwen2.5vl:7b",
        capture_ts="2026-01-01T00:00:00.000Z",
        emitted_ts="2026-01-01T00:00:06.000Z",
        latency_ms=6000.0,
        action="REJECT_TO_BIN",
        preconditions=["line.state == 'RUN'"],
    )


def test_build_submission_anchors_differ():
    cap = build_submission(_result(), anchor="capture", freshness_ms=500)
    emi = build_submission(_result(), anchor="emission", freshness_ms=500)
    assert cap["timestamp_decision"] == "2026-01-01T00:00:00.000Z"
    assert emi["timestamp_decision"] == "2026-01-01T00:00:06.000Z"
    assert cap["extensions"]["validity_anchor"] == "capture"


def test_build_submission_keeps_model_preconditions():
    sub = build_submission(_result(), anchor="capture")
    assert sub["preconditions"] == [{"expr": "line.state == 'RUN'"}]
    assert sub["extensions"]["precondition_source"] == "model"


def test_build_submission_harness_override_is_explicit():
    sub = build_submission(_result(), anchor="capture", preconditions=["gripper.state == 'OPEN'"])
    assert sub["preconditions"] == [{"expr": "gripper.state == 'OPEN'"}]
    assert sub["extensions"]["precondition_source"] == "harness"


def test_build_submission_never_injects_preconditions():
    bare = _result()
    bare.preconditions = []
    assert build_submission(bare, anchor="capture")["preconditions"] == []


def test_perception_result_roundtrip():
    r = _result()
    assert PerceptionResult.from_json(r.to_json()).action == r.action


def test_legacy_envelope_still_available_for_ablation():
    out = _ensure_ais_envelope(
        {"payload": {"action_type": "STOP", "target_entity": "robot_3"}},
        include_preconditions=True,
    )
    assert len(out["preconditions"]) == 3
    assert _ensure_ais_envelope({"payload": {}}, include_preconditions=False)["preconditions"] == []


def test_legacy_raw_conversion():
    leg = StructuredIntentProducer.raw_to_legacy_intent({"action": "GRASP", "x": 1}, {})
    assert leg["payload"]["action_type"] == "GRASP"


def test_validate_base_ais():
    ok = {
        "id": "x",
        "source": "ai",
        "timestamp_decision": "2026-01-01T00:00:00.000Z",
        "freshness_window_ms": 500,
        "payload": {"action_type": "RESUME", "target_entity": "robot_3"},
    }
    assert StructuredIntentProducer._validate_base_ais(ok)
    assert not StructuredIntentProducer._validate_base_ais({"id": "x"})
