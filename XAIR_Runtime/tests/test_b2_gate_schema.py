"""Unit tests for XAIR offline gate: schema blocking and precondition parse rules."""

from __future__ import annotations

from experiments.run_b2_validity_frontier import gate_publishes


def test_xair_blocks_invalid_schema():
    pub, reason = gate_publishes(
        "xair",
        elapsed_ms=100.0,
        freshness_ms=4000,
        preconds_hold=True,
        schema_valid=False,
    )
    assert pub is False
    assert reason == "schema_invalid"


def test_xair_schema_checked_before_freshness():
    pub, reason = gate_publishes(
        "xair",
        elapsed_ms=9000.0,
        freshness_ms=4000,
        preconds_hold=True,
        schema_valid=False,
    )
    assert pub is False
    assert reason == "schema_invalid"


def test_xair_fresh_then_precond():
    pub, reason = gate_publishes(
        "xair",
        elapsed_ms=100.0,
        freshness_ms=4000,
        preconds_hold=False,
        schema_valid=True,
    )
    assert pub is False
    assert reason == "precondition_failed"


def test_xair_validated():
    pub, reason = gate_publishes(
        "xair",
        elapsed_ms=100.0,
        freshness_ms=4000,
        preconds_hold=True,
        schema_valid=True,
    )
    assert pub is True
    assert reason == "validated"


def test_freshness_only_ignores_schema():
    pub, reason = gate_publishes(
        "freshness_only",
        elapsed_ms=100.0,
        freshness_ms=4000,
        preconds_hold=False,
        schema_valid=False,
    )
    assert pub is True
    assert reason == "fresh"


def test_direct_ignores_schema():
    pub, reason = gate_publishes(
        "direct",
        elapsed_ms=9000.0,
        freshness_ms=100,
        preconds_hold=False,
        schema_valid=False,
    )
    assert pub is True
    assert reason == "no_validation"
