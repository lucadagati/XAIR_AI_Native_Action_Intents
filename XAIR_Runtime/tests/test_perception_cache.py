"""Tests for the Phase P work planner: representative pilots and correct sweep shape."""

from __future__ import annotations

from experiments.perception_cache import CacheKey, plan_jobs, stratified_subset

SEVERITIES = ("none", "minor", "major")
USE_CASES = ("uc1_triage", "uc2_restart", "uc3_speed")


def _episodes(n: int = 90) -> list[dict]:
    rows = []
    for i in range(n):
        severity = SEVERITIES[i % 3]
        rows.append(
            {
                "frame_id": f"f{i:04d}",
                "use_case": USE_CASES[i % 3],
                "severity": severity,
                "defect_present": severity != "none",
                "path": f"frames/f{i:04d}.png",
            }
        )
    return rows


def test_stratified_subset_covers_every_severity():
    """
    A pilot that misses the defective arm can report high accuracy while never testing
    detection, so the subset must span the severity strata even when it is small.
    """
    picked = stratified_subset(_episodes(), 9)
    assert len(picked) == 9
    assert {p["severity"] for p in picked} == set(SEVERITIES)
    assert {p["use_case"] for p in picked} == set(USE_CASES)


def test_stratified_subset_is_deterministic():
    assert [p["frame_id"] for p in stratified_subset(_episodes(), 12)] == [
        p["frame_id"] for p in stratified_subset(_episodes(), 12)
    ]


def test_stratified_subset_never_duplicates_or_overruns():
    picked = stratified_subset(_episodes(), 30)
    ids = [p["frame_id"] for p in picked]
    assert len(ids) == len(set(ids)) == 30


def test_stratified_subset_passthrough_when_limit_exceeds_pool():
    eps = _episodes(10)
    assert len(stratified_subset(eps, 50)) == 10


def test_variants_sweep_only_on_the_primary_model():
    """Secondary models exist for the routing study, not for the prompt ablation."""
    jobs = plan_jobs(
        _episodes(6),
        models=["primary", "other"],
        variants=["blind", "blind_cot", "leaky"],
        primary_model="primary",
        limit=None,
    )
    primary = [j for j in jobs if j[1] == "primary"]
    other = [j for j in jobs if j[1] == "other"]
    assert len(primary) == 6 * 3
    assert len(other) == 6
    assert {j[2] for j in other} == {"blind"}


def test_jobs_are_grouped_by_model():
    """
    Model-major ordering keeps each model resident for its whole pass. Interleaving models
    per frame makes Ollama evict and reload weights between calls, which dominated the
    runtime when it regressed.
    """
    jobs = plan_jobs(
        _episodes(5),
        models=["a", "b", "c"],
        variants=["blind", "leaky"],
        primary_model="a",
        limit=None,
    )
    model_sequence = [j[1] for j in jobs]
    # Each model appears in exactly one contiguous run.
    runs = [m for i, m in enumerate(model_sequence) if i == 0 or m != model_sequence[i - 1]]
    assert len(runs) == len(set(runs)) == 3


def test_cache_key_identifies_one_call():
    a = CacheKey("f1", "m", "blind", "uc1_triage")
    assert a == CacheKey("f1", "m", "blind", "uc1_triage")
    assert a != CacheKey("f1", "m", "leaky", "uc1_triage")
