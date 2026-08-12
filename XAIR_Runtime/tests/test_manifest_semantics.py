"""
Semantic checks on the manufacturing-a1 manifest.

The important invariant is that each declared hazard predicate holds in the nominal
context and fails once the declared drift patch is applied. If that breaks, the harness
would be measuring admissibility against a predicate that was never satisfiable, which
is precisely the failure mode the ``defect.absent == true`` expression used to cause.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from xair.core.context_validator import ContextValidator  # noqa: E402
from xair.core.deep_merge import deep_merge  # noqa: E402

MANIFEST = ROOT / "experiments" / "datasets" / "manufacturing-a1" / "manifest.jsonl"
USE_CASES = ("uc1_triage", "uc2_restart", "uc3_speed", "uc4_conflict", "uc5_safety")
ACTIONS = {
    "ACCEPT",
    "REJECT_TO_BIN",
    "HOLD_FOR_OPERATOR",
    "SLOW_DOWN",
    "RESUME",
    "STOP",
    "E_STOP",
}


def _rows() -> list[dict]:
    if not MANIFEST.is_file():
        pytest.skip("manifest not built")
    return [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]


def _holds(context: dict, predicate: str) -> bool:
    return ContextValidator(context)._check(predicate)[0]


def test_manifest_is_populated():
    rows = _rows()
    assert len(rows) >= 100
    assert len({(r["source_dataset"], r["category"]) for r in rows}) >= 20


def test_hazard_holds_nominally_and_fails_under_drift():
    for r in _rows():
        for uc in USE_CASES:
            spec = r["use_cases"][uc]
            ctx = r["context"]
            predicate = spec["hazard_predicate"]
            assert _holds(ctx, predicate), f"{r['frame_id']}/{uc}: hazard false at capture"
            drifted = deep_merge(json.loads(json.dumps(ctx)), spec["drift_patch"])
            assert not _holds(drifted, predicate), f"{r['frame_id']}/{uc}: drift not detected"


def test_primary_use_case_fields_match_the_dict():
    for r in _rows():
        uc = r["use_case"]
        assert uc in USE_CASES
        spec = r["use_cases"][uc]
        assert r["ground_truth_action"] == spec["ground_truth_action"]
        assert r["hazard_predicate"] == spec["hazard_predicate"]
        assert r["drift_patch"] == spec["drift_patch"]


def test_ground_truth_actions_are_known():
    for r in _rows():
        for uc in USE_CASES:
            assert r["use_cases"][uc]["ground_truth_action"] in ACTIONS


def test_severity_is_consistent_with_defect_flag():
    for r in _rows():
        if r["defect_present"]:
            assert r["severity"] in ("minor", "major")
        else:
            assert r["severity"] == "none"
            assert r["defect_area"] == 0.0


def test_major_defects_have_larger_area_than_minor_within_category():
    by_cat: dict[str, dict[str, list[float]]] = {}
    for r in _rows():
        if not r["defect_present"] or r["defect_area"] <= 0:
            continue
        by_cat.setdefault(r["category"], {"minor": [], "major": []})
        by_cat[r["category"]][r["severity"]].append(r["defect_area"])
    checked = 0
    for cat, groups in by_cat.items():
        if groups["minor"] and groups["major"]:
            assert max(groups["minor"]) <= min(groups["major"]), cat
            checked += 1
    assert checked > 0


def test_use_cases_and_severity_are_balanced():
    rows = _rows()
    uc_counts = Counter(r["use_case"] for r in rows)
    spread = max(uc_counts.values()) - min(uc_counts.values())
    assert spread <= max(2, len(rows) // 100)
    defect_share = sum(1 for r in rows if r["defect_present"]) / len(rows)
    assert 0.4 <= defect_share <= 0.6


def test_frames_exist_on_disk():
    base = MANIFEST.parent
    for r in _rows()[:50]:
        assert (base / r["path"]).is_file(), r["path"]


def test_uc5_inverts_the_drift_for_real_defects():
    rows = [r for r in _rows() if r["severity"] == "major"]
    assert rows
    spec = rows[0]["use_cases"]["uc5_safety"]
    assert spec["ground_truth_action"] == "E_STOP"
    assert spec["hazard_predicate"] == "defect.absent == false"
    assert spec["drift_patch"] == {"defect": {"absent": True}}
