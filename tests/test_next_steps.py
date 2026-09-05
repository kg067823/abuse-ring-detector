from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

from abuse_ring_detector.case_management import ALLOWED_TRANSITIONS  # noqa: E402
from next_steps import NEXT_TRANSITIONS, suggest_next_steps  # noqa: E402


def test_ui_transition_map_mirrors_backend():
    backend = {status.value: {target.value for target in targets} for status, targets in ALLOWED_TRANSITIONS.items()}
    assert backend == NEXT_TRANSITIONS


def test_informational_steps_are_always_allowed():
    case = {"status": "NEW", "severity": "CRITICAL", "risk_score": 1.0, "estimated_exposure": 1500, "alert_count": 5, "related_customers": ["a", "b", "c"]}
    steps = suggest_next_steps(case, {"nodes": [{"type": "device_id", "id": "d"}], "edges": []}, {"items": []})
    informational = [s for s in steps if s["action"] == ""]
    assert informational, "expected at least one informational suggestion"
    assert all(s["allowed"] for s in informational)


def test_transitions_respect_state_machine():
    # NEW can only go to IN_REVIEW or CLOSED: escalation must not be offered.
    case = {"status": "NEW", "severity": "CRITICAL", "risk_score": 1.0, "estimated_exposure": 0, "alert_count": 5, "related_customers": []}
    steps = suggest_next_steps(case)
    transition_targets = {s["action"] for s in steps if s["action"] not in {"", "note"}}
    assert transition_targets <= NEXT_TRANSITIONS["NEW"]
    assert all(s["allowed"] for s in steps if s["action"] in transition_targets)


def test_closed_case_offers_only_note():
    steps = suggest_next_steps({"status": "CLOSED", "severity": "LOW", "risk_score": 0.1, "estimated_exposure": 0, "alert_count": 0, "related_customers": []})
    assert all(s["allowed"] for s in steps)
    assert all(s["action"] in {"", "note"} for s in steps)
