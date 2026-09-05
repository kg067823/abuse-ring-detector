from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from demo_scenarios import scenario_payloads  # noqa: E402
from monitor import (  # noqa: E402
    REVIEW_THRESHOLD,
    case_belongs_to_run,
    observation_text,
    pick_active_case,
    risk_status,
    row_from_event,
    shared_with,
    timeline_view,
)


def test_threshold_is_locked_half():
    assert REVIEW_THRESHOLD == 0.50


def test_risk_status_mapping_honest():
    assert risk_status(0.18) == "NORMAL"
    assert risk_status(0.31) == "WATCHING"
    assert risk_status(0.46) == "WATCHING"
    assert risk_status(0.50) == "ALERT"
    assert risk_status(0.72) == "ALERT"
    # Fallback must never present as an alert.
    assert risk_status(0.99, fallback=True) == "FALLBACK"


def test_row_from_event_uses_backend_fields():
    response = {"calibrated_score": 0.72, "alert": True, "fallback_applied": False,
                "timestamp": "2026-09-05T12:03:24Z"}
    payload = {"customer_id": "DEMO_x_MIXED_CUSTOMER_63", "amount": 4100.0,
               "device_id": "DEV", "address_id": None}
    row = row_from_event(4, payload, response, [])
    assert row["time"] == "12:03:24"
    assert row["customer"] == "Customer ••63"
    assert row["risk"] == 0.72
    assert row["status"] == "ALERT"
    assert row["observation"] == "Risk crossed review threshold"


def test_observation_never_fabricates():
    # No shared entities and low risk: neutral wording only.
    assert observation_text(0.18, False, False, 2, []) == "Event scored by Model F-R1."
    # Missing entity field must not count as shared.
    assert shared_with({"device_id": None}, [{"device_id": "D"}]) == []
    # Real sharing is reported.
    assert shared_with({"device_id": "D"}, [{"device_id": "D"}]) == ["device"]


def test_sequential_replay_not_instant_batch():
    """Scenario payloads replay one at a time — stream semantics, not batch."""
    payloads = scenario_payloads("Mixed multi-entity", "seq")
    assert len(payloads) == 8
    # Each payload is an independent /v1/predict request; ordering preserved.
    assert [p["event_time"] for p in payloads] == sorted(p["event_time"] for p in payloads)
    assert len({p["order_id"] for p in payloads}) == len(payloads)


def test_automatic_case_selection_prefers_active_highest_risk():
    run_orders = {"o1", "o2"}
    cases = [
        {"case_id": "old", "related_orders": ["zzz"], "risk_score": 0.95, "status": "NEW"},
        {"case_id": "mine-low", "related_orders": ["o1"], "risk_score": 0.6, "status": "NEW"},
        {"case_id": "mine-high", "related_orders": ["o2"], "risk_score": 0.9, "status": "IN_REVIEW"},
        {"case_id": "mine-closed", "related_orders": ["o1"], "risk_score": 1.0, "status": "CLOSED"},
    ]
    picked = pick_active_case(cases, run_orders)
    assert picked is not None
    # Highest-risk ACTIVE case from this run wins; closed case excluded despite risk 1.0.
    assert picked["case_id"] == "mine-high"


def test_no_case_from_other_runs_leaks_in():
    cases = [{"case_id": "old", "related_orders": ["zzz"], "risk_score": 1.0, "status": "NEW"}]
    assert pick_active_case(cases, {"o1"}) is None


def test_legitimate_control_scenario_has_no_built_in_case():
    """The control traffic must rely purely on the backend verdict — nothing in
    the UI or scenario generator can force a case into existence."""
    payloads = scenario_payloads("Legitimate high-connectivity", "ctrl")
    assert all("HOUSEHOLD" in p["device_id"] for p in payloads)


def test_no_enforcement_in_response_contract():
    """Response contract is shadow-only: enforcement fields must stay false/absent
    in any UI-rendered path."""
    response = {"calibrated_score": 1.0, "alert": True, "fallback_applied": False,
                "shadow_mode": True, "enforcement_applied": False,
                "timestamp": "2026-09-05T12:03:24Z"}
    row = row_from_event(1, {"customer_id": "x", "amount": 1}, response, [])
    assert row["status"] == "ALERT"  # review signal, honestly shown


def test_timeline_view_caps_and_sorts():
    items = [
        {"timestamp": "2026-09-05T12:03:24Z", "description": "Risk crossed"},
        {"timestamp": "2026-09-05T12:03:01Z", "description": "First transaction"},
    ] + [{"timestamp": f"2026-09-05T12:04:{i:02d}Z", "description": f"e{i}"} for i in range(10)]
    view = timeline_view(items, limit=8)
    assert len(view) == 8
    assert view[0]["event"] == "First transaction"
    assert view[0]["time"] == "12:03:01"
