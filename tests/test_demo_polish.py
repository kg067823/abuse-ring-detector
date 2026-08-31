from __future__ import annotations

import hashlib
from pathlib import Path

from app.demo_scenarios import SCENARIOS, scenario_payloads


def test_scenarios_are_stable_and_increasing():
    for name in SCENARIOS:
        first = scenario_payloads(name, "fixed")
        second = scenario_payloads(name, "fixed")
        assert first == second
        assert len(first) == 8
        assert len({row["order_id"] for row in first}) == 8
        assert [row["event_time"] for row in first] == sorted(row["event_time"] for row in first)
        assert all(row["order_id"].startswith("DEMO_fixed_") for row in first)


def test_mixed_scenario_progressively_connects_entities():
    rows = scenario_payloads("Mixed multi-entity", "fixed")
    assert len({row["device_id"] for row in rows}) == 1
    assert len({row["address_id"] for row in rows}) == 1
    assert len({row["ip_id"] for row in rows}) == 1
    assert len({row["payment_id"] for row in rows}) == 1


def test_legitimate_control_is_explicitly_named():
    rows = scenario_payloads("Legitimate high-connectivity", "fixed")
    assert all("HOUSEHOLD" in row["device_id"] for row in rows)
    assert all("HOUSEHOLD" in row["address_id"] for row in rows)


def test_frozen_artifact_checksum_unchanged():
    path = Path("artifacts/model_f_r1_bundle.pkl")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "3f8bae638c6e81d0b391f9b226385e855ceb09744d774ea2d24cb9d0375c7cff"
