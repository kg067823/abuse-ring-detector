from __future__ import annotations

import re

import pytest

from abuse_ring_detector.case_management import (
    CaseManager,
    CaseStatus,
    InMemoryCaseRepository,
    NoteCreate,
    Severity,
    StatusUpdate,
    severity_for,
)


class Response:
    calibrated_score = 0.8
    fallback_applied = False
    timestamp = "2026-01-01T00:00:00+00:00"
    correlation_id = "corr-1"


class Payload:
    order_id = "order-1"
    customer_id = "customer-1"
    amount = 125.0
    device_id = "device-secret"
    address_id = "address-secret"
    ip_id = ""
    payment_id = ""


def test_alert_creation_dedup_and_case_creation():
    manager = CaseManager()
    first = manager.ingest_prediction(Payload(), Response(), [])
    second = manager.ingest_prediction(Payload(), Response(), [])
    assert first is second
    assert len(manager.repository.alerts) == 1
    assert len(manager.repository.cases) == 1
    case = next(iter(manager.repository.cases.values()))
    assert case.alert_count == 1
    assert case.status == CaseStatus.NEW
    assert case.severity == Severity.HIGH


def test_low_score_and_fallback_do_not_create_alerts():
    manager = CaseManager()
    low = type("Low", (), {"calibrated_score": 0.49, "fallback_applied": False, "timestamp": "t", "correlation_id": ""})
    assert manager.ingest_prediction(Payload(), low(), []) is None
    fallback = type("Fallback", (), {"calibrated_score": 0.9, "fallback_applied": True, "timestamp": "t", "correlation_id": ""})
    assert manager.ingest_prediction(Payload(), fallback(), []) is None


def test_consolidation_uses_observable_entity_not_ring_id():
    manager = CaseManager()
    first = manager.ingest_prediction(Payload(), Response(), [])
    second_payload = type("P2", (), {"order_id": "order-2", "customer_id": "customer-2", "amount": 50.0, "device_id": "device-secret", "address_id": "", "ip_id": "", "payment_id": "", "ring_id": "ground-truth-must-not-be-used"})
    second = manager.ingest_prediction(second_payload(), Response(), [])
    assert second is not None
    assert len(manager.repository.cases) == 1
    case = next(iter(manager.repository.cases.values()))
    assert sorted(case.related_customers) == ["customer-1", "customer-2"]
    assert "ring_id" not in case.model_dump_json()


def test_status_transitions_and_immutable_history():
    manager = CaseManager()
    manager.ingest_prediction(Payload(), Response(), [])
    case = next(iter(manager.repository.cases.values()))
    manager.repository.transition(case.case_id, StatusUpdate(status=CaseStatus.IN_REVIEW, actor="analyst-1", reason="review"))
    manager.repository.transition(case.case_id, StatusUpdate(status=CaseStatus.LEGITIMATE, actor="analyst-1", reason="validated context"))
    assert len(case.history) == 3
    with pytest.raises(ValueError):
        manager.repository.transition(case.case_id, StatusUpdate(status=CaseStatus.NEW, actor="analyst-1", reason="invalid"))
    assert len(case.history) == 3


def test_notes_are_audited_and_case_graph_is_masked(tmp_path):
    repo = InMemoryCaseRepository(tmp_path / "case_audit.jsonl")
    manager = CaseManager(repo)
    manager.ingest_prediction(Payload(), Response(), [])
    case = next(iter(repo.cases.values()))
    repo.add_note(case.case_id, NoteCreate(actor="analyst-1", note="Observed shared device; review needed."))
    public = repo.public_case(case.case_id)
    assert "device-secret" not in str(public)
    graph = manager.graph(case.case_id)
    assert all(not re.search("secret", str(node), re.I) for node in graph["nodes"])
    assert "caused" not in str(public).lower()
    assert len((tmp_path / "case_audit.jsonl").read_text().splitlines()) >= 2


def test_severity_boundaries():
    assert severity_for(.49, 1, 1, 0) == Severity.LOW
    assert severity_for(.50, 1, 1, 0) == Severity.MEDIUM
    assert severity_for(.75, 1, 1, 0) == Severity.HIGH
    assert severity_for(.90, 1, 1, 0) == Severity.CRITICAL
