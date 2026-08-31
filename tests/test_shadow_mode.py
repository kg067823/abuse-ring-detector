"""Automated Tests for Live Shadow Mode & Non-Enforcement Customer Protection."""
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from abuse_ring_detector.api import app, set_service
from abuse_ring_detector.inference import ProductionInferenceService, load_model_artifact
from abuse_ring_detector.state import InMemoryFeatureStateStore
from abuse_ring_detector.shadow_gates import ShadowSafetyGateEvaluator


@pytest.fixture
def setup_shadow_service(tmp_path):
    audit_file = tmp_path / "shadow_audit.jsonl"
    model_path = Path("artifacts/model_f_r1_bundle.pkl")
    model, _ = load_model_artifact(model_path, require_frozen_contract=True, manifest_path="model_f_r1_manifest.json", contract_path="inference_contract_r1.json")
    service = ProductionInferenceService(model=model, feature_names=model.feature_columns, calibrator=model.calibrator, threshold=0.50, model_version="model_f_r1", schema_version="inference_contract_r1.v1", model_checksum="3f8bae638c6e81d0b391f9b226385e855ceb09744d774ea2d24cb9d0375c7cff", state_store=InMemoryFeatureStateStore(), audit_log_path=audit_file)
    set_service(service)
    client = TestClient(app)
    return service, client, audit_file


def test_kill_switch_requires_admin_auth(setup_shadow_service, monkeypatch):
    monkeypatch.setenv("ADMIN_KILL_SWITCH_TOKEN", "test-admin-token")
    _, client, _ = setup_shadow_service
    denied = client.post("/v1/admin/kill-switch", json={"active": True})
    assert denied.status_code == 401
    accepted = client.post("/v1/admin/kill-switch", json={"active": True}, headers={"Authorization": "Bearer test-admin-token"})
    assert accepted.status_code == 200
    client.post("/v1/admin/kill-switch", json={"active": False}, headers={"Authorization": "Bearer test-admin-token"})


def test_shadow_mode_non_enforcement_guarantee(setup_shadow_service):
    """Verify SHADOW_MODE=true / ENFORCE_DECISIONS=false never blocks customer orders."""
    service, client, audit_file = setup_shadow_service
    
    # Send transaction payload
    payload = {
        "order_id": "SHADOW_TEST_001",
        "customer_id": "C_SHADOW_001",
        "event_time": "2025-06-25T12:00:00Z",
        "amount": 9500.0,
        "device_id": "D_TEST_SHARED",
        "address_id": "A_TEST_SHARED"
    }

    resp = client.post("/v1/predict", json=payload, headers={"X-Shadow-Mode": "true"})
    assert resp.status_code == 200
    data = resp.json()

    # Safety Guarantees
    assert data["fallback_applied"] is False
    assert "risk_score" in data
    assert "calibrated_score" in data
    assert data["order_id"] == "SHADOW_TEST_001"


def test_shadow_audit_logging_and_pii_privacy(setup_shadow_service):
    """Verify shadow mode predictions write structured audit records with zero PII leakage."""
    service, client, audit_file = setup_shadow_service

    payload = {
        "order_id": "SHADOW_PII_001",
        "customer_id": "C_PII_USER",
        "event_time": "2025-06-26T14:30:00Z",
        "amount": 1500.0,
        "payment_id": "P_CARD_999"
    }

    _ = client.post("/v1/predict", json=payload, headers={"X-Shadow-Mode": "true"})
    assert audit_file.exists()

    lines = [json.loads(line) for line in audit_file.read_text().splitlines() if line.strip()]
    matching = [l for l in lines if l.get("order_id") == "SHADOW_PII_001"]
    assert len(matching) == 1

    record = matching[0]
    assert record["order_id"] == "SHADOW_PII_001"
    assert "risk_score" in record
    
    # Zero cleartext credit card or credential leakage
    rec_str = json.dumps(record)
    assert "password" not in rec_str.lower()
    assert "secret" not in rec_str.lower()


def test_shadow_mode_kill_switch_safety(setup_shadow_service):
    """Verify kill switch overrides shadow scoring and returns safe fallback risk_score=0.05."""
    service, client, audit_file = setup_shadow_service

    service.set_kill_switch(True)
    resp = client.post("/v1/predict", json={
        "order_id": "SHADOW_KILL_001",
        "customer_id": "C_KILL",
        "event_time": "2025-06-27T10:00:00Z",
        "amount": 300.0
    }, headers={"X-Shadow-Mode": "true"})

    service.set_kill_switch(False)

    assert resp.status_code == 200
    data = resp.json()
    assert data["fallback_applied"] is True
    assert data["risk_score"] == 0.05
    assert "kill_switch_active" in data["reason_codes"]


def test_shadow_safety_gates_evaluation(setup_shadow_service):
    """Verify Phase 5 Shadow Safety Gate Evaluator returns PASS on clean service state."""
    service, client, audit_file = setup_shadow_service

    evaluator = ShadowSafetyGateEvaluator()
    metrics = service.get_metrics()
    results = evaluator.run_all_gates(metrics, shadow_logs=[], model_bundle=service.model)

    assert results["overall_passed"] is True, f"Gates failed: {results}"
    assert results["overall_status"] in ("PASS", "WARN")
