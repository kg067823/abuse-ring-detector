"""Unit and Integration tests for production FastAPI service endpoints."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from abuse_ring_detector.api import app, set_service
from abuse_ring_detector.inference import ProductionInferenceService, load_model_artifact
from abuse_ring_detector.state import InMemoryFeatureStateStore


@pytest.fixture(scope="module")
def api_client(tmp_path_factory):
    model_f, _ = load_model_artifact(
        "artifacts/model_f_r1_bundle.pkl",
        require_frozen_contract=True,
        manifest_path="model_f_r1_manifest.json",
        contract_path="inference_contract_r1.json",
    )
    feature_names = model_f.feature_columns

    audit_log = tmp_path_factory.mktemp("logs") / "api_audit.jsonl"
    service = ProductionInferenceService(
        model=model_f,
        feature_names=feature_names,
        threshold=0.50,
        calibrator=model_f.calibrator,
        model_version="model_f_r1",
        schema_version="inference_contract_r1.v1",
        state_store=InMemoryFeatureStateStore(),
        audit_log_path=audit_log
    )

    set_service(service)
    client = TestClient(app)
    yield client
    set_service(None)


def test_health_probes(api_client: TestClient):
    resp_h = api_client.get("/health")
    assert resp_h.status_code == 200
    assert resp_h.json()["status"] == "healthy"

    resp_r = api_client.get("/readiness")
    assert resp_r.status_code == 200
    assert resp_r.json()["status"] == "ready"

    resp_l = api_client.get("/liveness")
    assert resp_l.status_code == 200
    assert resp_l.json()["status"] == "alive"


def test_predict_endpoint_valid_payload(api_client: TestClient):
    payload = {
        "order_id": "API_ORD_001",
        "customer_id": "C_API_100",
        "event_time": "2025-06-30T14:00:00",
        "amount": 450.0,
        "device_id": "D_API_01",
        "ip_id": "IP_API_01",
        "address_id": "ADDR_API_01",
        "payment_id": "PAY_API_01",
        "merchant_category": "electronics",
        "retry_count": 0.0
    }
    headers = {"X-Correlation-ID": "test_correlation_123"}
    resp = api_client.post("/v1/predict", json=payload, headers=headers)

    assert resp.status_code == 200
    data = resp.json()
    assert data["order_id"] == "API_ORD_001"
    assert "risk_score" in data
    assert "calibrated_score" in data
    assert "alert" in data
    assert data["model_version"] == "model_f_r1"
    assert data["correlation_id"] == "test_correlation_123"


def test_metrics_endpoint(api_client: TestClient):
    resp = api_client.get("/metrics")
    assert resp.status_code == 200
    text = resp.text
    assert "abuse_ring_detector_requests_total" in text
    assert "abuse_ring_detector_latency_seconds_bucket" in text
    assert "abuse_ring_detector_fallback_total" in text


def test_kill_switch_admin_endpoint(api_client: TestClient, monkeypatch):
    monkeypatch.setenv("ADMIN_KILL_SWITCH_TOKEN", "test-admin-token")
    # Activate kill switch
    resp_ks_on = api_client.post("/v1/admin/kill-switch", json={"active": True}, headers={"Authorization": "Bearer test-admin-token"})
    assert resp_ks_on.status_code == 200
    assert resp_ks_on.json()["kill_switch_active"] is True

    # Predict call should return fallback
    payload = {
        "order_id": "API_KS_001",
        "customer_id": "C_API_200",
        "event_time": "2025-06-30T15:00:00",
        "amount": 100.0
    }
    resp_p = api_client.post("/v1/predict", json=payload)
    assert resp_p.status_code == 200
    data = resp_p.json()
    assert data["fallback_applied"] is True
    assert data["risk_score"] == 0.05
    assert "kill_switch_active" in data["reason_codes"]

    # Deactivate kill switch
    resp_ks_off = api_client.post("/v1/admin/kill-switch", json={"active": False}, headers={"Authorization": "Bearer test-admin-token"})
    assert resp_ks_off.status_code == 200
    assert resp_ks_off.json()["kill_switch_active"] is False


def test_invalid_payload_error_interceptor(api_client: TestClient):
    bad_payload = {
        "order_id": "",
        "customer_id": "",
        "event_time": "invalid_date",
        "amount": -500.0
    }
    resp = api_client.post("/v1/predict", json=bad_payload)
    assert resp.status_code in (200, 422)
    data = resp.json()
    if resp.status_code == 200:
        assert data["fallback_applied"] is True
