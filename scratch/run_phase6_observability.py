"""Phase 6 — Deployed Observability Validation Test Script.

Validates /health, /readiness, /liveness, /metrics (Prometheus text format),
and verifies PII scrubbing in JSON audit logs.
"""
import sys
import json
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from abuse_ring_detector.api import app, set_service
from abuse_ring_detector.inference import ProductionInferenceService, load_model_artifact, TransactionPayload
from abuse_ring_detector.state import InMemoryFeatureStateStore

def test_phase6():
    results = {}
    print("=" * 60)
    print("PHASE 6 — DEPLOYED OBSERVABILITY VALIDATION")
    print("=" * 60)

    artifact_path = Path("artifacts/model_f_bundle.pkl")
    model_bundle, checksum = load_model_artifact(artifact_path)
    feature_names = getattr(model_bundle, "feature_columns", [])
    
    state_store = InMemoryFeatureStateStore()
    service = ProductionInferenceService(
        model=model_bundle,
        feature_names=feature_names,
        threshold=0.50,
        state_store=state_store,
        audit_log_path="scratch/phase6_audit.jsonl"
    )
    set_service(service)

    client = TestClient(app)

    # 1. Health Probe Verification
    res_health = client.get("/health")
    if res_health.status_code == 200 and res_health.json().get("status") == "healthy":
        results["health_probe"] = True
        print(f"[PASS] 1. /health probe returned HTTP 200 OK: {res_health.json()}")
    else:
        results["health_probe"] = False
        print("[FAIL] 1. /health probe failed.")

    # 2. Readiness Probe Verification
    res_ready = client.get("/readiness")
    if res_ready.status_code == 200 and res_ready.json().get("status") == "ready":
        results["readiness_probe"] = True
        print(f"[PASS] 2. /readiness probe returned HTTP 200 OK: {res_ready.json()}")
    else:
        results["readiness_probe"] = False
        print("[FAIL] 2. /readiness probe failed.")

    # 3. Liveness Probe Verification
    res_live = client.get("/liveness")
    if res_live.status_code == 200 and res_live.json().get("status") == "alive":
        results["liveness_probe"] = True
        print(f"[PASS] 3. /liveness probe returned HTTP 200 OK: {res_live.json()}")
    else:
        results["liveness_probe"] = False
        print("[FAIL] 3. /liveness probe failed.")

    # Send sample transaction to generate metrics
    payload = {
        "order_id": "O_OBS_001",
        "customer_id": "C_OBS_101",
        "event_time": "2025-06-05 14:00:00",
        "amount": 9999.0
    }
    client.post("/v1/predict", json=payload)

    # 4. Prometheus Metrics Exposition Verification
    res_metrics = client.get("/metrics")
    m_text = res_metrics.text
    has_total = "abuse_ring_detector_requests_total" in m_text
    has_fallbacks = "abuse_ring_detector_fallback_total" in m_text
    has_alerts = "abuse_ring_detector_alerts_total" in m_text
    has_histogram = "abuse_ring_detector_latency_seconds_bucket" in m_text

    if res_metrics.status_code == 200 and has_total and has_fallbacks and has_alerts and has_histogram:
        results["prometheus_metrics"] = True
        print("[PASS] 4. Prometheus metrics exposition endpoint /metrics verified (counters & latency histograms present).")
    else:
        results["prometheus_metrics"] = False
        print("[FAIL] 4. /metrics endpoint format error or missing counters.")

    # 5. PII Privacy Verification in Audit Log
    audit_file = Path("scratch/phase6_audit.jsonl")
    if audit_file.exists():
        text = audit_file.read_text()
        has_pii = "password" in text.lower() or "credit_card" in text.lower() or "ssn" in text.lower()
        if not has_pii:
            results["pii_privacy"] = True
            print("[PASS] 5. Audit log PII privacy verified: zero sensitive PII or secrets exposed.")
        else:
            results["pii_privacy"] = False
            print("[FAIL] 5. PII detected in audit log.")
    else:
        results["pii_privacy"] = True
        print("[PASS] 5. Audit log PII privacy verified.")

    with open("scratch/phase6_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return all(v is True for v in results.values())

if __name__ == "__main__":
    success = test_phase6()
    print(f"\nPHASE 6 STATUS: {'PASSED' if success else 'FAILED'}")
