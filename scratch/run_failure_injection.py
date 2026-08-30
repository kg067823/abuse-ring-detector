"""Phase 6 — Failure and Recovery Testing Script.

Injects chaos and failure scenarios into the production Model F backend:
1. Process Crash & Service Re-initialization
2. Redis Connection Failure & Graceful Fallback
3. Corrupted & Malformed Payload Ingestion
4. Emergency Kill Switch Safety Interception
5. Corrupted State Snapshot File Restoration
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fastapi.testclient import TestClient
from abuse_ring_detector.api import app, set_service
from abuse_ring_detector.config import load_config
from abuse_ring_detector.features import build_subgraph_extended_features
from abuse_ring_detector.inference import ProductionInferenceService, TransactionPayload
from abuse_ring_detector.models import fit_model
from abuse_ring_detector.state import InMemoryFeatureStateStore, RedisFeatureStateStore
from abuse_ring_detector.synthetic import generate_ecosystem

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("abuse_ring_detector").setLevel(logging.WARNING)


def create_test_service() -> ProductionInferenceService:
    config = load_config("configs/default.yaml")
    dataset = generate_ecosystem(config)
    orders = dataset.orders.head(100)
    labels = dataset.labels

    fs_all = build_subgraph_extended_features(orders, labels, config.graph["history_days"])
    feature_names = fs_all.X.columns.tolist()

    model_f = fit_model(fs_all.X, fs_all.y, config.model["backend"], config.seed)
    model_f.feature_columns = feature_names

    return ProductionInferenceService(
        model=model_f,
        feature_names=feature_names,
        threshold=0.50,
        model_version="v1.0.0-ModelF",
        schema_version="v1.0.0"
    )


def test_scenario_1_process_crash_and_recovery():
    print("\nScenario 1: Process Crash & State Restoration...")
    service = create_test_service()
    
    p1 = TransactionPayload(order_id="CRASH_001", customer_id="C_CRASH", event_time="2025-06-30T10:00:00", amount=150.0)
    r1 = service.score_transaction(p1)
    assert not r1.fallback_applied
    
    # Save snapshot
    snap_file = Path("scratch/test_crash_snap.pkl")
    service.state_store.save_snapshot(snap_file)
    assert snap_file.exists()

    # Simulate crash by resetting service
    service_recovered = create_test_service()
    service_recovered.state_store.load_snapshot(snap_file)

    p2 = TransactionPayload(order_id="CRASH_002", customer_id="C_CRASH", event_time="2025-06-30T10:05:00", amount=200.0)
    r2 = service_recovered.score_transaction(p2)
    assert not r2.fallback_applied

    if snap_file.exists():
        snap_file.unlink()
    print(" -> PASSED: State restored successfully after process crash simulation.")


def test_scenario_2_redis_loss_and_fallback():
    print("\nScenario 2: Redis Disconnection & Automatic Fallback...")
    store = RedisFeatureStateStore(redis_url="redis://invalid-host-9999:6379/0")
    assert not store.is_healthy()

    record = {
        "order_id": "REDIS_001",
        "customer_id": "C_REDIS",
        "event_time": "2025-06-30T11:00:00",
        "amount": 300.0,
        "device_id": "D_REDIS"
    }
    store.add_event(record)
    events = store.get_events()
    assert len(events) == 1
    assert events[0]["order_id"] == "REDIS_001"
    print(" -> PASSED: Redis loss fallback to local memory state store verified.")


def test_scenario_3_corrupted_payload_ingestion():
    print("\nScenario 3: Corrupted & Malformed Payload Ingestion...")
    service = create_test_service()
    set_service(service)
    client = TestClient(app)

    corrupted_payloads = [
        {"order_id": "CORRUPT_1", "customer_id": "C_BAD", "event_time": "INVALID_TIME", "amount": 100.0},
        {"order_id": "", "customer_id": "C_BAD", "event_time": "2025-06-30T12:00:00", "amount": -999.0},
        {"random_key": "junk_data"}
    ]

    for i, bad in enumerate(corrupted_payloads):
        resp = client.post("/v1/predict", json=bad)
        assert resp.status_code in (200, 422)
        if resp.status_code == 200:
            assert resp.json()["fallback_applied"] is True
    print(f" -> PASSED: All {len(corrupted_payloads)} corrupted payloads safely handled with zero crash.")


def test_scenario_4_kill_switch_interception():
    print("\nScenario 4: Emergency Kill Switch Safety Interception...")
    service = create_test_service()
    set_service(service)
    client = TestClient(app)

    # Enable kill switch
    client.post("/v1/admin/kill-switch", json={"active": True})

    payload = {
        "order_id": "KS_REQ_001",
        "customer_id": "C_KS",
        "event_time": "2025-06-30T13:00:00",
        "amount": 75.0
    }
    resp = client.post("/v1/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["fallback_applied"] is True
    assert data["risk_score"] == 0.05
    assert "kill_switch_active" in data["reason_codes"]

    # Disable kill switch
    client.post("/v1/admin/kill-switch", json={"active": False})
    print(" -> PASSED: Kill switch forced safe fallback response instantly.")


def test_scenario_5_corrupted_snapshot_restoration():
    print("\nScenario 5: Corrupted State Snapshot File Restoration...")
    service = create_test_service()
    corrupt_file = Path("scratch/corrupted_snapshot.pkl")
    corrupt_file.write_bytes(b"INVALID_PICKLE_HEADER_CORRUPTED_BYTES_123456789")

    success = service.state_store.load_snapshot(corrupt_file)
    assert success is False

    p = TransactionPayload(order_id="SAFE_001", customer_id="C_SAFE", event_time="2025-06-30T14:00:00", amount=50.0)
    r = service.score_transaction(p)
    assert not r.fallback_applied

    if corrupt_file.exists():
        corrupt_file.unlink()
    print(" -> PASSED: Corrupted snapshot file load failed safely without breaking service.")


def main():
    print("=========================================================")
    print("PHASE 6 — FAILURE AND RECOVERY TESTING & CHAOS VALIDATION")
    print("=========================================================")

    test_scenario_1_process_crash_and_recovery()
    test_scenario_2_redis_loss_and_fallback()
    test_scenario_3_corrupted_payload_ingestion()
    test_scenario_4_kill_switch_interception()
    test_scenario_5_corrupted_snapshot_restoration()

    print("\nALL 5 FAILURE AND RECOVERY SCENARIOS PASSED CLEANLY!")
    set_service(None)


if __name__ == "__main__":
    main()
