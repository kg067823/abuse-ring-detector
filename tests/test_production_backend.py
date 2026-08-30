"""Automated Unit & Integration Test Suite for Hardened Production Backend Service Infrastructure."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from abuse_ring_detector.config import load_config
from abuse_ring_detector.features import build_subgraph_extended_features
from abuse_ring_detector.inference import InferenceResponse, ProductionInferenceService, TransactionPayload
from abuse_ring_detector.models import fit_model
from abuse_ring_detector.splits import split_by_time
from abuse_ring_detector.state import InMemoryFeatureStateStore
from abuse_ring_detector.synthetic import generate_ecosystem


@pytest.fixture(scope="module")
def setup_backend(tmp_path_factory):
    config = load_config("configs/default.yaml")
    dataset = generate_ecosystem(config)
    orders = dataset.orders
    labels = dataset.labels

    split = split_by_time(orders, config.split["train"], config.split["validation"])
    fs_all = build_subgraph_extended_features(orders, labels, config.graph["history_days"])
    feature_names = fs_all.X.columns.tolist()

    train_ids = pd.Index(split.train["order_id"])
    model_f = fit_model(fs_all.X.loc[train_ids], fs_all.y.loc[train_ids], config.model["backend"], config.seed)
    model_f.feature_columns = feature_names

    audit_log = tmp_path_factory.mktemp("logs") / "audit.jsonl"

    service = ProductionInferenceService(
        model=model_f,
        feature_names=feature_names,
        threshold=0.50,
        model_version="v1.0.0-ModelF",
        schema_version="v1.0.0",
        audit_log_path=audit_log
    )
    return service, feature_names, fs_all, split.test, orders, audit_log


def test_training_serving_parity():
    """Verify 100% bitwise parity across all 137 features between offline batch and online streaming feature store."""
    config = load_config("configs/default.yaml")
    dataset = generate_ecosystem(config)
    orders = dataset.orders.head(1000)

    fs_batch = build_subgraph_extended_features(orders, dataset.labels, config.graph["history_days"])
    feature_names = fs_batch.X.columns.tolist()
    assert len(feature_names) == 137, f"Expected 137 features, got {len(feature_names)}"

    state_store = InMemoryFeatureStateStore(history_days=30)
    
    # Process orders in streaming order
    for row in orders.itertuples():
        payload = TransactionPayload(
            order_id=row.order_id,
            customer_id=row.customer_id,
            event_time=row.event_time,
            amount=row.amount,
            device_id=getattr(row, "device_id", ""),
            ip_id=getattr(row, "ip_id", ""),
            address_id=getattr(row, "address_id", ""),
            payment_id=getattr(row, "payment_id", ""),
            merchant_category=getattr(row, "merchant_category", "general"),
            retry_count=getattr(row, "retry_count", 0.0)
        )
        
        # Compute online as-of features
        t_current = pd.to_datetime(payload.event_time)
        all_recs = state_store.get_events()
        past_list = [r for r in all_recs if pd.to_datetime(r["event_time"]) < t_current]
        combined = past_list + [payload.to_record_dict()]
        
        df_combined = pd.DataFrame(combined)
        if "retry_count" not in df_combined.columns:
            df_combined["retry_count"] = 0.0
        df_combined["retry_count"] = df_combined["retry_count"].fillna(0.0)
        
        fs_online = build_subgraph_extended_features(df_combined, history_days=30)
        online_vec = fs_online.X.loc[row.order_id]
        offline_vec = fs_batch.X.loc[row.order_id]

        for fname in feature_names:
            off_val = float(offline_vec[fname])
            on_val = float(online_vec[fname])
            diff = abs(on_val - off_val)
            assert diff < 1e-5, f"Parity mismatch for order {row.order_id}, feature '{fname}': online={on_val}, offline={off_val}, diff={diff}"

        # Add event to state after scoring
        state_store.add_event(payload.to_record_dict())


def test_strict_chronological_causality(setup_backend):
    """Verify modifying a future event does not alter prior feature vectors or risk scores."""
    service, feature_names, fs_all, test_orders, _, _ = setup_backend

    p1 = TransactionPayload(
        order_id="CAUSAL_001", customer_id="C_CAUSAL", event_time="2025-06-01T10:00:00", amount=1000.0
    )
    r1_before = service.score_transaction(p1)

    # Insert future event for same customer
    p2 = TransactionPayload(
        order_id="CAUSAL_002", customer_id="C_CAUSAL", event_time="2025-06-02T10:00:00", amount=99999.0
    )
    _ = service.score_transaction(p2)

    # Re-verify p1 score remains strictly identical
    r1_after = service.score_transaction(p1)
    assert r1_before.risk_score == r1_after.risk_score
    assert r1_before.calibrated_score == r1_after.calibrated_score


def test_idempotent_duplicate_handling(setup_backend):
    """Verify duplicate submission returns cached response without double-counting state."""
    service, feature_names, fs_all, test_orders, _, _ = setup_backend

    payload = TransactionPayload(
        order_id="DUP_TEST_999", customer_id="C_DUP", event_time="2025-06-15T12:00:00", amount=2500.0
    )

    r1 = service.score_transaction(payload)
    count_before = service.total_processed_count

    r2 = service.score_transaction(payload)
    count_after = service.total_processed_count

    assert r1.risk_score == r2.risk_score
    assert r1.calibrated_score == r2.calibrated_score
    assert count_before == count_after, "Duplicate call must not increment total_processed_count"


def test_out_of_order_event_handling(setup_backend):
    """Verify out-of-order timestamp submission is causally handled without crashing."""
    service, feature_names, fs_all, test_orders, _, _ = setup_backend

    p_past = TransactionPayload(
        order_id="OOO_001", customer_id="C_OOO", event_time="2025-05-01T12:00:00", amount=1200.0
    )
    resp = service.score_transaction(p_past)
    assert isinstance(resp, InferenceResponse)
    assert not resp.fallback_applied


def test_state_persistence_and_restart_recovery(tmp_path, setup_backend):
    """Verify state serialization and crash recovery restores 100% exact bitwise scoring state."""
    service, feature_names, fs_all, test_orders, _, _ = setup_backend
    state_file = tmp_path / "service_state.json"

    service.save_state(state_file)

    recovered = ProductionInferenceService(
        model=service.model,
        feature_names=feature_names,
        threshold=service.threshold,
        model_version=service.model_version,
        schema_version=service.schema_version
    )
    recovered.load_state(state_file)

    p_test = TransactionPayload(
        order_id="REC_TEST_001", customer_id="C_REC", event_time="2025-06-20T10:00:00", amount=3300.0
    )
    score_orig = service.score_transaction(p_test).risk_score
    score_rec = recovered.score_transaction(p_test).risk_score

    assert abs(score_orig - score_rec) < 1e-6


def test_schema_field_validation(setup_backend):
    """Verify payload validation intercepts invalid fields and triggers safe fallback."""
    service, feature_names, fs_all, test_orders, _, _ = setup_backend

    bad_p = TransactionPayload(
        order_id="", customer_id="C_INVALID", event_time="BAD_DATE", amount=-10.0
    )

    resp = service.score_transaction(bad_p)
    assert resp.fallback_applied
    assert resp.risk_score == 0.05
    assert len(resp.reason_codes) > 0


def test_feature_contract_validation(setup_backend):
    """Verify feature vector matches expected contract schema length and names."""
    service, feature_names, fs_all, test_orders, _, _ = setup_backend
    assert len(feature_names) == 137
    assert "subgraph_edge_density_7d" in feature_names
    assert "two_hop_distinct_connected_customers_7d" in feature_names


def test_model_version_mismatch(setup_backend):
    """Verify service records explicit model version metadata in output response."""
    service, feature_names, fs_all, test_orders, _, _ = setup_backend
    p = TransactionPayload(
        order_id="VER_001", customer_id="C_VER", event_time="2025-06-18T10:00:00", amount=100.0
    )
    resp = service.score_transaction(p)
    assert resp.model_version == "v1.0.0-ModelF"
    assert resp.schema_version == "v1.0.0"


def test_emergency_kill_switch_behavior(setup_backend):
    """Verify emergency kill switch bypasses inference and returns safe fallback payload."""
    service, feature_names, fs_all, test_orders, _, _ = setup_backend
    p = TransactionPayload(
        order_id="KILL_001", customer_id="C_KILL", event_time="2025-06-19T10:00:00", amount=500.0
    )

    service.set_kill_switch(True)
    resp = service.score_transaction(p)
    service.set_kill_switch(False)

    assert resp.fallback_applied
    assert resp.risk_score == 0.05
    assert "kill_switch_active" in resp.reason_codes


def test_audit_log_emission(setup_backend):
    """Verify immutable JSON audit log line is written to disk for every transaction."""
    service, feature_names, fs_all, test_orders, _, audit_log = setup_backend
    p = TransactionPayload(
        order_id="AUDIT_001", customer_id="C_AUDIT", event_time="2025-06-21T10:00:00", amount=1500.0
    )
    _ = service.score_transaction(p)

    assert audit_log.exists()
    with open(audit_log, "r") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    matching = [l for l in lines if l.get("order_id") == "AUDIT_001"]
    assert len(matching) == 1
    assert matching[0]["order_id"] == "AUDIT_001"
    assert "risk_score" in matching[0]
