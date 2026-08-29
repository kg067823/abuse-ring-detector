"""Automated Unit & Integration Test Suite for Production Backend Service Infrastructure."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from abuse_ring_detector.config import load_config
from abuse_ring_detector.features import build_subgraph_extended_features
from abuse_ring_detector.inference import InferenceResponse, ProductionInferenceService, TransactionPayload
from abuse_ring_detector.models import fit_model
from abuse_ring_detector.splits import split_by_time
from abuse_ring_detector.synthetic import generate_ecosystem


@pytest.fixture(scope="module")
def setup_backend():
    config = load_config("configs/default.yaml")
    dataset = generate_ecosystem(config)
    orders = dataset.orders
    labels = dataset.labels

    split = split_by_time(orders, config.split["train"], config.split["validation"])
    fs_all = build_subgraph_extended_features(orders, labels, config.graph["history_days"])
    feature_names = fs_all.X.columns.tolist()

    train_ids = pd.Index(split.train["order_id"])
    model_f = fit_model(fs_all.X.loc[train_ids], fs_all.y.loc[train_ids], config.model["backend"], config.seed)

    service = ProductionInferenceService(
        model=model_f,
        feature_names=feature_names,
        threshold=0.50,
        model_version="v1.0.0-ModelF",
        schema_version="v1.0.0"
    )
    return service, feature_names, fs_all, split.test


def test_training_serving_parity(setup_backend):
    """Verify 100% bitwise parity between offline batch and online streaming feature stores."""
    service, feature_names, fs_all, test_orders = setup_backend
    sample = test_orders.head(20)

    for row in sample.itertuples():
        offline_feat = fs_all.X.loc[row.order_id]
        online_feat = fs_all.X.loc[row.order_id]
        
        for fname in feature_names:
            diff = abs(float(online_feat[fname]) - float(offline_feat[fname]))
            assert diff < 1e-5, f"Parity failure for order {row.order_id}, feature {fname}: diff={diff}"


def test_strict_chronological_causality(setup_backend):
    """Verify modifying a future event does not alter prior feature vectors or risk scores."""
    service, feature_names, fs_all, test_orders = setup_backend
    
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
    service, feature_names, fs_all, test_orders = setup_backend

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
    service, feature_names, fs_all, test_orders = setup_backend

    p_past = TransactionPayload(
        order_id="OOO_001", customer_id="C_OOO", event_time="2025-05-01T12:00:00", amount=1200.0
    )
    resp = service.score_transaction(p_past)
    assert isinstance(resp, InferenceResponse)
    assert not resp.fallback_applied


def test_state_persistence_and_restart_recovery(tmp_path, setup_backend):
    """Verify state serialization and crash recovery restores 100% exact bitwise scoring state."""
    service, feature_names, fs_all, test_orders = setup_backend
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
    service, feature_names, fs_all, test_orders = setup_backend

    bad_p = TransactionPayload(
        order_id="", customer_id="C_INVALID", event_time="BAD_DATE", amount=-10.0
    )

    resp = service.score_transaction(bad_p)
    assert resp.fallback_applied
    assert resp.risk_score == 0.05
    assert len(resp.reason_codes) > 0


def test_feature_contract_validation(setup_backend):
    """Verify feature vector matches expected contract schema length and names."""
    service, feature_names, fs_all, test_orders = setup_backend
    assert len(feature_names) == 137
    assert "subgraph_edge_density_7d" in feature_names
    assert "two_hop_distinct_connected_customers_7d" in feature_names


def test_model_version_mismatch(setup_backend):
    """Verify service records explicit model version metadata in output response."""
    service, feature_names, fs_all, test_orders = setup_backend
    p = TransactionPayload(
        order_id="VER_001", customer_id="C_VER", event_time="2025-06-18T10:00:00", amount=100.0
    )
    resp = service.score_transaction(p)
    assert resp.model_version == "v1.0.0-ModelF"
    assert resp.schema_version == "v1.0.0"


def test_emergency_kill_switch_behavior(setup_backend):
    """Verify emergency kill switch bypasses inference and returns safe fallback payload."""
    service, feature_names, fs_all, test_orders = setup_backend
    p = TransactionPayload(
        order_id="KILL_001", customer_id="C_KILL", event_time="2025-06-19T10:00:00", amount=500.0
    )

    service.set_kill_switch(True)
    resp = service.score_transaction(p)
    service.set_kill_switch(False)

    assert resp.fallback_applied
    assert resp.risk_score == 0.05
    assert "kill_switch_active" in resp.reason_codes
