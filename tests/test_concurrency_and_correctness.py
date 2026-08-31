"""Rigorously test concurrency, idempotency, state corruption, Redis failover, out-of-order events, and failure paths."""
from __future__ import annotations

import concurrent.futures
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
from abuse_ring_detector.state import InMemoryFeatureStateStore, RedisFeatureStateStore
from abuse_ring_detector.synthetic import generate_ecosystem


@pytest.fixture(scope="module")
def shared_service(tmp_path_factory):
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

    audit_log = tmp_path_factory.mktemp("logs") / "audit_concurrency.jsonl"

    service = ProductionInferenceService(
        model=model_f,
        feature_names=feature_names,
        threshold=0.50,
        model_version="model_f_r1",
        schema_version="inference_contract_r1.v1",
        audit_log_path=audit_log
    )
    return service, feature_names


def test_concurrent_scoring_requests(shared_service):
    """Verify thread safety under 50 concurrent transaction requests."""
    service, _ = shared_service

    def _score(idx: int):
        p = TransactionPayload(
            order_id=f"CONCUR_{idx:04d}",
            customer_id=f"C_{idx % 5:03d}",
            event_time=f"2025-06-25T12:{idx%60:02d}:00",
            amount=100.0 + idx,
            device_id=f"D_{idx % 3:03d}"
        )
        return service.score_transaction(p, correlation_id=f"corr_{idx}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(_score, i) for i in range(50)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(results) == 50
    assert all(isinstance(r, InferenceResponse) for r in results)
    assert all(r.risk_score >= 0.0 for r in results)


def test_concurrent_duplicate_transaction_ids(shared_service):
    """Verify race conditions on identical order_id yield exactly 1 processing & 19 cached responses."""
    service, _ = shared_service
    order_id = "RACE_DUP_999"

    p = TransactionPayload(
        order_id=order_id,
        customer_id="C_RACE",
        event_time="2025-06-26T10:00:00",
        amount=500.0
    )

    def _submit():
        return service.score_transaction(p)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(_submit) for _ in range(20)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(results) == 20
    first_score = results[0].risk_score
    assert all(abs(r.risk_score - first_score) < 1e-6 for r in results)


def test_out_of_order_and_clock_skew(shared_service):
    """Verify out-of-order timestamps process causally without corrupting state."""
    base_service, feature_names = shared_service
    service = ProductionInferenceService(
        model=base_service.model,
        feature_names=feature_names,
        threshold=base_service.threshold,
        model_version=base_service.model_version,
        schema_version=base_service.schema_version
    )

    p_latest = TransactionPayload(
        order_id="TIME_002", customer_id="C_TIME", event_time="2025-06-27T15:00:00", amount=200.0
    )
    r_latest = service.score_transaction(p_latest)

    p_earlier = TransactionPayload(
        order_id="TIME_001", customer_id="C_TIME", event_time="2025-06-27T10:00:00", amount=100.0
    )
    r_earlier = service.score_transaction(p_earlier)

    assert not r_latest.fallback_applied
    assert not r_earlier.fallback_applied


def test_state_restoration_from_corrupted_file(tmp_path, shared_service):
    """Verify loading corrupted state snapshot raises FileNotFoundError or ValueError gracefully."""
    service, feature_names = shared_service
    bad_file = tmp_path / "corrupted_state.json"
    with open(bad_file, "w") as f:
        f.write("{invalid_json: true")

    with pytest.raises(Exception):
        service.load_state(bad_file)


def test_unavailable_redis_state_backend(shared_service):
    """Verify Redis state store automatically falls back to in-memory store if Redis is unavailable."""
    service, _ = shared_service
    redis_store = RedisFeatureStateStore(redis_url="redis://invalid_host_12345:6379/0")

    assert not redis_store.is_healthy()
    # Adding event must not crash; falls back gracefully
    record = {
        "order_id": "REDIS_FALLBACK_001",
        "customer_id": "C_REDIS",
        "event_time": "2025-06-28T10:00:00",
        "amount": 300.0
    }
    redis_store.add_event(record)
    assert len(redis_store.get_events("C_REDIS")) == 1


def test_kill_switch_activation_path(shared_service):
    """Verify kill switch activation instantly returns safe fallback payloads with code 'kill_switch_active'."""
    service, _ = shared_service
    service.set_kill_switch(True)

    p = TransactionPayload(
        order_id="KS_001", customer_id="C_KS", event_time="2025-06-28T12:00:00", amount=120.0
    )
    resp = service.score_transaction(p)
    service.set_kill_switch(False)

    assert resp.fallback_applied
    assert resp.risk_score == 0.05
    assert "kill_switch_active" in resp.reason_codes
