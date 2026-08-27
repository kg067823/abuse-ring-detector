"""Leakage and correctness tests for streaming temporal edge velocity and burst features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from abuse_ring_detector.config import load_config
from abuse_ring_detector.features import (
    build_baseline_features,
    build_combined_features,
    build_full_features,
    build_temporal_velocity_features,
)
from abuse_ring_detector.splits import split_by_time
from abuse_ring_detector.synthetic import generate_ecosystem


def test_window_correctness_controlled_dataset():
    """Verify that 1h and 24h windows capture exactly the right historical events and exclude current and future events."""
    base_time = pd.Timestamp("2025-06-01 12:00:00")
    orders = pd.DataFrame({
        "order_id": ["O1", "O2", "O3", "O4", "O5", "O6"],
        "customer_id": ["C1", "C2", "C3", "C4", "C5", "C6"],
        "event_time": [
            base_time - pd.Timedelta(hours=2),        # T - 2h (in 24h, outside 1h)
            base_time - pd.Timedelta(minutes=59),     # T - 59m (in 24h, in 1h)
            base_time - pd.Timedelta(minutes=30),     # T - 30m (in 24h, in 1h)
            base_time - pd.Timedelta(minutes=1),      # T - 1m (in 24h, in 1h)
            base_time,                                # T (the evaluated event)
            base_time + pd.Timedelta(minutes=1),      # T + 1m (future event)
        ],
        "amount": [100.0, 150.0, 200.0, 250.0, 300.0, 350.0],
        "device_id": ["D0001", "D0001", "D0001", "D0001", "D0001", "D0001"],
        "address_id": ["A0001", "A0001", "A0002", "A0002", "A0001", "A0001"],
        "ip_id": ["IP0001", "IP0001", "IP0001", "IP0001", "IP0001", "IP0001"],
        "payment_id": ["P0001", "P0002", "P0003", "P0004", "P0005", "P0006"],
        "merchant_category": ["electronics"] * 6,
        "retry_count": [0] * 6,
    })

    fs = build_temporal_velocity_features(orders)
    X = fs.X

    # At O5 (time T):
    # Device D0001:
    # Prior events in 24h: O1, O2, O3, O4 -> 4 orders, 4 distinct customers (C1, C2, C3, C4)
    # Prior events in 1h: O2, O3, O4 -> 3 orders, 3 distinct customers (C2, C3, C4)
    assert X.loc["O5", "device_order_count_24h"] == 4.0
    assert X.loc["O5", "device_distinct_customers_24h"] == 4.0
    assert X.loc["O5", "device_order_count_1h"] == 3.0
    assert X.loc["O5", "device_distinct_customers_1h"] == 3.0

    # Address A0001:
    # Prior events in 24h: O1, O2 -> 2 orders, 2 distinct customers (C1, C2)
    # Prior events in 1h: O2 -> 1 order, 1 distinct customer (C2)
    assert X.loc["O5", "address_order_count_24h"] == 2.0
    assert X.loc["O5", "address_distinct_customers_24h"] == 2.0
    assert X.loc["O5", "address_order_count_1h"] == 1.0
    assert X.loc["O5", "address_distinct_customers_1h"] == 1.0

    # O1 (time T-2h):
    # No prior events exist
    assert X.loc["O1", "device_order_count_24h"] == 0.0
    assert X.loc["O1", "device_distinct_customers_24h"] == 0.0
    assert X.loc["O1", "device_order_count_1h"] == 0.0
    assert X.loc["O1", "device_distinct_customers_1h"] == 0.0


def test_current_event_exclusion():
    """Verify that current event attributes do not count into their own prior feature values."""
    orders = pd.DataFrame({
        "order_id": ["O1"],
        "customer_id": ["C1"],
        "event_time": [pd.Timestamp("2025-01-01 10:00:00")],
        "amount": [100.0],
        "device_id": ["D0001"],
        "address_id": ["A0001"],
        "ip_id": ["IP0001"],
        "payment_id": ["P0001"],
        "merchant_category": ["electronics"],
        "retry_count": [0],
    })

    fs = build_temporal_velocity_features(orders)
    row = fs.X.loc["O1"]

    for prefix in ["device", "address", "ip", "payment"]:
        assert row[f"{prefix}_order_count_1h"] == 0.0
        assert row[f"{prefix}_order_count_24h"] == 0.0
        assert row[f"{prefix}_distinct_customers_1h"] == 0.0
        assert row[f"{prefix}_distinct_customers_24h"] == 0.0


def test_future_event_exclusion():
    """Verify modifying future transactions leaves earlier temporal features strictly unchanged."""
    orders_v1 = pd.DataFrame({
        "order_id": ["O1", "O2", "O3"],
        "customer_id": ["C1", "C2", "C3"],
        "event_time": pd.to_datetime(["2025-01-01 10:00", "2025-01-01 10:30", "2025-01-01 11:00"]),
        "amount": [100.0, 200.0, 300.0],
        "device_id": ["D1", "D1", "D1"],
        "address_id": ["A1", "A1", "A1"],
        "ip_id": ["IP1", "IP1", "IP1"],
        "payment_id": ["P1", "P2", "P3"],
        "merchant_category": ["electronics", "fashion", "home"],
        "retry_count": [0, 0, 0],
    })

    # Version 2 modifies future transaction O3's device and amount
    orders_v2 = orders_v1.copy()
    orders_v2.loc[2, "device_id"] = "D9999"
    orders_v2.loc[2, "amount"] = 99999.0

    fs1 = build_temporal_velocity_features(orders_v1)
    fs2 = build_temporal_velocity_features(orders_v2)

    pd.testing.assert_series_equal(fs1.X.loc["O1"], fs2.X.loc["O1"])
    pd.testing.assert_series_equal(fs1.X.loc["O2"], fs2.X.loc["O2"])


def test_target_isolation():
    """Verify target and ground-truth columns are absent from temporal feature matrix."""
    orders = pd.DataFrame({
        "order_id": ["O1", "O2"],
        "customer_id": ["C1", "C2"],
        "event_time": pd.to_datetime(["2025-01-01 10:00", "2025-01-01 10:30"]),
        "amount": [100.0, 200.0],
        "device_id": ["D1", "D1"],
        "address_id": ["A1", "A1"],
        "ip_id": ["IP1", "IP1"],
        "payment_id": ["P1", "P2"],
        "merchant_category": ["electronics", "fashion"],
        "retry_count": [0, 0],
    })
    labels = pd.DataFrame({
        "order_id": ["O1", "O2"],
        "is_abuse": [False, True],
        "ring_id": [pd.NA, "R0001"],
        "abuse_type": [pd.NA, "shared_device"],
        "loss_amount": [0.0, 150.0],
        "reason_codes": ["", "burst_velocity"],
    })

    fs = build_temporal_velocity_features(orders, labels)
    forbidden = {"is_abuse", "ring_id", "abuse_type", "loss_amount", "reason_codes", "customer_id"}
    assert forbidden.isdisjoint(fs.X.columns)


def test_determinism():
    """Verify that identical inputs produce bitwise identical temporal features."""
    cfg = load_config("configs/default.yaml")
    ds1 = generate_ecosystem(cfg)
    ds2 = generate_ecosystem(cfg)

    fs1 = build_temporal_velocity_features(ds1.orders)
    fs2 = build_temporal_velocity_features(ds2.orders)

    pd.testing.assert_frame_equal(fs1.X, fs2.X)


def test_chronological_split_consistency():
    """Verify train, validation, and test split temporal integrity with full features."""
    cfg = load_config("configs/default.yaml")
    ds = generate_ecosystem(cfg)
    split = split_by_time(ds.orders, cfg.split["train"], cfg.split["validation"])

    assert split.train.event_time.max() <= split.validation.event_time.min()
    assert split.validation.event_time.max() <= split.test.event_time.min()

    fs_full = build_full_features(ds.orders, ds.labels)
    assert len(fs_full.X) == len(ds.orders)
    assert fs_full.X.shape[1] == 67  # 19 baseline + 18 graph + 30 temporal
