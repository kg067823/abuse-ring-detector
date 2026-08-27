"""Unit and leakage tests for customer-relative temporal velocity features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from abuse_ring_detector.config import Config, RingConfig
from abuse_ring_detector.features import (
    build_customer_relative_features,
    build_extended_features,
)
from abuse_ring_detector.splits import split_by_time
from abuse_ring_detector.synthetic import generate_ecosystem


def _create_controlled_customer_dataset() -> pd.DataFrame:
    """Create a minimal deterministic sequence of events for customer C1 and C2."""
    t0 = pd.Timestamp("2025-06-15 12:00:00")
    orders = pd.DataFrame([
        # C1 events:
        {"order_id": "O001", "customer_id": "C1", "event_time": t0 - pd.Timedelta(days=15), "device_id": "D1", "ip_id": "IP1", "address_id": "A1", "payment_id": "P1", "amount": 100.0, "retry_count": 0, "merchant_category": "electronics"},
        {"order_id": "O002", "customer_id": "C1", "event_time": t0 - pd.Timedelta(days=5), "device_id": "D1", "ip_id": "IP1", "address_id": "A1", "payment_id": "P1", "amount": 200.0, "retry_count": 0, "merchant_category": "electronics"},
        {"order_id": "O003", "customer_id": "C1", "event_time": t0 - pd.Timedelta(hours=12), "device_id": "D1", "ip_id": "IP1", "address_id": "A1", "payment_id": "P1", "amount": 150.0, "retry_count": 0, "merchant_category": "electronics"},
        {"order_id": "O004", "customer_id": "C1", "event_time": t0 - pd.Timedelta(minutes=30), "device_id": "D2", "ip_id": "IP1", "address_id": "A1", "payment_id": "P1", "amount": 300.0, "retry_count": 0, "merchant_category": "electronics"},
        # Target event for C1 at t0:
        {"order_id": "O005", "customer_id": "C1", "event_time": t0, "device_id": "D2", "ip_id": "IP1", "address_id": "A1", "payment_id": "P1", "amount": 500.0, "retry_count": 0, "merchant_category": "electronics"},
        # Future event for C1 at t0 + 10m:
        {"order_id": "O006", "customer_id": "C1", "event_time": t0 + pd.Timedelta(minutes=10), "device_id": "D2", "ip_id": "IP1", "address_id": "A1", "payment_id": "P1", "amount": 600.0, "retry_count": 0, "merchant_category": "electronics"},
        # C2 event (new customer):
        {"order_id": "O007", "customer_id": "C2", "event_time": t0, "device_id": "D1", "ip_id": "IP2", "address_id": "A2", "payment_id": "P2", "amount": 100.0, "retry_count": 0, "merchant_category": "clothing"},
    ])
    return orders


def test_controlled_window_correctness():
    """1. Verify exact counting across 1h, 24h, 7d, 30d sliding windows."""
    orders = _create_controlled_customer_dataset()
    fs = build_customer_relative_features(orders)
    X = fs.X

    # Prior events for O005 (C1 at t0):
    # O001 (t0 - 15d): in 30d (yes), in 7d (no), in 24h (no), in 1h (no). D1.
    # O002 (t0 - 5d):  in 30d (yes), in 7d (yes), in 24h (no), in 1h (no). D1.
    # O003 (t0 - 12h): in 30d (yes), in 7d (yes), in 24h (yes), in 1h (no). D1.
    # O004 (t0 - 30m): in 30d (yes), in 7d (yes), in 24h (yes), in 1h (yes). D2.
    row_o5 = X.loc["O005"]
    assert row_o5["customer_order_count_30d"] == 4.0
    assert row_o5["customer_order_count_7d"] == 3.0
    assert row_o5["customer_order_count_24h"] == 2.0
    assert row_o5["customer_order_count_1h"] == 1.0

    # Device-specific counts for O005 (device is D2):
    # D2 was used only at O004 (t0 - 30m)
    assert row_o5["customer_device_orders_1h"] == 1.0
    assert row_o5["customer_device_orders_24h"] == 1.0
    assert row_o5["customer_device_orders_7d"] == 1.0

    # Address-specific counts for O005 (address is A1):
    # A1 was used in O001, O002, O003, O004
    assert row_o5["customer_address_orders_1h"] == 1.0
    assert row_o5["customer_address_orders_24h"] == 2.0
    assert row_o5["customer_address_orders_7d"] == 3.0


def test_current_event_exclusion():
    """2. Verify that the current transaction at time T is excluded from its own features."""
    orders = _create_controlled_customer_dataset()
    fs = build_customer_relative_features(orders)
    X = fs.X

    # First event O001 must have 0.0 prior customer stats
    row_o1 = X.loc["O001"]
    assert row_o1["customer_order_count_1h"] == 0.0
    assert row_o1["customer_order_count_24h"] == 0.0
    assert row_o1["customer_order_count_7d"] == 0.0
    assert row_o1["customer_order_count_30d"] == 0.0
    assert row_o1["customer_device_orders_1h"] == 0.0
    assert row_o1["customer_device_velocity_ratio"] == 0.0

    # C2 first event O007 must also have 0.0 prior customer stats
    row_o7 = X.loc["O007"]
    assert row_o7["customer_order_count_1h"] == 0.0
    assert row_o7["customer_order_count_24h"] == 0.0
    assert row_o7["customer_order_count_7d"] == 0.0
    assert row_o7["customer_device_orders_1h"] == 0.0


def test_future_event_exclusion():
    """3. Verify modifying future transactions has zero effect on earlier feature rows."""
    orders1 = _create_controlled_customer_dataset()
    orders2 = _create_controlled_customer_dataset()
    # Modify future event O006
    orders2.loc[orders2["order_id"] == "O006", "device_id"] = "D_FUTURE_MUTATED"
    orders2.loc[orders2["order_id"] == "O006", "amount"] = 999999.0

    X1 = build_customer_relative_features(orders1).X
    X2 = build_customer_relative_features(orders2).X

    # Features for O001 through O005 must be exactly identical
    earlier_ids = ["O001", "O002", "O003", "O004", "O005", "O007"]
    pd.testing.assert_frame_equal(X1.loc[earlier_ids], X2.loc[earlier_ids])


def test_target_isolation():
    """4. Verify ground truth labels are strictly absent from features."""
    config = Config(customers=100, orders=200, date_range_days=30,
                    rings=RingConfig(count=10, min_size=3, max_size=5))
    dataset = generate_ecosystem(config)
    fs_cust = build_customer_relative_features(dataset.orders, dataset.labels)
    fs_ext = build_extended_features(dataset.orders, dataset.labels)

    forbidden = {"is_abuse", "ring_id", "abuse_type", "loss_amount", "reason_codes", "is_abusive"}
    for col in forbidden:
        assert col not in fs_cust.X.columns, f"{col} leaked into customer relative features"
        assert col not in fs_ext.X.columns, f"{col} leaked into extended features"


def test_deterministic_output():
    """5. Verify deterministic, bitwise identical feature matrices across repeated runs."""
    config = Config(customers=100, orders=200, date_range_days=30,
                    rings=RingConfig(count=10, min_size=3, max_size=5))
    dataset = generate_ecosystem(config)

    fs1 = build_customer_relative_features(dataset.orders, dataset.labels)
    fs2 = build_customer_relative_features(dataset.orders, dataset.labels)

    pd.testing.assert_frame_equal(fs1.X, fs2.X)


def test_zero_history_and_denominator_stability():
    """6. Verify all denominators prevent division by zero or NaN values."""
    orders = _create_controlled_customer_dataset()
    fs = build_customer_relative_features(orders)
    X = fs.X

    assert not X.isna().any().any(), "Found NaN in customer relative features"
    assert not np.isinf(X.to_numpy()).any(), "Found Inf in customer relative features"


def test_chronological_split_consistency():
    """7. Verify full extended feature builder shape and strict temporal splitting."""
    config = Config(customers=200, orders=500, date_range_days=60,
                    rings=RingConfig(count=20, min_size=3, max_size=5))
    dataset = generate_ecosystem(config)
    split = split_by_time(dataset.orders, 0.7, 0.15)

    fs_ext = build_extended_features(dataset.orders, dataset.labels)
    assert fs_ext.X.shape[1] == 97, f"Expected 97 features, got {fs_ext.X.shape[1]}"
    assert len(fs_ext.X) == len(dataset.orders)

    # Train, validation, test rows match split index
    train_x = fs_ext.X.loc[split.train.order_id]
    val_x = fs_ext.X.loc[split.validation.order_id]
    test_x = fs_ext.X.loc[split.test.order_id]

    assert len(train_x) == len(split.train)
    assert len(val_x) == len(split.validation)
    assert len(test_x) == len(split.test)
