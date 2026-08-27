"""Comprehensive test suite for 2-hop streaming graph features (Model E).

Tests include:
1. Exact controlled-window correctness (7d and 30d).
2. Current-event exclusion.
3. Future-event exclusion.
4. Label/target isolation.
5. Deterministic output.
6. Zero-history behavior.
7. No NaN / Inf.
8. Chronological split consistency.
9. 2-hop path correctness on a hand-built graph.
10. Verification that modifying a future second-hop event does not modify an earlier feature vector.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from abuse_ring_detector.features import (
    build_two_hop_features,
    build_two_hop_extended_features,
)


@pytest.fixture
def sample_orders_and_labels():
    """Create a hand-crafted chronological order stream for testing 2-hop graph connectivity."""
    t0 = pd.Timestamp("2025-01-01 10:00:00")
    orders = pd.DataFrame([
        # Day 1: Customer A uses Device 1, Address 1, IP 1, Payment 1
        {"order_id": "O1", "customer_id": "C_A", "event_time": t0, "amount": 100.0,
         "device_id": "D1", "address_id": "A1", "ip_id": "IP1", "payment_id": "P1"},
        # Day 2 (within 7d): Customer B uses Device 1 (shares with A), but Address 2, IP 2, Payment 2
        {"order_id": "O2", "customer_id": "C_B", "event_time": t0 + pd.Timedelta(days=1), "amount": 120.0,
         "device_id": "D1", "address_id": "A2", "ip_id": "IP2", "payment_id": "P2"},
        # Day 3 (within 7d): Customer C uses Address 2 (shares with B), but Device 3, IP 3, Payment 3
        # At this point, C is 2-hops from A via B (A -> D1 -> B -> A2 -> C)
        {"order_id": "O3", "customer_id": "C_C", "event_time": t0 + pd.Timedelta(days=2), "amount": 150.0,
         "device_id": "D3", "address_id": "A2", "ip_id": "IP3", "payment_id": "P3"},
        # Day 4 (within 7d): Customer A places another order with D1, A1
        # A should now see 2-hop connected Address A2 via D1 (from B), and peer devices D1, D3 (if C connected)
        {"order_id": "O4", "customer_id": "C_A", "event_time": t0 + pd.Timedelta(days=3), "amount": 200.0,
         "device_id": "D1", "address_id": "A1", "ip_id": "IP1", "payment_id": "P1"},
        # Day 40 (beyond 30d): Customer D uses D1
        {"order_id": "O5", "customer_id": "C_D", "event_time": t0 + pd.Timedelta(days=40), "amount": 110.0,
         "device_id": "D1", "address_id": "A4", "ip_id": "IP4", "payment_id": "P4"},
    ])
    labels = pd.DataFrame([
        {"order_id": "O1", "is_abuse": 0, "loss_amount": 0.0},
        {"order_id": "O2", "is_abuse": 1, "loss_amount": 120.0},
        {"order_id": "O3", "is_abuse": 1, "loss_amount": 150.0},
        {"order_id": "O4", "is_abuse": 1, "loss_amount": 200.0},
        {"order_id": "O5", "is_abuse": 0, "loss_amount": 0.0},
    ])
    return orders, labels


def test_two_hop_feature_count():
    """Verify that build_two_hop_features outputs exactly 20 features and Model E has 117 features."""
    t0 = pd.Timestamp("2025-01-01 10:00:00")
    orders = pd.DataFrame([
        {"order_id": "O1", "customer_id": "C1", "event_time": t0, "amount": 100.0,
         "merchant_category": "electronics", "retry_count": 0, "hour_of_day": 10, "day_of_week": 2,
         "device_id": "D1", "address_id": "A1", "ip_id": "IP1", "payment_id": "P1"},
    ])
    labels = pd.DataFrame([{"order_id": "O1", "is_abuse": 0, "loss_amount": 0.0}])

    fs_2hop = build_two_hop_features(orders, labels)
    assert len(fs_2hop.X.columns) == 20

    fs_e = build_two_hop_extended_features(orders, labels)
    assert len(fs_e.X.columns) == 117


def test_hand_built_two_hop_path_correctness(sample_orders_and_labels):
    """Test 1 & 9: Verify exact 2-hop path correctness on a hand-built graph."""
    orders, labels = sample_orders_and_labels
    fs = build_two_hop_features(orders, labels)
    X = fs.X

    # O1: No prior history
    assert X.loc["O1", "two_hop_shared_device_customers_7d"] == 0.0
    assert X.loc["O1", "two_hop_distinct_connected_customers_7d"] == 0.0
    assert X.loc["O1", "two_hop_connected_addresses_via_device_7d"] == 0.0

    # O2 (C_B using D1): 1-hop peer customer C_A on D1
    assert X.loc["O2", "two_hop_shared_device_customers_7d"] == 1.0  # C_A
    assert X.loc["O2", "two_hop_distinct_connected_customers_7d"] == 1.0
    # C_A has address A1, which is not C_B's current A2 -> 1 connected address via device
    assert X.loc["O2", "two_hop_connected_addresses_via_device_7d"] == 1.0
    assert X.loc["O2", "two_hop_total_peer_addresses_7d"] == 1.0

    # O3 (C_C using A2): 1-hop peer customer C_B on A2
    assert X.loc["O3", "two_hop_shared_address_customers_7d"] == 1.0  # C_B
    # C_B has device D1, which is not C_C's current D3 -> 1 connected device via address
    assert X.loc["O3", "two_hop_connected_devices_via_address_7d"] == 1.0
    assert X.loc["O3", "two_hop_total_peer_devices_7d"] == 1.0

    # O4 (C_A using D1, A1): 1-hop peer customer C_B on D1
    assert X.loc["O4", "two_hop_shared_device_customers_7d"] == 1.0  # C_B
    # C_B previously used address A2 -> 1 connected address via device
    assert X.loc["O4", "two_hop_connected_addresses_via_device_7d"] == 1.0
    assert X.loc["O4", "two_hop_total_peer_addresses_7d"] == 1.0


def test_controlled_window_correctness(sample_orders_and_labels):
    """Test 1: Verify 7-day and 30-day lookback window cutoff behavior."""
    orders, labels = sample_orders_and_labels
    fs = build_two_hop_features(orders, labels)
    X = fs.X

    # O5 occurs on Day 40 (beyond 30 days of O1, O2, O3, O4)
    # D1 was used by C_A and C_B 37+ days ago, so it should be expired
    assert X.loc["O5", "two_hop_shared_device_customers_7d"] == 0.0
    assert X.loc["O5", "two_hop_distinct_connected_customers_7d"] == 0.0
    assert X.loc["O5", "two_hop_distinct_connected_customers_30d"] == 0.0
    assert X.loc["O5", "two_hop_peer_cluster_size_7d"] == 0.0


def test_current_event_exclusion(sample_orders_and_labels):
    """Test 2: Ensure current event entities and customer are strictly excluded from its own peer features."""
    orders, labels = sample_orders_and_labels
    fs = build_two_hop_features(orders, labels)
    X = fs.X

    # For O1, even though it provides D1 and A1, its peer counts must be 0
    assert X.loc["O1", "two_hop_distinct_connected_customers_7d"] == 0.0
    assert X.loc["O1", "two_hop_connected_addresses_via_device_7d"] == 0.0
    assert X.loc["O1", "two_hop_total_peer_orders_7d"] == 0.0


def test_future_event_exclusion(sample_orders_and_labels):
    """Test 3: Future events must not affect earlier feature vectors."""
    orders, labels = sample_orders_and_labels

    # Run on first 3 events
    fs_sub = build_two_hop_features(orders.iloc[:3], labels.iloc[:3])
    # Run on all 5 events
    fs_full = build_two_hop_features(orders, labels)

    pd.testing.assert_frame_equal(fs_sub.X, fs_full.X.iloc[:3])


def test_future_event_modification_invariance(sample_orders_and_labels):
    """Test 10: Modifying a future event must not alter any earlier feature vector."""
    orders, labels = sample_orders_and_labels

    fs1 = build_two_hop_features(orders, labels)

    modified_orders = orders.copy()
    # Modify O3 (future relative to O1 and O2)
    modified_orders.loc[2, "device_id"] = "MODIFIED_DEV_999"
    modified_orders.loc[2, "address_id"] = "MODIFIED_ADDR_999"

    fs2 = build_two_hop_features(modified_orders, labels)

    # O1 and O2 must be identical
    pd.testing.assert_series_equal(fs1.X.loc["O1"], fs2.X.loc["O1"])
    pd.testing.assert_series_equal(fs1.X.loc["O2"], fs2.X.loc["O2"])


def test_label_target_isolation(sample_orders_and_labels):
    """Test 4: Labels must never leak into feature matrix X."""
    orders, labels = sample_orders_and_labels

    # Run with labels
    fs_with_labels = build_two_hop_features(orders, labels)
    # Run without labels
    fs_without_labels = build_two_hop_features(orders, None)

    pd.testing.assert_frame_equal(fs_with_labels.X, fs_without_labels.X)

    # Invert labels and verify X is identical
    inverted_labels = labels.copy()
    inverted_labels["is_abuse"] = 1 - inverted_labels["is_abuse"]
    fs_inverted = build_two_hop_features(orders, inverted_labels)
    pd.testing.assert_frame_equal(fs_with_labels.X, fs_inverted.X)


def test_deterministic_output(sample_orders_and_labels):
    """Test 5: Repeated executions must yield byte-for-byte identical feature matrices."""
    orders, labels = sample_orders_and_labels

    fs1 = build_two_hop_features(orders, labels)
    fs2 = build_two_hop_features(orders, labels)

    pd.testing.assert_frame_equal(fs1.X, fs2.X)


def test_zero_history_behavior():
    """Test 6: Single first order has all zero 2-hop features."""
    t0 = pd.Timestamp("2025-01-01 10:00:00")
    orders = pd.DataFrame([
        {"order_id": "O1", "customer_id": "C1", "event_time": t0, "amount": 100.0,
         "device_id": "D1", "address_id": "A1", "ip_id": "IP1", "payment_id": "P1"},
    ])
    fs = build_two_hop_features(orders)
    row = fs.X.iloc[0]
    assert (row == 0.0).all()


def test_no_nan_or_inf():
    """Test 7: No NaN or Inf values in generated feature matrix."""
    t0 = pd.Timestamp("2025-01-01 10:00:00")
    orders = pd.DataFrame([
        {"order_id": f"O{i}", "customer_id": f"C{i % 5}",
         "event_time": t0 + pd.Timedelta(hours=i * 2), "amount": float(i * 10),
         "device_id": f"D{i % 3}", "address_id": f"A{i % 4}", "ip_id": f"IP{i % 2}", "payment_id": f"P{i % 3}"}
        for i in range(30)
    ])
    fs = build_two_hop_features(orders)
    assert not fs.X.isna().any().any()
    assert not np.isinf(fs.X.to_numpy()).any()


def test_chronological_split_consistency(sample_orders_and_labels):
    """Test 8: Feature values computed on full dataset match order-by-order temporal slicing."""
    orders, labels = sample_orders_and_labels
    fs = build_two_hop_features(orders, labels)

    # Slice at index 3
    train_orders = orders.iloc[:3]
    fs_train = build_two_hop_features(train_orders)

    pd.testing.assert_frame_equal(fs_train.X, fs.X.iloc[:3])
