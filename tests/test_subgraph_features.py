"""Unit, causal, temporal, and property tests for streaming suspicious subgraph features."""
import numpy as np
import pandas as pd
import pytest

from abuse_ring_detector.features import build_subgraph_features, build_subgraph_extended_features


@pytest.fixture
def base_toy_orders():
    t0 = pd.Timestamp("2025-01-01 10:00:00")
    t1 = pd.Timestamp("2025-01-01 12:00:00")
    t2 = pd.Timestamp("2025-01-01 14:00:00")
    t3 = pd.Timestamp("2025-01-01 16:00:00")
    
    return pd.DataFrame([
        # C1 uses D1, IP1, A1, P1
        {"order_id": "O001", "customer_id": "C1", "event_time": t0, "amount": 100.0,
         "device_id": "D1", "ip_id": "IP1", "address_id": "A1", "payment_id": "P1"},
        # C2 uses D1, IP1, A2, P2 (shares D1 & IP1 with C1 -> 2 shared modalities)
        {"order_id": "O002", "customer_id": "C2", "event_time": t1, "amount": 200.0,
         "device_id": "D1", "ip_id": "IP1", "address_id": "A2", "payment_id": "P2"},
        # C3 uses D3, IP3, A2, P3 (shares A2 with C2 -> 2-hop from C1)
        {"order_id": "O003", "customer_id": "C3", "event_time": t2, "amount": 300.0,
         "device_id": "D3", "ip_id": "IP3", "address_id": "A2", "payment_id": "P3"},
        # C1 places second order with D1, IP1, A1, P1 at t3
        {"order_id": "O004", "customer_id": "C1", "event_time": t3, "amount": 400.0,
         "device_id": "D1", "ip_id": "IP1", "address_id": "A1", "payment_id": "P1"},
    ])


def test_subgraph_current_event_exclusion(base_toy_orders):
    """Test 1 & 2: Verify current event entities and relationships are excluded from its own feature calculation."""
    fs = build_subgraph_features(base_toy_orders)
    row1 = fs.X.loc["O001"]
    # At t0, C1 has no prior history, so component size is strictly local
    assert row1["subgraph_customer_count_24h"] == 1.0
    assert row1["subgraph_edge_count_24h"] == 0.0
    assert row1["subgraph_multi_entity_conspirator_count_7d"] == 0.0


def test_subgraph_future_event_exclusion(base_toy_orders):
    """Test 3: Future events (O002, O003, O004) must not influence O001 features."""
    fs1 = build_subgraph_features(base_toy_orders)
    # Truncate dataset to only O001
    fs_single = build_subgraph_features(base_toy_orders.iloc[:1])
    pd.testing.assert_series_equal(fs1.X.loc["O001"], fs_single.X.loc["O001"])


def test_subgraph_label_isolation(base_toy_orders):
    """Test 4: Verify zero target/label leakage."""
    labels = pd.DataFrame({"order_id": base_toy_orders["order_id"], "is_abuse": [0, 1, 1, 1]})
    fs_no_labels = build_subgraph_features(base_toy_orders)
    fs_with_labels = build_subgraph_features(base_toy_orders, labels=labels)
    pd.testing.assert_frame_equal(fs_no_labels.X, fs_with_labels.X)


def test_subgraph_determinism(base_toy_orders):
    """Test 5: Repeated executions must be byte-for-byte identical."""
    fs1 = build_subgraph_features(base_toy_orders)
    fs2 = build_subgraph_features(base_toy_orders)
    pd.testing.assert_frame_equal(fs1.X, fs2.X)


def test_subgraph_zero_history_cold_start():
    """Test 6: Brand new cold start orders must produce valid clean defaults."""
    orders = pd.DataFrame([
        {"order_id": "O_NEW", "customer_id": "C_NEW", "event_time": pd.Timestamp("2025-01-01 10:00:00"),
         "amount": 50.0, "device_id": "D_NEW", "ip_id": "IP_NEW", "address_id": "A_NEW", "payment_id": "P_NEW"}
    ])
    fs = build_subgraph_features(orders)
    assert not fs.X.isna().any().any()
    assert not np.isinf(fs.X.values).any()
    assert fs.X.loc["O_NEW", "subgraph_edge_count_24h"] == 0.0
    assert fs.X.loc["O_NEW", "subgraph_customer_count_7d"] == 1.0


def test_subgraph_no_nan_or_inf(base_toy_orders):
    """Test 7: No NaN or Inf across entire table."""
    fs = build_subgraph_features(base_toy_orders)
    assert not fs.X.isna().any().any()
    assert not np.isinf(fs.X.values).any()


def test_subgraph_first_hop_construction(base_toy_orders):
    """Test 8: O002 should correctly identify 1-hop connection to C1 via D1 and IP1."""
    fs = build_subgraph_features(base_toy_orders)
    row2 = fs.X.loc["O002"]
    # C2 sees C1 as peer via D1 and IP1
    assert row2["subgraph_customer_count_7d"] == 2.0  # C1 and C2
    assert row2["subgraph_shared_modality_count_7d"] == 2.0  # D1 and IP1
    assert row2["subgraph_multi_entity_conspirator_count_7d"] == 1.0  # C1 shares 2 modalities


def test_subgraph_second_hop_construction(base_toy_orders):
    """Test 9: O004 should capture the 2-hop connected cluster (C1, C2, C3)."""
    fs = build_subgraph_features(base_toy_orders)
    row4 = fs.X.loc["O004"]
    # At t3, O004 (C1) sees C2 (1-hop via D1, IP1) and C3 (2-hop via A2 from C2)
    assert row4["subgraph_customer_count_7d"] == 3.0  # C1, C2, C3
    assert row4["subgraph_edge_count_7d"] >= 4.0


def test_subgraph_component_size_calculation(base_toy_orders):
    """Test 10: Component node count includes both customers and participating entities."""
    fs = build_subgraph_features(base_toy_orders)
    row4 = fs.X.loc["O004"]
    # Nodes = Customers (C1, C2, C3) + Entities (D1, IP1, A1, P1, A2, P2, D3, IP3, P3)
    assert row4["subgraph_node_count_7d"] >= 5.0
    assert row4["subgraph_node_count_7d"] == row4["subgraph_customer_count_7d"] + row4["subgraph_entity_count_7d"]


def test_subgraph_density_calculation(base_toy_orders):
    """Test 11: Bipartite density satisfies 0 <= density <= 1."""
    fs = build_subgraph_features(base_toy_orders)
    for col in ["subgraph_edge_density_24h", "subgraph_edge_density_7d"]:
        assert (fs.X[col] >= 0.0).all()
        assert (fs.X[col] <= 1.0).all()


def test_subgraph_expansion_calculation():
    """Test 12: 1-hour fast expansion dynamics."""
    t0 = pd.Timestamp("2025-01-01 10:00:00")
    t_burst = pd.Timestamp("2025-01-01 10:30:00")  # within 30 min (1h window)
    
    orders = pd.DataFrame([
        {"order_id": "O001", "customer_id": "C1", "event_time": t0, "amount": 100.0,
         "device_id": "D1", "ip_id": "IP1", "address_id": "A1", "payment_id": "P1"},
        {"order_id": "O002", "customer_id": "C2", "event_time": t_burst, "amount": 200.0,
         "device_id": "D1", "ip_id": "IP2", "address_id": "A2", "payment_id": "P2"},
    ])
    fs = build_subgraph_features(orders)
    row2 = fs.X.loc["O002"]
    assert row2["subgraph_new_nodes_1h"] >= 1.0
    assert row2["subgraph_new_edges_1h"] >= 1.0


def test_subgraph_future_perturbation_invariance(base_toy_orders):
    """Test 13: Modifying a future event must produce zero change in earlier feature vectors."""
    fs_orig = build_subgraph_features(base_toy_orders)
    
    # Mutate future event O004
    mutated_orders = base_toy_orders.copy()
    mutated_orders.loc[3, "device_id"] = "D_MUTATED"
    mutated_orders.loc[3, "amount"] = 99999.0
    
    fs_mutated = build_subgraph_features(mutated_orders)
    
    # Earlier orders O001, O002, O003 must be identical
    pd.testing.assert_frame_equal(fs_orig.X.loc[["O001", "O002", "O003"]],
                                  fs_mutated.X.loc[["O001", "O002", "O003"]])


def test_subgraph_chronological_split_consistency():
    """Test 14: Subgraph extended feature set has 137 features and preserves row alignment."""
    t0 = pd.Timestamp("2025-01-01 10:00:00")
    orders = pd.DataFrame([
        {"order_id": f"O{i:03d}", "customer_id": f"C{i%3}", "event_time": t0 + pd.Timedelta(hours=i),
         "amount": 100.0 + i, "merchant_category": "electronics", "retry_count": 0,
         "device_id": f"D{i%2}", "ip_id": f"IP{i%2}", "address_id": f"A{i%2}", "payment_id": f"P{i%2}"}
        for i in range(10)
    ])
    fs = build_subgraph_extended_features(orders)
    assert len(fs.X.columns) == 137
    assert len(fs.X) == 10
    assert (fs.X.index == orders["order_id"]).all()


def test_subgraph_controlled_handbuilt_graph_correctness():
    """Test 15: Exact hand-built graph verification."""
    t0 = pd.Timestamp("2025-01-01 10:00:00")
    orders = pd.DataFrame([
        {"order_id": "O1", "customer_id": "C1", "event_time": t0, "amount": 100.0,
         "device_id": "D1", "ip_id": "IP1", "address_id": "A1", "payment_id": "P1"},
        {"order_id": "O2", "customer_id": "C2", "event_time": t0 + pd.Timedelta(minutes=5), "amount": 150.0,
         "device_id": "D1", "ip_id": "IP1", "address_id": "A1", "payment_id": "P1"},
    ])
    fs = build_subgraph_features(orders)
    row2 = fs.X.loc["O2"]
    assert row2["subgraph_customer_count_7d"] == 2.0
    assert row2["subgraph_entity_count_7d"] == 4.0
    assert row2["subgraph_edge_count_7d"] == 4.0  # (C1, D1), (C1, IP1), (C1, A1), (C1, P1)
    assert row2["subgraph_shared_modality_count_7d"] == 4.0
    assert row2["subgraph_multi_entity_conspirator_count_7d"] == 1.0

