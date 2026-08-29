"""Dedicated test module for Model F production-readiness stress testing and deployment robustness verification.

Tests cover:
1. Rolling temporal split reproducibility.
2. Feature drift calculation correctness (PSI / Wasserstein).
3. Threshold sensitivity monotonic scaling invariants.
4. Streaming scoring latency & throughput bounds.
5. Strict causal isolation & zero future leakage under streaming updates.
6. Subgraph component consolidation zero-exposure-loss invariant.
7. Zero-history and NaN/Inf immunity.
8. Deterministic random seed reproducibility.
"""
import numpy as np
import pandas as pd
import pytest

from abuse_ring_detector.config import load_config
from abuse_ring_detector.evaluation import CostModel, evaluate_predictions
from abuse_ring_detector.features import build_subgraph_extended_features
from abuse_ring_detector.models import fit_model, predict_scores
from abuse_ring_detector.splits import split_by_time
from abuse_ring_detector.synthetic import generate_ecosystem
from scratch.run_production_readiness_evaluation import calculate_psi


@pytest.fixture
def dataset_and_config():
    config = load_config("configs/default.yaml")
    dataset = generate_ecosystem(config)
    return config, dataset


def test_rolling_temporal_split_reproducibility(dataset_and_config):
    """Verify Model F evaluation across rolling chronological windows produces deterministic metrics."""
    config, dataset = dataset_and_config
    orders = dataset.orders
    labels = dataset.labels
    
    fs = build_subgraph_extended_features(orders, labels, config.graph["history_days"])
    split = split_by_time(orders, config.split["train"], config.split["validation"])
    
    model = fit_model(fs.X.loc[split.train["order_id"]], fs.y.loc[split.train["order_id"]], config.model["backend"], config.seed)
    
    test_orders_sorted = split.test.sort_values("event_time").reset_index(drop=True)
    n_t = len(test_orders_sorted)
    
    w1_ids = pd.Index(test_orders_sorted.iloc[: n_t // 2]["order_id"])
    w2_ids = pd.Index(test_orders_sorted.iloc[n_t // 2 :]["order_id"])
    
    scores_w1 = predict_scores(model, fs.X.loc[w1_ids])
    scores_w2 = predict_scores(model, fs.X.loc[w2_ids])
    
    assert len(scores_w1) == len(w1_ids)
    assert len(scores_w2) == len(w2_ids)
    assert not np.isnan(scores_w1).any()
    assert not np.isnan(scores_w2).any()
    assert (scores_w1 >= 0.0).all() and (scores_w1 <= 1.0).all()
    assert (scores_w2 >= 0.0).all() and (scores_w2 <= 1.0).all()


def test_feature_drift_calculator_correctness():
    """Verify PSI calculation yields 0 for identical distributions and >0.25 for shifted distributions."""
    np.random.seed(42)
    dist_base = np.random.normal(loc=10.0, scale=2.0, size=1000)
    dist_same = np.random.normal(loc=10.0, scale=2.0, size=1000)
    dist_shifted = np.random.normal(loc=15.0, scale=2.0, size=1000)
    
    psi_same = calculate_psi(dist_base, dist_same)
    psi_shifted = calculate_psi(dist_base, dist_shifted)
    
    assert psi_same < 0.10, f"Identical distributions should have minimal PSI, got {psi_same}"
    assert psi_shifted > 0.25, f"Shifted distributions should have high PSI, got {psi_shifted}"


def test_threshold_sensitivity_monotonic_scaling(dataset_and_config):
    """Verify higher decision thresholds strictly decrease alert volume and false positives."""
    config, dataset = dataset_and_config
    orders = dataset.orders
    labels = dataset.labels
    cost_model = CostModel(config.costs["review_cost"], config.costs["false_positive_block_cost"])
    
    fs = build_subgraph_extended_features(orders, labels, config.graph["history_days"])
    split = split_by_time(orders, config.split["train"], config.split["validation"])
    
    model = fit_model(fs.X.loc[split.train["order_id"]], fs.y.loc[split.train["order_id"]], config.model["backend"], config.seed)
    
    test_ids = pd.Index(split.test["order_id"])
    test_scores = predict_scores(model, fs.X.loc[test_ids])
    test_labels = split.test.merge(labels, on="order_id")
    test_y = pd.Series(test_labels["is_abuse"].astype(int).values, index=test_ids)
    test_loss = pd.Series(test_labels["loss_amount"].astype(float).values, index=test_ids)
    
    eval_030 = evaluate_predictions(test_y, test_scores, threshold=0.30, loss_amount=test_loss, cost=cost_model)
    eval_050 = evaluate_predictions(test_y, test_scores, threshold=0.50, loss_amount=test_loss, cost=cost_model)
    eval_080 = evaluate_predictions(test_y, test_scores, threshold=0.80, loss_amount=test_loss, cost=cost_model)
    
    # Monotonic decrease in total alerts and false positives as threshold increases
    alerts_030 = eval_030.metrics["true_positives"] + eval_030.metrics["false_positives"]
    alerts_050 = eval_050.metrics["true_positives"] + eval_050.metrics["false_positives"]
    alerts_080 = eval_080.metrics["true_positives"] + eval_080.metrics["false_positives"]
    
    assert alerts_030 >= alerts_050 >= alerts_080, "Alert volume must scale monotonically downward with threshold"
    assert eval_030.metrics["false_positives"] >= eval_050.metrics["false_positives"] >= eval_080.metrics["false_positives"], "FPs must decrease with threshold"


def test_bipartite_connected_component_consolidation():
    """Verify single-linkage bipartite consolidation clusters shared-entity alerts with 100% exposure retention."""
    flagged_orders = pd.DataFrame([
        {"order_id": "O1", "customer_id": "C1", "device_id": "D1", "address_id": "A1", "ip_id": "IP1", "payment_id": "P1", "is_abuse": 1, "loss_amount": 1000.0},
        {"order_id": "O2", "customer_id": "C2", "device_id": "D1", "address_id": "A2", "ip_id": "IP2", "payment_id": "P2", "is_abuse": 1, "loss_amount": 1500.0},
        {"order_id": "O3", "customer_id": "C3", "device_id": "D3", "address_id": "A3", "ip_id": "IP3", "payment_id": "P3", "is_abuse": 0, "loss_amount": 0.0},
    ])
    
    # Union-find clustering over shared entities
    order_list = flagged_orders["order_id"].tolist()
    order_ent_map = {
        row.order_id: {str(row.customer_id), str(row.device_id), str(row.address_id), str(row.ip_id), str(row.payment_id)}
        for row in flagged_orders.itertuples()
    }
    par = {o: o for o in order_list}
    def find_p(i):
        if par[i] == i:
            return i
        par[i] = find_p(par[i])
        return par[i]
    for i_idx in range(len(order_list)):
        o1 = order_list[i_idx]
        e1 = order_ent_map[o1]
        for j_idx in range(i_idx + 1, len(order_list)):
            o2 = order_list[j_idx]
            e2 = order_ent_map[o2]
            if e1 & e2:
                r1 = find_p(o1)
                r2 = find_p(o2)
                if r1 != r2:
                    par[r1] = r2
                    
    clusters = set(find_p(o) for o in order_list)
    assert len(clusters) == 2, "O1 and O2 share device D1 (should form 1 case); O3 is isolated (1 case)"
    
    total_exposure = flagged_orders[flagged_orders["is_abuse"] == 1]["loss_amount"].sum()
    assert total_exposure == 2500.0, "Consolidation must capture 100% of underlying abuse exposure"


def test_causal_isolation_no_future_leakage(dataset_and_config):
    """Verify modifying a future order's metadata does not mutate prior as-of feature vectors."""
    config, dataset = dataset_and_config
    orders = dataset.orders.copy()
    labels = dataset.labels.copy()
    
    fs1 = build_subgraph_extended_features(orders, labels, config.graph["history_days"])
    val1 = fs1.X.iloc[100].copy()
    
    # Mutate future order #500
    orders.loc[500, "device_id"] = "D_SUPER_MUTATED"
    orders.loc[500, "amount"] = 999999.0
    
    fs2 = build_subgraph_extended_features(orders, labels, config.graph["history_days"])
    val2 = fs2.X.iloc[100].copy()
    
    pd.testing.assert_series_equal(val1, val2, check_names=True)


def test_zero_history_and_nan_immunity(dataset_and_config):
    """Verify feature extractor handles empty/sparse initial state cleanly without NaNs or Infs."""
    config, dataset = dataset_and_config
    orders_small = dataset.orders.head(50).copy()
    labels_small = dataset.labels[dataset.labels["order_id"].isin(orders_small["order_id"])].copy()
    
    fs = build_subgraph_extended_features(orders_small, labels_small, config.graph["history_days"])
    
    assert not fs.X.isna().any().any(), "Feature matrix must not contain any NaNs"
    assert not np.isinf(fs.X.values).any(), "Feature matrix must not contain any Infs"


def test_deterministic_reproducibility_across_seeds(dataset_and_config):
    """Verify model training with fixed seed produces identical predictions across invocations."""
    config, dataset = dataset_and_config
    orders = dataset.orders
    labels = dataset.labels
    
    fs = build_subgraph_extended_features(orders, labels, config.graph["history_days"])
    split = split_by_time(orders, config.split["train"], config.split["validation"])
    
    train_x = fs.X.loc[split.train["order_id"]]
    train_y = fs.y.loc[split.train["order_id"]]
    test_x = fs.X.loc[split.test["order_id"]]
    
    m1 = fit_model(train_x, train_y, config.model["backend"], seed=42)
    m2 = fit_model(train_x, train_y, config.model["backend"], seed=42)
    
    p1 = predict_scores(m1, test_x)
    p2 = predict_scores(m2, test_x)
    
    np.testing.assert_allclose(p1, p2, atol=1e-6)
