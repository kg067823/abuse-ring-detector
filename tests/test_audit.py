"""Audit test suite validating threshold fairness, metric math, latency, Top-K, and leakage safety."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from abuse_ring_detector.config import Config, RingConfig
from abuse_ring_detector.evaluation import CostModel, choose_threshold, evaluate_predictions
from abuse_ring_detector.features import build_baseline_features, build_graph_features
from abuse_ring_detector.ring_evaluation import evaluate_rings, evaluate_top_k
from abuse_ring_detector.synthetic import generate_ecosystem


def test_threshold_selection_validation_only():
    """Verify threshold selection uses validation data only and is unaffected by test labels."""
    y_val = pd.Series([0, 1, 0, 1, 0, 0, 1, 0])
    scores_val = np.array([0.1, 0.7, 0.2, 0.8, 0.4, 0.3, 0.9, 0.2])
    loss_val = pd.Series([0, 100, 0, 200, 0, 0, 150, 0])
    cost = CostModel(2.0, 10.0)

    val_eval = evaluate_predictions(y_val, scores_val, loss_amount=loss_val, cost=cost)
    chosen_tau = choose_threshold(val_eval)

    # Threshold must be chosen from validation threshold table
    assert chosen_tau in [0.5, 0.6, 0.7, 0.8, 0.9]
    # Check that it minimizes validation expected loss
    min_row = val_eval.threshold_table.sort_values(["expected_loss", "threshold"]).iloc[0]
    assert chosen_tau == min_row.threshold


def test_pr_auc_uses_continuous_probabilities():
    """Verify that PR-AUC is calculated from continuous probability scores."""
    y_true = pd.Series([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])

    eval_res = evaluate_predictions(y_true, scores, threshold=0.5)
    expected_pr_auc = average_precision_score(y_true, scores)

    assert np.isclose(eval_res.metrics["pr_auc"], expected_pr_auc)


def test_ring_coverage_and_rules_correctness():
    """Verify ring detection rules A (any), B (20%), C (50%), and coverage calculations."""
    test_orders = pd.DataFrame({
        "order_id": ["O1", "O2", "O3", "O4"],
        "customer_id": ["C1", "C2", "C3", "C4"],
        "event_time": pd.to_datetime(["2025-06-01 10:00", "2025-06-01 11:00", "2025-06-01 12:00", "2025-06-01 13:00"]),
        "amount": [100.0, 200.0, 300.0, 400.0],
    })
    labels = pd.DataFrame({
        "order_id": ["O1", "O2", "O3", "O4"],
        "is_abuse": [True, True, True, True],
        "ring_id": ["R1", "R1", "R2", "R2"],
        "abuse_type": ["shared_device", "shared_device", "shared_address", "shared_address"],
        "loss_amount": [80.0, 160.0, 250.0, 350.0],
    })
    # Scores: R1 has C1 flagged (0.8) and C2 unflagged (0.3); R2 has both unflagged (0.2, 0.4)
    scores = np.array([0.8, 0.3, 0.2, 0.4])

    res = evaluate_rings(test_orders, labels, scores, threshold=0.5)

    # R1: 2 active members (C1, C2), 1 flagged (C1) -> coverage = 0.50
    # Rule A: True, Rule B: True, Rule C: True
    # R2: 2 active members (C3, C4), 0 flagged -> coverage = 0.0
    # Rule A: False, Rule B: False, Rule C: False
    pr = res.per_ring.set_index("ring_id")
    assert pr.loc["R1", "coverage"] == 0.50
    assert bool(pr.loc["R1", "detected_rule_a"]) is True
    assert bool(pr.loc["R1", "detected_rule_b"]) is True
    assert bool(pr.loc["R1", "detected_rule_c"]) is True

    assert pr.loc["R2", "coverage"] == 0.0
    assert bool(pr.loc["R2", "detected_rule_a"]) is False
    assert bool(pr.loc["R2", "detected_rule_b"]) is False
    assert bool(pr.loc["R2", "detected_rule_c"]) is False

    assert res.metrics["rule_a_recall"] == 0.50
    assert res.metrics["mean_member_coverage"] == 0.25
    assert res.metrics["total_exposure_captured"] == 80.0


def test_top_k_ordering_and_exposure_correctness():
    """Verify Top-K correctly ranks by score and calculates cumulative exposure captured."""
    eval_df = pd.DataFrame({
        "order_id": ["O1", "O2", "O3", "O4"],
        "customer_id": ["C1", "C2", "C3", "C4"],
        "event_time": pd.to_datetime(["2025-06-01 10:00", "2025-06-01 11:00", "2025-06-01 12:00", "2025-06-01 13:00"]),
        "amount": [100.0, 200.0, 300.0, 400.0],
        "is_abuse": [True, False, True, False],
        "ring_id": ["R1", pd.NA, "R2", pd.NA],
        "loss_amount": [50.0, 0.0, 150.0, 0.0],
        "predicted_score": [0.9, 0.8, 0.7, 0.2],
    })

    top_k_df = evaluate_top_k(eval_df, active_ring_ids=["R1", "R2"], top_k_list=[1, 2, 4], total_test_loss=200.0)

    res_map = top_k_df.set_index("k")
    assert res_map.loc[1, "precision"] == 1.0
    assert res_map.loc[1, "exposure_captured"] == 50.0
    assert res_map.loc[2, "precision"] == 0.5
    assert res_map.loc[2, "exposure_captured"] == 50.0
    assert res_map.loc[4, "precision"] == 0.5
    assert res_map.loc[4, "exposure_captured"] == 200.0


def test_latency_as_of_and_calculation():
    """Verify latency calculation and ensure unflagged rings have latency = None."""
    test_orders = pd.DataFrame({
        "order_id": ["O1", "O2", "O3"],
        "customer_id": ["C1", "C1", "C2"],
        "event_time": pd.to_datetime(["2025-06-01 10:00", "2025-06-01 16:00", "2025-06-01 12:00"]),
        "amount": [100.0, 200.0, 300.0],
    })
    labels = pd.DataFrame({
        "order_id": ["O1", "O2", "O3"],
        "is_abuse": [True, True, True],
        "ring_id": ["R1", "R1", "R2"],
        "abuse_type": ["shared_device", "shared_device", "shared_address"],
        "loss_amount": [50.0, 100.0, 150.0],
    })
    scores = np.array([0.3, 0.9, 0.2])

    res = evaluate_rings(test_orders, labels, scores, threshold=0.5)
    pr = res.per_ring.set_index("ring_id")

    assert pr.loc["R1", "latency_hours"] == 6.0
    assert pd.isna(pr.loc["R2", "latency_hours"])


def test_financial_loss_formula_consistency():
    """Verify financial loss formula: sum(FN) + FP * (review_cost + false_positive_block_cost)."""
    y_true = pd.Series([1, 1, 0, 0])
    scores = np.array([0.9, 0.2, 0.8, 0.1])  # Alerts: O1 (TP), O3 (FP). FN: O2. TN: O4.
    loss_amount = pd.Series([100.0, 250.0, 0.0, 0.0])
    cost = CostModel(review_cost=2.0, false_positive_block_cost=10.0)

    eval_res = evaluate_predictions(y_true, scores, threshold=0.5, loss_amount=loss_amount, cost=cost)

    # Missed loss (FN = O2): 250.0
    # FP = 1 (O3) -> cost = 1 * (2.0 + 10.0) = 12.0
    # Expected loss = 250.0 + 12.0 = 262.0
    assert np.isclose(eval_res.metrics["expected_loss"], 262.0)


def test_feature_as_of_timestamp_leakage_safety():
    """Verify that modifying future transactions does not alter earlier feature representations."""
    orders_v1 = pd.DataFrame({
        "order_id": ["O1", "O2"],
        "customer_id": ["C1", "C2"],
        "event_time": pd.to_datetime(["2025-01-01 10:00", "2025-01-02 10:00"]),
        "amount": [100.0, 200.0],
        "device_id": ["D1", "D1"],
        "address_id": ["A1", "A2"],
        "ip_id": ["IP1", "IP2"],
        "payment_id": ["P1", "P2"],
        "merchant_category": ["electronics", "fashion"],
        "retry_count": [0, 0],
    })

    orders_v2 = orders_v1.copy()
    orders_v2.loc[1, "device_id"] = "D999"
    orders_v2.loc[1, "amount"] = 9999.0

    fs1_base = build_baseline_features(orders_v1)
    fs2_base = build_baseline_features(orders_v2)

    fs1_graph = build_graph_features(orders_v1)
    fs2_graph = build_graph_features(orders_v2)

    pd.testing.assert_series_equal(fs1_base.X.loc["O1"], fs2_base.X.loc["O1"])
    pd.testing.assert_series_equal(fs1_graph.X.loc["O1"], fs2_graph.X.loc["O1"])
