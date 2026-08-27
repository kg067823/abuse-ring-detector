"""Unit tests for ring-level evaluation, member coverage, and Top-K risk prioritisation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from abuse_ring_detector.config import Config, RingConfig
from abuse_ring_detector.features import build_baseline_features, build_graph_features
from abuse_ring_detector.ring_evaluation import (
    compare_baseline_vs_graph,
    evaluate_rings,
    evaluate_top_k,
)
from abuse_ring_detector.splits import split_by_time
from abuse_ring_detector.synthetic import generate_ecosystem


def _create_mock_test_data():
    """Create a minimal deterministic test fixture for ring evaluation."""
    times = pd.date_range("2025-06-01", periods=10, freq="1D")
    orders = pd.DataFrame({
        "order_id": [f"O{i:03d}" for i in range(10)],
        "customer_id": ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"],
        "event_time": times,
        "amount": [1000.0] * 10,
    })
    labels = pd.DataFrame({
        "order_id": [f"O{i:03d}" for i in range(10)],
        "is_abuse": [True, True, True, True, True, False, False, False, False, False],
        "ring_id": ["R001", "R001", "R001", "R002", "R002", pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
        "abuse_type": ["mixed", "mixed", "mixed", "shared_device", "shared_device", pd.NA, pd.NA, pd.NA, pd.NA, pd.NA],
        "loss_amount": [500.0, 500.0, 500.0, 300.0, 300.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "reason_codes": ["coordination"] * 5 + [""] * 5,
    })
    return orders, labels


def test_ring_coverage_calculation():
    """1. Ring coverage calculation is correct: flagged_members / active_members."""
    orders, labels = _create_mock_test_data()
    # R001 has 3 members: C1, C2, C3
    # R002 has 2 members: C4, C5
    # Flag C1 and C2 (2/3 for R001), flag none for R002
    scores = np.array([0.9, 0.8, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05])
    threshold = 0.5

    result = evaluate_rings(orders, labels, scores, threshold=threshold)
    per_ring = result.per_ring.set_index("ring_id")

    # R001 coverage should be 2/3
    assert pytest.approx(per_ring.loc["R001", "coverage"], abs=1e-4) == 2.0 / 3.0
    assert per_ring.loc["R001", "flagged_members"] == 2
    assert per_ring.loc["R001", "test_members"] == 3

    # R002 coverage should be 0/2
    assert per_ring.loc["R002", "coverage"] == 0.0
    assert per_ring.loc["R002", "flagged_members"] == 0
    assert per_ring.loc["R002", "test_members"] == 2


def test_zero_flagged_members_not_detected():
    """2. A ring with zero flagged members is not detected under any rule."""
    orders, labels = _create_mock_test_data()
    # No scores above threshold
    scores = np.zeros(len(orders))
    result = evaluate_rings(orders, labels, scores, threshold=0.5)
    per_ring = result.per_ring.set_index("ring_id")

    for r_id in ["R001", "R002"]:
        assert per_ring.loc[r_id, "detected_rule_a"] is False or per_ring.loc[r_id, "detected_rule_a"] == 0
        assert per_ring.loc[r_id, "detected_rule_b"] is False or per_ring.loc[r_id, "detected_rule_b"] == 0
        assert per_ring.loc[r_id, "detected_rule_c"] is False or per_ring.loc[r_id, "detected_rule_c"] == 0
        assert per_ring.loc[r_id, "coverage"] == 0.0
        assert per_ring.loc[r_id, "exposure_captured"] == 0.0

    assert result.metrics["rule_a_recall"] == 0.0
    assert result.metrics["rule_b_recall"] == 0.0
    assert result.metrics["rule_c_recall"] == 0.0
    assert result.metrics["total_exposure_captured"] == 0.0


def test_sufficient_flagged_members_detected():
    """3. A ring with sufficient flagged members is detected under Rules A, B, and C."""
    orders, labels = _create_mock_test_data()
    # Flag all members of R001 (C1, C2, C3) and 1 member of R002 (C4, which is 50% of R002)
    scores = np.array([0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    result = evaluate_rings(orders, labels, scores, threshold=0.5)
    per_ring = result.per_ring.set_index("ring_id")

    # R001 has 100% coverage -> detected under A, B, C
    assert per_ring.loc["R001", "detected_rule_a"] is True or per_ring.loc["R001", "detected_rule_a"] == 1
    assert per_ring.loc["R001", "detected_rule_b"] is True or per_ring.loc["R001", "detected_rule_b"] == 1
    assert per_ring.loc["R001", "detected_rule_c"] is True or per_ring.loc["R001", "detected_rule_c"] == 1
    assert per_ring.loc["R001", "coverage"] == 1.0

    # R002 has 50% coverage (1/2) -> detected under A (>=1), B (>=20%), C (>=50%)
    assert per_ring.loc["R002", "detected_rule_a"] is True or per_ring.loc["R002", "detected_rule_a"] == 1
    assert per_ring.loc["R002", "detected_rule_b"] is True or per_ring.loc["R002", "detected_rule_b"] == 1
    assert per_ring.loc["R002", "detected_rule_c"] is True or per_ring.loc["R002", "detected_rule_c"] == 1
    assert per_ring.loc["R002", "coverage"] == 0.5

    assert result.metrics["rule_a_recall"] == 1.0
    assert result.metrics["rule_b_recall"] == 1.0
    assert result.metrics["rule_c_recall"] == 1.0


def test_top_k_respects_score_ordering():
    """4. Top-K analysis strictly respects descending score ordering."""
    orders, labels = _create_mock_test_data()
    # Indices 0-4 are abuse (scores 0.1 to 0.5), indices 5-9 are non-abuse (scores 0.6 to 0.99)
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
    result = evaluate_rings(orders, labels, scores, threshold=0.5, top_k_list=[1, 2, 5, 6])
    top_k = result.top_k.set_index("k")

    # Top-1 order is O009 (score 0.99, non-abuse) -> Precision = 0, TP = 0, FP = 1
    assert top_k.loc[1, "precision"] == 0.0
    assert top_k.loc[1, "tp_count"] == 0
    assert top_k.loc[1, "fp_count"] == 1

    # Top-2 orders are O009, O008 (both non-abuse) -> Precision = 0
    assert top_k.loc[2, "precision"] == 0.0
    assert top_k.loc[2, "tp_count"] == 0

    # Top-5 orders are O009, O008, O007, O006, O005 (all 5 non-abuse) -> TP = 0, FP = 5
    assert top_k.loc[5, "tp_count"] == 0
    assert top_k.loc[5, "fp_count"] == 5

    # Top-6 orders include O004 (abuse R002 with score 0.5) -> TP = 1, FP = 5
    assert top_k.loc[6, "tp_count"] == 1
    assert top_k.loc[6, "rings_touched"] == 1


def test_evaluation_only_uses_held_out_test_examples():
    """5. Evaluation strictly operates on the test split orders provided."""
    config = Config(customers=200, orders=500, date_range_days=60,
                    rings=RingConfig(count=100, min_size=3, max_size=6))
    dataset = generate_ecosystem(config)
    split = split_by_time(dataset.orders, 0.7, 0.15)

    test_orders = split.test
    test_scores = np.random.default_rng(42).uniform(0, 1, len(test_orders))

    result = evaluate_rings(test_orders, dataset.labels, test_scores, threshold=0.5)

    # All evaluated order IDs must be strictly subset of test_orders
    evaluated_order_count = result.metrics["total_test_abuse_orders"]
    test_abuse_count = dataset.labels.set_index("order_id").loc[test_orders.order_id, "is_abuse"].sum()
    assert evaluated_order_count == test_abuse_count


def test_ground_truth_cannot_influence_predictions():
    """6. Ground truth columns (ring_id, is_abuse) are absent from features."""
    config = Config(customers=200, orders=500, date_range_days=60,
                    rings=RingConfig(count=100, min_size=3, max_size=6))
    dataset = generate_ecosystem(config)
    base_fs = build_baseline_features(dataset.orders, dataset.labels)
    graph_fs = build_graph_features(dataset.orders, dataset.labels)

    forbidden_cols = {"ring_id", "is_abuse", "abuse_type", "loss_amount", "reason_codes", "is_abusive"}
    for col in forbidden_cols:
        assert col not in base_fs.X.columns, f"{col} leaked into baseline features"
        assert col not in graph_fs.X.columns, f"{col} leaked into graph features"


def test_empty_and_no_positive_cases_do_not_crash():
    """7. Empty or no-positive test cases run cleanly without errors."""
    orders, labels = _create_mock_test_data()
    # Modify labels to have zero abuse
    labels_no_abuse = labels.copy()
    labels_no_abuse["is_abuse"] = False
    labels_no_abuse["ring_id"] = pd.NA
    labels_no_abuse["loss_amount"] = 0.0

    scores = np.random.uniform(0, 1, len(orders))
    result = evaluate_rings(orders, labels_no_abuse, scores, threshold=0.5)

    assert result.metrics["total_test_rings"] == 0
    assert result.metrics["total_test_exposure"] == 0.0
    assert result.metrics["rule_a_recall"] == 0.0
    assert result.metrics["mean_member_coverage"] == 0.0
    assert len(result.per_ring) == 0
    assert len(result.by_ring_type) == 0


def test_detection_latency_calculation():
    """8. Latency accurately measures first flagged timestamp minus first abuse timestamp."""
    orders, labels = _create_mock_test_data()
    # R001 orders are O000 (day 0), O001 (day 1), O002 (day 2)
    # Flag only O002 (day 2, i.e., 48 hours after first abuse event O000)
    scores = np.array([0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    result = evaluate_rings(orders, labels, scores, threshold=0.5)
    per_ring = result.per_ring.set_index("ring_id")

    # R001 latency should be 48.0 hours
    assert pytest.approx(per_ring.loc["R001", "latency_hours"], abs=1e-4) == 48.0

    # R002 was not flagged -> latency is None / NaN
    assert per_ring.loc["R002", "latency_hours"] is None or np.isnan(per_ring.loc["R002", "latency_hours"])


def test_rule_threshold_distinctions():
    """9. Verify distinct outcomes across Rule A, Rule B, and Rule C."""
    # Create a 5-member ring: C1, C2, C3, C4, C5
    orders = pd.DataFrame({
        "order_id": [f"O{i}" for i in range(5)],
        "customer_id": [f"C{i}" for i in range(5)],
        "event_time": pd.date_range("2025-06-01", periods=5, freq="1D"),
        "amount": [100.0] * 5,
    })
    labels = pd.DataFrame({
        "order_id": [f"O{i}" for i in range(5)],
        "is_abuse": [True] * 5,
        "ring_id": ["R1"] * 5,
        "abuse_type": ["mixed"] * 5,
        "loss_amount": [50.0] * 5,
    })

    # Flag 1 member (1/5 = 20% coverage) -> Detected under Rule A (>=1) and Rule B (>=20%), but NOT Rule C (>=50%)
    scores_1 = np.array([0.9, 0.1, 0.1, 0.1, 0.1])
    res_1 = evaluate_rings(orders, labels, scores_1, threshold=0.5)
    r1 = res_1.per_ring.iloc[0]
    assert r1.detected_rule_a is True or r1.detected_rule_a == 1
    assert r1.detected_rule_b is True or r1.detected_rule_b == 1
    assert r1.detected_rule_c is False or r1.detected_rule_c == 0

    # Create a 10-member ring: C0..C9. Flag 1 member (1/10 = 10% coverage) -> Rule A YES, Rule B NO, Rule C NO
    orders_10 = pd.DataFrame({
        "order_id": [f"O{i}" for i in range(10)],
        "customer_id": [f"C{i}" for i in range(10)],
        "event_time": pd.date_range("2025-06-01", periods=10, freq="1D"),
        "amount": [100.0] * 10,
    })
    labels_10 = pd.DataFrame({
        "order_id": [f"O{i}" for i in range(10)],
        "is_abuse": [True] * 10,
        "ring_id": ["R1"] * 10,
        "abuse_type": ["mixed"] * 10,
        "loss_amount": [50.0] * 10,
    })
    scores_10 = np.array([0.9] + [0.1] * 9)
    res_10 = evaluate_rings(orders_10, labels_10, scores_10, threshold=0.5)
    r10 = res_10.per_ring.iloc[0]
    assert r10.detected_rule_a is True or r10.detected_rule_a == 1
    assert r10.detected_rule_b is False or r10.detected_rule_b == 0
    assert r10.detected_rule_c is False or r10.detected_rule_c == 0


def test_compare_baseline_vs_graph_structure():
    """10. Verify comparison table contains all required metrics."""
    orders, labels = _create_mock_test_data()
    scores_b = np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    scores_g = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    res_b = evaluate_rings(orders, labels, scores_b, threshold=0.5)
    res_g = evaluate_rings(orders, labels, scores_g, threshold=0.5)

    comp = compare_baseline_vs_graph(res_b, res_g)
    assert set(comp.columns) == {"Metric", "Baseline", "Graph-enhanced"}
    metric_names = set(comp["Metric"])
    expected = {
        "Event Precision", "Event Recall", "PR-AUC",
        "Any-member Ring Recall (Rule A)", "20% Coverage Ring Recall (Rule B)",
        "50% Coverage Ring Recall (Rule C)", "Mean Member Coverage",
        "Top-5 Exposure Captured", "Top-10 Exposure Captured",
        "Top-20 Exposure Captured", "Top-50 Exposure Captured",
    }
    assert expected.issubset(metric_names)
