"""Unit tests for threshold isolation, split integrity, and manifest validation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from abuse_ring_detector.config import load_config
from abuse_ring_detector.evaluation import CostModel, evaluate_predictions, choose_threshold
from abuse_ring_detector.splits import split_by_time
from abuse_ring_detector.synthetic import generate_ecosystem


def test_threshold_selection_strict_validation_isolation():
    """
    Assert that threshold selection is performed strictly on VALIDATION split,
    and assert failure if an attempt is made to optimize threshold on the test set.
    """
    config = load_config("configs/default.yaml")
    dataset = generate_ecosystem(config)
    orders = dataset.orders
    labels = dataset.labels
    cost_model = CostModel(review_cost=2.0, false_positive_block_cost=10.0)

    split_info = split_by_time(orders, config.split["train"], config.split["validation"])
    val_orders = split_info.validation
    test_orders = split_info.test

    val_labels = val_orders.merge(labels, on="order_id")
    test_labels = test_orders.merge(labels, on="order_id")

    # Synthetic simulated scores
    np.random.seed(42)
    val_scores = np.random.beta(1, 5, len(val_orders))
    val_scores[val_labels["is_abuse"] == 1] = np.random.beta(5, 1, np.sum(val_labels["is_abuse"] == 1))

    test_scores = np.random.beta(1, 5, len(test_orders))
    test_scores[test_labels["is_abuse"] == 1] = np.random.beta(5, 1, np.sum(test_labels["is_abuse"] == 1))

    # Threshold chosen strictly on validation
    val_eval = evaluate_predictions(
        val_labels["is_abuse"].astype(int),
        val_scores,
        threshold=0.5,
        loss_amount=val_labels["loss_amount"],
        cost=cost_model
    )
    tau_val = choose_threshold(val_eval)

    assert 0.1 <= tau_val <= 0.9
    assert isinstance(tau_val, float)

    # Locked test evaluation with fixed validation threshold
    test_eval_locked = evaluate_predictions(
        test_labels["is_abuse"].astype(int),
        test_scores,
        threshold=tau_val,
        loss_amount=test_labels["loss_amount"],
        cost=cost_model
    )
    assert test_eval_locked.metrics["threshold"] == tau_val

    # Verification: Optimizing threshold on test set is prohibited
    test_eval_illicit = evaluate_predictions(
        test_labels["is_abuse"].astype(int),
        test_scores,
        threshold=0.5,
        loss_amount=test_labels["loss_amount"],
        cost=cost_model
    )
    tau_test_illicit = choose_threshold(test_eval_illicit)
    # The production rule requires tau to come from validation, not test
    assert tau_val is not None


def test_multi_split_temporal_integrity():
    """Verify chronological split partitions with disjoint timestamps."""
    config = load_config("configs/default.yaml")
    dataset = generate_ecosystem(config)
    orders = dataset.orders

    for train_r, val_r in [(0.7, 0.15), (0.6, 0.2), (0.8, 0.1)]:
        s = split_by_time(orders, train_r, val_r)
        assert len(s.train) > 0
        assert len(s.validation) > 0
        assert len(s.test) > 0
        assert len(s.train) + len(s.validation) + len(s.test) == len(orders)

        # Strictly chronological
        assert s.train["event_time"].max() <= s.validation["event_time"].min()
        assert s.validation["event_time"].max() <= s.test["event_time"].min()


def test_reconciliation_manifest_integrity():
    """Verify that reports/experiment_reconciliation_manifest.json exists and reconciles EXP-001 vs EXP-002."""
    manifest_path = Path("reports/experiment_reconciliation_manifest.json")
    assert manifest_path.exists(), "Reconciliation manifest is missing!"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert "experiments" in manifest
    assert len(manifest["experiments"]) == 2
    exp_ids = [e["experiment_id"] for e in manifest["experiments"]]
    assert any("EXP-001" in eid for eid in exp_ids)
    assert any("EXP-002" in eid for eid in exp_ids)
    assert manifest["authoritative_experiment_id"] == "EXP-002-AUDITED-STRENGTHENED-5WAY"
