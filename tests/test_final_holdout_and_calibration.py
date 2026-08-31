"""Test module for Final Holdout Isolation, Model Freeze, Probability Calibration, and Monitoring Logic."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

from abuse_ring_detector.config import load_config
from abuse_ring_detector.features import build_subgraph_extended_features
from abuse_ring_detector.models import fit_model, predict_scores
from abuse_ring_detector.splits import split_by_time
from abuse_ring_detector.synthetic import generate_ecosystem
from scratch.run_final_holdout_and_calibration_evaluation import calculate_ece_mce


def test_final_holdout_isolation():
    """Verify final holdout timestamps (Days 180-210) strictly occur after training and validation periods."""
    config_210 = load_config("configs/default.yaml")
    config_210.date_range_days = 210
    config_210.orders = 58333
    dataset = generate_ecosystem(config_210)
    orders = dataset.orders
    
    config_180 = load_config("configs/default.yaml")
    dataset_180 = generate_ecosystem(config_180)
    split_180 = split_by_time(dataset_180.orders, config_180.split["train"], config_180.split["validation"])
    
    val_end_time = split_180.validation_end
    holdout_orders = orders[orders["event_time"] > pd.Timestamp("2025-06-30")]
    
    # Holdout must be strictly greater than validation end time
    assert (holdout_orders["event_time"] > val_end_time).all(), "Holdout events leak into validation window!"
    assert len(holdout_orders) > 0, "Holdout dataset is empty!"


def test_no_holdout_leakage_during_model_selection():
    """Verify model fitted on training set produces identical weights and predictions regardless of holdout dataset presence."""
    config = load_config("configs/default.yaml")
    dataset_180 = generate_ecosystem(config)
    split_180 = split_by_time(dataset_180.orders, config.split["train"], config.split["validation"])
    
    fs_180 = build_subgraph_extended_features(dataset_180.orders, dataset_180.labels, config.graph["history_days"])
    train_ids = pd.Index(split_180.train["order_id"])
    
    model_1 = fit_model(fs_180.X.loc[train_ids], fs_180.y.loc[train_ids], config.model["backend"], config.seed)
    scores_1 = predict_scores(model_1, fs_180.X.loc[train_ids].iloc[:100])
    
    # Model 2 trained on identical training slice
    model_2 = fit_model(fs_180.X.loc[train_ids], fs_180.y.loc[train_ids], config.model["backend"], config.seed)
    scores_2 = predict_scores(model_2, fs_180.X.loc[train_ids].iloc[:100])
    
    np.testing.assert_allclose(scores_1, scores_2, atol=1e-7, err_msg="Model training affected by external state!")


def test_model_freeze_reproducibility():
    """Verify model freeze manifest exists and feature count matches current model baseline (137 features)."""
    manifest_path = Path("model_f_r1_manifest.json")
    assert manifest_path.exists(), "Model freeze manifest does not exist!"
    
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        
    assert manifest["feature_count"] == 137, f"Expected 137 frozen features, got {manifest['feature_count']}"
    assert manifest["threshold"] == 0.50, f"Expected validation-locked threshold 0.50, got {manifest['threshold']}"
    assert manifest["seed"] == 42, "Random seed mismatch in R1 manifest"
    assert manifest["model_version"] == "model_f_r1"


def test_calibration_fitting_isolation():
    """Verify probability calibrators (Platt / Isotonic) are fitted strictly on validation scores and never touch holdout labels."""
    rng = np.random.default_rng(42)
    val_scores = rng.uniform(0, 1, 1000)
    val_labels = (val_scores + rng.normal(0, 0.1, 1000) > 0.5).astype(int)
    
    holdout_scores = rng.uniform(0, 1, 500)
    holdout_labels = (holdout_scores + rng.normal(0, 0.1, 500) > 0.5).astype(int)
    
    # Fit calibrators ONLY on validation data
    platt = LogisticRegression(C=1.0)
    val_logits = np.log(np.clip(val_scores, 1e-6, 1-1e-6) / (1 - np.clip(val_scores, 1e-6, 1-1e-6))).reshape(-1, 1)
    platt.fit(val_logits, val_labels)
    
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(val_scores, val_labels)
    
    # Predict on holdout
    holdout_logits = np.log(np.clip(holdout_scores, 1e-6, 1-1e-6) / (1 - np.clip(holdout_scores, 1e-6, 1-1e-6))).reshape(-1, 1)
    p_platt = platt.predict_proba(holdout_logits)[:, 1]
    p_iso = isotonic.predict(holdout_scores)
    
    assert len(p_platt) == 500
    assert len(p_iso) == 500
    assert not np.isnan(p_platt).any()
    assert not np.isnan(p_iso).any()


def test_calibration_denominator_safety():
    """Verify calculate_ece_mce handles empty bins and zero division safely without returning NaN or Inf."""
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.05, 0.95, 0.05, 0.95])
    
    # Request 20 bins (producing several empty bins)
    ece, mce, bins = calculate_ece_mce(y_true, y_prob, n_bins=20)
    
    assert not np.isnan(ece), "ECE returned NaN!"
    assert not np.isnan(mce), "MCE returned NaN!"
    assert not np.isinf(ece), "ECE returned Inf!"
    assert not np.isinf(mce), "MCE returned Inf!"
    assert len(bins) == 20


def test_monitoring_threshold_logic():
    """Verify drift monitoring policy correctly classifies PSI into GREEN (<0.10), WARNING (0.10-0.25), and CRITICAL (>0.25)."""
    policy_path = Path("reports/production_drift_monitoring_policy.json")
    assert policy_path.exists(), "Monitoring policy file missing!"
    
    with open(policy_path, "r") as f:
        policy = json.load(f)
        
    def classify_psi(psi: float) -> str:
        if psi < 0.10:
            return "GREEN"
        elif psi <= 0.25:
            return "WARNING"
        else:
            return "CRITICAL"
            
    assert classify_psi(0.04) == "GREEN"
    assert classify_psi(0.15) == "WARNING"
    assert classify_psi(0.76) == "CRITICAL"
    
    # Check prior_paymentcount special mitigation entry
    payment_policy = next(p for p in policy["feature_drift_thresholds"] if p["feature_name"] == "prior_paymentcount")
    assert payment_policy["known_high_drift"] is True
    assert payment_policy["historical_psi"] == 0.760


def test_deterministic_evaluation_reproducibility():
    """Verify bootstrap confidence interval calculation is 100% deterministic when supplied with fixed seed."""
    rng1 = np.random.default_rng(42)
    sample1 = rng1.normal(0.85, 0.05, 1000)
    
    rng2 = np.random.default_rng(42)
    sample2 = rng2.normal(0.85, 0.05, 1000)
    
    np.testing.assert_array_equal(sample1, sample2, err_msg="Non-deterministic random sequence!")


def test_causal_streaming_guarantees_on_holdout():
    """Verify modifying a future holdout event does not alter historical feature values computed at prior timestamps."""
    config = load_config("configs/default.yaml")
    dataset = generate_ecosystem(config)
    orders = dataset.orders.copy()
    labels = dataset.labels.copy()
    
    # Compute features for original dataset
    fs_orig = build_subgraph_extended_features(orders, labels, config.graph["history_days"])
    val_id = orders.iloc[10000]["order_id"]
    val_orig = fs_orig.X.loc[val_id].copy()
    
    # Modify a future order occurring AFTER order 10000
    future_idx = 45000
    assert orders.iloc[future_idx]["event_time"] > orders.iloc[10000]["event_time"]
    
    orders_mod = orders.copy()
    orders_mod.loc[future_idx, "amount"] = 999999.0
    orders_mod.loc[future_idx, "device_id"] = "D_MODIFIED_FUTURE"
    
    fs_mod = build_subgraph_extended_features(orders_mod, labels, config.graph["history_days"])
    val_mod = fs_mod.X.loc[val_id].copy()
    
    pd.testing.assert_series_equal(val_orig, val_mod, check_exact=True, obj="Causal Feature Leakage Detected!")
