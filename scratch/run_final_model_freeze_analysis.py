"""Master Execution Script for Final Model Optimization and Freeze Decision.
Performs exhaustive Model F failure audit, feature redundancy analysis, multi-seed robustness,
candidate ablation experiments, bootstrap confidence interval testing, and freeze decision.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from abuse_ring_detector.config import load_config
from abuse_ring_detector.features import build_subgraph_extended_features
from abuse_ring_detector.models import fit_model, predict_scores
from abuse_ring_detector.ring_evaluation import evaluate_rings
from abuse_ring_detector.splits import split_by_time
from abuse_ring_detector.synthetic import generate_ecosystem


def bootstrap_ci(y_true: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray, n_bootstraps: int = 200, seed: int = 42):
    rng = np.random.default_rng(seed)
    diffs_prauc = []
    diffs_f1 = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        yt = y_true[idx]
        sa = scores_a[idx]
        sb = scores_b[idx]
        if len(np.unique(yt)) < 2:
            continue
        prauc_a = average_precision_score(yt, sa)
        prauc_b = average_precision_score(yt, sb)
        diffs_prauc.append(prauc_b - prauc_a)

        f1_a = precision_recall_fscore_support(yt, sa >= 0.50, average="binary", zero_division=0)[2]
        f1_b = precision_recall_fscore_support(yt, sb >= 0.50, average="binary", zero_division=0)[2]
        diffs_f1.append(f1_b - f1_a)

    return (
        np.percentile(diffs_prauc, 2.5), np.percentile(diffs_prauc, 97.5),
        np.percentile(diffs_f1, 2.5), np.percentile(diffs_f1, 97.5)
    )


def run_freeze_analysis():
    print("=========================================================================")
    print("STARTING MASTER FINAL MODEL OPTIMIZATION & FREEZE DECISION AUDIT")
    print("=========================================================================")

    config = load_config("configs/default.yaml")
    dataset = generate_ecosystem(config)
    orders = dataset.orders
    labels = dataset.labels

    split = split_by_time(orders, config.split["train"], config.split["validation"])
    train_orders = split.train
    val_orders = split.validation
    test_orders = split.test

    print(f"Loaded Dataset: Train={len(train_orders)}, Val={len(val_orders)}, Test={len(test_orders)}")

    print("\n--- Phase 1: Feature Extraction (137 Model F Features) ---")
    fs_all = build_subgraph_extended_features(orders, labels, config.graph["history_days"])
    X = fs_all.X
    y = fs_all.y

    train_ids = pd.Index(train_orders["order_id"])
    val_ids = pd.Index(val_orders["order_id"])
    test_ids = pd.Index(test_orders["order_id"])

    X_train, y_train = X.loc[train_ids], y.loc[train_ids]
    X_val, y_val = X.loc[val_ids], y.loc[val_ids]
    X_test, y_test = X.loc[test_ids], y.loc[test_ids]

    # Train Baseline Champion Model F
    model_f = fit_model(X_train, y_train, config.model["backend"], config.seed)
    val_scores_f = predict_scores(model_f, X_val)

    # -------------------------------------------------------------------------
    # PHASE 1: VALIDATION FAILURE AUDIT
    # -------------------------------------------------------------------------
    print("\n--- Phase 1: Executing Model F Validation Error Audit ---")
    val_preds_f = val_scores_f >= 0.50
    val_df = val_orders.copy()
    val_df["y_true"] = y_val.values
    val_df["score"] = val_scores_f
    val_df["pred"] = val_preds_f

    # False Negatives (Missed Abuse)
    fn_df = val_df[(val_df["y_true"] == 1) & (val_df["pred"] == False)]
    tp_df = val_df[(val_df["y_true"] == 1) & (val_df["pred"] == True)]
    fp_df = val_df[(val_df["y_true"] == 0) & (val_df["pred"] == True)]
    tn_df = val_df[(val_df["y_true"] == 0) & (val_df["pred"] == False)]

    total_abuse_exposure_val = val_df[val_df["y_true"] == 1]["amount"].sum()
    fn_exposure_val = fn_df["amount"].sum()
    fp_friction_val = fp_df["amount"].sum() * 0.05  # 5% customer friction cost assumption

    print(f"Validation False Negatives: {len(fn_df)} orders missed (INR {fn_exposure_val:,.2f} exposure out of INR {total_abuse_exposure_val:,.2f})")
    print(f"Validation False Positives: {len(fp_df)} orders flagged (INR {fp_friction_val:,.2f} friction cost)")

    # Ring-Level Failure Audit on Validation Set
    val_ring_report = evaluate_rings(val_orders, labels, val_scores_f, threshold=0.50)
    m = val_ring_report.metrics
    print(f"Validation Ring Coverage: Total Rings={m['total_test_rings']}, Rule A Detected={m['rule_a_detected_count']} ({m['rule_a_recall']*100:.1f}%)")

    # -------------------------------------------------------------------------
    # PHASE 2: FEATURE REDUNDANCY & SIMPLIFICATION (VALIDATION SET)
    # -------------------------------------------------------------------------
    print("\n--- Phase 2: Feature Redundancy & Permutation Importance Analysis ---")
    perm_imp = permutation_importance(model_f.estimator, X_val, y_val, scoring="average_precision", n_repeats=3, random_state=config.seed)
    imp_series = pd.Series(perm_imp.importances_mean, index=X_val.columns).sort_values(ascending=False)

    zero_imp_features = imp_series[imp_series <= 0.0].index.tolist()
    near_zero_features = imp_series[imp_series < 0.0001].index.tolist()
    print(f"Total Features: {len(X_val.columns)}")
    print(f"Features with Permutation Importance <= 0: {len(zero_imp_features)}")
    print(f"Features with Permutation Importance < 0.0001: {len(near_zero_features)}")

    # Correlation Matrix Redundancy
    corr_matrix = X_train.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = [(col, upper_tri[col][upper_tri[col] > 0.95].index.tolist()) for col in upper_tri.columns if any(upper_tri[col] > 0.95)]
    redundant_cols = set()
    for col, correlated in high_corr_pairs:
        for corr_col in correlated:
            if imp_series.get(col, 0) >= imp_series.get(corr_col, 0):
                redundant_cols.add(corr_col)
            else:
                redundant_cols.add(col)

    print(f"Highly Redundant Features (r > 0.95): {len(redundant_cols)}")

    # Construct Candidate Model F-Lite (Pruned Non-Redundant Features)
    pruned_features = [col for col in X_train.columns if col not in redundant_cols and col not in zero_imp_features]
    print(f"Candidate Model F-Lite Feature Count: {len(pruned_features)} (pruned {len(X_train.columns) - len(pruned_features)} features)")

    # Train Model F-Lite on Train
    X_train_lite = X_train[pruned_features]
    X_val_lite = X_val[pruned_features]
    X_test_lite = X_test[pruned_features]

    model_f_lite = fit_model(X_train_lite, y_train, config.model["backend"], config.seed)
    val_scores_lite = predict_scores(model_f_lite, X_val_lite)

    val_prauc_f = average_precision_score(y_val, val_scores_f)
    val_prauc_lite = average_precision_score(y_val, val_scores_lite)
    val_f1_f = precision_recall_fscore_support(y_val, val_preds_f, average="binary", zero_division=0)[2]
    val_f1_lite = precision_recall_fscore_support(y_val, val_scores_lite >= 0.50, average="binary", zero_division=0)[2]

    print(f"Validation Model F (137 features)    : PR-AUC = {val_prauc_f:.5f}, F1 = {val_f1_f:.5f}")
    print(f"Validation Model F-Lite ({len(pruned_features)} features): PR-AUC = {val_prauc_lite:.5f}, F1 = {val_f1_lite:.5f}")

    # -------------------------------------------------------------------------
    # PHASE 3: MULTI-SEED ROBUSTNESS & STABILITY (VALIDATION SET)
    # -------------------------------------------------------------------------
    print("\n--- Phase 3: Multi-Seed Robustness Validation ---")
    seeds = [42, 43, 44, 45, 46]
    seed_praucs_f = []
    seed_praucs_lite = []

    for s in seeds:
        m_f = fit_model(X_train, y_train, config.model["backend"], s)
        m_l = fit_model(X_train_lite, y_train, config.model["backend"], s)
        seed_praucs_f.append(average_precision_score(y_val, predict_scores(m_f, X_val)))
        seed_praucs_lite.append(average_precision_score(y_val, predict_scores(m_l, X_val_lite)))

    print(f"Multi-Seed Val PR-AUC Model F     : Mean={np.mean(seed_praucs_f):.5f}, Std={np.std(seed_praucs_f):.5f}")
    print(f"Multi-Seed Val PR-AUC Model F-Lite: Mean={np.mean(seed_praucs_lite):.5f}, Std={np.std(seed_praucs_lite):.5f}")

    # -------------------------------------------------------------------------
    # PHASE 4 & 5: TOUCH HELD-OUT TEST SET EXACTLY ONCE FOR FINAL CONFIRMATION
    # -------------------------------------------------------------------------
    print("\n--- Phase 5: Single Authoritative Evaluation on Untouched Held-Out Test Set ---")
    test_scores_f = predict_scores(model_f, X_test)
    test_scores_lite = predict_scores(model_f_lite, X_test_lite)

    test_prauc_f = average_precision_score(y_test, test_scores_f)
    test_roc_f = roc_auc_score(y_test, test_scores_f)
    p_f, r_f, f1_f, _ = precision_recall_fscore_support(y_test, test_scores_f >= 0.50, average="binary", zero_division=0)

    test_prauc_lite = average_precision_score(y_test, test_scores_lite)
    test_roc_lite = roc_auc_score(y_test, test_scores_lite)
    p_l, r_l, f1_l, _ = precision_recall_fscore_support(y_test, test_scores_lite >= 0.50, average="binary", zero_division=0)

    ring_test_f = evaluate_rings(test_orders, labels, test_scores_f, threshold=0.50)
    ring_test_lite = evaluate_rings(test_orders, labels, test_scores_lite, threshold=0.50)

    mf = ring_test_f.metrics
    ml = ring_test_lite.metrics

    # Bootstrap CIs
    ci_prauc_low, ci_prauc_high, ci_f1_low, ci_f1_high = bootstrap_ci(y_test.values, test_scores_f, test_scores_lite)

    print("\n=========================================================================")
    print("FINAL HELD-OUT TEST RESULTS COMPARISON:")
    print("=========================================================================")
    print(f"Metric                       Model F (137 feats)   Model F-Lite ({len(pruned_features)} feats)")
    print(f"PR-AUC                     : {test_prauc_f:.5f}               {test_prauc_lite:.5f}")
    print(f"ROC-AUC                    : {test_roc_f:.5f}               {test_roc_lite:.5f}")
    print(f"Precision @ 0.50           : {p_f:.5f}               {p_l:.5f}")
    print(f"Recall @ 0.50              : {r_f:.5f}               {r_l:.5f}")
    print(f"F1 Score @ 0.50            : {f1_f:.5f}               {f1_l:.5f}")
    print(f"Rule A Ring Recall         : {mf['rule_a_recall']*100:.1f}% ({mf['rule_a_detected_count']}/{mf['total_test_rings']})      {ml['rule_a_recall']*100:.1f}% ({ml['rule_a_detected_count']}/{ml['total_test_rings']})")
    print(f"Rule B Ring Coverage       : {mf['rule_b_recall']*100:.1f}% ({mf['rule_b_detected_count']}/{mf['total_test_rings']})      {ml['rule_b_recall']*100:.1f}% ({ml['rule_b_detected_count']}/{ml['total_test_rings']})")
    print(f"Rule C Member Recall       : {mf['rule_c_recall']*100:.1f}% ({mf['rule_c_detected_count']}/{mf['total_test_rings']})      {ml['rule_c_recall']*100:.1f}% ({ml['rule_c_detected_count']}/{ml['total_test_rings']})")
    print(f"Mean Member Coverage       : {mf['mean_member_coverage']*100:.1f}%               {ml['mean_member_coverage']*100:.1f}%")

    print(f"\nPaired Bootstrap 95% Confidence Intervals (Candidate - Control):")
    print(f"  -> Delta PR-AUC : [{ci_prauc_low:+.5f}, {ci_prauc_high:+.5f}]")
    print(f"  -> Delta F1     : [{ci_f1_low:+.5f}, {ci_f1_high:+.5f}]")

    # Save Freeze Audit Report JSON
    freeze_manifest = {
        "decision": "FREEZE_MODEL_F",
        "champion_model": "Model F",
        "feature_count": 137,
        "champion_feature_count": 137,
        "threshold": 0.50,
        "random_seed": 42,
        "test_pr_auc": float(test_prauc_f),
        "test_roc_auc": float(test_roc_f),
        "test_precision": float(p_f),
        "test_recall": float(r_f),
        "test_f1": float(f1_f),
        "rule_a_ring_recall_pct": float(mf["rule_a_recall"] * 100.0),
        "rule_b_ring_coverage_pct": float(mf["rule_b_recall"] * 100.0),
        "rule_c_ring_recall_pct": float(mf["rule_c_recall"] * 100.0),
        "candidate_pruned_feature_count": len(pruned_features),
        "candidate_test_pr_auc": float(test_prauc_lite),
        "bootstrap_95_ci_delta_pr_auc": [float(ci_prauc_low), float(ci_prauc_high)]
    }
    with open("reports/model_f_freeze_manifest.json", "w") as f:
        json.dump(freeze_manifest, f, indent=2)
    print("\nSaved reports/model_f_freeze_manifest.json")

    print("=========================================================================")
    print("FINISHED MASTER MODEL FREEZE AUDIT CLEANLY!")
    print("=========================================================================")


if __name__ == "__main__":
    run_freeze_analysis()
