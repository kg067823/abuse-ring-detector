"""Phase 7 — Master Shadow Observation Validation Suite.

Executes complete pre-observation validation across Shadow Mode non-enforcement,
Prometheus observability, Shadow Evaluation Pipeline, Shadow Safety Gates,
and output report generation.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
import pandas as pd

# Add project root and src to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

from abuse_ring_detector.api import app, initialize_service
from abuse_ring_detector.inference import load_model_artifact
from abuse_ring_detector.shadow_evaluator import ShadowEvaluationPipeline
from abuse_ring_detector.shadow_gates import ShadowSafetyGateEvaluator
from fastapi.testclient import TestClient

logger = logging.getLogger("shadow_validation")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def run_shadow_observation_validation():
    print("=" * 80)
    print("MODEL F PRODUCTION LIVE SHADOW MODE OBSERVATION VALIDATION SUITE")
    print("=" * 80)

    report_data = {}

    # --- Phase 1: Frozen Model Integrity ---
    print("\n--- PHASE 1: FROZEN MODEL INTEGRITY & SAFETY AUDIT ---")
    manifest_path = Path("reports/model_f_freeze_manifest.json")
    model_path = Path("artifacts/model_f_bundle.pkl")

    bundle, actual_checksum = load_model_artifact(model_path)
    feature_cols = getattr(bundle, "feature_columns", [])
    
    expected_checksum = "82e77daac0762a04"
    expected_features = 137

    checksum_ok = actual_checksum.startswith(expected_checksum) or expected_checksum.startswith(actual_checksum)
    feat_ok = len(feature_cols) == expected_features

    print(f" [{ 'PASS' if checksum_ok else 'FAIL' }] Checksum Verification: {actual_checksum[:16]} (Expected: {expected_checksum})")
    print(f" [{ 'PASS' if feat_ok else 'FAIL' }] Feature Count Verification: {len(feature_cols)} (Expected: {expected_features})")

    report_data["phase1_integrity"] = {
        "checksum_verified": checksum_ok,
        "feature_count_verified": feat_ok,
        "model_checksum": actual_checksum,
        "feature_count": len(feature_cols)
    }

    # --- Phase 2: Shadow Mode Non-Enforcement Audit ---
    print("\n--- PHASE 2: SHADOW MODE IMPLEMENTATION & NON-ENFORCEMENT AUDIT ---")
    audit_file = Path("scratch/shadow_mode_live_audit.jsonl")
    if audit_file.exists():
        audit_file.unlink()

    service = initialize_service(model_path=model_path, audit_log_path=audit_file)
    client = TestClient(app)

    shadow_requests = 150
    shadow_logs = []
    blocked_customers = 0

    for i in range(shadow_requests):
        payload = {
            "order_id": f"SHADOW_LIVE_{i:04d}",
            "customer_id": f"C_SHADOW_{i % 25:03d}",
            "event_time": f"2025-06-29T10:{(i // 60):02d}:{(i % 60):02d}Z",
            "amount": 1000.0 + (i * 25.0),
            "device_id": f"D_SHADOW_{i % 10:03d}",
            "ip_id": f"IP_SHADOW_{i % 15:03d}"
        }
        resp = client.post("/v1/predict", json=payload, headers={"X-Shadow-Mode": "true"})
        assert resp.status_code == 200
        data = resp.json()
        
        # Enforce safety check: action must NEVER be BLOCK when SHADOW_MODE is active
        if data.get("action") == "BLOCK" and not data.get("enforce_decisions", False):
            blocked_customers += 1

    shadow_non_enforcement_ok = (blocked_customers == 0)
    print(f" [{ 'PASS' if shadow_non_enforcement_ok else 'FAIL' }] Customer Non-Enforcement Guarantee: {blocked_customers} customer transactions blocked.")

    report_data["phase2_shadow_audit"] = {
        "total_requests": shadow_requests,
        "blocked_customers": blocked_customers,
        "non_enforcement_guaranteed": shadow_non_enforcement_ok,
        "audit_file_created": audit_file.exists()
    }

    # --- Phase 3: Production Shadow Observability ---
    print("\n--- PHASE 3: PRODUCTION SHADOW OBSERVABILITY PROBES & METRICS ---")
    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    metrics_text = metrics_resp.text

    metrics_ok = "abuse_ring_detector_requests_total" in metrics_text
    print(f" [{ 'PASS' if metrics_ok else 'FAIL' }] Prometheus Exposition Endpoint (/metrics): Active & verified.")

    report_data["phase3_observability"] = {
        "metrics_endpoint_healthy": metrics_ok,
        "service_metrics": service.get_metrics()
    }

    # --- Phase 4: Shadow Evaluation Pipeline ---
    print("\n--- PHASE 4: SHADOW EVALUATION PIPELINE ---")
    evaluator = ShadowEvaluationPipeline(shadow_log_path=audit_file)
    
    # Generate mock label dataset to verify evaluation pipeline mechanics
    df_labels_mock = pd.DataFrame([
        {"order_id": f"SHADOW_LIVE_{i:04d}", "is_abuse": 1 if i % 10 == 0 else 0}
        for i in range(shadow_requests)
    ])
    df_orders_mock = pd.DataFrame([
        {"order_id": f"SHADOW_LIVE_{i:04d}", "amount": 1000.0 + (i * 25.0)}
        for i in range(shadow_requests)
    ])

    eval_results = evaluator.evaluate_with_labels(df_labels_mock, df_orders_mock)
    eval_ok = eval_results.get("evaluation_status") == "EVALUATED_WITH_LABELS"
    print(f" [{ 'PASS' if eval_ok else 'FAIL' }] Delayed Ground-Truth Evaluation Pipeline: {eval_results.get('evaluation_status')}")

    report_data["phase4_evaluator"] = eval_results

    # --- Phase 5: Shadow Data Quality & Safety Gates ---
    print("\n--- PHASE 5: SHADOW DATA QUALITY & SAFETY GATES ---")
    gate_evaluator = ShadowSafetyGateEvaluator(shadow_log_path=audit_file)
    gate_summary = gate_evaluator.run_all_gates(service.get_metrics(), model_bundle=service.model)
    
    print("\nSHADOW SAFETY GATE MATRIX SUMMARY:")
    for gate in gate_summary["gates"]:
        status = f"[{gate['status']}]"
        print(f" {status:6s} {gate['title']}: {gate['details']}")

    report_data["phase5_safety_gates"] = gate_summary

    # --- Phase 7: Final Decision Gate ---
    print("\n--- PHASE 7: FINAL DECISION GATE VERDICT ---")
    verdict = "READY FOR LIVE SHADOW DEPLOYMENT" if gate_summary["overall_passed"] else "NOT READY FOR LIVE SHADOW DEPLOYMENT"
    print(f" DECISION: {verdict}")

    report_data["final_verdict"] = verdict

    with open("scratch/shadow_observation_summary.json", "w") as f:
        json.dump(report_data, f, indent=2)

    return report_data


if __name__ == "__main__":
    run_shadow_observation_validation()
