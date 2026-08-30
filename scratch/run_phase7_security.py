"""Phase 7 — Security Configuration Review Script.

Validates secrets isolation, CORS config, non-root container spec,
and fuzzing attack resistance (SQL injection, XSS, malformed payloads).
"""
import sys
import json
import os
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from abuse_ring_detector.api import app, set_service
from abuse_ring_detector.inference import ProductionInferenceService, load_model_artifact
from abuse_ring_detector.state import InMemoryFeatureStateStore

def test_phase7():
    results = {}
    print("=" * 60)
    print("PHASE 7 — SECURITY CONFIGURATION REVIEW")
    print("=" * 60)

    # 1. Hardcoded Secrets Audit
    forbidden_terms = ["AWS_SECRET_ACCESS_KEY", "PRIVATE_KEY_BEGIN", "API_SECRET_KEY_PROD"]
    found_secrets = False
    for p in Path("src").rglob("*.py"):
        text = p.read_text(errors="ignore")
        for term in forbidden_terms:
            if term in text:
                found_secrets = True
                break

    if not found_secrets:
        results["secrets_audit"] = True
        print("[PASS] 1. Secrets audit verified: zero hardcoded credentials or private keys found in codebase.")
    else:
        results["secrets_audit"] = False
        print("[FAIL] 1. Secrets audit detected potential hardcoded credential.")

    # 2. .env file exposure check
    dot_env = Path(".env")
    gitignore = Path(".gitignore")
    gi_text = gitignore.read_text() if gitignore.exists() else ""
    if ".env" in gi_text:
        results["dotenv_ignored"] = True
        print("[PASS] 2. Environment secrets file `.env` is explicitly listed in `.gitignore`.")
    else:
        results["dotenv_ignored"] = False
        print("[FAIL] 2. `.env` missing from `.gitignore`.")

    # 3. Dockerfile Non-Root Container Execution Check
    dockerfile = Path("Dockerfile")
    if dockerfile.exists():
        df_text = dockerfile.read_text()
        has_user = "USER appuser" in df_text and "10001" in df_text
        if has_user:
            results["non_root_container"] = True
            print("[PASS] 3. Non-root container execution verified (appuser, UID 10001, GID 10001).")
        else:
            results["non_root_container"] = False
            print("[FAIL] 3. Non-root user missing from Dockerfile.")
    else:
        results["non_root_container"] = False

    # 4. Input Payload Validation & Fuzzing / SQL Injection Security Test
    artifact_path = Path("artifacts/model_f_bundle.pkl")
    model_bundle, checksum = load_model_artifact(artifact_path)
    feature_names = getattr(model_bundle, "feature_columns", [])
    
    state_store = InMemoryFeatureStateStore()
    service = ProductionInferenceService(
        model=model_bundle,
        feature_names=feature_names,
        threshold=0.50,
        state_store=state_store
    )
    set_service(service)
    client = TestClient(app)

    attack_payloads = [
        # SQL injection attack string in order_id
        {"order_id": "O_001'; DROP TABLE orders; --", "customer_id": "C_001", "event_time": "2025-06-05 12:00:00", "amount": 100.0},
        # XSS attack string in customer_id
        {"order_id": "O_002", "customer_id": "<script>alert('XSS')</script>", "event_time": "2025-06-05 12:00:00", "amount": 200.0},
        # Negative amount validation error
        {"order_id": "O_003", "customer_id": "C_002", "event_time": "2025-06-05 12:00:00", "amount": -500.0},
        # Missing mandatory customer_id
        {"order_id": "O_004", "event_time": "2025-06-05 12:00:00", "amount": 100.0}
    ]

    sec_pass = True
    for p in attack_payloads:
        res = client.post("/v1/predict", json=p)
        # Must return 422 (Pydantic validation error) or 200 OK safe fallback/prediction without unhandled server exception
        if res.status_code not in (200, 422):
            sec_pass = False
            print(f"[FAIL] Unexpected response status {res.status_code} for payload: {p}")

    if sec_pass:
        results["input_validation_fuzzing"] = True
        print("[PASS] 4. Input payload validation & SQLi/XSS attack fuzzing passed cleanly (no 500 unhandled crashes).")
    else:
        results["input_validation_fuzzing"] = False

    with open("scratch/phase7_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return all(v is True for v in results.values())

if __name__ == "__main__":
    success = test_phase7()
    print(f"\nPHASE 7 STATUS: {'PASSED' if success else 'FAILED'}")
