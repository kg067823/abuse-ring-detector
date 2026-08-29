# Production Deployment Runbook & Operational Retraining / Rollback Policy

## 1. Trigger Criteria for Model Retraining

Automatic or manual retraining is triggered when any of the following conditions occur:

- **30-Day Lagged Precision Drop**: Ground-truth precision on delayed chargeback labels drops below **75.0%**.
- **Significant Feature Drift**: Monitored PSI exceeds **0.25 (CRITICAL)** for key graph or velocity features.
- **Scheduled Calendar Interval**: Standard **30-day automated retraining cadence**.

---

## 2. Conditions That Block Retraining

Retraining pipelines are automatically **BLOCKED** and aborted if:

- **Data Volume Deficiency**: Candidate training dataset contains fewer than **25,000 valid orders**.
- **Label Depletion**: Total positive abuse events in retraining split is fewer than **300 labels**.
- **Validation Regression**: Candidate model validation PR-AUC is lower than current active model PR-AUC by $\ge 0.03$.
- **Validation Precision Regression**: Precision at $	au=0.50$ drops below **80.0%**.

---

## 3. Model & Feature Versioning Strategy

- **Model Artifact Identifier**: `v{MAJOR}.{MINOR}.{PATCH}-ModelF` (e.g. `v1.0.0-ModelF`).
- **Feature Schema Identifier**: `v{MAJOR}.{MINOR}.{PATCH}` (e.g. `v1.0.0`).
- **Storage Path**: `s3://abuse-ring-detector/models/v1.0.0-ModelF/model.pkl`.
- **Freeze Manifest Linkage**: Every deployed model version must contain an immutable `reports/model_f_freeze_manifest.json`.

---

## 4. Model Rollback Procedure

If production monitoring detects anomalies post-deployment:

1. **Trigger Rollback**: Risk operator executes blue-green router traffic shift to previous stable version (e.g. `v0.9.0-ModelE`).
2. **State Sync**: The previous model version loads feature store state from `scratch/service_state_backup.json`.
3. **Verification**: Verify 100% scoring traffic routes cleanly with zero scoring failures.

---

## 5. Safe Fallback Behavior & Emergency Kill-Switch

- **Emergency Scoring Kill-Switch**: Setting `kill_switch_active = True` immediately bypasses model inference.
- **Fallback Response Payload**:
  - `risk_score`: `0.05` (baseline population risk)
  - `calibrated_score`: `0.05`
  - `alert`: `False`
  - `fallback_applied`: `True`
  - `reason_codes`: `["kill_switch_active"]` or `["schema_validation_error"]`
- **Zero Unhandled Exceptions**: Inference API intercepts all runtime exceptions and returns HTTP 200 with fallback payload.

---

## 6. Audit Logging Requirements

Every inference scoring event must emit an immutable JSON audit log line containing:

```json
{
  "timestamp": "2026-08-29T14:30:00Z",
  "order_id": "O0042501",
  "customer_id": "C001234",
  "model_version": "v1.0.0-ModelF",
  "schema_version": "v1.0.0",
  "risk_score": 0.8842,
  "calibrated_score": 0.8710,
  "alert": true,
  "threshold": 0.50,
  "latency_ms": 0.045,
  "fallback_applied": false,
  "reason_codes": []
}
```
