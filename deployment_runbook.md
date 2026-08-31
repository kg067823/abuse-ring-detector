# Production Deployment Runbook & Operational Standard Operating Procedures (SOP)

## 1. Prerequisites & Environment Dependencies
- **Runtime Environment**: Python 3.11+ or Docker 24.0+ / Docker Compose v2.20+
- **Inference Service**: FastAPI application (`src/abuse_ring_detector/api.py`)
- **State Backend**: Redis v7.0+ (AOF persistence enabled, `volatile-lru` eviction policy)
- **Model Artifact**: `artifacts/model_f_r1_bundle.pkl` (Model F-R1 reconstruction, 137 features, new SHA-256 recorded in `model_f_r1_manifest.json`)
- **Operating System**: Linux (Ubuntu 22.04 LTS / Debian 12 / Container slim) or Windows 10/11

---

## 2. Environment Variables Configuration (`.env`)

| Variable Name | Default Value | Description |
| :--- | :--- | :--- |
| `HOST` | `0.0.0.0` | API Server listening host interface |
| `PORT` | `8000` | API Server listening port |
| `WORKERS` | `4` | Uvicorn worker process count |
| `ENVIRONMENT` | `production` | Deployment stage environment identifier |
| `REDIS_URL` | `redis://localhost:6379/0` | Primary Redis feature state store connection string |
| `MODEL_PATH` | `artifacts/model_f_r1_bundle.pkl` | Path to reconstructed Model F-R1 bundle |
| `AUDIT_LOG_PATH`| `logs/audit.jsonl` | Filepath for immutable JSON audit logging |
| `LOCKED_THRESHOLD`| `0.50` | Non-negotiable frozen operating threshold $\tau = 0.50$ |
| `FALLBACK_RISK_SCORE`| `0.05` | Population baseline risk score served during emergency fallbacks |
| `KILL_SWITCH` | `false` | Initial startup state of emergency scoring kill-switch |

---

## 3. Deployment Startup Procedure

### Option A: Docker Compose Deployment (Recommended)
```bash
# 1. Copy environment template
cp .env.example .env

# 2. Build and start production stack in detached mode
docker-compose up -d --build

# 3. Verify container statuses
docker-compose ps
```

### Option B: Local / Direct Process Startup
```bash
# 1. Activate Python virtual environment
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows

# 2. Start Uvicorn ASGI server with 4 worker processes
uvicorn abuse_ring_detector.api:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 4. Operational Health & Readiness Verification

After service startup, execute health probes to verify operational readiness:

### 1. Liveness Probe (`GET /liveness`)
```bash
curl -s http://localhost:8000/liveness
# Expected Response (HTTP 200 OK): {"status": "alive"}
```

### 2. Service Health Probe (`GET /health`)
```bash
curl -s http://localhost:8000/health
# Expected Response (HTTP 200 OK):
# {
#   "status": "healthy",
#   "service": "abuse-ring-detector",
#   "model_version": "v1.0.0-ModelF",
#   "schema_version": "v1.0.0",
#   "kill_switch_active": false
# }
```

### 3. Readiness Probe (`GET /readiness`)
```bash
curl -s http://localhost:8000/readiness
# Expected Response (HTTP 200 OK):
# {
#   "status": "ready",
#   "model_loaded": true,
#   "feature_count": 137,
#   "state_backend": "redis",
#   "state_backend_healthy": true
# }
```

---

## 5. Model & Checksum Validation
Verify that the loaded model matches the non-negotiable frozen Model F manifest:
- **Model Architecture**: Graph-Temporal Subgraph HistGradientBoostingClassifier
- **Feature Vector Dimension**: Exactly **137 features**
- **Operating Threshold**: $\tau = 0.50$
- **Model Artifact Checksum**: Full SHA-256 from `model_f_r1_manifest.json`; the historical checksum is not accepted.

---

## 6. Emergency Kill-Switch Operational Procedure

If production issues arise (e.g. upstream data corruption, third-party API outage):

### Activate Emergency Kill-Switch (Bypasses Model Scoring)
```bash
curl -X POST http://localhost:8000/v1/admin/kill-switch \
     -H "Content-Type: application/json" \
     -d '{"active": true}'
```
*Effect*: All `/v1/predict` requests instantly return HTTP 200 OK with safe fallback risk score `0.05`, `alert=false`, `fallback_applied=true`, `reason_codes=["kill_switch_active"]`.

### Deactivate Emergency Kill-Switch (Resumes Full Model F Inference)
```bash
curl -X POST http://localhost:8000/v1/admin/kill-switch \
     -H "Content-Type: application/json" \
     -d '{"active": false}'
```

---

## 7. Redis & State Backend Failure Recovery Procedure

1. **Automatic Local Fallback**: If Redis connection is interrupted, `RedisFeatureStateStore` automatically falls back to local thread-safe state store (`InMemoryFeatureStateStore`). The API returns HTTP 200 predictions without dropping requests.
2. **Redis Restoral**: Once Redis instance is restarted or reconnected, the state backend automatically re-establishes connectivity without requiring API service restarts.
3. **State Snapshot Restoration**:
   ```python
   # To restore feature state snapshot after full stack maintenance:
   service.load_state("scratch/service_state_backup.json")
   ```

---

## 8. Service Shutdown & Graceful Draining

```bash
# Gracefully stop API services and save shutdown snapshot
docker-compose stop -t 30 api

# Stop all services including Redis
docker-compose down
```

---

## 9. Model Retraining & Rollback Trigger Policy

### Retraining Trigger Criteria
- **30-Day Lagged Precision Drop**: Ground-truth precision on delayed chargeback labels drops below **75.0%**.
- **Significant Feature Drift**: Monitored PSI exceeds **0.25 (CRITICAL)** for key graph or velocity features.
- **Scheduled Cadence**: Standard **30-day automated retraining cycle**.

### Conditions That BLOCK Retraining
- **Data Volume Deficiency**: Candidate dataset contains fewer than **25,000 valid orders**.
- **Label Depletion**: Total positive abuse events in training split is fewer than **300 labels**.
- **Validation Regression**: Candidate model validation PR-AUC is lower than current active model PR-AUC by $\ge 0.03$.
- **Precision Regression**: Precision at $\tau=0.50$ drops below **80.0%**.

### Model Rollback Procedure
1. Shift traffic via load balancer / router to previous stable release (e.g. `v0.9.0-ModelE`).
2. Restore service state snapshot from backup.
3. Verify `/health` and `/readiness` status endpoints on previous release.

---

## 10. Audit Logging & Prometheus Metrics Exposition

### Audit Log Inspection (`logs/audit.jsonl`)
Every scoring event appends an immutable JSON record:
```json
{
  "timestamp": "2026-08-30T06:25:41Z",
  "order_id": "O_OBS_001",
  "customer_id": "C_OBS_101",
  "model_version": "v1.0.0-ModelF",
  "schema_version": "v1.0.0",
  "risk_score": 0.0027,
  "calibrated_score": 0.0027,
  "alert": false,
  "threshold": 0.50,
  "latency_ms": 0.14,
  "fallback_applied": false,
  "reason_codes": []
}
```

### Prometheus Metrics Endpoint (`GET /metrics`)
```bash
curl -s http://localhost:8000/metrics
# Output format: Prometheus exposition text
# - abuse_ring_detector_requests_total
# - abuse_ring_detector_fallback_total
# - abuse_ring_detector_alerts_total
# - abuse_ring_detector_latency_seconds_bucket
```
