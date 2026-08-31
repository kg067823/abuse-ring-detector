# R1 Pre-Production Validation Report

## Status

This report supersedes stale historical release reports that refer to the
unrecoverable historical Model F artifact. Those reports are archival context
only and are not R1 evidence.

```text
R1_PREPRODUCTION_VALIDATION: PASS (pre-production engineering evidence complete; live production remains blocked)
LIVE_PRODUCTION_OBSERVATION: NOT_STARTED
QUALIFYING_DAYS: 0/7
CANARY_STAGE_1: BLOCKED
ENFORCEMENT: DISABLED
```

This report covers local/container and synthetic replay evidence only. It does
not claim live production traffic, customer-impact validation, or seven
qualifying production observation days.

## Authoritative R1 contract

- Version: `model_f_r1`
- Identity/design: `graph_temporal_custrel_subgraph`
- Artifact: `artifacts/model_f_r1_bundle.pkl`
- SHA-256: `3f8bae638c6e81d0b391f9b226385e855ceb09744d774ea2d24cb9d0375c7cff`
- Features: 137 ordered features
- Seed: 42
- Threshold: 0.50
- Calibration: new `isotonic_regression`
- Safety: `SHADOW_MODE=true`, `ENFORCE_DECISIONS=false`

The historical Model F artifact was not recovered and its checksum is not used
by R1.

## Measured model evidence

The R1 builder measured the following on reconstructed synthetic data:

| Split | Orders | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 8,180 | 0.82680 | 0.94418 | 0.91252 | 0.67500 | 0.77599 |
| Test | 8,180 | 0.79812 | 0.93815 | 0.91874 | 0.63994 | 0.75440 |

These are R1 measurements, not historical Model F results. Rebuilding twice
produced byte-identical artifacts with the SHA-256 above.

## Dependency and test environment

`pyproject.toml` and `requirements.txt` declare FastAPI, Uvicorn, Redis,
HTTPX, pytest, and the pinned R1 ML stack. The supported environment is Python
3.11. The host session used Python 3.9 and could not import FastAPI. The authoritative `.venv` uses Python 3.11.15 and the complete repository
suite passed: 125 passed, 0 failed, 0 skipped, in 196.25 seconds. Two
non-failing dependency deprecation warnings were emitted.

## Docker, Redis, and API evidence

Measured previously against the Compose stack:

- Docker build: PASS.
- Non-root `appuser` UID 10001: PASS.
- Redis healthcheck and connectivity: PASS.
- `/health`: HTTP 200.
- `/liveness`: HTTP 200.
- `/readiness`: HTTP 200 with R1 checksum, 137 features, threshold 0.50, and healthy Redis.
- `/metrics`: HTTP 200 with shadow/enforcement and request/fallback counters.
- `/v1/predict`: representative request succeeded with `enforcement_applied=false`.

`/v1/explain` now provides masked, deterministic observed signals with an
explicit non-causal caveat while leaving the prediction response contract
unchanged.

## Validation harness

`scratch/run_r1_preproduction_validation.py` emits
`r1_preproduction_validation.json` with artifact, probe, replay, explanation,
and live-gate statuses. In the latest run, contract/probes passed but the
runtime replay was blocked by a local Docker connection reset during teardown;
this is recorded as a harness/environment issue, not converted to PASS.

## Safety and failure behavior

- Missing/corrupt/mismatched R1 artifact, manifest, contract, feature order,
  calibration, or checksum fails closed at initialization.
- Redis unavailable before startup blocks readiness.
- Kill switch remains a safe fallback.
- Malformed requests cannot enable enforcement.
- Shadow responses expose no customer-blocking action.
- Seven-day gate remains not started; synthetic/staging data does not qualify.

## Monitoring and state limitations

The API exposes request, fallback, alert, shadow, enforcement, blocked,
modified, and latency metrics. R1 identity/checksum are visible in readiness
and explanation output. Current counters are process-local and the hand-built
histogram is not a shared Prometheus backend. Redis feature reads and atomic
cross-worker deduplication require further production hardening; no multi-worker
state-consistency PASS is claimed.

## Remaining blockers

- No real production ingress or delayed production ground truth.
- No seven qualifying live observation days.
- Canary Stage 1 and enforcement remain blocked.
- Full Python 3.11 suite is complete locally; Docker external replay still
  requires a fresh daemon/container lifecycle because a stale unhealthy
  container caused connection resets during one run.
- Authenticated kill-switch administration is not implemented.
- Redis restart/reconnect and aggregate multi-worker metrics require dedicated
  follow-up validation.

## Next step

Run the full suite in a persistent Python 3.11 environment and rerun the
external Docker/Redis harness without teardown interruption. Then attach real
production shadow telemetry for seven consecutive qualifying days before any
canary decision.
