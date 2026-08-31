# Production Readiness Report — Frozen Model F

## Current verdict

**Model F-R1 reconstructed; full production readiness remains pending shadow/runtime validation.**

The repository now contains a newly generated Model F-R1 bundle and R1 contract. This is a reconstruction from repository source and generated synthetic data, not recovery of the historical Model F artifact. Docker startup, readiness, and a representative HTTP request were validated locally; broader load, recovery, and live production validation remain pending.

## Frozen contract status

| Contract item | Required value | Current evidence |
|---|---:|---|
| Champion | Model F | Referenced by historical reports only |
| Model | `graph_temporal_custrel_subgraph` | No authoritative bundle present |
| Features | Exactly 137 | No authoritative feature list present |
| Seed | 42 | Referenced by historical reports only |
| Threshold | 0.50 | Configured as a locked runtime constant |
| Checksum | `3f8bae638c6e81d0b391f9b226385e855ceb09744d774ea2d24cb9d0375c7cff` | Verified against Model F-R1 artifact bytes |
| Calibration | Frozen calibration | Contract metadata absent |

The service now fails closed rather than training a replacement model when the
R1 artifact, manifest, contract, or calibration object is missing or incompatible.

## Completed engineering changes

- Added package metadata required by the Docker builder.
- Added explicit `SHADOW_MODE=true` and `ENFORCE_DECISIONS=false` deployment configuration.
- Added frozen bundle validation hooks for checksum, feature count, model identity, and seed.
- Removed the production startup path that silently trained a synthetic fallback.
- Production initialization now requires healthy Redis; it no longer silently selects a per-worker in-memory store.
- Readiness now reports the verified model identity/checksum and fails when the model or required state backend is unavailable.
- Added a non-bypassable `seven_day_gate.py` status command that reports no live observation until genuine evidence exists.

## Not yet evidenced

The following must not be represented as completed until the authoritative bundle,
manifest, and contract are supplied and real commands succeed:

- Clean Docker build and startup.
- Model checksum verification and exact 137-feature contract verification.
- Redis-backed multi-worker consistency.
- External HTTP chronological replay.
- High/low score shadow safety with zero blocked or modified transactions.
- API/Redis restart and recovery behavior.
- 100/250/500 RPS load test results.
- Audit and Prometheus evidence from the deployed service.

## Seven-day operational gate

```text
LIVE_PRODUCTION_OBSERVATION: NOT STARTED
QUALIFYING_DAYS: 0/7
CANARY_STAGE_1: BLOCKED
```

No staging replay or synthetic data is converted into live production evidence.

## Final recommendation

**CONDITIONAL GO** only after the authoritative frozen artifact and contract are
made available and the real Docker/HTTP/Redis validation package passes. Even
then, customer enforcement remains disabled until seven genuine qualifying
production observation days are complete.
