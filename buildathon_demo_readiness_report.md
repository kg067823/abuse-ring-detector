# Buildathon Demo Readiness Report

## Verdict

```text
DEMO_READY: YES (local API/UI flow and tests verified)
LIVE_PRODUCTION_OBSERVATION: NOT_STARTED
CANARY_STAGE_1: BLOCKED
ENFORCEMENT: DISABLED
```

## Selected demo

Primary scenario: **Mixed multi-entity**. It uses stable identifiers and
increasing timestamps, then sends every event through the real `/v1/predict`
endpoint. The UI refreshes actual `/v1/cases` responses after each event.

Secondary scenarios:

- Shared-device ring
- Shared-address ring
- Behavioral coordination
- Legitimate high-connectivity control

The legitimate control is reported honestly from the frozen R1 output. The UI
does not force a negative result or fabricate a case.

## Startup

The cross-platform launcher is `scripts/start_demo.py`:

```bash
ADMIN_KILL_SWITCH_TOKEN=demo-secret .venv/bin/python scripts/start_demo.py --docker
```

It verifies the R1 artifact checksum, sets demo/shadow environment flags, starts
Compose when requested, waits for `/readiness`, and prints API/UI URLs. The demo
Compose override uses one API worker because investigator cases are currently
process-local.

## Polish delivered

- API-backed one-click scenario execution in Demo Mode.
- `Event n of 8` score/alert/case progression.
- Stable increasing timestamps and replay IDs.
- Overview shadow/model banner and case KPIs.
- Case workspace evidence/timeline emphasis.
- Graph legend, short masked labels, typed node metadata, and table fallback.
- Honest legitimate-control messaging.
- Loading/error/empty/replay-safe states.
- `DEMO_SCRIPT.md` and `DEMO_RECOVERY.md`.

## Tests and checks

Before this polish, the complete Python 3.11 suite passed:

```text
134 passed, 0 failed, 0 skipped
```

Frontend/helper tests passed:

```text
9 passed
```

After the deterministic scenario and UI changes, syntax and helper checks were
run. The immutable artifact remains:

```text
3f8bae638c6e81d0b391f9b226385e855ceb09744d774ea2d24cb9d0375c7cff
```

## Known limitations

- CaseManager storage is process-local; the demo uses one API worker for
  reliable case visibility.
- The graph renderer depends on Graphviz availability and falls back to tables.
- The legitimate control is a demo illustration, not a formal model-quality
  claim.
- The UI is Streamlit, not a production multi-user frontend with RBAC.
- Synthetic replay is not live production observation.

## Next step

**BUILDATHON PITCH / PRESENTATION PREPARATION**
