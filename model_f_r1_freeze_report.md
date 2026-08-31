# Model F-R1 Freeze Report

## Status

**Model F-R1 is a new reconstruction/new freeze, not recovery of historical Model F.**

The original artifact and checksum `82e77daac0762a04` were not recoverable from the repository and are not used by this artifact.

## Reconstruction methodology

- Source pipeline: `build_subgraph_extended_features()`
- Data generation: repository `generate_ecosystem()` using `configs/default.yaml`
- Seed: `42`
- Split: chronological 70% train / 15% validation / 15% test
- Training: training partition only
- Calibration: new validation-only Platt and isotonic fits; lower validation Brier selected
- Threshold: `0.50`
- Artifact: `artifacts/model_f_r1_bundle.pkl`
- Manifest: `model_f_r1_manifest.json`
- Serving contract: `inference_contract_r1.json`

## Feature contract

Exactly 137 ordered features are recorded in the manifest and contract:

```text
19 baseline + 18 graph + 30 temporal + 30 customer-relative +
20 two-hop + 20 subgraph = 137
```

## New artifact identity

The measured artifact checksum is
`3f8bae638c6e81d0b391f9b226385e855ceb09744d774ea2d24cb9d0375c7cff`.
The calibration method, split counts, environment, and performance metrics are
recorded in `model_f_r1_manifest.json`. This is the full SHA-256 digest of the
exact serialized artifact bytes and is not the historical checksum.

## Measured metrics

Metrics in the manifest were calculated from the newly generated R1 corpus and
newly fitted R1 model/calibrator. Historical README and release-report metrics
are not reused as R1 results.

## Reproducibility and limitations

The corpus and feature construction are deterministic under the recorded source,
configuration, seed, and dependency environment. This does not establish
identity with the missing historical artifact. The generated corpus is a
repository reconstruction, not the historical training snapshot. Dependency
versions and artifact digest are recorded for this freeze.

The online feature implementation and shared Redis state still require runtime
validation before any production readiness claim. This release remains shadow-only.

## Production/shadow status

```text
SHADOW_MODE=true
ENFORCE_DECISIONS=false
LIVE_PRODUCTION_OBSERVATION: NOT STARTED
QUALIFYING_DAYS: 0/7
CANARY_STAGE_1: BLOCKED
```

No customer blocking or modification is enabled.

## Exact next step

Run the dedicated R1 tests, full pytest suite, Docker/Redis/API smoke checks,
and independent reproducibility check. Report any failures as blockers. Do not
start canary enforcement or the seven-day live gate.
