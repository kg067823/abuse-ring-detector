# Model F Artifact Recovery and Reconstruction Assessment

**Assessment date:** 2026-08-31  
**Repository:** `abuse-ring-detector`  
**Assessed history:** `bfe5113`, `abda93a`, `ee43a90`, `07ba3eb`, `5cd6937`, `1369f09`

## Final determination

**C — BLOCKED — MODEL CANNOT BE RECOVERED OR RECONSTRUCTED**

The original frozen Model F binary with documented checksum
`82e77daac0762a04` is not present in the checkout, any reachable branch/tag,
or the inspected unreachable Git objects. The repository contains enough code
on the later feature branches to describe and approximately retrain a model
with the documented 137-feature architecture, but it does **not** contain
enough immutable inputs and fitted state to reproduce the frozen artifact and
its calibrated serving behavior reproducibly.

No replacement model was trained, serialized, or substituted.

## Repository and Git investigation

The following were inspected:

- Current working tree, index, ignored files, and generated directories.
- `git status --short --ignored --untracked-files=all`.
- All local and remote branches: `main`, `newFeatureBranch`, `origin/main`,
  `origin/featureBranch`, `origin/newFeatureBranch`, and agent worktree refs.
- All reachable commits and trees using `git log --all`, `git rev-list --objects
  --all --reflog`, `git ls-tree -r`, and historical `git show`.
- Git unreachable objects using `git fsck --full --no-reflogs --unreachable`,
  including all unreachable trees and blobs.
- File searches across the repository excluding Git internals and agent
  worktrees, including ignored paths.
- Candidate binary extensions: `.pkl`, `.pickle`, `.joblib`, `.onnx`, `.h5`,
  `.pt`, `.bin`, and `.ipynb`.
- References to `model_f_bundle`, `82e77daac0762a04`,
  `graph_temporal_custrel_subgraph`, `inference_contract`, freeze manifests,
  calibration, serialization, and model export.

### What was found

The later feature branches contain source code and scripts, including:

- `src/abuse_ring_detector/features.py`
  - `build_subgraph_extended_features()` documents Model F as 137 features.
  - Composition is Model E's 117 features plus 20 subgraph features.
  - The documented composition is baseline 19 + graph 18 + temporal 30 +
    customer-relative 30 + two-hop 20 + subgraph 20.
- `configs/default.yaml`
  - Seed 42.
  - 20,000 customers, 50,000 orders, 180 days.
  - Chronological 70/15/15 split.
  - 30-day graph history.
  - HistGradientBoosting backend, `max_iter=160`, `learning_rate=0.08`,
    `max_leaf_nodes=15`.
- `scratch/generate_model_artifact.py`
  - Generates a new dataset, fits a new model, and calls
    `save_model_artifact()`.
- `scratch/run_final_model_freeze_analysis.py`
  - Runs a fresh fit and writes a freeze report, but does not serialize the
    fitted estimator or calibrator.
- `scratch/run_final_holdout_and_calibration_evaluation.py`
  - Fits Platt and isotonic calibrators on validation scores and chooses by
    validation Brier score.
  - Writes reports only; it does not persist fitted calibration parameters.
- `src/abuse_ring_detector/models.py`
  - Fits a new estimator with the configured backend and seed.
  - Stores feature columns and minimal metadata, but no frozen fitted model is
    committed.
- `src/abuse_ring_detector/inference.py`
  - Has generic pickle save/load helpers, but no original pickle is present.

### What was not found

No copy of any of the following exists in the current tree, reachable Git
history, ignored artifact directories, or inspected unreachable Git objects:

- `artifacts/model_f_bundle.pkl` or another fitted Model F binary.
- `model_f_freeze_manifest.json` as an authoritative committed manifest.
- `inference_contract.json`.
- Fitted Platt coefficients or isotonic knots/calibrator artifact.
- The exact 210-day training/validation/holdout dataset snapshot used by the
  final calibration script.
- A checked-in feature-order artifact/hash tied to the frozen binary.
- A complete immutable environment/dependency lock for the frozen fit.
- A deterministic script that captures the original fitted estimator bytes and
  calibration object rather than fitting anew.

The existing `artifacts/` and `reports/` rules ignore generated contents; the
reachable trees contain only `.gitkeep` placeholders. Local generated files are
dataset outputs and reports, not the missing model binary or contract.

## Why exact recovery is impossible

A model checksum identifies fitted artifact bytes; it cannot be derived from a
reported metric table or source code. Re-running `fit_model()` produces a new
serialization and cannot establish identity with checksum
`82e77daac0762a04`.

The final serving behavior also depends on fitted calibration state. The source
only describes fitting Platt and isotonic calibration and selecting the lower
validation Brier score. It does not preserve the selected calibrator object,
its parameters, or the exact data snapshot used to fit it. The freeze manifest
writer records feature names and metrics when run, but that generated output is
absent and the script does not save the fitted estimator/calibrator.

The generated synthetic data is deterministic in principle, but the repository
still lacks an authoritative frozen data snapshot and an immutable environment
lock. More importantly, deterministic regeneration of a new fit is not proof
of the original artifact's bytes, serialization protocol, calibration state, or
checksum. The current production `StreamingFeatureStore` also does not by
itself constitute proof of serving parity with all 137 offline features.

Historical validation scripts and reports are evidence of prior claims, not
recoverable artifacts. Some checks are hardcoded or tautological, and the
shadow evidence identifies itself as staging replay rather than live
production. They cannot supply the missing fitted state.

## Reconstruction decision

**Deterministic reconstruction of the original frozen artifact: NO.**

It is possible to execute a fresh, repository-derived experiment using the
later feature branch's source, seed, configuration, and synthetic generator.
That would be a new model and would require explicitly choosing and persisting
calibration parameters. It could be labeled **Model F reconstructed/re-frozen
from repository source**, but it cannot honestly be called recovery of the
original Model F and cannot inherit the old checksum.

Because the user explicitly forbids silently replacing the old checksum and
requires an exact recovery/reconstruction determination, no new model was
created.

## Contract values

| Item | Documented value | Verified from an artifact? |
|---|---:|---:|
| Model identity | `graph_temporal_custrel_subgraph` | No |
| Feature count | `137` | Source composition only; no artifact |
| Seed | `42` | Configuration/source only |
| Threshold | `0.50` | Documentation/source only |
| Previous checksum | `82e77daac0762a04` | No binary to verify |
| Calibration | Platt/isotonic selection procedure described | No fitted object |

## Required next input

To resume an authoritative production artifact milestone, one of the following
must be supplied by an authorized owner:

1. The original Model F bundle and its checksum/manifest/contract; or
2. Explicit authorization for a new reconstruction/re-freeze, together with an
   agreed policy that it is a new model, receives a new checksum, and undergoes
   fresh validation and calibration.

Without one of those inputs, production serving and any claim of Model F
artifact identity remain blocked. The seven-day live gate and canary
enforcement must remain blocked.
