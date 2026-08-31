# Presentation Metrics — Model F-R1

Use this table as the single source for judge-facing numbers. Every result is
from the reconstructed/newly frozen R1 artifact, not historical Model F, and
not live production.

| Metric | Value | Provenance |
|---|---:|---|
| Model version | `model_f_r1` | `model_f_r1_manifest.json` |
| Model identity | `graph_temporal_custrel_subgraph` | R1 manifest/contract |
| Artifact SHA-256 | `3f8bae638c6e81d0b391f9b226385e855ceb09744d774ea2d24cb9d0375c7cff` | Exact artifact bytes |
| Ordered feature count | 137 | R1 manifest/contract |
| Seed | 42 | R1 manifest |
| Threshold | 0.50 | R1 contract |
| Calibration | `isotonic_regression` | Validation-only new R1 fit |
| Held-out test orders | 8,180 | Chronological unseen test split |
| Test PR-AUC | 0.79812 | R1 manifest, reconstructed synthetic data |
| Test ROC-AUC | 0.93815 | R1 manifest, reconstructed synthetic data |
| Test precision @ 0.50 | 0.91874 | R1 manifest, reconstructed synthetic data |
| Test recall @ 0.50 | 0.63994 | R1 manifest, reconstructed synthetic data |
| Test F1 @ 0.50 | 0.75440 | R1 manifest, reconstructed synthetic data |
| Test calibrated Brier | 0.02840 | R1 manifest |
| Test calibrated ECE | 0.00752 | R1 manifest |
| Test true positives | 407 | R1 manifest |
| Test false positives | 36 | R1 manifest |
| Test false negatives | 229 | R1 manifest |
| Artifact rebuild reproducibility | Byte-identical twice | Freeze/pre-production reports |
| Python 3.11 suite | 141 passed, 0 failed, 0 skipped | Current `.venv` regression run at presentation-preparation time |

## How to say the metrics

- **PR-AUC 0.798:** “On the held-out chronological synthetic test split, the
  model maintained strong ranking quality for an imbalanced alerting problem.”
- **Precision 0.919:** “At the locked 0.50 review threshold, about 92% of model
  alerts corresponded to positive synthetic test labels.”
- **Recall 0.640:** “The same fixed operating point recovered about 64% of positive
  synthetic test events; we present this together with precision rather than
  hiding the trade-off.”
- **F1 0.754:** “The fixed threshold balances the measured precision and recall
  at an F1 of about 0.75.”
- **ECE 0.00752:** “The newly fitted calibrator kept test calibration error low
  in this reconstructed evaluation.”
- **138 tests:** “The engineering suite passed 138 automated tests, separate from
  model-quality metrics.”

## Do not use

Do not quote old Model E/F tables, old 16-character checksum
`82e77daac0762a04`, historical holdout numbers, historical latency/RPS claims,
ring-level coverage claims absent from the R1 manifest, production ROI, or live
production performance as current R1 evidence. The original Model F artifact is
unrecoverable; R1 is a new reconstruction/freeze.
