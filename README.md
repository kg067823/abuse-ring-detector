# AbuseRing Detector — POC v0.1

A defensive, synthetic proof-of-concept for detecting coordinated merchant abuse. The experiment compares a leakage-safe behavioural baseline with a graph-enhanced model using customer/device/IP/address/payment relationships.

## What this tests

Individual accounts can look legitimate while coordinated relationships reveal shared devices, addresses, payment instruments, velocity, and similar behaviour. This POC deliberately includes legitimate households, workgroups, high-volume customers, retries, and promotions as hard negatives. It does **not** claim production fraud-detection performance.

## Quickstart

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m abuse_ring_detector.cli run-poc --config configs/default.yaml --output-dir artifacts/run
```

For a fast smoke run, create a YAML config with a few hundred customers/orders and ring count 100. Generated CSV/GZIP tables, manifest, threshold tables, and Markdown report appear in the run directory.

## Pipeline

1. Synthetic ecosystem generation over six months.
2. Normal hard-negative behaviour plus configurable shared-device, shared-address, behavioural, and mixed rings.
3. Strict chronological split (70% train, 15% validation, 15% test).
4. As-of baseline features and historical customer/entity graph features.
5. HistGradientBoosting by default; optional XGBoost when installed.
6. Validation-only threshold selection using explicit review/block and false-negative loss assumptions.
7. Test metrics, threshold table, financial-loss estimate, ring statistics, and evidence report.

Feature rows are computed before the current event is added to state. Same-timestamp/future events are excluded. The graph model uses numeric structural features rather than raw IDs or community identifiers.

## Outputs and schemas

- `customers.csv.gz`: customer profile and latent generation fields.
- `orders.csv.gz`: order events and relationship IDs.
- `returns.csv.gz`: return events.
- `labels.csv.gz`: order-level ground truth abuse/ring/loss labels, never model input.
- `ground_truth.csv.gz`: customer-level abusive membership labels, never model input.
- `rings.csv.gz`, `ring_memberships.csv.gz`: synthetic ring metadata.
- `run_manifest.json`: counts, split cutoffs, model metrics, costs, and limitations.
- `report.md`, `*_thresholds.csv`: human-readable evaluation outputs.

## Limitations

Synthetic behaviour and labels are not a substitute for merchant data. Entity resolution, temporal drift, ring overlap, graph scale, and legitimate household/business sharing all require real-world validation. NetworkX is intentionally used only for this POC. Poor or non-improving graph metrics should be reported honestly rather than tuned away.

## Next steps

Add tests and notebooks for leakage checks, calibrated alert budgets, ring-level recall, graph ablations, SHAP/permutation importance, and a thin Streamlit artifact viewer after the core experiment is stable.
