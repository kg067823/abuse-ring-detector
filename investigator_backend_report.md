# AbuseRing Investigator Case Management Backend

## Scope

This is a product layer over the frozen Model F-R1 shadow inference service.
It does not retrain, change, or wrap the model's features, calibration, or
threshold. It does not enable customer enforcement and does not represent live
production traffic.

All demo records are labeled `DEMO / SYNTHETIC`.

## Domain model

The backend defines typed Pydantic models for:

- `Alert`: a calibrated R1 score at or above 0.50, linked to an order, masked
  customer identity, correlation ID, evidence, amount, and R1 provenance.
- `InvestigationCase`: grouped alerts with status, severity, related customers,
  orders/entities, exposure, evidence, graph context, timeline, notes, and
  immutable mutation history.
- `EvidenceItem`: an observed signal with description, value, comparison,
  timestamp/window, entity type, source feature names, and provenance.
- `Entity`: masked customer/order/device/address/IP/payment graph node metadata.
- `RingConnection`: sanitized graph edge metadata.
- `TimelineEvent`: deterministic chronological investigation events.
- `AnalystDecision`: actor, decision, reason, and timestamp.

`ring_id`, labels, abuse types, and historical loss fields are not part of the
case-management grouping or public case payload.

## Alert lifecycle

A successful shadow prediction with calibrated score `>= 0.50` creates one
alert keyed deterministically by order ID. Fallback responses and lower scores
do not create alerts. Repeated idempotent predictions return the existing alert
rather than creating a duplicate.

Alert creation is internal review data only:

```text
SHADOW_MODE=true
ENFORCE_DECISIONS=false
enforcement_applied=false
```

## Case consolidation

Cases use deterministic single-linkage over observable prior-time entity keys:
`device_id`, `address_id`, `ip_id`, and `payment_id`. Two alerts consolidate
when they share a non-empty typed entity key or customer identity already in a
case. Empty values are never grouping keys. This is a transparent graph
relationship strategy and does not inspect synthetic ground-truth ring IDs.

The current demo repository is in-memory behind `CaseRepository`; a future
PostgreSQL implementation can replace it without changing the API/service
contract.

## Severity policy

- `CRITICAL`: score `>= 0.90`, or at least three connected customers with two or
  more observable entity modalities and a burst signal.
- `HIGH`: score `>= 0.75`, or score `>= 0.50` with meaningful connection,
  modality, or burst evidence.
- `MEDIUM`: score `>= 0.50` with limited evidence.
- `LOW`: below the alert threshold.

Exposure is the sum of observed transaction amounts in the case. It is not a
chargeback or ground-truth loss estimate.

## Evidence and explanations

Evidence descriptions use “observed” and “associated with elevated risk”. They
explicitly avoid causal claims. The evidence includes a threshold-crossing
signal and prior observable shared-entity signals. `POST /v1/explain` remains a
separate R1-compatible endpoint and returns masked IDs, deterministic observed
signals, provenance, and a non-causal caveat.

No raw customer, device, address, IP, or payment identifiers are returned by the
case API. Public IDs are stable pseudonyms generated with SHA-256 prefixes.

## Graph and timeline APIs

- `GET /v1/cases/{case_id}/graph` returns sanitized customer/order/entity nodes
  and typed edges such as `customer-placed-order` and `customer-used-device`.
- `GET /v1/cases/{case_id}/timeline` returns deterministic chronological
  timeline events.
- `GET /v1/cases/{case_id}/evidence` returns sorted evidence items.

The graph output is designed for a future investigator visualization. It is not
a causal or ground-truth ring declaration.

## Case APIs

- `GET /v1/alerts`
- `GET /v1/cases`
- `GET /v1/cases/{case_id}`
- `GET /v1/cases/{case_id}/graph`
- `GET /v1/cases/{case_id}/timeline`
- `GET /v1/cases/{case_id}/evidence`
- `PATCH /v1/cases/{case_id}/status`
- `POST /v1/cases/{case_id}/notes`

Status changes and note additions validate the workflow and append immutable
history records. Case mutation audit entries use masked case/actor IDs.

## Demo mode

The current APIs label returned collections `DEMO / SYNTHETIC`. A deterministic
replay should send synthetic transactions through the actual R1 HTTP prediction
path and then query these case endpoints. No hard-coded production case output
is used. The existing frozen R1 artifact remains the source of scores.

## Tests

`tests/test_case_management.py` covers:

- score/fallback alert creation and duplicate suppression;
- observable shared-entity consolidation without `ring_id`;
- severity boundaries;
- explicit status transitions and immutable history;
- notes and mutation audit;
- masked graph output and non-causal case payloads.

The pre-existing Python 3.11 suite remains the compatibility gate.

## Limitations

- The repository implementation is an in-memory demo repository, not a durable
  production system of record.
- Analyst mutation authentication is not yet separate from the existing API
  admin-token mechanism.
- Cases are created from the API process's in-memory manager and are not
  replicated across workers or persisted in Redis/PostgreSQL.
- The demo collection label is not live production evidence.
- The seven-day gate remains untouched:

```text
LIVE_PRODUCTION_OBSERVATION: NOT_STARTED
QUALIFYING_DAYS: 0/7
CANARY_STAGE_1: BLOCKED
ENFORCEMENT: DISABLED
```
