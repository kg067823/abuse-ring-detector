"""Shadow-only investigator case management for Model F-R1 alerts.

This module is deliberately independent of model training and enforcement. It
turns observable shadow alerts into deterministic, analyst-facing cases.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field, field_validator

from .explain import mask_identifier

R1_VERSION = "model_f_r1"
R1_CHECKSUM = "3f8bae638c6e81d0b391f9b226385e855ceb09744d774ea2d24cb9d0375c7cff"
R1_THRESHOLD = 0.50
ENTITY_FIELDS = ("device_id", "address_id", "ip_id", "payment_id")


class CaseStatus(str, Enum):
    NEW = "NEW"
    IN_REVIEW = "IN_REVIEW"
    ESCALATED = "ESCALATED"
    CONFIRMED_ABUSE = "CONFIRMED_ABUSE"
    LEGITIMATE = "LEGITIMATE"
    CLOSED = "CLOSED"


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Entity(BaseModel):
    id: str
    type: str
    first_seen: str | None = None
    last_seen: str | None = None
    risk_metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceItem(BaseModel):
    evidence_id: str
    evidence_type: str
    description: str
    value: float | int | str | bool
    comparison: str | None = None
    timestamp: str
    window: str
    entity_type: str | None = None
    source_features: list[str] = Field(default_factory=list)
    provenance: str = "observed_signal"


class RingConnection(BaseModel):
    source: str
    target: str
    relationship: str
    first_seen: str | None = None
    last_seen: str | None = None
    event_count: int = 1


class TimelineEvent(BaseModel):
    event_id: str
    timestamp: str
    event_type: str
    description: str
    source: str = "shadow_observation"


class AnalystDecision(BaseModel):
    actor: str
    decision: str
    reason: str
    timestamp: str


class Alert(BaseModel):
    alert_id: str
    order_id: str
    customer_id: str
    risk_score: float
    threshold: float = R1_THRESHOLD
    model_version: str = R1_VERSION
    model_checksum: str = R1_CHECKSUM
    created_at: str
    correlation_id: str = ""
    entity_keys: dict[str, str] = Field(default_factory=dict)
    amount: float = 0.0
    evidence: list[EvidenceItem] = Field(default_factory=list)
    shadow_alert: bool = True
    enforcement_applied: bool = False

    @field_validator("risk_score")
    @classmethod
    def score_bounds(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("risk_score must be between 0 and 1")
        return value


class InvestigationCase(BaseModel):
    case_id: str
    status: CaseStatus = CaseStatus.NEW
    severity: Severity = Severity.MEDIUM
    risk_score: float
    created_at: str
    updated_at: str
    primary_customer: str
    related_customers: list[str] = Field(default_factory=list)
    related_orders: list[str] = Field(default_factory=list)
    related_entities: list[Entity] = Field(default_factory=list)
    alert_count: int = 0
    estimated_exposure: float = 0.0
    model_version: str = R1_VERSION
    model_checksum: str = R1_CHECKSUM
    explanation_summary: str = "Observed evidence associated with elevated risk."
    evidence: list[EvidenceItem] = Field(default_factory=list)
    graph_context: list[RingConnection] = Field(default_factory=list)
    timeline: list[TimelineEvent] = Field(default_factory=list)
    analyst_disposition: AnalystDecision | None = None
    analyst_notes: list[str] = Field(default_factory=list)
    history: list[dict[str, Any]] = Field(default_factory=list)


class StatusUpdate(BaseModel):
    status: CaseStatus
    reason: str = Field(min_length=1, max_length=1000)
    actor: str = Field(min_length=1, max_length=200)


class NoteCreate(BaseModel):
    note: str = Field(min_length=1, max_length=4000)
    actor: str = Field(min_length=1, max_length=200)


class CaseRepository(Protocol):
    def list_cases(self) -> list[InvestigationCase]: ...
    def get_case(self, case_id: str) -> InvestigationCase | None: ...
    def list_alerts(self) -> list[Alert]: ...


ALLOWED_TRANSITIONS: dict[CaseStatus, set[CaseStatus]] = {
    CaseStatus.NEW: {CaseStatus.IN_REVIEW, CaseStatus.CLOSED},
    CaseStatus.IN_REVIEW: {CaseStatus.ESCALATED, CaseStatus.CONFIRMED_ABUSE, CaseStatus.LEGITIMATE, CaseStatus.CLOSED},
    CaseStatus.ESCALATED: {CaseStatus.CONFIRMED_ABUSE, CaseStatus.LEGITIMATE, CaseStatus.CLOSED},
    CaseStatus.CONFIRMED_ABUSE: {CaseStatus.CLOSED},
    CaseStatus.LEGITIMATE: {CaseStatus.CLOSED},
    CaseStatus.CLOSED: set(),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable(prefix: str, value: str) -> str:
    return f"{prefix}-{hashlib.sha256(value.encode()).hexdigest()[:16]}"


def severity_for(score: float, alerts: int, customers: int, modalities: int, burst: int = 0) -> Severity:
    if score >= 0.90 or (customers >= 3 and modalities >= 2 and burst >= 3):
        return Severity.CRITICAL
    if score >= 0.75 or (score >= 0.50 and (customers >= 2 or modalities >= 2 or burst >= 3)):
        return Severity.HIGH
    if score >= 0.50:
        return Severity.MEDIUM
    return Severity.LOW


def _public_case(case: InvestigationCase) -> dict[str, Any]:
    """Serialize a case with only pseudonymous identifiers."""
    data = case.model_dump(mode="json")
    data["primary_customer"] = mask_identifier(case.primary_customer)
    data["related_customers"] = [mask_identifier(v) for v in case.related_customers]
    data["related_orders"] = [mask_identifier(v) for v in case.related_orders]
    return data


class InMemoryCaseRepository:
    """Thread-safe demo repository with append-only mutation history."""

    def __init__(self, audit_path: str | Path | None = None):
        self._lock = threading.RLock()
        self.cases: dict[str, InvestigationCase] = {}
        self.alerts: dict[str, Alert] = {}
        self.audit_path = Path(audit_path) if audit_path else None
        if self.audit_path:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)

    def _audit(self, action: str, case_id: str, actor: str = "system", **details: Any) -> None:
        record = {"timestamp": _now(), "action": action, "case_id": mask_identifier(case_id), "actor": mask_identifier(actor), **details}
        if self.audit_path:
            with self.audit_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    def list_cases(self) -> list[InvestigationCase]:
        with self._lock:
            return sorted(self.cases.values(), key=lambda case: (case.created_at, case.case_id), reverse=True)

    def get_case(self, case_id: str) -> InvestigationCase | None:
        with self._lock:
            return self.cases.get(case_id)

    def list_alerts(self) -> list[Alert]:
        with self._lock:
            return sorted(self.alerts.values(), key=lambda alert: (alert.created_at, alert.alert_id), reverse=True)

    def transition(self, case_id: str, update: StatusUpdate) -> InvestigationCase:
        with self._lock:
            case = self.cases[case_id]
            if update.status not in ALLOWED_TRANSITIONS[case.status]:
                raise ValueError(f"invalid case transition {case.status.value}->{update.status.value}")
            previous = case.status.value
            case.status = update.status
            case.updated_at = _now()
            event = {"timestamp": case.updated_at, "action": "status_change", "actor": mask_identifier(update.actor), "previous": previous, "new": update.status.value, "reason": update.reason}
            case.history.append(event)
            if update.status in {CaseStatus.CONFIRMED_ABUSE, CaseStatus.LEGITIMATE}:
                case.analyst_disposition = AnalystDecision(actor=mask_identifier(update.actor), decision=update.status.value, reason=update.reason, timestamp=case.updated_at)
            self._audit("status_change", case_id, update.actor, previous=previous, new=update.status.value, reason=update.reason)
            return case

    def add_note(self, case_id: str, note: NoteCreate) -> InvestigationCase:
        with self._lock:
            case = self.cases[case_id]
            safe_note = note.note.replace("\n", " ")
            case.analyst_notes.append(safe_note)
            case.updated_at = _now()
            case.history.append({"timestamp": case.updated_at, "action": "note_added", "actor": mask_identifier(note.actor)})
            self._audit("note_added", case_id, note.actor)
            return case

    def public_case(self, case_id: str) -> dict[str, Any]:
        case = self.get_case(case_id)
        if case is None:
            raise KeyError(case_id)
        return _public_case(case)


class CaseManager:
    """Converts shadow alerts to cases using only observable graph evidence."""

    def __init__(self, repository: InMemoryCaseRepository | None = None):
        self.repository = repository or InMemoryCaseRepository()

    def ingest_prediction(self, payload: Any, response: Any, history: list[dict[str, Any]] | None = None) -> Alert | None:
        if response.fallback_applied or response.calibrated_score < R1_THRESHOLD:
            return None
        alert_id = _stable("ALERT", payload.order_id)
        if alert_id in self.repository.alerts:
            return self.repository.alerts[alert_id]
        entities = {field: str(getattr(payload, field, "")) for field in ENTITY_FIELDS if str(getattr(payload, field, ""))}
        now = response.timestamp or _now()
        previous = history or []
        evidence = self._evidence(payload, response, entities, previous, now)
        alert = Alert(alert_id=alert_id, order_id=payload.order_id, customer_id=payload.customer_id, risk_score=response.calibrated_score, created_at=now, correlation_id=response.correlation_id, entity_keys=entities, amount=float(payload.amount), evidence=evidence)
        self.repository.alerts[alert_id] = alert
        self._consolidate(alert, previous)
        return alert

    def _evidence(self, payload: Any, response: Any, entities: dict[str, str], history: list[dict[str, Any]], now: str) -> list[EvidenceItem]:
        evidence = [EvidenceItem(evidence_id=_stable("EVID", payload.order_id + ":threshold"), evidence_type="score_threshold", description="Observed calibrated risk score crossed the review threshold.", value=round(float(response.calibrated_score), 6), comparison=f">= {R1_THRESHOLD:.2f}", timestamp=now, window="event", source_features=[])]
        for field, value in sorted(entities.items()):
            peers = {str(row.get("customer_id")) for row in history if str(row.get(field, "")) == value and str(row.get("customer_id")) != payload.customer_id}
            if peers:
                evidence.append(EvidenceItem(evidence_id=_stable("EVID", payload.order_id + field), evidence_type="shared_entity", description=f"Observed {len(peers)} other customer(s) sharing the same {field} in prior state.", value=len(peers), comparison="prior observable events", timestamp=now, window="30d", entity_type=field, source_features=[f"graph_{field}_customer_count"]))
        return evidence

    def _related(self, alert: Alert, case: InvestigationCase) -> bool:
        if alert.customer_id in case.related_customers:
            return True
        existing_alerts = [
            candidate for candidate in self.repository.alerts.values()
            if candidate.order_id in case.related_orders
        ]
        for existing in existing_alerts:
            for field in ENTITY_FIELDS:
                left = alert.entity_keys.get(field)
                right = existing.entity_keys.get(field)
                if left and right and left == right:
                    return True
        return False

    def _consolidate(self, alert: Alert, history: list[dict[str, Any]]) -> InvestigationCase:
        candidates = [case for case in self.repository.cases.values() if self._related(alert, case)]
        if candidates:
            case = sorted(candidates, key=lambda item: item.case_id)[0]
            case.related_customers = sorted(set(case.related_customers + [alert.customer_id]))
            case.related_orders = sorted(set(case.related_orders + [alert.order_id]))
            case.alert_count = len(case.related_orders)
            case.risk_score = max(case.risk_score, alert.risk_score)
            case.estimated_exposure = round(case.estimated_exposure + alert.amount, 2)
            case.evidence.extend(alert.evidence)
            case.updated_at = _now()
            case.history.append({"timestamp": case.updated_at, "action": "alert_consolidated", "alert_id": alert.alert_id})
            case.severity = severity_for(case.risk_score, case.alert_count, len(case.related_customers), len(alert.entity_keys))
            return case
        now = alert.created_at
        case_id = _stable("CASE", alert.alert_id)
        case = InvestigationCase(case_id=case_id, risk_score=alert.risk_score, created_at=now, updated_at=now, primary_customer=alert.customer_id, related_customers=[alert.customer_id], related_orders=[alert.order_id], alert_count=1, estimated_exposure=alert.amount, severity=severity_for(alert.risk_score, 1, 1, len(alert.entity_keys)), evidence=list(alert.evidence), timeline=[TimelineEvent(event_id=_stable("TIME", alert.alert_id), timestamp=now, event_type="risk_threshold_crossed", description="Observed risk score crossed the review threshold.")], history=[{"timestamp": now, "action": "case_created", "actor": "system"}])
        self.repository.cases[case_id] = case
        self.repository._audit("case_created", case_id)
        return case

    def get_public_cases(self, status: CaseStatus | None = None, severity: Severity | None = None, min_risk: float | None = None) -> list[dict[str, Any]]:
        cases = self.repository.list_cases()
        return [_public_case(case) for case in cases if (status is None or case.status == status) and (severity is None or case.severity == severity) and (min_risk is None or case.risk_score >= min_risk)]

    def public_alerts(self, min_risk: float | None = None) -> list[dict[str, Any]]:
        result = []
        for alert in self.repository.list_alerts():
            data = alert.model_dump(mode="json", exclude={"entity_keys"})
            data["order_id"] = mask_identifier(alert.order_id)
            data["customer_id"] = mask_identifier(alert.customer_id)
            result.append(data)
        return [item for item in result if min_risk is None or item["risk_score"] >= min_risk]

    def graph(self, case_id: str) -> dict[str, Any]:
        case = self.repository.get_case(case_id)
        if case is None:
            raise KeyError(case_id)
        nodes: dict[str, dict[str, Any]] = {}
        edges: dict[tuple[str, str, str], RingConnection] = {}
        for order_id in case.related_orders:
            alert = next((a for a in self.repository.alerts.values() if a.order_id == order_id), None)
            if not alert:
                continue
            customer = mask_identifier(alert.customer_id); order = mask_identifier(alert.order_id)
            nodes.setdefault(customer, {"id": customer, "type": "customer", "risk_metadata": {"risk_score": alert.risk_score}})
            nodes.setdefault(order, {"id": order, "type": "order", "risk_metadata": {"risk_score": alert.risk_score}})
            key = (customer, order, "customer-placed-order")
            edges[key] = RingConnection(source=customer, target=order, relationship=key[2])
            for field, value in alert.entity_keys.items():
                entity = mask_identifier(f"{field}:{value}"); nodes.setdefault(entity, {"id": entity, "type": field, "risk_metadata": {}})
                key = (customer, entity, f"customer-used-{field.removesuffix('_id')}")
                edges[key] = RingConnection(source=customer, target=entity, relationship=key[2])
        return {"case_id": mask_identifier(case_id), "nodes": sorted(nodes.values(), key=lambda x: x["id"]), "edges": [edge.model_dump(mode="json") for edge in sorted(edges.values(), key=lambda x: (x.source, x.target, x.relationship))]}

    def timeline(self, case_id: str) -> list[dict[str, Any]]:
        case = self.repository.get_case(case_id)
        if case is None:
            raise KeyError(case_id)
        return [event.model_dump(mode="json") for event in sorted(case.timeline, key=lambda event: (event.timestamp, event.event_id))]

    def evidence(self, case_id: str) -> list[dict[str, Any]]:
        case = self.repository.get_case(case_id)
        if case is None:
            raise KeyError(case_id)
        return [item.model_dump(mode="json") for item in sorted(case.evidence, key=lambda item: item.evidence_id)]
