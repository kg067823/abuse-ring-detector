"""Production FastAPI Inference Application for Model F Real-Time Abuse Detection.

Provides schema validation, health/readiness/liveness probes, correlation IDs,
emergency kill-switch administration, Prometheus/JSON metrics, and safe fallback handling.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .case_management import (
    CaseManager,
    CaseStatus,
    NoteCreate,
    Severity,
    StatusUpdate,
)
from .explain import explain_prediction, mask_identifier
from .inference import (
    R1_MODEL_VERSION,
    R1_THRESHOLD,
    InferenceResponse,
    ProductionInferenceService,
    TransactionPayload,
    load_model_artifact,
)
from .state import InMemoryFeatureStateStore, RedisFeatureStateStore

logger = logging.getLogger("abuse_ring_detector.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Global Service Singleton
_inference_service: ProductionInferenceService | None = None
_case_manager = CaseManager()


def set_service(service: ProductionInferenceService | None) -> None:
    """Set global inference service instance (used for custom injection in tests)."""
    global _inference_service
    _inference_service = service


def initialize_service(
    model_path: str | Path | None = None,
    redis_url: str | None = None,
    audit_log_path: str | Path = "logs/audit.jsonl"
) -> ProductionInferenceService:
    """Initialize or load frozen Model F inference service."""
    global _inference_service

    model_version = R1_MODEL_VERSION
    schema_version = "inference_contract_r1.v1"
    configured_path = model_path or os.getenv("MODEL_PATH") or "artifacts/model_f_r1_bundle.pkl"
    artifact_path = Path(configured_path)
    manifest_path = Path(os.getenv("MODEL_MANIFEST_PATH", "model_f_r1_manifest.json"))
    contract_path = Path(os.getenv("INFERENCE_CONTRACT_PATH", "inference_contract_r1.json"))

    # Production initialization is fail-closed: the frozen artifact must be
    # present and contract-compatible. Never train a replacement at startup.
    model, checksum = load_model_artifact(
        artifact_path,
        require_frozen_contract=True,
        manifest_path=manifest_path,
        contract_path=contract_path,
    )
    feature_names = list(getattr(model, "feature_columns", []))
    logger.info("Loaded verified frozen Model F artifact from %s (checksum=%s)", artifact_path, checksum)

    # Determine state backend
    redis_uri = redis_url or os.getenv("REDIS_URL")
    if redis_uri:
        state_store = RedisFeatureStateStore(redis_url=redis_uri)
        if state_store.is_healthy():
            logger.info("Connected to Redis persistent feature state backend")
        else:
            raise RuntimeError("configured Redis state backend is unavailable")
    else:
        # A production deployment must declare a shared state backend. The
        # in-memory store remains available only to direct unit-test injection.
        raise RuntimeError("REDIS_URL is required for production initialization")

    checksum = locals().get("checksum", None)
    _inference_service = ProductionInferenceService(
        model=model,
        feature_names=feature_names,
        calibrator=getattr(model, "calibrator", None),
        threshold=R1_THRESHOLD,
        model_version=model_version,
        schema_version=schema_version,
        state_store=state_store,
        audit_log_path=audit_log_path,
        model_checksum=checksum
    )

    # This milestone is shadow-only. Refuse unsafe production configuration.
    shadow_mode = os.getenv("SHADOW_MODE", "true").lower() in ("true", "1", "yes")
    enforce_decisions = os.getenv("ENFORCE_DECISIONS", "false").lower() in ("true", "1", "yes")
    if not shadow_mode or enforce_decisions:
        raise RuntimeError("unsafe decision configuration: require SHADOW_MODE=true and ENFORCE_DECISIONS=false")

    # Check for initial kill switch env flag
    if os.getenv("KILL_SWITCH", "").lower() in ("true", "1", "yes"):
        _inference_service.set_kill_switch(True)
        logger.warning("Kill switch ACTIVATED via environment variable!")

    return _inference_service


def get_service() -> ProductionInferenceService:
    global _inference_service
    if _inference_service is None:
        _inference_service = initialize_service()
    return _inference_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for graceful startup and shutdown."""
    logger.info("Starting AbuseRing Detector Inference Service...")
    # Tests and embedding applications may inject a verified service before
    # creating the client. Production startup still initializes strictly from
    # the configured frozen artifact.
    if _inference_service is None:
        initialize_service()
    yield
    logger.info("Shutting down AbuseRing Detector Inference Service...")
    # Flush or close state backend connections if needed
    service = get_service()
    if hasattr(service.state_store, "save_snapshot"):
        try:
            service.save_state("scratch/service_state_shutdown_backup.json")
        except Exception as e:
            logger.warning(f"Error saving shutdown snapshot: {e}")


app = FastAPI(
    title="AbuseRing Detector — Model F Production API",
    description="Real-time transaction fraud scoring service powered by 137-feature graph-temporal Model F",
    version="1.0.0",
    lifespan=lifespan
)


# --- Request/Response Pydantic Models ---

class TransactionApiPayload(BaseModel):
    order_id: str = Field(..., description="Unique order identifier")
    customer_id: str = Field(..., description="Unique customer account identifier")
    event_time: str = Field(..., description="ISO-8601 UTC timestamp of order")
    amount: float = Field(..., ge=0.0, description="Transaction amount in INR")
    currency: str = Field("INR", description="Transaction currency")
    device_id: str = Field("", description="Device identifier")
    ip_id: str = Field("", description="IP address identifier")
    address_id: str = Field("", description="Shipping address identifier")
    payment_id: str = Field("", description="Payment instrument identifier")
    merchant_category: str = Field("general", description="Merchant business category")
    retry_count: float = Field(0.0, description="Payment retry count")


class KillSwitchRequest(BaseModel):
    active: bool = Field(..., description="Enable or disable emergency kill switch")


class AdminAuthError(HTTPException):
    def __init__(self):
        super().__init__(status_code=401, detail="admin authentication required")


def require_admin_token(authorization: str | None) -> None:
    expected = os.getenv("ADMIN_KILL_SWITCH_TOKEN")
    if not expected or authorization != f"Bearer {expected}":
        raise AdminAuthError()


# --- Middlewares & Exception Handlers ---

@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    corr_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
    request.state.correlation_id = corr_id
    response: Response = await call_next(request)
    response.headers["X-Correlation-ID"] = corr_id
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    corr_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))
    logger.error(f"Unhandled exception during request processing [{corr_id}]: {exc}", exc_info=True)
    
    # Return safe fallback response
    service = get_service()
    fallback_resp = service._build_fallback_response(
        order_id="UNKNOWN",
        t0=time.perf_counter(),
        reasons=[f"unhandled_exception: {str(exc)}"],
        correlation_id=corr_id
    )
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=fallback_resp.to_dict(),
        headers={"X-Correlation-ID": corr_id}
    )


# --- Routes ---

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Liveness & health probe."""
    service = get_service()
    return {
        "status": "healthy",
        "service": "abuse-ring-detector",
        "model_version": service.model_version,
        "schema_version": service.schema_version,
        "kill_switch_active": service.kill_switch_active
    }


@app.get("/readiness", status_code=status.HTTP_200_OK)
def readiness_check():
    """Readiness probe checking verified model and state store health."""
    try:
        service = get_service()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="frozen model is not verified",
        )
    if service.model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model artifact not loaded")

    backend_healthy = service.state_store.is_healthy()
    if not backend_healthy:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="state backend unavailable")
    return {
        "status": "ready",
        "model_loaded": True,
        "feature_count": len(service.feature_names),
        "threshold": service.threshold,
        "model_checksum": service.model_checksum,
        "state_backend": "redis" if isinstance(service.state_store, RedisFeatureStateStore) else "in_memory",
        "state_backend_healthy": backend_healthy,
        "shadow_mode": True,
        "enforce_decisions": False,
    }


@app.get("/liveness", status_code=status.HTTP_200_OK)
def liveness_check():
    return {"status": "alive"}


@app.post("/v1/predict", response_model=InferenceResponse, status_code=status.HTTP_200_OK)
def predict_transaction(payload: TransactionApiPayload, request: Request, x_correlation_id: str | None = Header(None)):
    corr_id = x_correlation_id or getattr(request.state, "correlation_id", str(uuid.uuid4()))
    service = get_service()

    domain_payload = TransactionPayload(
        order_id=payload.order_id,
        customer_id=payload.customer_id,
        event_time=payload.event_time,
        amount=payload.amount,
        currency=payload.currency,
        device_id=payload.device_id,
        ip_id=payload.ip_id,
        address_id=payload.address_id,
        payment_id=payload.payment_id,
        merchant_category=payload.merchant_category,
        retry_count=payload.retry_count
    )

    resp = service.score_transaction(domain_payload, correlation_id=corr_id)
    history = service.state_store.get_events()
    _case_manager.ingest_prediction(domain_payload, resp, history)
    return resp


@app.post("/v1/explain", status_code=status.HTTP_200_OK)
def explain_transaction(payload: TransactionApiPayload, request: Request, x_correlation_id: str | None = Header(None)):
    """Return masked, non-causal observed signals for a shadow investigation."""
    corr_id = x_correlation_id or getattr(request.state, "correlation_id", str(uuid.uuid4()))
    service = get_service()
    domain_payload = TransactionPayload(
        order_id=payload.order_id, customer_id=payload.customer_id, event_time=payload.event_time,
        amount=payload.amount, currency=payload.currency, device_id=payload.device_id,
        ip_id=payload.ip_id, address_id=payload.address_id, payment_id=payload.payment_id,
        merchant_category=payload.merchant_category, retry_count=payload.retry_count,
    )
    if domain_payload.validate():
        raise HTTPException(status_code=422, detail="invalid transaction payload")
    feature_row = service.feature_store.compute_as_of_features(domain_payload, service.feature_names)
    response = service.score_transaction(domain_payload, correlation_id=corr_id)
    result = explain_prediction(feature_row, response.calibrated_score, model_version=response.model_version, model_checksum=service.model_checksum)
    result.update({
        "order_id": mask_identifier(payload.order_id),
        "customer_id": mask_identifier(payload.customer_id),
        "event_time": str(payload.event_time),
        "threshold": response.threshold,
        "shadow_alert": response.alert,
        "correlation_id": corr_id,
    })
    return result


@app.get("/v1/alerts")
def list_alerts(min_risk: float | None = None):
    return {"items": _case_manager.public_alerts(min_risk=min_risk), "demo_label": "DEMO / SYNTHETIC"}


@app.get("/v1/cases")
def list_cases(status_filter: CaseStatus | None = None, severity: Severity | None = None, min_risk: float | None = None):
    return {"items": _case_manager.get_public_cases(status=status_filter, severity=severity, min_risk=min_risk), "demo_label": "DEMO / SYNTHETIC"}


@app.get("/v1/cases/{case_id}")
def get_case(case_id: str):
    try:
        return _case_manager.repository.public_case(case_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="case not found")


@app.get("/v1/cases/{case_id}/graph")
def case_graph(case_id: str):
    try:
        return _case_manager.graph(case_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="case not found")


@app.get("/v1/cases/{case_id}/timeline")
def case_timeline(case_id: str):
    try:
        return {"items": _case_manager.timeline(case_id)}
    except KeyError:
        raise HTTPException(status_code=404, detail="case not found")


@app.get("/v1/cases/{case_id}/evidence")
def case_evidence(case_id: str):
    try:
        return {"items": _case_manager.evidence(case_id)}
    except KeyError:
        raise HTTPException(status_code=404, detail="case not found")


@app.patch("/v1/cases/{case_id}/status")
def update_case_status(case_id: str, update: StatusUpdate, authorization: str | None = Header(None)):
    require_admin_token(authorization)
    try:
        _case_manager.repository.transition(case_id, update)
        return _case_manager.repository.public_case(case_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="case not found")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.post("/v1/cases/{case_id}/notes")
def add_case_note(case_id: str, note: NoteCreate, authorization: str | None = Header(None)):
    require_admin_token(authorization)
    try:
        _case_manager.repository.add_note(case_id, note)
        return _case_manager.repository.public_case(case_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="case not found")


@app.get("/metrics", status_code=status.HTTP_200_OK)
def get_metrics():
    """Retrieve operational and latency metrics in Prometheus exposition format."""
    service = get_service()
    m = service.get_metrics()

    lines = [
        "# HELP abuse_ring_detector_requests_total Total number of inference requests scored.",
        "# TYPE abuse_ring_detector_requests_total counter",
        f"abuse_ring_detector_requests_total {m.get('total_processed_count', 0)}",
        "# HELP abuse_ring_detector_fallback_total Total number of fallback responses served.",
        "# TYPE abuse_ring_detector_fallback_total counter",
        f"abuse_ring_detector_fallback_total {m.get('fallback_count', 0)}",
        "# HELP abuse_ring_detector_alerts_total Total number of high risk alerts triggered.",
        "# TYPE abuse_ring_detector_alerts_total counter",
        f"abuse_ring_detector_alerts_total {m.get('alert_count', 0)}",
        "# HELP abuse_ring_detector_shadow_alerts_total High-risk shadow decisions.",
        "# TYPE abuse_ring_detector_shadow_alerts_total counter",
        f"abuse_ring_detector_shadow_alerts_total {m.get('alert_count', 0)}",
        "# HELP abuse_ring_detector_blocked_transactions_total Customer transactions blocked (must remain zero in shadow mode).",
        "# TYPE abuse_ring_detector_blocked_transactions_total counter",
        f"abuse_ring_detector_blocked_transactions_total {m.get('blocked_transactions', 0)}",
        "# HELP abuse_ring_detector_modified_transactions_total Customer transactions modified (must remain zero in shadow mode).",
        "# TYPE abuse_ring_detector_modified_transactions_total counter",
        f"abuse_ring_detector_modified_transactions_total {m.get('modified_transactions', 0)}",
        "# HELP abuse_ring_detector_shadow_mode Shadow mode enabled (1/0).",
        "# TYPE abuse_ring_detector_shadow_mode gauge",
        "abuse_ring_detector_shadow_mode 1",
        "# HELP abuse_ring_detector_enforcement_enabled Customer enforcement enabled (1/0).",
        "# TYPE abuse_ring_detector_enforcement_enabled gauge",
        "abuse_ring_detector_enforcement_enabled 0",
        "# HELP abuse_ring_detector_latency_seconds_bucket Latency histogram in seconds.",
        "# TYPE abuse_ring_detector_latency_seconds_bucket histogram",
        f"abuse_ring_detector_latency_seconds_bucket{{le=\"0.05\"}} {m.get('latencies_ms', {}).get('p50', 0)/1000.0}",
        f"abuse_ring_detector_latency_seconds_bucket{{le=\"0.10\"}} {m.get('latencies_ms', {}).get('p95', 0)/1000.0}",
        f"abuse_ring_detector_latency_seconds_bucket{{le=\"0.20\"}} {m.get('latencies_ms', {}).get('p99', 0)/1000.0}",
        f"abuse_ring_detector_latency_seconds_bucket{{le=\"+Inf\"}} {m.get('total_processed_count', 0)}"
    ]
    return Response(content="\n".join(lines) + "\n", media_type="text/plain")


@app.post("/v1/admin/kill-switch", status_code=status.HTTP_200_OK)
def toggle_kill_switch(req: KillSwitchRequest, authorization: str | None = Header(None)):
    """Dynamically activate or deactivate emergency scoring kill-switch."""
    require_admin_token(authorization)
    service = get_service()
    service.set_kill_switch(req.active)
    logger.warning("Kill switch status updated; active=%s", req.active)
    return {
        "status": "updated",
        "kill_switch_active": req.active,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
    }
