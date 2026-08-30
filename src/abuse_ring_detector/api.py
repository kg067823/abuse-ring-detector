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

from .config import load_config
from .features import build_subgraph_extended_features
from .inference import (
    InferenceResponse,
    ProductionInferenceService,
    TransactionPayload,
    compute_model_checksum,
    load_model_artifact,
)
from .models import fit_model
from .splits import split_by_time
from .state import InMemoryFeatureStateStore, RedisFeatureStateStore
from .synthetic import generate_ecosystem

logger = logging.getLogger("abuse_ring_detector.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Global Service Singleton
_inference_service: ProductionInferenceService | None = None


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

    model = None
    feature_names = []
    model_version = "v1.0.0-ModelF"
    schema_version = "v1.0.0"

    # Attempt to load model from disk if provided or default path exists
    candidate_paths = [
        model_path,
        Path("artifacts/model_f_bundle.pkl"),
        Path("artifacts/full-run/model_f_bundle.pkl")
    ]
    
    for p in candidate_paths:
        if p and Path(p).exists():
            try:
                bundle, checksum = load_model_artifact(p)
                model = bundle
                feature_names = getattr(bundle, "feature_columns", [])
                logger.info(f"Loaded frozen Model F artifact from {p} (checksum={checksum})")
                break
            except Exception as e:
                logger.warning(f"Failed to load model artifact from {p}: {e}")

    # Fallback to generating baseline synthetic model if no artifact file is on disk
    if model is None:
        logger.info("Initializing baseline Model F champion model from default config...")
        config = load_config("configs/default.yaml")
        dataset = generate_ecosystem(config)
        split = split_by_time(dataset.orders, config.split["train"], config.split["validation"])
        fs_all = build_subgraph_extended_features(dataset.orders, dataset.labels, config.graph["history_days"])
        feature_names = fs_all.X.columns.tolist()
        train_ids = pd.Index(split.train["order_id"]) if hasattr(split.train, "order_id") else split.train.index
        model = fit_model(fs_all.X.loc[train_ids], fs_all.y.loc[train_ids], config.model["backend"], config.seed)
        model.feature_columns = feature_names

    # Determine state backend
    redis_uri = redis_url or os.getenv("REDIS_URL")
    if redis_uri:
        state_store = RedisFeatureStateStore(redis_url=redis_uri)
        if state_store.is_healthy():
            logger.info("Connected to Redis persistent feature state backend")
        else:
            logger.warning("Redis backend unavailable; falling back to thread-safe local in-memory store")
            state_store = InMemoryFeatureStateStore()
    else:
        state_store = InMemoryFeatureStateStore()

    _inference_service = ProductionInferenceService(
        model=model,
        feature_names=feature_names,
        threshold=0.50,
        model_version=model_version,
        schema_version=schema_version,
        state_store=state_store,
        audit_log_path=audit_log_path
    )

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
    """Readiness probe checking model loading and state store health."""
    service = get_service()
    if service.model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model artifact not loaded")
    
    backend_healthy = service.state_store.is_healthy()
    return {
        "status": "ready",
        "model_loaded": True,
        "feature_count": len(service.feature_names),
        "state_backend": "redis" if isinstance(service.state_store, RedisFeatureStateStore) else "in_memory",
        "state_backend_healthy": backend_healthy
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
    return resp


@app.get("/metrics", status_code=status.HTTP_200_OK)
def get_metrics():
    """Retrieve operational and latency metrics in Prometheus exposition format."""
    service = get_service()
    m = service.get_metrics()

    lines = [
        "# HELP abuse_ring_detector_requests_total Total number of inference requests scored.",
        "# TYPE abuse_ring_detector_requests_total counter",
        f"abuse_ring_detector_requests_total {m.get('total_requests', 0)}",
        "# HELP abuse_ring_detector_fallback_total Total number of fallback responses served.",
        "# TYPE abuse_ring_detector_fallback_total counter",
        f"abuse_ring_detector_fallback_total {m.get('total_fallbacks', 0)}",
        "# HELP abuse_ring_detector_alerts_total Total number of high risk alerts triggered.",
        "# TYPE abuse_ring_detector_alerts_total counter",
        f"abuse_ring_detector_alerts_total {m.get('total_alerts', 0)}",
        "# HELP abuse_ring_detector_latency_seconds_bucket Latency histogram in seconds.",
        "# TYPE abuse_ring_detector_latency_seconds_bucket histogram",
        f"abuse_ring_detector_latency_seconds_bucket{{le=\"0.05\"}} {m.get('latency_p50_ms', 0)/1000.0}",
        f"abuse_ring_detector_latency_seconds_bucket{{le=\"0.10\"}} {m.get('latency_p95_ms', 0)/1000.0}",
        f"abuse_ring_detector_latency_seconds_bucket{{le=\"0.20\"}} {m.get('latency_p99_ms', 0)/1000.0}",
        f"abuse_ring_detector_latency_seconds_bucket{{le=\"+Inf\"}} {m.get('total_requests', 0)}"
    ]
    return Response(content="\n".join(lines) + "\n", media_type="text/plain")


@app.post("/v1/admin/kill-switch", status_code=status.HTTP_200_OK)
def toggle_kill_switch(req: KillSwitchRequest):
    """Dynamically activate or deactivate emergency scoring kill-switch."""
    service = get_service()
    service.set_kill_switch(req.active)
    logger.warning(f"Kill switch status updated to active={req.active}")
    return {
        "status": "updated",
        "kill_switch_active": req.active,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
    }
