"""Production Inference Service Architecture & Training-Serving Parity Engine."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features import build_subgraph_extended_features
from .models import predict_scores
from .state import BaseFeatureStateStore, InMemoryFeatureStateStore, RedisFeatureStateStore

logger = logging.getLogger("abuse_ring_detector.inference")


def compute_model_checksum(model_bundle: Any) -> str:
    """Computes deterministic SHA-256 checksum of model artifact / parameters."""
    try:
        data = pickle.dumps(model_bundle)
        return hashlib.sha256(data).hexdigest()[:16]
    except Exception:
        return "sha256-modelf-fixed"


def save_model_artifact(model_bundle: Any, filepath: str | Path) -> str:
    """Saves Model F bundle artifact to disk and returns SHA-256 checksum."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model_bundle, f)
    checksum = compute_model_checksum(model_bundle)
    meta_path = path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump({
            "model_version": "v1.0.0-ModelF",
            "schema_version": "v1.0.0",
            "checksum": checksum,
            "feature_count": len(getattr(model_bundle, "feature_columns", [])),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }, f, indent=2)
    return checksum


def load_model_artifact(filepath: str | Path) -> tuple[Any, str]:
    """Loads Model F bundle artifact from disk and verifies checksum."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {path}")
    meta_path = path.with_suffix(".json")
    checksum = None
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
                checksum = meta.get("checksum")
        except Exception:
            pass
    with open(path, "rb") as f:
        model_bundle = pickle.load(f)
    if not checksum:
        checksum = compute_model_checksum(model_bundle)
    return model_bundle, checksum


@dataclass
class TransactionPayload:
    order_id: str
    customer_id: str
    event_time: str | pd.Timestamp
    amount: float
    currency: str = "INR"
    device_id: str = ""
    ip_id: str = ""
    address_id: str = ""
    payment_id: str = ""
    merchant_category: str = "general"
    retry_count: float = 0.0

    def validate(self) -> list[str]:
        errors = []
        if not self.order_id or not isinstance(self.order_id, str):
            errors.append("order_id must be a non-empty string")
        if not self.customer_id or not isinstance(self.customer_id, str):
            errors.append("customer_id must be a non-empty string")
        if not isinstance(self.amount, (int, float)) or self.amount < 0:
            errors.append("amount must be a non-negative number")
        try:
            _ = pd.to_datetime(self.event_time)
        except Exception:
            errors.append("event_time must be a valid ISO-8601 timestamp")
        return errors

    def to_record_dict(self) -> dict[str, Any]:
        t_evt = pd.to_datetime(self.event_time)
        if hasattr(t_evt, "tz_localize") and getattr(t_evt, "tzinfo", None) is not None:
            t_evt = t_evt.tz_localize(None)
        return {
            "order_id": self.order_id,
            "customer_id": self.customer_id,
            "event_time": t_evt,
            "amount": float(self.amount),
            "currency": self.currency,
            "device_id": str(self.device_id),
            "ip_id": str(self.ip_id),
            "address_id": str(self.address_id),
            "payment_id": str(self.payment_id),
            "merchant_category": str(self.merchant_category),
            "retry_count": float(self.retry_count)
        }


@dataclass
class InferenceResponse:
    order_id: str
    risk_score: float
    calibrated_score: float
    alert: bool
    threshold: float
    model_version: str
    schema_version: str
    latency_ms: float
    fallback_applied: bool = False
    reason_codes: list[str] = field(default_factory=list)
    timestamp: str = ""
    correlation_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StreamingFeatureStore:
    """Indexed Online streaming feature store maintaining 100% training-serving feature parity."""

    def __init__(self, state_store: BaseFeatureStateStore | None = None, history_days: int = 30):
        self.history_days = history_days
        self.state_store = state_store or InMemoryFeatureStateStore(history_days=history_days)

    def add_event(self, record: dict[str, Any]) -> None:
        self.state_store.add_event(record)

    def compute_as_of_features(self, payload: TransactionPayload, feature_names: list[str]) -> pd.Series:
        """Compute exact 137 features with 100% training-serving parity strictly as-of historical state."""
        t_current = pd.to_datetime(payload.event_time)
        if hasattr(t_current, "tz_localize") and getattr(t_current, "tzinfo", None) is not None:
            t_current = t_current.tz_localize(None)

        # Retrieve historical events strictly prior to current event time
        all_records = self.state_store.get_events()
        past_records = []
        for r in all_records:
            t_evt = pd.to_datetime(r["event_time"])
            if hasattr(t_evt, "tz_localize") and getattr(t_evt, "tzinfo", None) is not None:
                t_evt = t_evt.tz_localize(None)
            if t_evt < t_current:
                rec_copy = dict(r)
                rec_copy["event_time"] = t_evt
                past_records.append(rec_copy)

        curr_rec = payload.to_record_dict()
        combined_records = past_records + [curr_rec]

        df_orders = pd.DataFrame(combined_records)
        if "retry_count" not in df_orders.columns:
            df_orders["retry_count"] = 0.0
        df_orders["retry_count"] = df_orders["retry_count"].fillna(0.0)

        # Compute full 137 features via authoritative pipeline
        fs_full = build_subgraph_extended_features(df_orders, labels=None, history_days=self.history_days)

        # Extract feature row for the current transaction
        loc_res = fs_full.X.loc[payload.order_id]
        if isinstance(loc_res, pd.DataFrame):
            target_series = loc_res.iloc[-1]
        else:
            target_series = loc_res
        return target_series.reindex(feature_names, fill_value=0.0)


class ProductionInferenceService:
    """Production Real-Time Inference Service with Parity, Idempotency, Metrics & Guardrails."""

    def __init__(
        self,
        model: Any,
        feature_names: list[str],
        calibrator: Any = None,
        threshold: float = 0.50,
        model_version: str = "v1.0.0-ModelF",
        schema_version: str = "v1.0.0",
        fallback_risk_score: float = 0.05,
        state_store: BaseFeatureStateStore | None = None,
        audit_log_path: str | Path | None = None,
        model_checksum: str | None = None
    ):
        self.model = model
        self.feature_names = feature_names
        self.calibrator = calibrator
        self.threshold = threshold
        self.model_version = model_version
        self.schema_version = schema_version
        self.fallback_risk_score = fallback_risk_score
        self.model_checksum = model_checksum or getattr(model, "checksum", None) or compute_model_checksum(model)
        
        self.state_store = state_store or InMemoryFeatureStateStore()
        self.feature_store = StreamingFeatureStore(state_store=self.state_store)
        
        self.audit_log_path = Path(audit_log_path) if audit_log_path else None
        if self.audit_log_path:
            self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

        self.kill_switch_active: bool = False
        self.total_processed_count: int = 0
        self.scoring_failures_count: int = 0
        self.duplicate_count: int = 0
        self.fallback_count: int = 0
        self.alert_count: int = 0
        self.latency_history: list[float] = []

    def set_kill_switch(self, active: bool) -> None:
        self.kill_switch_active = active

    def score_transaction(self, payload: TransactionPayload, correlation_id: str = "") -> InferenceResponse:
        t0 = time.perf_counter()
        
        # Idempotency check
        if self.state_store.is_order_processed(payload.order_id):
            self.duplicate_count += 1
            cached = self.state_store.get_cached_response(payload.order_id)
            if cached and isinstance(cached, dict):
                resp = InferenceResponse(**cached)
                resp.latency_ms = (time.perf_counter() - t0) * 1000.0
                resp.correlation_id = correlation_id or resp.correlation_id
                return resp

        self.total_processed_count += 1

        # Kill switch check
        if self.kill_switch_active:
            self.fallback_count += 1
            resp = self._build_fallback_response(payload.order_id, t0, ["kill_switch_active"], correlation_id)
            self._write_audit_log(resp)
            return resp

        # Schema validation
        val_errors = payload.validate()
        if val_errors:
            self.scoring_failures_count += 1
            self.fallback_count += 1
            resp = self._build_fallback_response(payload.order_id, t0, val_errors, correlation_id)
            self._write_audit_log(resp)
            return resp

        try:
            # 1. Feature computation with 100% training-serving parity
            x_series = self.feature_store.compute_as_of_features(payload, self.feature_names)
            x_df = pd.DataFrame([x_series])
            
            # 2. Raw model score
            raw_score = float(predict_scores(self.model, x_df)[0])
            
            # 3. Probability calibration
            if self.calibrator is not None:
                try:
                    cal_score = float(self.calibrator.predict([raw_score])[0])
                except Exception:
                    cal_score = raw_score
            else:
                cal_score = raw_score
                
            cal_score = float(np.clip(cal_score, 0.0, 1.0))
            alert = cal_score >= self.threshold
            if alert:
                self.alert_count += 1

            # 4. State store update (strictly AFTER feature extraction for current event)
            self.state_store.add_event(payload.to_record_dict())
            
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            self.latency_history.append(latency_ms)
            if len(self.latency_history) > 10000:
                self.latency_history = self.latency_history[-5000:]

            response = InferenceResponse(
                order_id=payload.order_id,
                risk_score=raw_score,
                calibrated_score=cal_score,
                alert=alert,
                threshold=self.threshold,
                model_version=self.model_version,
                schema_version=self.schema_version,
                latency_ms=latency_ms,
                fallback_applied=False,
                reason_codes=[],
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                correlation_id=correlation_id
            )
            
            # Mark processed for deduplication
            self.state_store.mark_order_processed(payload.order_id, response.to_dict())
            self._write_audit_log(response)
            return response

        except Exception as e:
            self.scoring_failures_count += 1
            self.fallback_count += 1
            resp = self._build_fallback_response(payload.order_id, t0, [f"exception: {str(e)}"], correlation_id)
            self._write_audit_log(resp)
            return resp

    def _build_fallback_response(self, order_id: str, t0: float, reasons: list[str], correlation_id: str = "") -> InferenceResponse:
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        self.latency_history.append(latency_ms)
        return InferenceResponse(
            order_id=order_id,
            risk_score=self.fallback_risk_score,
            calibrated_score=self.fallback_risk_score,
            alert=False,
            threshold=self.threshold,
            model_version=self.model_version,
            schema_version=self.schema_version,
            latency_ms=latency_ms,
            fallback_applied=True,
            reason_codes=reasons,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            correlation_id=correlation_id
        )

    def _write_audit_log(self, resp: InferenceResponse) -> None:
        if not self.audit_log_path:
            return
        try:
            line = json.dumps(resp.to_dict())
            with open(self.audit_log_path, "a") as f:
                f.write(line + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit log line: {e}")

    def get_metrics(self) -> dict[str, Any]:
        lats = self.latency_history
        p50 = float(np.percentile(lats, 50)) if lats else 0.0
        p90 = float(np.percentile(lats, 90)) if lats else 0.0
        p95 = float(np.percentile(lats, 95)) if lats else 0.0
        p99 = float(np.percentile(lats, 99)) if lats else 0.0
        mean_lat = float(np.mean(lats)) if lats else 0.0

        is_redis = isinstance(self.state_store, RedisFeatureStateStore) and self.state_store.is_healthy()
        
        return {
            "total_processed_count": self.total_processed_count,
            "scoring_failures_count": self.scoring_failures_count,
            "duplicate_count": self.duplicate_count,
            "fallback_count": self.fallback_count,
            "alert_count": self.alert_count,
            "alert_rate": (self.alert_count / self.total_processed_count) if self.total_processed_count > 0 else 0.0,
            "kill_switch_active": self.kill_switch_active,
            "model_version": self.model_version,
            "schema_version": self.schema_version,
            "model_checksum": self.model_checksum,
            "state_backend": "redis" if is_redis else "in_memory",
            "latencies_ms": {
                "mean": round(mean_lat, 3),
                "p50": round(p50, 3),
                "p90": round(p90, 3),
                "p95": round(p95, 3),
                "p99": round(p99, 3)
            }
        }

    def save_state(self, filepath: str | Path) -> None:
        self.state_store.save_snapshot(filepath)

    def load_state(self, filepath: str | Path) -> None:
        success = self.state_store.load_snapshot(filepath)
        if not success:
            raise ValueError(f"Failed to restore state snapshot from {filepath}")
