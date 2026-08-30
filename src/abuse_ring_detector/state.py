"""Production Persistent Feature State Abstraction Engine.

Supports Local In-Memory and Redis state backends with concurrency safety,
atomic operations, state versioning, snapshot restoration, and graceful fallback.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("abuse_ring_detector.state")


class BaseFeatureStateStore(ABC):
    """Abstract Base Class for Production Feature State Storage."""

    @abstractmethod
    def add_event(self, record: dict[str, Any]) -> None:
        """Atomically persist a transaction event into the streaming feature state."""
        pass

    @abstractmethod
    def get_events(self, customer_id: str | None = None) -> list[dict[str, Any]]:
        """Retrieve all events, optionally filtered by customer_id."""
        pass

    @abstractmethod
    def is_order_processed(self, order_id: str) -> bool:
        """Check if an order_id has already been processed (idempotency check)."""
        pass

    @abstractmethod
    def mark_order_processed(self, order_id: str, response_payload: dict[str, Any] | None = None) -> None:
        """Mark an order_id as processed and store cached response for deduplication."""
        pass

    @abstractmethod
    def get_cached_response(self, order_id: str) -> dict[str, Any] | None:
        """Retrieve cached response for a previously processed order_id."""
        pass

    @abstractmethod
    def is_healthy(self) -> bool:
        """Check health status of state backend."""
        pass

    @abstractmethod
    def save_snapshot(self, filepath: str | Path) -> None:
        """Save full snapshot of state to disk."""
        pass

    @abstractmethod
    def load_snapshot(self, filepath: str | Path) -> None:
        """Restore full state snapshot from disk."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all stored state."""
        pass


class InMemoryFeatureStateStore(BaseFeatureStateStore):
    """Thread-safe In-Memory Feature State Store with indexing and atomic operations."""

    def __init__(self, history_days: int = 30):
        self.history_days = history_days
        self._lock = threading.RLock()
        self.records: list[dict[str, Any]] = []
        self.processed_orders: dict[str, dict[str, Any]] = {}
        self.customer_records: dict[str, list[dict[str, Any]]] = {}
        self.entity_records: dict[tuple[str, str], list[dict[str, Any]]] = {}

    def add_event(self, record: dict[str, Any]) -> None:
        with self._lock:
            order_id = record["order_id"]
            rec_copy = dict(record)
            if isinstance(rec_copy.get("event_time"), str):
                rec_copy["event_time"] = pd.to_datetime(rec_copy["event_time"])

            self.records.append(rec_copy)

            cust_id = rec_copy["customer_id"]
            if cust_id not in self.customer_records:
                self.customer_records[cust_id] = []
            self.customer_records[cust_id].append(rec_copy)

            for col in ("device_id", "ip_id", "address_id", "payment_id"):
                val = str(rec_copy.get(col, ""))
                if val:
                    key = (col, val)
                    if key not in self.entity_records:
                        self.entity_records[key] = []
                    self.entity_records[key].append(rec_copy)

    def get_events(self, customer_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if customer_id is not None:
                return list(self.customer_records.get(customer_id, []))
            return list(self.records)

    def is_order_processed(self, order_id: str) -> bool:
        with self._lock:
            return order_id in self.processed_orders

    def mark_order_processed(self, order_id: str, response_payload: dict[str, Any] | None = None) -> None:
        with self._lock:
            self.processed_orders[order_id] = response_payload or {"processed": True}

    def get_cached_response(self, order_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self.processed_orders.get(order_id)

    def is_healthy(self) -> bool:
        return True

    def save_snapshot(self, filepath: str | Path) -> None:
        with self._lock:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            export_records = []
            for r in self.records:
                r_dict = dict(r)
                if isinstance(r_dict["event_time"], (pd.Timestamp, float, int)):
                    r_dict["event_time"] = str(r_dict["event_time"])
                export_records.append(r_dict)

            state_data = {
                "version": "v1.0.0",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "records": export_records,
                "processed_orders": self.processed_orders
            }
            with open(path, "w") as f:
                json.dump(state_data, f, indent=2)

    def load_snapshot(self, filepath: str | Path) -> bool:
        with self._lock:
            path = Path(filepath)
            if not path.exists():
                logger.warning(f"State snapshot not found at {path}")
                return False

            try:
                with open(path, "r") as f:
                    state_data = json.load(f)

                self.clear()
                for r in state_data.get("records", []):
                    r["event_time"] = pd.to_datetime(r["event_time"])
                    self.add_event(r)

                self.processed_orders = state_data.get("processed_orders", {})
                return True
            except Exception as e:
                logger.error(f"Failed to load state snapshot from {path}: {e}")
                return False

    def clear(self) -> None:
        with self._lock:
            self.records.clear()
            self.processed_orders.clear()
            self.customer_records.clear()
            self.entity_records.clear()


class RedisFeatureStateStore(BaseFeatureStateStore):
    """Production Redis/KeyDB-Compatible Persistent Feature State Store with automatic local fallback."""

    def __init__(self, redis_url: str | None = None, history_days: int = 30, key_prefix: str = "ard:v1"):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.history_days = history_days
        self.key_prefix = key_prefix
        self.redis_client: Any = None
        self.local_fallback = InMemoryFeatureStateStore(history_days=history_days)
        self.is_connected = False
        self._connect()

    def _connect(self) -> None:
        try:
            import redis
            r = redis.Redis.from_url(self.redis_url, socket_timeout=1.0, socket_connect_timeout=1.0, decode_responses=True)
            r.ping()
            self.redis_client = r
            self.is_connected = True
        except Exception:
            self.redis_client = None
            self.is_connected = False

    def add_event(self, record: dict[str, Any]) -> None:
        # Always maintain local store for fast as-of feature computations
        self.local_fallback.add_event(record)

        if not self.is_connected or self.redis_client is None:
            return

        try:
            order_id = record["order_id"]
            cust_id = record["customer_id"]
            event_time_str = str(record["event_time"])

            payload = {
                "order_id": order_id,
                "customer_id": cust_id,
                "event_time": event_time_str,
                "amount": float(record["amount"]),
                "device_id": str(record.get("device_id", "")),
                "ip_id": str(record.get("ip_id", "")),
                "address_id": str(record.get("address_id", "")),
                "payment_id": str(record.get("payment_id", "")),
                "merchant_category": str(record.get("merchant_category", "general"))
            }

            p_json = json.dumps(payload)
            ttl_seconds = self.history_days * 86400

            pipe = self.redis_client.pipeline()
            pipe.set(f"{self.key_prefix}:order:{order_id}", p_json, ex=ttl_seconds)
            pipe.rpush(f"{self.key_prefix}:cust:{cust_id}", p_json)
            pipe.expire(f"{self.key_prefix}:cust:{cust_id}", ttl_seconds)

            for col in ("device_id", "ip_id", "address_id", "payment_id"):
                val = str(record.get(col, ""))
                if val:
                    key = f"{self.key_prefix}:entity:{col}:{val}"
                    pipe.rpush(key, p_json)
                    pipe.expire(key, ttl_seconds)

            pipe.execute()
        except Exception:
            self.is_connected = False

    def get_events(self, customer_id: str | None = None) -> list[dict[str, Any]]:
        return self.local_fallback.get_events(customer_id)

    def is_order_processed(self, order_id: str) -> bool:
        if self.is_connected and self.redis_client is not None:
            try:
                return bool(self.redis_client.exists(f"{self.key_prefix}:dedup:{order_id}"))
            except Exception:
                self.is_connected = False
        return self.local_fallback.is_order_processed(order_id)

    def mark_order_processed(self, order_id: str, response_payload: dict[str, Any] | None = None) -> None:
        self.local_fallback.mark_order_processed(order_id, response_payload)
        if self.is_connected and self.redis_client is not None:
            try:
                val = json.dumps(response_payload or {"processed": True})
                ttl_seconds = self.history_days * 86400
                self.redis_client.set(f"{self.key_prefix}:dedup:{order_id}", val, ex=ttl_seconds)
            except Exception:
                self.is_connected = False

    def get_cached_response(self, order_id: str) -> dict[str, Any] | None:
        if self.is_connected and self.redis_client is not None:
            try:
                val = self.redis_client.get(f"{self.key_prefix}:dedup:{order_id}")
                if val:
                    return json.loads(val)
            except Exception:
                self.is_connected = False
        return self.local_fallback.get_cached_response(order_id)

    def is_healthy(self) -> bool:
        if self.redis_client is None:
            return False
        try:
            return bool(self.redis_client.ping())
        except Exception:
            return False

    def save_snapshot(self, filepath: str | Path) -> None:
        self.local_fallback.save_snapshot(filepath)

    def load_snapshot(self, filepath: str | Path) -> None:
        self.local_fallback.load_snapshot(filepath)

    def clear(self) -> None:
        self.local_fallback.clear()
        if self.is_connected and self.redis_client is not None:
            try:
                keys = self.redis_client.keys(f"{self.key_prefix}:*")
                if keys:
                    self.redis_client.delete(*keys)
            except Exception:
                self.is_connected = False
