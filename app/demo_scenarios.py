"""Stable, increasing-timestamp demo payloads for the real R1 API."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

SCENARIOS = ["Shared-device ring", "Shared-address ring", "Mixed multi-entity", "Behavioral coordination", "Legitimate high-connectivity"]


def scenario_payloads(name: str, run_id: str = "demo-001") -> list[dict]:
    base = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    prefix = {"Shared-device ring": "DEVICE", "Shared-address ring": "ADDRESS", "Mixed multi-entity": "MIXED", "Behavioral coordination": "BURST", "Legitimate high-connectivity": "HOUSEHOLD"}.get(name, "DEMO")
    rows = []
    for i in range(8):
        entity = {
            "device_id": f"{prefix}_DEVICE_SHARED" if name != "Legitimate high-connectivity" else "HOUSEHOLD_DEVICE",
            "address_id": f"{prefix}_ADDRESS_SHARED" if name in {"Shared-address ring", "Mixed multi-entity"} else ("HOUSEHOLD_ADDRESS" if name == "Legitimate high-connectivity" else f"{prefix}_ADDRESS_{i}"),
            "ip_id": f"{prefix}_IP_SHARED" if name == "Mixed multi-entity" else f"{prefix}_IP_{i % 2}",
            "payment_id": f"{prefix}_PAYMENT_SHARED" if name == "Mixed multi-entity" else f"{prefix}_PAYMENT_{i}",
        }
        rows.append({
            "order_id": f"DEMO_{run_id}_{prefix}_{i:02d}",
            "customer_id": f"DEMO_{run_id}_{prefix}_CUSTOMER_{i:02d}",
            "event_time": (base + timedelta(minutes=i * 7)).isoformat().replace("+00:00", "Z"),
            "amount": float(95 + i * 17),
            **entity,
        })
    return rows
