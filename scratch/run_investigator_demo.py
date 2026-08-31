"""Replay deterministic demo scenarios through the real R1 HTTP API."""
from __future__ import annotations

import argparse
import json
import os
import time
from urllib.request import Request, urlopen

BASE = os.getenv("ABUSERING_API_URL", "http://localhost:8000").rstrip("/")


def post(payload: dict) -> dict:
    req = Request(BASE + "/v1/predict", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", "X-Correlation-ID": payload["order_id"]})
    with urlopen(req, timeout=30) as response:
        return json.loads(response.read())


def scenario(name: str) -> list[dict]:
    base = int(time.time()) % 100000
    common = {"amount": 125.0, "event_time": "2026-01-01T10:00:00Z"}
    if name == "Shared-address ring":
        return [{**common, "order_id": f"DEMO_ADDR_{base}_{i}", "customer_id": f"DEMO_C_{i}", "address_id": "DEMO_ADDRESS_SHARED", "device_id": f"DEMO_DEVICE_{i}"} for i in range(4)]
    if name == "Mixed multi-entity":
        return [{**common, "order_id": f"DEMO_MIX_{base}_{i}", "customer_id": f"DEMO_C_{i}", "address_id": "DEMO_ADDRESS_SHARED", "device_id": f"DEMO_DEVICE_{i % 2}", "ip_id": "DEMO_IP_SHARED"} for i in range(5)]
    if name == "Behavioral coordination":
        return [{**common, "order_id": f"DEMO_BURST_{base}_{i}", "customer_id": "DEMO_C_{i % 3}", "device_id": "DEMO_DEVICE_BURST"} for i in range(6)]
    if name == "Legitimate high-connectivity":
        return [{**common, "order_id": f"DEMO_LEGIT_{base}_{i}", "customer_id": f"DEMO_FAMILY_{i}", "address_id": "DEMO_HOUSEHOLD", "device_id": "DEMO_HOUSEHOLD_DEVICE"} for i in range(4)]
    return [{**common, "order_id": f"DEMO_DEVICE_{base}_{i}", "customer_id": f"DEMO_C_{i}", "device_id": "DEMO_DEVICE_SHARED"} for i in range(4)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="Shared-device ring")
    args = parser.parse_args()
    results = [post(payload) for payload in scenario(args.scenario)]
    print(json.dumps({"label": "DEMO / SYNTHETIC", "scenario": args.scenario, "results": results}, indent=2))


if __name__ == "__main__":
    main()
