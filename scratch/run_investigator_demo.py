"""Replay deterministic demo scenarios through the real R1 HTTP API."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from demo_scenarios import SCENARIOS, scenario_payloads

BASE = os.getenv("ABUSERING_API_URL", "http://localhost:8000").rstrip("/")


def request(path: str, payload: dict | None = None) -> dict:
    if payload is None:
        req = Request(BASE + path)
    else:
        req = Request(BASE + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", "X-Correlation-ID": payload["order_id"]})
    with urlopen(req, timeout=30) as response:
        return json.loads(response.read())


def run_scenario(name: str, run_id: str = "demo-001") -> dict:
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario: {name}")
    progression = []
    for index, payload in enumerate(scenario_payloads(name, run_id), start=1):
        result = request("/v1/predict", payload)
        cases = request("/v1/cases").get("items", [])
        progression.append({"event": index, "total": 8, "order_id": payload["order_id"], "score": result.get("calibrated_score"), "alert": result.get("alert"), "fallback": result.get("fallback_applied"), "cases": len(cases), "correlation_id": result.get("correlation_id")})
        print(f"Event {index}/8 · score={result.get('calibrated_score', 0):.3f} · alerts={result.get('alert')} · cases={len(cases)}")
    return {"label": "DEMO / SYNTHETIC", "scenario": name, "progression": progression, "cases": request("/v1/cases").get("items", [])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=SCENARIOS, default="Mixed multi-entity")
    parser.add_argument("--run-id", default="demo-001")
    args = parser.parse_args()
    print(json.dumps(run_scenario(args.scenario, args.run_id), indent=2))


if __name__ == "__main__":
    main()
