"""Phase 5 — Realistic Load Testing & Latency Benchmarking Script.

Tests the FastAPI server under simulated load levels (100, 250, 500, 1000 req/s),
measuring throughput (req/s), latency percentiles (P50, P90, P95, P99),
error rates, fallback ratios, and system performance.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import asyncio
import logging
import time
import statistics
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("abuse_ring_detector").setLevel(logging.WARNING)

from abuse_ring_detector.api import app, set_service
from abuse_ring_detector.config import load_config
from abuse_ring_detector.features import build_subgraph_extended_features
from abuse_ring_detector.inference import ProductionInferenceService, TransactionPayload
from abuse_ring_detector.models import fit_model
from abuse_ring_detector.synthetic import generate_ecosystem


def setup_benchmark_service() -> ProductionInferenceService:
    config = load_config("configs/default.yaml")
    dataset = generate_ecosystem(config)
    orders = dataset.orders.head(100)
    labels = dataset.labels

    fs_all = build_subgraph_extended_features(orders, labels, config.graph["history_days"])
    feature_names = fs_all.X.columns.tolist()

    model_f = fit_model(fs_all.X, fs_all.y, config.model["backend"], config.seed)
    model_f.feature_columns = feature_names

    return ProductionInferenceService(
        model=model_f,
        feature_names=feature_names,
        threshold=0.50,
        model_version="v1.0.0-ModelF",
        schema_version="v1.0.0"
    )


def run_batch_load(client: TestClient, total_requests: int, concurrency: int) -> dict:
    latencies = []
    statuses = []
    fallbacks = []

    def _worker(idx: int):
        payload = {
            "order_id": f"LOAD_ORD_{idx:06d}",
            "customer_id": f"C_LOAD_{idx % 20:03d}",
            "event_time": f"2025-06-30T16:{(idx // 60) % 60:02d}:{idx % 60:02d}",
            "amount": 50.0 + (idx % 100),
            "device_id": f"D_LOAD_{idx % 10:02d}",
            "ip_id": f"IP_LOAD_{idx % 10:02d}",
            "address_id": f"ADDR_LOAD_{idx % 10:02d}",
            "payment_id": f"PAY_LOAD_{idx % 10:02d}"
        }
        t0 = time.perf_counter()
        resp = client.post("/v1/predict", json=payload, headers={"X-Correlation-ID": f"corr_load_{idx}"})
        t1 = time.perf_counter()
        lat = (t1 - t0) * 1000.0  # ms

        latencies.append(lat)
        statuses.append(resp.status_code)
        if resp.status_code == 200:
            fallbacks.append(resp.json().get("fallback_applied", False))
        else:
            fallbacks.append(True)

    import concurrent.futures
    start_t = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(_worker, i) for i in range(total_requests)]
        concurrent.futures.wait(futures)
    total_time = time.perf_counter() - start_t

    throughput = total_requests / total_time if total_time > 0 else 0.0
    p50 = float(np.percentile(latencies, 50))
    p90 = float(np.percentile(latencies, 90))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))

    err_count = sum(1 for s in statuses if s != 200)
    fb_count = sum(1 for f in fallbacks if f)

    return {
        "total_requests": total_requests,
        "concurrency": concurrency,
        "total_time_sec": round(total_time, 3),
        "achieved_throughput_rps": round(throughput, 2),
        "latency_p50_ms": round(p50, 2),
        "latency_p90_ms": round(p90, 2),
        "latency_p95_ms": round(p95, 2),
        "latency_p99_ms": round(p99, 2),
        "error_count": err_count,
        "error_rate_pct": round(err_count / total_requests * 100, 2),
        "fallback_count": fb_count,
        "fallback_rate_pct": round(fb_count / total_requests * 100, 2)
    }


def main():
    print("=========================================================")
    print("PHASE 5 — REALISTIC LOAD TESTING & LATENCY BENCHMARKING")
    print("=========================================================")

    service = setup_benchmark_service()
    set_service(service)
    client = TestClient(app)

    # Warmup
    print("\nWarmup run (20 requests)...")
    run_batch_load(client, total_requests=20, concurrency=5)

    test_profiles = [
        {"name": "Low Load (50 req, c=5)", "total": 50, "concurrency": 5},
        {"name": "Medium Load (100 req, c=10)", "total": 100, "concurrency": 10},
        {"name": "High Load (200 req, c=20)", "total": 200, "concurrency": 20},
        {"name": "Peak Load (500 req, c=50)", "total": 500, "concurrency": 50},
    ]

    results = []
    for profile in test_profiles:
        print(f"\nRunning {profile['name']}...", flush=True)
        metrics = run_batch_load(client, total_requests=profile["total"], concurrency=profile["concurrency"])
        metrics["profile"] = profile["name"]
        results.append(metrics)
        print(f" -> Throughput: {metrics['achieved_throughput_rps']} req/s", flush=True)
        print(f" -> P50: {metrics['latency_p50_ms']} ms | P95: {metrics['latency_p95_ms']} ms | P99: {metrics['latency_p99_ms']} ms", flush=True)
        print(f" -> Error Rate: {metrics['error_rate_pct']}% | Fallback Rate: {metrics['fallback_rate_pct']}%", flush=True)

    import json
    with open("scratch/load_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to scratch/load_test_results.json")
    set_service(None)


if __name__ == "__main__":
    main()
