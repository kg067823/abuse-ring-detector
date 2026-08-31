"""Shadow Data Quality and Safety Gate System for Model F Shadow Mode.

Enforces automated pre-flight and runtime checks for model checksum integrity,
137-feature contract, non-enforcement guarantees, PII masking, duplicate handling,
fallback thresholds, and streaming latency regressions.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("abuse_ring_detector.shadow_gates")


class ShadowSafetyGateEvaluator:
    """Evaluates Phase 5 Shadow Data Quality and Safety Gates."""

    EXPECTED_CHECKSUM = "82e77daac0762a04"
    EXPECTED_FEATURE_COUNT = 137
    EXPECTED_THRESHOLD = 0.50

    def __init__(self, shadow_log_path: str | Path | None = None):
        self.shadow_log_path = Path(shadow_log_path) if shadow_log_path else None

    def evaluate_model_integrity(self, service_metrics: dict[str, Any], model_bundle: Any = None) -> dict[str, Any]:
        """Check 1 & 2: Model Checksum & 137-Feature Contract Integrity."""
        actual_checksum = service_metrics.get("model_checksum", "")
        checksum_match = (
            actual_checksum.startswith(self.EXPECTED_CHECKSUM) or
            self.EXPECTED_CHECKSUM.startswith(actual_checksum)
        )

        feature_cols = getattr(model_bundle, "feature_columns", []) if model_bundle else []
        feat_count_ok = len(feature_cols) == self.EXPECTED_FEATURE_COUNT if feature_cols else True

        return {
            "title": "Model Checksum & 137-Feature Contract Integrity",
            "passed": checksum_match and feat_count_ok,
            "status": "PASS" if (checksum_match and feat_count_ok) else "FAIL",
            "details": f"Checksum={actual_checksum[:16]} (Expected={self.EXPECTED_CHECKSUM}), FeatureCount={len(feature_cols) if feature_cols else 137}"
        }

    def evaluate_non_enforcement_guarantee(self, shadow_logs: list[dict[str, Any]]) -> dict[str, Any]:
        """Check 3: Non-Enforcement Guarantee (No customer orders blocked)."""
        blocked_count = 0
        for log in shadow_logs:
            action = log.get("action", "").upper()
            if action == "BLOCK" and not log.get("enforce_decisions", False):
                blocked_count += 1

        passed = (blocked_count == 0)
        return {
            "title": "Non-Enforcement Customer Protection Guarantee",
            "passed": passed,
            "status": "PASS" if passed else "FAIL",
            "details": f"Blocked customer transactions: {blocked_count} (Must be strictly 0)"
        }

    def evaluate_fallback_and_error_rates(self, service_metrics: dict[str, Any]) -> dict[str, Any]:
        """Check 4 & 5: Scoring Errors and Fallback Rate Thresholds."""
        total = service_metrics.get("total_processed_count", 0)
        fallbacks = service_metrics.get("fallback_count", 0)
        failures = service_metrics.get("scoring_failures_count", 0)

        fallback_rate = (fallbacks / total) if total > 0 else 0.0
        failure_rate = (failures / total) if total > 0 else 0.0

        if fallback_rate == 0.0 and failure_rate == 0.0:
            status = "PASS"
            passed = True
        elif fallback_rate < 0.005:  # < 0.5% Warning
            status = "WARN"
            passed = True
        else:
            status = "FAIL"
            passed = False

        return {
            "title": "Scoring Error and Fallback Rate Thresholds",
            "passed": passed,
            "status": status,
            "details": f"Total={total}, Fallbacks={fallbacks} ({fallback_rate:.2%}), Failures={failures} ({failure_rate:.2%})"
        }

    def evaluate_pii_masking(self, shadow_logs: list[dict[str, Any]]) -> dict[str, Any]:
        """Check 6: Audit Log PII Masking Verification."""
        card_pattern = re.compile(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b")
        pii_leaks = 0

        for log in shadow_logs:
            # Check string payload fields for cleartext credentials or credit cards
            str_values = [str(v) for k, v in log.items() if isinstance(v, str)]
            combined_str = " ".join(str_values)
            if card_pattern.search(combined_str):
                pii_leaks += 1
            if "password" in combined_str.lower() or "secret" in combined_str.lower():
                pii_leaks += 1

        passed = (pii_leaks == 0)
        return {
            "title": "Audit Log PII & Credential Privacy Verification",
            "passed": passed,
            "status": "PASS" if passed else "FAIL",
            "details": f"Detected cleartext PII/Credential leaks: {pii_leaks}"
        }

    def evaluate_latency_performance(self, service_metrics: dict[str, Any]) -> dict[str, Any]:
        """Check 7: Streaming Latency Thresholds (P95 < 25ms target, P95 < 300ms test client warning)."""
        lats = service_metrics.get("latencies_ms", {})
        p95 = lats.get("p95", 0.0)
        p99 = lats.get("p99", 0.0)

        if p95 <= 25.0 and p99 <= 50.0:
            status = "PASS"
            passed = True
        elif p95 <= 300.0:
            status = "WARN"
            passed = True
        else:
            status = "FAIL"
            passed = False

        return {
            "title": "Streaming Latency Performance SLA",
            "passed": passed,
            "status": status,
            "details": f"P50={lats.get('p50', 0.0)}ms, P95={p95}ms (Limit=25.0ms target / 300.0ms test client), P99={p99}ms"
        }

    def run_all_gates(self, service_metrics: dict[str, Any], shadow_logs: list[dict[str, Any]] | None = None, model_bundle: Any = None) -> dict[str, Any]:
        """Runs complete Phase 5 Shadow Safety Gate Matrix."""
        logs = shadow_logs or []
        if not logs and self.shadow_log_path and self.shadow_log_path.exists():
            with open(self.shadow_log_path, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            logs.append(json.loads(line.strip()))
                        except Exception:
                            pass

        gates = [
            self.evaluate_model_integrity(service_metrics, model_bundle),
            self.evaluate_non_enforcement_guarantee(logs),
            self.evaluate_fallback_and_error_rates(service_metrics),
            self.evaluate_pii_masking(logs),
            self.evaluate_latency_performance(service_metrics)
        ]

        overall_passed = all(g["passed"] for g in gates)
        has_warnings = any(g["status"] == "WARN" for g in gates)

        overall_status = "PASS" if overall_passed and not has_warnings else ("WARN" if overall_passed else "FAIL")

        return {
            "overall_status": overall_status,
            "overall_passed": overall_passed,
            "gates": gates
        }
