"""7-Day Live Production Shadow Gate Evaluator.

Enforces strict, un-cheatable verification of the mandatory 7-day live shadow observation period.
Explicitly rejects staging replay, synthetic transactions, local simulations, and manual test requests
from counting toward the 7 live production observation days.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

DATA_SOURCE_TYPE = Literal["REAL_LIVE_PRODUCTION", "STAGING_REPLAY", "SIMULATION", "UNAVAILABLE"]

@dataclass
class DailyObservationRecord:
    date_str: str
    data_source: DATA_SOURCE_TYPE
    total_transactions: int
    error_count: int
    fallback_count: int
    blocked_transactions: int
    p95_latency_ms: float
    model_checksum: str
    feature_count: int
    incidents_count: int = 0
    pii_violations: int = 0

    @property
    def is_qualifying_live_production_day(self) -> bool:
        """A day qualifies towards the 7-day gate ONLY if it is real live production traffic."""
        return self.data_source == "REAL_LIVE_PRODUCTION"

    @property
    def passes_safety_thresholds(self) -> bool:
        if self.blocked_transactions > 0:
            return False
        if self.model_checksum != "82e77daac0762a04":
            return False
        if self.feature_count != 137:
            return False
        if self.pii_violations > 0:
            return False
        
        # Rate thresholds
        if self.total_transactions == 0:
            return False
        error_rate = self.error_count / self.total_transactions
        fallback_rate = self.fallback_count / self.total_transactions
        
        if error_rate > 0.001:  # > 0.1%
            return False
        if fallback_rate > 0.001:  # > 0.1%
            return False
        if self.p95_latency_ms > 300.0:  # > 300ms SLA
            return False
        return True


@dataclass
class SevenDayGateVerdict:
    status: Literal[
        "NOT STARTED — LIVE TRAFFIC NOT ATTACHED",
        "IN PROGRESS — LIVE SHADOW OBSERVATION",
        "NO-GO — SHADOW GATE FAILED",
        "GO — ELIGIBLE FOR 5% CANARY"
    ]
    qualifying_live_days_completed: int
    total_records_inspected: int
    rejection_reasons: list[str] = field(default_factory=list)
    canary_eligible: bool = False


class SevenDayShadowGateEvaluator:
    """Evaluates daily observation records for promotion to Canary Stage 1."""

    def __init__(self, target_checksum: str = "82e77daac0762a04", target_features: int = 137):
        self.target_checksum = target_checksum
        self.target_features = target_features

    def evaluate_records(self, records: list[DailyObservationRecord]) -> SevenDayGateVerdict:
        rejection_reasons = []
        qualifying_consecutive_live_days = 0
        live_days_found = 0

        for rec in records:
            if not rec.is_qualifying_live_production_day:
                rejection_reasons.append(
                    f"Day {rec.date_str} source '{rec.data_source}' rejected: Not real live production data."
                )
                qualifying_consecutive_live_days = 0  # reset consecutive counter
                continue

            live_days_found += 1
            if not rec.passes_safety_thresholds:
                rejection_reasons.append(
                    f"Day {rec.date_str} live production record failed safety thresholds (blocked={rec.blocked_transactions}, p95={rec.p95_latency_ms}ms)."
                )
                qualifying_consecutive_live_days = 0
            else:
                qualifying_consecutive_live_days += 1

        if live_days_found == 0:
            return SevenDayGateVerdict(
                status="NOT STARTED — LIVE TRAFFIC NOT ATTACHED",
                qualifying_live_days_completed=0,
                total_records_inspected=len(records),
                rejection_reasons=rejection_reasons or ["No live production traffic streams attached."],
                canary_eligible=False
            )

        if any("failed safety thresholds" in r for r in rejection_reasons):
            return SevenDayGateVerdict(
                status="NO-GO — SHADOW GATE FAILED",
                qualifying_live_days_completed=qualifying_consecutive_live_days,
                total_records_inspected=len(records),
                rejection_reasons=rejection_reasons,
                canary_eligible=False
            )

        if qualifying_consecutive_live_days >= 7:
            return SevenDayGateVerdict(
                status="GO — ELIGIBLE FOR 5% CANARY",
                qualifying_live_days_completed=qualifying_consecutive_live_days,
                total_records_inspected=len(records),
                rejection_reasons=[],
                canary_eligible=True
            )
        else:
            return SevenDayGateVerdict(
                status="IN PROGRESS — LIVE SHADOW OBSERVATION",
                qualifying_live_days_completed=qualifying_consecutive_live_days,
                total_records_inspected=len(records),
                rejection_reasons=[f"Only {qualifying_consecutive_live_days} / 7 consecutive live production days completed."],
                canary_eligible=False
            )
