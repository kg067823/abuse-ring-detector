"""Unit tests for the 7-Day Live Shadow Gate Evaluator."""

import pytest
from abuse_ring_detector.seven_day_gate import (
    DailyObservationRecord,
    SevenDayShadowGateEvaluator,
)

def test_staging_replay_days_are_rejected():
    evaluator = SevenDayShadowGateEvaluator()
    staging_records = [
        DailyObservationRecord(
            date_str=f"2026-08-{10+i:02d}",
            data_source="STAGING_REPLAY",
            total_transactions=1000,
            error_count=0,
            fallback_count=0,
            blocked_transactions=0,
            p95_latency_ms=15.0,
            model_checksum="82e77daac0762a04",
            feature_count=137
        )
        for i in range(7)
    ]
    verdict = evaluator.evaluate_records(staging_records)
    assert verdict.status == "NOT STARTED — LIVE TRAFFIC NOT ATTACHED"
    assert verdict.qualifying_live_days_completed == 0
    assert verdict.canary_eligible is False

def test_in_progress_live_observation():
    evaluator = SevenDayShadowGateEvaluator()
    live_records = [
        DailyObservationRecord(
            date_str=f"2026-08-{10+i:02d}",
            data_source="REAL_LIVE_PRODUCTION",
            total_transactions=5000,
            error_count=0,
            fallback_count=0,
            blocked_transactions=0,
            p95_latency_ms=18.0,
            model_checksum="82e77daac0762a04",
            feature_count=137
        )
        for i in range(3)
    ]
    verdict = evaluator.evaluate_records(live_records)
    assert verdict.status == "IN PROGRESS — LIVE SHADOW OBSERVATION"
    assert verdict.qualifying_live_days_completed == 3
    assert verdict.canary_eligible is False

def test_full_7_day_live_gate_pass():
    evaluator = SevenDayShadowGateEvaluator()
    live_records = [
        DailyObservationRecord(
            date_str=f"2026-08-{10+i:02d}",
            data_source="REAL_LIVE_PRODUCTION",
            total_transactions=5000,
            error_count=0,
            fallback_count=0,
            blocked_transactions=0,
            p95_latency_ms=18.0,
            model_checksum="82e77daac0762a04",
            feature_count=137
        )
        for i in range(7)
    ]
    verdict = evaluator.evaluate_records(live_records)
    assert verdict.status == "GO — ELIGIBLE FOR 5% CANARY"
    assert verdict.qualifying_live_days_completed == 7
    assert verdict.canary_eligible is True

def test_customer_blocking_triggers_no_go():
    evaluator = SevenDayShadowGateEvaluator()
    records = [
        DailyObservationRecord(
            date_str="2026-08-10",
            data_source="REAL_LIVE_PRODUCTION",
            total_transactions=5000,
            error_count=0,
            fallback_count=0,
            blocked_transactions=1,  # VIOLATION!
            p95_latency_ms=18.0,
            model_checksum="82e77daac0762a04",
            feature_count=137
        )
    ]
    verdict = evaluator.evaluate_records(records)
    assert verdict.status == "NO-GO — SHADOW GATE FAILED"
    assert verdict.canary_eligible is False
