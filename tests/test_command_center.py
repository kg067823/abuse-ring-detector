from __future__ import annotations

from app.command_center_helpers import aggregate_cases, filter_items, graph_counts, timeline_sorted


def test_aggregate_cases_does_not_invent_metrics():
    result = aggregate_cases([
        {"status": "NEW", "severity": "CRITICAL", "risk_score": .9, "estimated_exposure": 100},
        {"status": "CLOSED", "severity": "HIGH", "risk_score": .7, "estimated_exposure": 50},
    ])
    assert result["open"] == 1
    assert result["critical"] == 1
    assert result["exposure"] == 150
    assert result["avg_risk"] == .8


def test_filters_sort_highest_risk_first():
    rows = [{"case_id": "a", "risk_score": .6, "severity": "HIGH", "status": "NEW"}, {"case_id": "b", "risk_score": .9, "severity": "CRITICAL", "status": "NEW"}]
    assert [row["case_id"] for row in filter_items(rows, severity="NEW")] == []
    assert [row["case_id"] for row in filter_items(rows, severity="CRITICAL")] == ["b"]
    assert [row["case_id"] for row in filter_items(rows)] == ["b", "a"]


def test_timeline_and_graph_helpers():
    assert [e["event_id"] for e in timeline_sorted([{"event_id":"b","timestamp":"02"},{"event_id":"a","timestamp":"01"}])] == ["a", "b"]
    assert graph_counts({"nodes": [1], "edges": [1, 2]}) == {"nodes": 1, "edges": 2}
