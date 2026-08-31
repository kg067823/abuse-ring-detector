"""Pure helpers used by the Command Center and its tests."""
from __future__ import annotations

from collections import Counter
from typing import Any


def aggregate_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "open": sum(case.get("status") != "CLOSED" for case in cases),
        "critical": sum(case.get("severity") == "CRITICAL" for case in cases),
        "exposure": round(sum(float(case.get("estimated_exposure", 0)) for case in cases), 2),
        "avg_risk": round(sum(float(case.get("risk_score", 0)) for case in cases) / len(cases), 4) if cases else 0.0,
        "by_severity": dict(Counter(case.get("severity", "UNKNOWN") for case in cases)),
        "by_status": dict(Counter(case.get("status", "UNKNOWN") for case in cases)),
    }


def filter_items(items: list[dict[str, Any]], query: str = "", severity: str = "ALL", status: str = "ALL") -> list[dict[str, Any]]:
    query = query.strip().lower()
    result = []
    for item in items:
        if severity != "ALL" and item.get("severity") != severity:
            continue
        if status != "ALL" and item.get("status") != status:
            continue
        if query and query not in str(item).lower():
            continue
        result.append(item)
    return sorted(result, key=lambda row: (float(row.get("risk_score", 0)), str(row.get("created_at", ""))), reverse=True)


def timeline_sorted(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(events, key=lambda row: (row.get("timestamp", ""), row.get("event_id", "")))


def graph_counts(graph: dict[str, Any]) -> dict[str, int]:
    return {"nodes": len(graph.get("nodes", [])), "edges": len(graph.get("edges", []))}
