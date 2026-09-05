"""Pure logic for the AbuseRing Live Monitor screen.

Everything here is deterministic and Streamlit-free so the monitoring-console
behavior (status mapping, replay pacing, automatic case selection) is unit-
testable without a UI runtime.
"""
from __future__ import annotations

from typing import Any

REVIEW_THRESHOLD = 0.50

SCENARIOS = [
    "Mixed multi-entity",
    "Shared-device ring",
    "Shared-address ring",
    "Behavioral coordination",
    "Legitimate high-connectivity",
]


def risk_status(risk: float, fallback: bool = False) -> str:
    """Map a backend score to the honest monitoring status.

    NORMAL < WATCHING < ALERT. Fallback responses (model unavailable) are
    never presented as alerts.
    """
    if fallback:
        return "FALLBACK"
    if risk < 0.30:
        return "NORMAL"
    if risk < REVIEW_THRESHOLD:
        return "WATCHING"
    return "ALERT"


def status_color(status: str) -> str:
    return {
        "NORMAL": "#34d399",
        "WATCHING": "#fbbf24",
        "ALERT": "#f87171",
        "FALLBACK": "#93a3b3",
    }.get(status, "#93a3b3")


def mask_customer(customer_id: str) -> str:
    return f"Customer ••{customer_id[-2:]}"


def observation_text(risk: float, fallback: bool, alert: bool, index: int, shared: list[str]) -> str:
    """Plain-English per-event observation from actual data. Never fabricated."""
    if fallback:
        return "Event scored by Model F-R1 (fallback path)."
    if alert:
        return "Risk crossed review threshold"
    if index == 1:
        return "First event — limited history"
    if shared:
        if len(shared) >= 2:
            return "Network connectivity increasing"
        if shared[0] == "device":
            return "Another customer connected to shared device"
        if shared[0] == "address":
            return "Shared address relationship observed"
        if shared[0] == "IP":
            return "Shared IP relationship observed"
        return f"Shared {shared[0]} relationship observed"
    if risk >= 0.30:
        return "Elevated score, below review threshold"
    return "Event scored by Model F-R1."


def shared_with(payload: dict[str, Any], previous: list[dict[str, Any]]) -> list[str]:
    """Entity types this event shares with earlier events in the stream.

    Missing entity fields never count as shared (None == None must not match).
    """
    labels = (("device_id", "device"), ("address_id", "address"), ("ip_id", "IP"), ("payment_id", "payment"))
    shared: list[str] = []
    for key, label in labels:
        value = payload.get(key)
        if value is None:
            continue
        if any(prev.get(key) == value for prev in previous):
            shared.append(label)
    return shared


def row_from_event(index: int, payload: dict[str, Any], response: dict[str, Any], previous: list[dict[str, Any]]) -> dict[str, Any]:
    """Build one feed row from a real backend response."""
    risk = float(response.get("calibrated_score", 0) or 0)
    fallback = bool(response.get("fallback_applied"))
    alert = bool(response.get("alert"))
    status = risk_status(risk, fallback)
    ts = str(response.get("timestamp", ""))
    return {
        "time": ts[11:19] if len(ts) >= 19 else ts,
        "customer": mask_customer(str(payload.get("customer_id", ""))),
        "amount": float(payload.get("amount", 0) or 0),
        "risk": risk,
        "status": status,
        "observation": observation_text(risk, fallback, alert, index, shared_with(payload, previous)),
    }


def case_belongs_to_run(case: dict[str, Any], run_orders: set[str]) -> bool:
    """A case counts for the demo only if it contains an order from this run."""
    return bool(run_orders & set(case.get("related_orders", []) or []))


def masked_orders_for_run(alerts: list[dict[str, Any]], raw_order_ids: set[str]) -> set[str]:
    """Translate this run's raw order ids to backend-masked order ids.

    Case payloads list masked order ids; the raw id survives only as the
    alert's correlation_id. Missing this mapping makes the demo show "no case"
    even when the backend created one.
    """
    masked: set[str] = set()
    for alert in alerts:
        if str(alert.get("correlation_id", "")) in raw_order_ids:
            order_id = alert.get("order_id")
            if order_id:
                masked.add(str(order_id))
    return masked


def pick_active_case(cases: list[dict[str, Any]], run_orders: set[str]) -> dict[str, Any] | None:
    """Auto-select the most relevant active case for this run.

    Preference: highest-risk non-closed case from this run; falls back to any
    case from this run. Returns None when the backend produced nothing.
    """
    relevant = [c for c in cases if case_belongs_to_run(c, run_orders)]
    if not relevant:
        return None
    active = [c for c in relevant if c.get("status") != "CLOSED"]
    pool = active or relevant
    return max(pool, key=lambda c: float(c.get("risk_score", 0) or 0))


def shared_entity_count(graph: dict[str, Any]) -> int:
    return sum(1 for n in graph.get("nodes", []) if n.get("type") in {"device_id", "address_id", "ip_id", "payment_id"})


def graph_dot(graph: dict[str, Any]) -> str:
    """Simplified Graphviz source: customers + shared entities, orders hidden.

    Labels are deterministic short masks (Customer A, Device X, IP 18…) mapped
    from the backend pseudonyms.
    """
    kinds = {
        "customer": ("Customer", "circle", "#4fd1c5"),
        "device_id": ("Device", "diamond", "#fbbf24"),
        "address_id": ("Address", "diamond", "#fbbf24"),
        "ip_id": ("IP", "diamond", "#fbbf24"),
        "payment_id": ("Payment", "diamond", "#fbbf24"),
    }
    counters: dict[str, int] = {}
    labels: dict[str, tuple[str, str, str]] = {}
    for node in sorted(graph.get("nodes", []), key=lambda n: n.get("id", "")):
        ntype = node.get("type", "")
        if ntype not in kinds:
            continue
        base, shape, color = kinds[ntype]
        counters[base] = counters.get(base, 0) + 1
        idx = counters[base]
        suffix = chr(ord("A") + idx - 1) if base != "IP" else str(idx)
        labels[node["id"]] = (f"{base} {suffix}", shape, color)
    lines = ["graph rankdir=LR {", 'node [fontname="Helvetica" fontsize=13 style=filled];']
    for nid, (label, shape, color) in labels.items():
        fill = "#1d2f3a" if color == "#4fd1c5" else "#332a12"
        lines.append(f'"{nid}" [label="{label}" shape={shape} color="{color}" fillcolor="{fill}" fontcolor="{color}"];')
    for edge in graph.get("edges", []):
        if edge.get("source") in labels and edge.get("target") in labels:
            rel = str(edge.get("relationship", "")).replace("customer-used-", "uses ")
            lines.append(f'"{edge["source"]}" -- "{edge["target"]}" [label="{rel}" fontcolor="#7d8ea0" fontsize=10 color="#3a4a5a"];')
    lines.append("}")
    return "\n".join(lines)


def timeline_view(items: list[dict[str, Any]], limit: int = 8) -> list[dict[str, str]]:
    """Backend timeline → compact view rows (time + description), no invention."""
    rows = []
    for event in sorted(items, key=lambda e: e.get("timestamp", ""))[:limit]:
        ts = str(event.get("timestamp", ""))
        rows.append({"time": ts[11:19] if len(ts) >= 19 else ts, "event": event.get("description", "—")})
    return rows
