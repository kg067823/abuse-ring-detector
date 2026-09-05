"""Analyst next-step suggestions derived from returned case state.

Pure logic, no API calls: the UI renders buttons only for transitions the
backend's ALLOWED_TRANSITIONS map permits, mirrored here and pinned by test
so the console can never offer an action the backend would reject.
"""
from __future__ import annotations

from typing import Any

# Mirrors src/abuse_ring_detector/case_management.py ALLOWED_TRANSITIONS.
# tests/test_next_steps.py asserts the two stay identical.
NEXT_TRANSITIONS: dict[str, set[str]] = {
    "NEW": {"IN_REVIEW", "CLOSED"},
    "IN_REVIEW": {"ESCALATED", "CONFIRMED_ABUSE", "LEGITIMATE", "CLOSED"},
    "ESCALATED": {"CONFIRMED_ABUSE", "LEGITIMATE", "CLOSED"},
    "CONFIRMED_ABUSE": {"CLOSED"},
    "LEGITIMATE": {"CLOSED"},
    "CLOSED": set(),
}

_RISK_VERBS = {
    "ESCALATED": "Escalate to the risk operations team",
    "CONFIRMED_ABUSE": "Confirm coordinated abuse",
    "LEGITIMATE": "Mark legitimate and release",
    "IN_REVIEW": "Open the review",
    "CLOSED": "Close the case",
}


def suggest_next_steps(
    case: dict[str, Any],
    graph: dict[str, Any] | None = None,
    evidence: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return ordered next-step suggestions for an investigation case.

    Each suggestion: {title, detail, action, allowed, priority}.
    action is a target status, or "note" for the always-available note.
    allowed reflects the backend transition map; the UI hides disallowed ones.
    """
    status = str(case.get("status", ""))
    severity = str(case.get("severity", ""))
    risk = float(case.get("risk_score", 0) or 0)
    exposure = float(case.get("estimated_exposure", 0) or 0)
    alert_count = int(case.get("alert_count", 0) or 0)
    related = list(case.get("related_customers", []) or [])
    allowed = NEXT_TRANSITIONS.get(status, set())
    items: list[dict[str, Any]] = []

    def add(title: str, detail: str, action: str, priority: int) -> None:
        # Informational steps (action "") are always shown; transition steps
        # are shown only when the backend state machine permits them.
        items.append({
            "title": title,
            "detail": detail,
            "action": action,
            "allowed": True if not action else action in allowed,
            "priority": priority,
        })

    # 1. Severity-driven escalation path — only when the state machine allows
    # NEW -> ESCALATED directly; otherwise the workflow loop below offers the
    # correct first step (open the review).
    if "ESCALATED" in allowed and (severity == "CRITICAL" or (severity == "HIGH" and risk >= 0.9)):
        add(
            "Escalate to risk operations",
            f"Severity {severity} with risk {risk:.2f} and {alert_count} linked alert(s).",
            "ESCALATED",
            1,
        )

    # 2. Exposure-driven prioritization.
    if exposure >= 1000:
        add(
            "Prioritize for exposure review",
            f"Estimated exposure ₹{exposure:,.0f} crosses the ₹1,000 review line in this window.",
            "",
            1,
        )

    # 3. Graph-driven follow-ups.
    if graph:
        nodes = graph.get("nodes", [])
        devices = [n for n in nodes if n.get("type") == "device_id"]
        payments = [n for n in nodes if n.get("type") == "payment_id"]
        if len(devices) <= 2 and len(related) >= 3:
            add(
                "Trace the shared device cluster",
                f"{len(related)} customers converge on {len(devices) or 'a shared'} device identifier(s).",
                "",
                2,
            )
        if payments:
            add(
                "Correlate payment instruments",
                f"{len(payments)} payment identifier(s) appear in this network — check for recycled instruments.",
                "",
                3,
            )

    # 4. Evidence gaps worth closing.
    if evidence:
        kinds = {str(e.get("evidence_type")) for e in evidence.get("items", [])}
        if "shared_entity" not in kinds:
            add("Request shared-entity evidence", "No shared-infrastructure evidence attached yet.", "", 3)

    # 5. Workflow transitions (state machine).
    for target in ("IN_REVIEW", "CONFIRMED_ABUSE", "LEGITIMATE", "CLOSED"):
        if target in allowed and target not in {i["action"] for i in items}:
            verb = _RISK_VERBS[target]
            if target == "IN_REVIEW":
                add(verb, "Move the case into active review to unlock escalate/confirm/legitimate actions.", target, 2)
            elif target == "CLOSED":
                add(verb, "Archive the case once its disposition is recorded in history.", target, 4)
            else:
                add(verb, "Record the analyst disposition on the case.", target, 3)

    # 6. Always available: analyst note.
    items.append({
        "title": "Add an analyst note",
        "detail": "Notes are appended to the case history with your actor label.",
        "action": "note",
        "allowed": True,
        "priority": 5,
    })

    return sorted(items, key=lambda i: (i["priority"], i["title"]))
