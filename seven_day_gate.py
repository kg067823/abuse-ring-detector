"""Non-bypassable live shadow observation gate.

This command intentionally reports the post-deployment operational gate as
pending until an operator supplies genuine production evidence. It never treats
synthetic replay, staging output, or an empty evidence file as seven days.
"""
from __future__ import annotations

import json
from pathlib import Path


def evaluate_gate(evidence_path: str | Path | None = None) -> dict[str, object]:
    """Return the current gate status without fabricating qualifying days."""
    evidence = Path(evidence_path) if evidence_path else None
    evidence_present = bool(evidence and evidence.exists() and evidence.stat().st_size > 0)
    # Evidence ingestion is deliberately not implemented until the production
    # telemetry schema and authoritative frozen contract are supplied.
    result = {
        "LIVE_PRODUCTION_OBSERVATION": "NOT STARTED",
        "QUALIFYING_DAYS": "0/7",
        "CANARY_STAGE_1": "BLOCKED",
        "evidence_present": evidence_present,
        "reason": "No genuine production shadow observation evidence supplied.",
    }
    return result


def main() -> int:
    result = evaluate_gate()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
