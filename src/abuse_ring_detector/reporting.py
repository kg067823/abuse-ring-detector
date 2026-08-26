"""Run artifact persistence and concise experiment report."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if str(path).endswith(".gz"):
        frame.to_csv(path, index=False, compression="gzip")
    else:
        frame.to_csv(path, index=False)


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str))


def write_report(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# AbuseRing Detector POC Report", "", "Synthetic data only; this is not production fraud detection.", ""]
    for section, value in summary.items():
        lines += [f"## {section.replace('_', ' ').title()}", "", f"```json\n{json.dumps(value, indent=2, default=str)}\n```", ""]
    path.write_text("\n".join(lines))
