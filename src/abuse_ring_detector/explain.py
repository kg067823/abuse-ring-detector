"""Evidence-backed, non-causal alert explanations."""
from __future__ import annotations

import hashlib
import re

import pandas as pd


def mask_identifier(value: str) -> str:
    """Return a stable pseudonym suitable for investigator/demo output."""
    digest = hashlib.sha256(str(value).encode()).hexdigest()[:10]
    return f"id_{digest}"


def explain_prediction(feature_row: pd.Series, score: float, top_k: int = 6, *, model_version: str = "model_f_r1", model_checksum: str = "") -> dict:
    """Return deterministic observed signals, never causal attribution."""
    numeric = feature_row.drop(labels=[c for c in ["is_abuse", "ring_id"] if c in feature_row], errors="ignore")
    numeric = pd.to_numeric(numeric, errors="coerce").dropna()
    numeric = numeric[~numeric.index.to_series().str.contains(r"(id|customer|address|device|payment|ip)", flags=re.I, regex=True)]
    ranked = numeric.abs().sort_values(ascending=False).head(top_k)
    evidence = [{"feature": name, "value": float(numeric[name]), "type": "observed_signal", "as_of_rule": "strictly earlier events only"} for name in ranked.index]
    return {
        "explanation_version": "explanation_r1.v1",
        "risk_score": float(score),
        "model_version": model_version,
        "model_checksum": model_checksum,
        "status": "SHADOW_REVIEW_ONLY",
        "enforcement_applied": False,
        "evidence": evidence,
        "rule_evidence": [],
        "caveat": "These are observed historical correlations and supporting signals, not causal proof and not model attribution.",
        "limitations": ["synthetic/reconstructed training data", "investigator validation required"],
    }
