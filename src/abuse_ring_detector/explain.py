"""Evidence-backed, non-causal alert explanations."""
from __future__ import annotations

import pandas as pd


def explain_prediction(feature_row: pd.Series, score: float, top_k: int = 6) -> dict:
    numeric = feature_row.drop(labels=[c for c in ["is_abuse", "ring_id"] if c in feature_row], errors="ignore")
    ranked = numeric.abs().sort_values(ascending=False).head(top_k)
    evidence = [{"feature": name, "value": float(feature_row[name])} for name in ranked.index if pd.notna(feature_row[name])]
    return {"risk_score": float(score), "evidence": evidence,
            "caveat": "These are historical correlations and supporting signals, not causal proof."}
