"""Model F-R1 calibration and contract helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class R1Calibration:
    """Persisted calibration adapter with the exact validation-time transform."""

    method: str
    estimator: Any

    def predict(self, raw_scores: Any) -> np.ndarray:
        scores = np.asarray(raw_scores, dtype=float)
        if self.method == "platt_scaling":
            clipped = np.clip(scores, 1e-6, 1.0 - 1e-6)
            logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
            return np.asarray(self.estimator.predict_proba(logits)[:, 1], dtype=float)
        if self.method == "isotonic_regression":
            return np.asarray(self.estimator.predict(scores), dtype=float)
        raise ValueError(f"unsupported R1 calibration method: {self.method}")
