"""Portable model factory."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .schemas import ModelBundle


def fit_model(X: pd.DataFrame, y: pd.Series, backend: str = "hist_gradient_boosting", seed: int = 42, params: dict[str, Any] | None = None) -> ModelBundle:
    params = params or {}
    requested = backend
    if backend in {"auto", "xgboost"}:
        if backend == "xgboost":
            try:
                from xgboost import XGBClassifier
                estimator = XGBClassifier(random_state=seed, eval_metric="logloss", n_jobs=2, **params)
                actual = "xgboost"
            except ImportError:
                estimator = HistGradientBoostingClassifier(random_state=seed, **params)
                actual = "hist_gradient_boosting_fallback"
        else:
            try:
                from xgboost import XGBClassifier
                estimator = XGBClassifier(random_state=seed, eval_metric="logloss", n_jobs=2, **params)
                actual = "xgboost"
            except ImportError:
                estimator = HistGradientBoostingClassifier(random_state=seed, **params)
                actual = "hist_gradient_boosting"
    else:
        estimator = HistGradientBoostingClassifier(random_state=seed, **params)
        actual = "hist_gradient_boosting"
    estimator.fit(X, y)
    return ModelBundle(estimator=estimator, feature_columns=list(X.columns), backend=actual,
                       feature_manifest=pd.DataFrame(), metadata={"requested_backend": requested, "seed": seed})


def predict_scores(bundle: ModelBundle, X: pd.DataFrame) -> np.ndarray:
    missing = set(bundle.feature_columns) - set(X.columns)
    if missing:
        raise ValueError(f"missing feature columns: {sorted(missing)}")
    return bundle.estimator.predict_proba(X[bundle.feature_columns])[:, 1]
