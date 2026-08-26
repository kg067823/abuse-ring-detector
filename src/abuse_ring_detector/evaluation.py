"""Evaluation, threshold selection, and explicit financial cost analysis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


@dataclass
class CostModel:
    review_cost: float = 2.0
    false_positive_block_cost: float = 10.0


@dataclass
class EvaluationResult:
    metrics: dict[str, float]
    threshold_table: pd.DataFrame


def threshold_table(y_true: pd.Series, scores: np.ndarray, loss_amount: pd.Series | None = None, cost: CostModel | None = None, thresholds=None) -> pd.DataFrame:
    cost = cost or CostModel()
    thresholds = thresholds if thresholds is not None else [.5, .6, .7, .8, .9]
    loss_amount = loss_amount if loss_amount is not None else pd.Series(np.ones(len(y_true)), index=y_true.index)
    rows = []
    for threshold in thresholds:
        predicted = scores >= threshold
        fp = ((predicted) & (np.asarray(y_true) == 0))
        fn = ((~predicted) & (np.asarray(y_true) == 1))
        rows.append({"threshold": threshold, "alerts": int(predicted.sum()), "alert_rate": float(predicted.mean()),
                     "precision": precision_score(y_true, predicted, zero_division=0), "recall": recall_score(y_true, predicted, zero_division=0),
                     "f1": f1_score(y_true, predicted, zero_division=0), "expected_loss": float(loss_amount.to_numpy()[fn].sum() + fp.sum() * (cost.review_cost + cost.false_positive_block_cost))})
    return pd.DataFrame(rows)


def evaluate_predictions(y_true: pd.Series, scores: np.ndarray, threshold: float = .5, loss_amount: pd.Series | None = None, cost: CostModel | None = None) -> EvaluationResult:
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    cost = cost or CostModel()
    if loss_amount is None:
        loss_amount = pd.Series(np.ones(len(y_true)), index=y_true.index)
    fn_mask = (~predicted) & (np.asarray(y_true) == 1)
    fp_mask = predicted & (np.asarray(y_true) == 0)
    expected_loss = float(loss_amount.to_numpy()[fn_mask].sum() + fp_mask.sum() * (cost.review_cost + cost.false_positive_block_cost))
    metrics = {"threshold": threshold, "precision": float(precision_score(y_true, predicted, zero_division=0)),
               "recall": float(recall_score(y_true, predicted, zero_division=0)), "f1": float(f1_score(y_true, predicted, zero_division=0)),
               "pr_auc": float(average_precision_score(y_true, scores)), "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else float("nan"),
               "false_positive_rate": float(fp / max(fp + tn, 1)), "false_negative_rate": float(fn / max(fn + tp, 1)),
               "true_positives": int(tp), "false_positives": int(fp), "false_negatives": int(fn), "alerts": int(predicted.sum()),
               "expected_loss": expected_loss}
    table = threshold_table(y_true, scores, loss_amount, cost)
    return EvaluationResult(metrics=metrics, threshold_table=table)


def choose_threshold(validation: EvaluationResult) -> float:
    row = validation.threshold_table.sort_values(["expected_loss", "threshold"]).iloc[0]
    return float(row.threshold)
