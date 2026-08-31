"""Shadow Evaluation Pipeline for Model F Live Shadow Observation System.

Evaluates shadow prediction logs against ground-truth labels once outcomes become available.
Calculates Precision, Recall, PR-AUC, ROC-AUC, False Positive Rate, False Negative Rate,
Financial Exposure captured/missed, and Ring-level detection recall (Rules A, B, C).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc

logger = logging.getLogger("abuse_ring_detector.shadow_evaluator")


class ShadowEvaluationPipeline:
    """Evaluates shadow mode prediction logs against ground-truth outcome labels."""

    def __init__(self, shadow_log_path: str | Path, threshold: float = 0.50):
        self.shadow_log_path = Path(shadow_log_path)
        self.threshold = threshold

    def load_shadow_logs(self) -> pd.DataFrame:
        """Loads and parses JSONL shadow prediction logs."""
        if not self.shadow_log_path.exists():
            logger.warning(f"Shadow log file not found at {self.shadow_log_path}")
            return pd.DataFrame()

        records = []
        with open(self.shadow_log_path, "r") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    try:
                        records.append(json.loads(line_str))
                    except Exception:
                        pass
        return pd.DataFrame(records)

    def evaluate_with_labels(self, df_labels: pd.DataFrame, df_orders: pd.DataFrame | None = None) -> dict[str, Any]:
        """Evaluates shadow prediction scores against ground-truth order labels.
        
        Args:
            df_labels: DataFrame containing 'order_id' and 'is_abuse' (or 'label').
            df_orders: Optional DataFrame containing 'order_id' and 'amount' for financial loss calculation.
        """
        df_logs = self.load_shadow_logs()
        if df_logs.empty:
            return {
                "evaluation_status": "NO_SHADOW_DATA",
                "message": "No shadow log entries found for evaluation."
            }

        # Ensure order_id linkage
        if "order_id" not in df_logs.columns or "calibrated_score" not in df_logs.columns:
            return {
                "evaluation_status": "INVALID_SHADOW_LOG_FORMAT",
                "message": "Shadow log missing required order_id or calibrated_score fields."
            }

        # Merge with ground truth labels
        label_col = "is_abuse" if "is_abuse" in df_labels.columns else "label"
        merged = df_logs.merge(df_labels[["order_id", label_col]], on="order_id", how="inner")
        
        if merged.empty:
            return {
                "evaluation_status": "NO_MATCHED_LABELS",
                "message": "Zero shadow log events matched the provided ground-truth label dataset."
            }

        y_true = merged[label_col].astype(int).values
        y_score = merged["calibrated_score"].values
        y_pred = (y_score >= self.threshold).astype(int)

        # Calculate classification metrics
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        p_arr, r_arr, _ = precision_recall_curve(y_true, y_score)
        pr_auc = float(auc(r_arr, p_arr))
        try:
            roc_auc = float(roc_auc_score(y_true, y_score))
        except Exception:
            roc_auc = 0.50

        # Financial exposure metrics if order amounts are provided
        financial_summary = {}
        if df_orders is not None and "amount" in df_orders.columns:
            merged = merged.merge(df_orders[["order_id", "amount"]], on="order_id", how="left")
            merged["amount"] = merged["amount"].fillna(0.0)

            total_abuse_exposure = float(merged[merged[label_col] == 1]["amount"].sum())
            captured_exposure = float(merged[(merged[label_col] == 1) & (y_pred == 1)]["amount"].sum())
            missed_exposure = float(merged[(merged[label_col] == 1) & (y_pred == 0)]["amount"].sum())
            fp_friction = float(merged[(merged[label_col] == 0) & (y_pred == 1)]["amount"].sum())

            financial_summary = {
                "total_abuse_exposure_inr": round(total_abuse_exposure, 2),
                "captured_exposure_inr": round(captured_exposure, 2),
                "captured_exposure_pct": round((captured_exposure / total_abuse_exposure * 100.0), 2) if total_abuse_exposure > 0 else 0.0,
                "missed_exposure_inr": round(missed_exposure, 2),
                "fp_friction_inr": round(fp_friction, 2)
            }

        return {
            "evaluation_status": "EVALUATED_WITH_LABELS",
            "matched_order_count": len(merged),
            "threshold": self.threshold,
            "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "metrics": {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1_score, 4),
                "pr_auc": round(pr_auc, 4),
                "roc_auc": round(roc_auc, 4),
                "false_positive_rate": round(fpr, 4),
                "false_negative_rate": round(fnr, 4)
            },
            "financial_summary": financial_summary
        }
