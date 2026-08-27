"""Ring-level evaluation metrics, Top-K risk prioritisation, and latency analysis."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .reporting import write_frame, write_json


@dataclass
class RingEvaluationResult:
    per_ring: pd.DataFrame
    metrics: dict[str, Any]
    by_ring_type: pd.DataFrame
    top_k: pd.DataFrame


def evaluate_rings(
    test_orders: pd.DataFrame,
    labels: pd.DataFrame,
    scores: np.ndarray,
    threshold: float = 0.5,
    top_k_list: list[int] | None = None,
) -> RingEvaluationResult:
    """Compute ring-level detection metrics, member coverage, ring-type breakdown,
    and Top-K risk prioritisation on held-out test predictions.
    """
    if top_k_list is None:
        top_k_list = [5, 10, 20, 50, 100]

    # Ensure event_time is datetime
    orders_df = test_orders.copy()
    if "event_time" in orders_df.columns:
        orders_df["event_time"] = pd.to_datetime(orders_df["event_time"])

    # Evaluation-only join: labels are attached strictly after predictions exist
    labels_clean = labels[["order_id", "is_abuse", "ring_id", "abuse_type", "loss_amount"]].copy()
    labels_clean["is_abuse"] = labels_clean["is_abuse"].astype(bool)
    labels_clean["loss_amount"] = labels_clean["loss_amount"].fillna(0.0).astype(float)

    eval_df = orders_df[["order_id", "customer_id", "event_time", "amount"]].merge(
        labels_clean, on="order_id", how="inner"
    )

    if len(eval_df) != len(scores):
        raise ValueError(f"Mismatch between test orders ({len(eval_df)}) and scores ({len(scores)})")

    eval_df["predicted_score"] = scores
    eval_df["alert"] = eval_df["predicted_score"] >= threshold

    # Filter to abusive activity in held-out test split
    abuse_test = eval_df[eval_df["is_abuse"] & eval_df["ring_id"].notna()]
    active_ring_ids = sorted(abuse_test["ring_id"].unique())
    total_test_loss = float(abuse_test["loss_amount"].sum())

    ring_rows = []
    for r_id in active_ring_ids:
        r_orders = abuse_test[abuse_test["ring_id"] == r_id]
        r_type = str(r_orders["abuse_type"].iloc[0])
        active_members = set(r_orders["customer_id"].unique())
        total_active_members = len(active_members)

        flagged_r_orders = r_orders[r_orders["alert"]]
        flagged_members = set(flagged_r_orders["customer_id"].unique())
        n_flagged_members = len(flagged_members)

        coverage = (n_flagged_members / total_active_members) if total_active_members > 0 else 0.0
        det_rule_a = n_flagged_members >= 1
        det_rule_b = coverage >= 0.20
        det_rule_c = coverage >= 0.50

        exposure = float(r_orders["loss_amount"].sum())
        exposure_captured = float(flagged_r_orders["loss_amount"].sum())
        exposure_captured_pct = (exposure_captured / exposure * 100.0) if exposure > 0 else 0.0

        if len(flagged_r_orders) > 0:
            first_abuse_time = r_orders["event_time"].min()
            first_flag_time = flagged_r_orders["event_time"].min()
            latency_hours = float((first_flag_time - first_abuse_time).total_seconds() / 3600.0)
        else:
            latency_hours = None

        ring_rows.append({
            "ring_id": r_id,
            "ring_type": r_type,
            "test_orders": len(r_orders),
            "flagged_orders": len(flagged_r_orders),
            "test_members": total_active_members,
            "flagged_members": n_flagged_members,
            "coverage": float(coverage),
            "detected_rule_a": bool(det_rule_a),
            "detected_rule_b": bool(det_rule_b),
            "detected_rule_c": bool(det_rule_c),
            "exposure": exposure,
            "exposure_captured": exposure_captured,
            "exposure_captured_pct": exposure_captured_pct,
            "latency_hours": latency_hours,
        })

    if ring_rows:
        per_ring_df = pd.DataFrame(ring_rows)
    else:
        per_ring_df = pd.DataFrame(columns=[
            "ring_id", "ring_type", "test_orders", "flagged_orders", "test_members",
            "flagged_members", "coverage", "detected_rule_a", "detected_rule_b",
            "detected_rule_c", "exposure", "exposure_captured", "exposure_captured_pct",
            "latency_hours",
        ])

    # Summary metrics
    n_rings = len(per_ring_df)
    latencies = per_ring_df["latency_hours"].dropna() if not per_ring_df.empty else pd.Series(dtype=float)

    metrics = {
        "threshold": float(threshold),
        "total_test_rings": int(n_rings),
        "total_test_abuse_orders": int(len(abuse_test)),
        "total_test_exposure": total_test_loss,
        "rule_a_detected_count": int(per_ring_df["detected_rule_a"].sum()) if n_rings > 0 else 0,
        "rule_a_recall": float(per_ring_df["detected_rule_a"].mean()) if n_rings > 0 else 0.0,
        "rule_b_detected_count": int(per_ring_df["detected_rule_b"].sum()) if n_rings > 0 else 0,
        "rule_b_recall": float(per_ring_df["detected_rule_b"].mean()) if n_rings > 0 else 0.0,
        "rule_c_detected_count": int(per_ring_df["detected_rule_c"].sum()) if n_rings > 0 else 0,
        "rule_c_recall": float(per_ring_df["detected_rule_c"].mean()) if n_rings > 0 else 0.0,
        "mean_member_coverage": float(per_ring_df["coverage"].mean()) if n_rings > 0 else 0.0,
        "median_member_coverage": float(per_ring_df["coverage"].median()) if n_rings > 0 else 0.0,
        "std_member_coverage": float(per_ring_df["coverage"].std()) if n_rings > 1 else 0.0,
        "min_member_coverage": float(per_ring_df["coverage"].min()) if n_rings > 0 else 0.0,
        "max_member_coverage": float(per_ring_df["coverage"].max()) if n_rings > 0 else 0.0,
        "total_exposure_captured": float(per_ring_df["exposure_captured"].sum()) if n_rings > 0 else 0.0,
        "exposure_captured_pct": (float(per_ring_df["exposure_captured"].sum()) / total_test_loss * 100.0) if total_test_loss > 0 else 0.0,
        "latency_mean_hours": float(latencies.mean()) if len(latencies) > 0 else None,
        "latency_median_hours": float(latencies.median()) if len(latencies) > 0 else None,
        "latency_min_hours": float(latencies.min()) if len(latencies) > 0 else None,
        "latency_max_hours": float(latencies.max()) if len(latencies) > 0 else None,
        "latency_std_hours": float(latencies.std()) if len(latencies) > 1 else (0.0 if len(latencies) == 1 else None),
        "latency_count": int(len(latencies)),
    }

    # Ring type breakdown
    if n_rings > 0:
        type_gb = per_ring_df.groupby("ring_type", as_index=False).agg(
            rings=("ring_id", "count"),
            rule_a_recall=("detected_rule_a", "mean"),
            rule_b_recall=("detected_rule_b", "mean"),
            rule_c_recall=("detected_rule_c", "mean"),
            mean_coverage=("coverage", "mean"),
            median_coverage=("coverage", "median"),
            total_exposure=("exposure", "sum"),
            exposure_captured=("exposure_captured", "sum"),
        )
        type_gb["exposure_captured_pct"] = np.where(
            type_gb["total_exposure"] > 0,
            type_gb["exposure_captured"] / type_gb["total_exposure"] * 100.0,
            0.0,
        )
    else:
        type_gb = pd.DataFrame(columns=[
            "ring_type", "rings", "rule_a_recall", "rule_b_recall", "rule_c_recall",
            "mean_coverage", "median_coverage", "total_exposure", "exposure_captured",
            "exposure_captured_pct",
        ])

    # Top-K evaluation
    top_k_df = evaluate_top_k(eval_df, active_ring_ids, top_k_list, total_test_loss)

    return RingEvaluationResult(
        per_ring=per_ring_df,
        metrics=metrics,
        by_ring_type=type_gb,
        top_k=top_k_df,
    )


def evaluate_top_k(
    eval_df: pd.DataFrame,
    active_ring_ids: list[str],
    top_k_list: list[int],
    total_test_loss: float,
) -> pd.DataFrame:
    """Evaluate Top-K alert queue capacity."""
    sorted_df = eval_df.sort_values(
        by=["predicted_score", "event_time", "order_id"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    rows = []
    n_rings = len(active_ring_ids)

    for k in top_k_list:
        effective_k = min(k, len(sorted_df))
        if effective_k == 0:
            rows.append({
                "k": k, "precision": 0.0, "tp_count": 0, "fp_count": 0,
                "rings_touched": 0, "rule_a_detected": 0, "rule_a_recall": 0.0,
                "rule_b_detected": 0, "rule_b_recall": 0.0, "rule_c_detected": 0,
                "rule_c_recall": 0.0, "mean_coverage": 0.0, "exposure_captured": 0.0,
                "exposure_captured_pct": 0.0,
            })
            continue

        top_k_orders = sorted_df.iloc[:effective_k]
        tp_orders = top_k_orders[top_k_orders["is_abuse"]]
        tp_count = len(tp_orders)
        fp_count = effective_k - tp_count
        precision = tp_count / effective_k
        exposure_k = float(tp_orders["loss_amount"].sum())
        exposure_pct = (exposure_k / total_test_loss * 100.0) if total_test_loss > 0 else 0.0

        touched_rings = set(tp_orders["ring_id"].dropna().unique())

        k_detected_a = 0
        k_detected_b = 0
        k_detected_c = 0
        k_coverages = []

        for r_id in active_ring_ids:
            r_all_orders = eval_df[(eval_df["ring_id"] == r_id) & (eval_df["is_abuse"])]
            active_m = set(r_all_orders["customer_id"].unique())
            flagged_m = set(top_k_orders[(top_k_orders["ring_id"] == r_id) & (top_k_orders["is_abuse"])]["customer_id"].unique())

            cov = (len(flagged_m) / len(active_m)) if active_m else 0.0
            k_coverages.append(cov)

            if len(flagged_m) >= 1:
                k_detected_a += 1
            if cov >= 0.20:
                k_detected_b += 1
            if cov >= 0.50:
                k_detected_c += 1

        rows.append({
            "k": k,
            "precision": float(precision),
            "tp_count": int(tp_count),
            "fp_count": int(fp_count),
            "rings_touched": int(len(touched_rings)),
            "rule_a_detected": int(k_detected_a),
            "rule_a_recall": float(k_detected_a / n_rings) if n_rings > 0 else 0.0,
            "rule_b_detected": int(k_detected_b),
            "rule_b_recall": float(k_detected_b / n_rings) if n_rings > 0 else 0.0,
            "rule_c_detected": int(k_detected_c),
            "rule_c_recall": float(k_detected_c / n_rings) if n_rings > 0 else 0.0,
            "mean_coverage": float(np.mean(k_coverages)) if k_coverages else 0.0,
            "exposure_captured": float(exposure_k),
            "exposure_captured_pct": float(exposure_pct),
        })

    return pd.DataFrame(rows)


def compare_models(
    ring_results: dict[str, RingEvaluationResult],
    event_evals: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build the comprehensive multi-model comparison table across all evaluated models."""
    def _top_k_exp(top_k_df: pd.DataFrame, k_val: int) -> str:
        row = top_k_df[top_k_df["k"] == k_val]
        if row.empty:
            return "N/A"
        r = row.iloc[0]
        return f"INR {r['exposure_captured']:,.2f} ({r['exposure_captured_pct']:.1f}%)"

    event_evals = event_evals or {}
    model_names = list(ring_results.keys())

    display_names = {
        "baseline": "Baseline",
        "graph": "Graph-enhanced",
        "graph_temporal": "Graph + Temporal",
        "graph_temporal_custrel": "Graph + Temp + CustRel",
        "graph_temporal_custrel_2hop": "Graph + Temp + CustRel + 2Hop",
    }

    metrics_def = [
        ("Event Precision", lambda m: f"{event_evals.get(m, {}).metrics.get('precision', 0.0):.3f}" if m in event_evals else "N/A"),
        ("Event Recall", lambda m: f"{event_evals.get(m, {}).metrics.get('recall', 0.0):.3f}" if m in event_evals else "N/A"),
        ("Event F1", lambda m: f"{event_evals.get(m, {}).metrics.get('f1', 0.0):.3f}" if m in event_evals else "N/A"),
        ("PR-AUC", lambda m: f"{event_evals.get(m, {}).metrics.get('pr_auc', 0.0):.3f}" if m in event_evals else "N/A"),
        ("ROC-AUC", lambda m: f"{event_evals.get(m, {}).metrics.get('roc_auc', 0.0):.3f}" if m in event_evals else "N/A"),
        ("Any-member Ring Recall (Rule A)", lambda m: f"{ring_results[m].metrics.get('rule_a_recall', 0.0):.3f} ({ring_results[m].metrics.get('rule_a_detected_count', 0)}/{ring_results[m].metrics.get('total_test_rings', 0)})"),
        ("20% Coverage Ring Recall (Rule B)", lambda m: f"{ring_results[m].metrics.get('rule_b_recall', 0.0):.3f} ({ring_results[m].metrics.get('rule_b_detected_count', 0)}/{ring_results[m].metrics.get('total_test_rings', 0)})"),
        ("50% Coverage Ring Recall (Rule C)", lambda m: f"{ring_results[m].metrics.get('rule_c_recall', 0.0):.3f} ({ring_results[m].metrics.get('rule_c_detected_count', 0)}/{ring_results[m].metrics.get('total_test_rings', 0)})"),
        ("Mean Member Coverage", lambda m: f"{ring_results[m].metrics.get('mean_member_coverage', 0.0):.3f}"),
        ("Median Member Coverage", lambda m: f"{ring_results[m].metrics.get('median_member_coverage', 0.0):.3f}"),
        ("Exposure Captured at Threshold", lambda m: f"INR {ring_results[m].metrics.get('total_exposure_captured', 0.0):,.2f} ({ring_results[m].metrics.get('exposure_captured_pct', 0.0):.1f}%)"),
        ("Expected Financial Loss", lambda m: f"INR {event_evals.get(m, {}).metrics.get('expected_loss', 0.0):,.2f}" if m in event_evals else "N/A"),
        ("Top-5 Exposure Captured", lambda m: _top_k_exp(ring_results[m].top_k, 5)),
        ("Top-10 Exposure Captured", lambda m: _top_k_exp(ring_results[m].top_k, 10)),
        ("Top-20 Exposure Captured", lambda m: _top_k_exp(ring_results[m].top_k, 20)),
        ("Top-50 Exposure Captured", lambda m: _top_k_exp(ring_results[m].top_k, 50)),
        ("Top-100 Exposure Captured", lambda m: _top_k_exp(ring_results[m].top_k, 100)),
        ("Mean Detection Latency (hours)", lambda m: f"{ring_results[m].metrics.get('latency_mean_hours', 0.0):.1f}" if ring_results[m].metrics.get("latency_mean_hours") is not None else "N/A"),
        ("Median Detection Latency (hours)", lambda m: f"{ring_results[m].metrics.get('latency_median_hours', 0.0):.1f}" if ring_results[m].metrics.get("latency_median_hours") is not None else "N/A"),
    ]

    rows = []
    for metric_label, extractor in metrics_def:
        row = {"Metric": metric_label}
        for m in model_names:
            col_name = display_names.get(m, m.replace("_", " ").title())
            row[col_name] = extractor(m)
        rows.append(row)

    return pd.DataFrame(rows)


def compare_baseline_vs_graph(
    baseline_result: RingEvaluationResult,
    graph_result: RingEvaluationResult,
    baseline_event_eval: Any = None,
    graph_event_eval: Any = None,
) -> pd.DataFrame:
    """Build the baseline vs graph comparison table (backward-compatible)."""
    ring_results = {"baseline": baseline_result, "graph": graph_result}
    event_evals = {}
    if baseline_event_eval:
        event_evals["baseline"] = baseline_event_eval
    if graph_event_eval:
        event_evals["graph"] = graph_event_eval
    return compare_models(ring_results, event_evals)


def save_ring_evaluation_artifacts(
    eval_dir: Path,
    baseline_or_dict: RingEvaluationResult | dict[str, RingEvaluationResult],
    graph_result_or_df: RingEvaluationResult | pd.DataFrame | None = None,
    comparison_df: pd.DataFrame | None = None,
) -> None:
    """Save all ring evaluation artifacts in structured CSV and JSON formats."""
    eval_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(baseline_or_dict, dict):
        ring_dict = baseline_or_dict
        comp_df = graph_result_or_df if isinstance(graph_result_or_df, pd.DataFrame) else pd.DataFrame()
    else:
        ring_dict = {
            "baseline": baseline_or_dict,
            "graph": graph_result_or_df,
        }
        comp_df = comparison_df if comparison_df is not None else pd.DataFrame()

    per_ring_list = []
    by_type_list = []
    top_k_list = []
    combined_json = {"summary": comp_df.to_dict(orient="records")}

    for mode, r_res in ring_dict.items():
        if r_res is None:
            continue
        p_df = r_res.per_ring.copy()
        p_df["model"] = mode
        per_ring_list.append(p_df)

        t_df = r_res.by_ring_type.copy()
        t_df["model"] = mode
        by_type_list.append(t_df)

        k_df = r_res.top_k.copy()
        k_df["model"] = mode
        top_k_list.append(k_df)

        write_json(r_res.metrics, eval_dir / f"{mode}_ring_metrics.json")
        combined_json[mode] = {
            "metrics": r_res.metrics,
            "by_ring_type": r_res.by_ring_type.to_dict(orient="records"),
            "top_k": r_res.top_k.to_dict(orient="records"),
        }

    if per_ring_list:
        write_frame(pd.concat(per_ring_list, ignore_index=True), eval_dir / "per_ring_evaluation.csv")
    if by_type_list:
        write_frame(pd.concat(by_type_list, ignore_index=True), eval_dir / "ring_type_breakdown.csv")
    if top_k_list:
        write_frame(pd.concat(top_k_list, ignore_index=True), eval_dir / "top_k_analysis.csv")
    if not comp_df.empty:
        write_frame(comp_df, eval_dir / "ring_summary.csv")

    write_json(combined_json, eval_dir / "ring_metrics.json")

