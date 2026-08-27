"""Dataset quality audit, split breakdown, and entity overlap reporting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .schemas import SyntheticDataset, TimeSplit
from .splits import split_by_time


@dataclass
class DatasetQualityReport:
    split_summary: pd.DataFrame
    rings_by_type: pd.DataFrame
    ring_statistics: dict[str, Any]
    entity_overlap: dict[str, float]
    metrics: dict[str, Any]


def compute_dataset_quality(dataset: SyntheticDataset, split: TimeSplit | None = None) -> DatasetQualityReport:
    """Compute comprehensive dataset quality metrics across train, validation, and test splits."""
    if split is None:
        split = split_by_time(dataset.orders, 0.70, 0.15)

    labels_map = dataset.labels.set_index("order_id")

    split_dfs = {
        "Train": split.train,
        "Validation": split.validation,
        "Test": split.test,
    }

    # 1. Summary across splits
    summary_rows = []
    active_rings_by_split = {}
    type_counts_by_split = {}

    all_returns = dataset.returns if not dataset.returns.empty else pd.DataFrame(columns=["order_id"])

    for split_name, o_df in split_dfs.items():
        o_ids = o_df["order_id"]
        sub_labels = labels_map.loc[o_ids]
        abuse_mask = sub_labels["is_abuse"].astype(bool)
        abuse_orders = sub_labels[abuse_mask]
        active_rings = set(abuse_orders["ring_id"].dropna().unique())
        active_rings_by_split[split_name] = active_rings

        type_counts = abuse_orders.groupby("abuse_type")["ring_id"].nunique().to_dict()
        type_counts_by_split[split_name] = type_counts

        ret_count = int(all_returns["order_id"].isin(set(o_ids)).sum()) if not all_returns.empty else 0
        total_loss = float(abuse_orders["loss_amount"].sum())

        summary_rows.append({
            "Split": split_name,
            "Orders": len(o_df),
            "Unique Customers": int(o_df["customer_id"].nunique()),
            "Returns": ret_count,
            "Abuse Orders": int(abuse_mask.sum()),
            "Abuse Rate (%)": float(abuse_mask.mean() * 100.0),
            "Abuse Exposure (INR)": total_loss,
            "Active Rings": int(len(active_rings)),
        })

    summary_df = pd.DataFrame(summary_rows)

    # 2. Ring-type breakdown across splits
    all_ring_types = ["shared_device", "shared_address", "behavioral", "mixed"]
    type_rows = []
    total_rings_df = dataset.rings

    for r_type in all_ring_types:
        total_in_dataset = int((total_rings_df["ring_type"] == r_type).sum()) if not total_rings_df.empty else 0
        type_rows.append({
            "Ring Type": r_type,
            "Train Active": int(type_counts_by_split["Train"].get(r_type, 0)),
            "Validation Active": int(type_counts_by_split["Validation"].get(r_type, 0)),
            "Test Active": int(type_counts_by_split["Test"].get(r_type, 0)),
            "Total Synthetic Rings": total_in_dataset,
        })

    # Add Total row
    type_rows.append({
        "Ring Type": "Total",
        "Train Active": int(len(active_rings_by_split["Train"])),
        "Validation Active": int(len(active_rings_by_split["Validation"])),
        "Test Active": int(len(active_rings_by_split["Test"])),
        "Total Synthetic Rings": len(total_rings_df),
    })
    types_df = pd.DataFrame(type_rows)

    # 3. Structural Ring Statistics
    ring_sizes = total_rings_df["customer_count"].dropna() if "customer_count" in total_rings_df.columns else pd.Series(dtype=float)
    if "start_time" in total_rings_df.columns and "end_time" in total_rings_df.columns:
        start_t = pd.to_datetime(total_rings_df["start_time"])
        end_t = pd.to_datetime(total_rings_df["end_time"])
        durations = (end_t - start_t).dt.total_seconds() / 86400.0
    else:
        durations = pd.Series(dtype=float)

    # Abuse orders per ring
    abuse_only = dataset.labels[dataset.labels["is_abuse"] & dataset.labels["ring_id"].notna()]
    orders_per_ring = abuse_only.groupby("ring_id")["order_id"].count() if not abuse_only.empty else pd.Series(dtype=float)

    ring_stats = {
        "total_rings": int(len(total_rings_df)),
        "mean_ring_size": float(ring_sizes.mean()) if len(ring_sizes) > 0 else 0.0,
        "median_ring_size": float(ring_sizes.median()) if len(ring_sizes) > 0 else 0.0,
        "min_ring_size": int(ring_sizes.min()) if len(ring_sizes) > 0 else 0,
        "max_ring_size": int(ring_sizes.max()) if len(ring_sizes) > 0 else 0,
        "mean_duration_days": float(durations.mean()) if len(durations) > 0 else 0.0,
        "median_duration_days": float(durations.median()) if len(durations) > 0 else 0.0,
        "min_duration_days": float(durations.min()) if len(durations) > 0 else 0.0,
        "max_duration_days": float(durations.max()) if len(durations) > 0 else 0.0,
        "mean_orders_per_ring": float(orders_per_ring.mean()) if len(orders_per_ring) > 0 else 0.0,
        "median_orders_per_ring": float(orders_per_ring.median()) if len(orders_per_ring) > 0 else 0.0,
    }

    # 4. Entity Overlap Analysis between Legitimate and Abusive
    merged_orders = dataset.orders.merge(dataset.labels[["order_id", "is_abuse"]], on="order_id")
    legit_orders = merged_orders[~merged_orders["is_abuse"]]
    abuse_orders_df = merged_orders[merged_orders["is_abuse"]]

    overlap = {}
    for entity in ["device_id", "address_id", "ip_id", "payment_id"]:
        legit_entities = set(legit_orders[entity].dropna().unique())
        abuse_entities = set(abuse_orders_df[entity].dropna().unique())
        shared_count = len(legit_entities.intersection(abuse_entities))
        overlap[f"{entity[:-3]}_overlap_count"] = shared_count
        overlap[f"{entity[:-3]}_abuse_entities_in_legit_pct"] = float(shared_count / max(len(abuse_entities), 1) * 100.0)
        overlap[f"{entity[:-3]}_legit_multi_customer_count"] = int((legit_orders.groupby(entity)["customer_id"].nunique() > 1).sum())

    metrics = {
        "total_customers": len(dataset.customers),
        "total_orders": len(dataset.orders),
        "total_returns": len(dataset.returns),
        "total_rings": len(dataset.rings),
        "train_active_rings": int(len(active_rings_by_split["Train"])),
        "validation_active_rings": int(len(active_rings_by_split["Validation"])),
        "test_active_rings": int(len(active_rings_by_split["Test"])),
        "test_active_by_type": type_counts_by_split["Test"],
    }

    return DatasetQualityReport(
        split_summary=summary_df,
        rings_by_type=types_df,
        ring_statistics=ring_stats,
        entity_overlap=overlap,
        metrics=metrics,
    )
