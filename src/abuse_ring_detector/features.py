"""Leakage-safe streaming behavioural and graph feature builders."""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterable

import numpy as np
import pandas as pd

from .schemas import FeatureSet

ENTITY_COLUMNS = ["device_id", "ip_id", "address_id", "payment_id"]


def _manifest(columns: Iterable[str], source: str) -> pd.DataFrame:
    return pd.DataFrame({"feature": list(columns), "source": source,
                         "as_of_rule": "strictly earlier events only", "target_independent": True})


def build_baseline_features(orders: pd.DataFrame, labels: pd.DataFrame | None = None, history_days: int | None = 30) -> FeatureSet:
    """Build one feature row per order, reading state before the current event."""
    data = orders.sort_values(["event_time", "order_id"], kind="mergesort").reset_index(drop=True)
    labels_by_id = labels.set_index("order_id") if labels is not None else None
    customer_state: dict[str, dict] = defaultdict(lambda: {"times": deque(), "amounts": [], "categories": set(), **{f"n_{c}": set() for c in ENTITY_COLUMNS}})
    rows = []
    for row in data.itertuples(index=False):
        state = customer_state[row.customer_id]
        cutoff = pd.Timestamp(row.event_time) - (pd.Timedelta(days=history_days) if history_days else pd.Timedelta.max)
        while state["times"] and state["times"][0][0] <= cutoff:
            state["times"].popleft()
        prior_times = [t for t, _ in state["times"]]
        prior_amounts = [a for _, a in state["times"]]
        previous = prior_times[-1] if prior_times else None
        values = {
            "order_id": row.order_id, "customer_id": row.customer_id,
            "prior_order_count": len(prior_times), "prior_spend": float(sum(prior_amounts)),
            "prior_avg_amount": float(np.mean(prior_amounts)) if prior_amounts else 0.0,
            "prior_amount_std": float(np.std(prior_amounts)) if len(prior_amounts) > 1 else 0.0,
            "velocity_per_day": len(prior_times) / max(history_days or 1, 1),
            "hours_since_prior": (pd.Timestamp(row.event_time) - previous).total_seconds() / 3600 if previous else 9999.0,
            "amount": float(row.amount), "amount_vs_prior_avg": float(row.amount) / max(float(np.mean(prior_amounts)) if prior_amounts else float(row.amount), 1.0),
            "account_age_days": max(0.0, (pd.Timestamp(row.event_time) - pd.Timestamp("2025-01-01")).total_seconds() / 86400),
            "prior_category_count": len(state["categories"]), "retry_count": float(row.retry_count),
        }
        for entity in ENTITY_COLUMNS:
            values[f"prior_{entity[:-3]}count"] = float(len(state[f"n_{entity}"]))
            values[f"{entity[:-3]}_is_new"] = float(getattr(row, entity) not in state[f"n_{entity}"])
        rows.append(values)
        state["times"].append((pd.Timestamp(row.event_time), float(row.amount)))
        state["categories"].add(row.merchant_category)
        for entity in ENTITY_COLUMNS:
            state[f"n_{entity}"].add(getattr(row, entity))
    X = pd.DataFrame(rows).set_index("order_id")
    y = X.index.to_series().map(labels_by_id.is_abuse).astype(int) if labels_by_id is not None else pd.Series(index=X.index, dtype=int)
    return FeatureSet(X=X.drop(columns="customer_id"), y=y, ids=X.index.to_series(), manifest=_manifest(X.drop(columns="customer_id").columns, "orders/customer history"))


def build_graph_features(orders: pd.DataFrame, labels: pd.DataFrame | None = None, history_days: int = 30) -> FeatureSet:
    """Build numeric customer/entity network features with strict as-of state."""
    data = orders.sort_values(["event_time", "order_id"], kind="mergesort").reset_index(drop=True)
    labels_by_id = labels.set_index("order_id") if labels is not None else None
    customer_entities: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    entity_customers: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    customer_times: dict[str, deque] = defaultdict(deque)
    rows = []
    for row in data.itertuples(index=False):
        now = pd.Timestamp(row.event_time)
        cutoff = now - pd.Timedelta(days=history_days)
        times = customer_times[row.customer_id]
        while times and times[0] <= cutoff:
            times.popleft()
        features = {"order_id": row.order_id, "customer_id": row.customer_id,
                    "graph_prior_orders": float(len(times)), "graph_velocity": len(times) / max(history_days, 1)}
        neighbor_accounts: set[str] = set()
        component_nodes: set[str] = {row.customer_id}
        for entity in ENTITY_COLUMNS:
            value = str(getattr(row, entity))
            prior_customers = entity_customers[entity].get(value, set())
            features[f"{entity[:-3]}_shared_accounts"] = float(len(prior_customers))
            features[f"{entity[:-3]}_is_reused"] = float(bool(prior_customers))
            features[f"{entity[:-3]}_customer_degree"] = float(len(prior_customers))
            neighbor_accounts.update(prior_customers)
            for neighbor in prior_customers:
                component_nodes.update(customer_entities[neighbor].get(entity, set()))
        features["graph_neighbor_count"] = float(len(neighbor_accounts))
        features["graph_component_size_approx"] = float(max(1, len(component_nodes)))
        features["graph_shared_entity_count"] = float(sum(features[f"{e[:-3]}_is_reused"] for e in ENTITY_COLUMNS))
        features["graph_shared_ratio"] = features["graph_shared_entity_count"] / len(ENTITY_COLUMNS)
        rows.append(features)
        times.append(now)
        for entity in ENTITY_COLUMNS:
            value = str(getattr(row, entity))
            customer_entities[row.customer_id][entity].add(value)
            entity_customers[entity][value].add(row.customer_id)
    X = pd.DataFrame(rows).set_index("order_id")
    y = X.index.to_series().map(labels_by_id.is_abuse).astype(int) if labels_by_id is not None else pd.Series(index=X.index, dtype=int)
    return FeatureSet(X=X.drop(columns="customer_id"), y=y, ids=X.index.to_series(), manifest=_manifest(X.drop(columns="customer_id").columns, "historical customer-entity graph"))


def build_combined_features(orders: pd.DataFrame, labels: pd.DataFrame | None = None, history_days: int = 30) -> FeatureSet:
    baseline = build_baseline_features(orders, labels, history_days)
    graph = build_graph_features(orders, labels, history_days)
    X = baseline.X.join(graph.X.drop(columns=[c for c in graph.X if c in baseline.X], errors="ignore"), how="inner")
    return FeatureSet(X=X, y=baseline.y.loc[X.index], ids=X.index.to_series(), manifest=pd.concat([baseline.manifest, graph.manifest], ignore_index=True))
