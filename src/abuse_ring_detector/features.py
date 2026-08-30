"""Leakage-safe streaming behavioural, graph, and temporal edge velocity feature builders."""
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


def _clean_entity(val: Any) -> str:
    if val is None or pd.isna(val):
        return ""
    s = str(val).strip()
    if s.lower() in ("", "nan", "none", "null", "<na>", "0", "0.0"):
        return ""
    return s


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
            "prior_category_count": len(state["categories"]), "retry_count": float(getattr(row, "retry_count", 0.0)),
        }
        for entity in ENTITY_COLUMNS:
            ent_val = _clean_entity(getattr(row, entity, ""))
            values[f"prior_{entity[:-3]}count"] = float(len(state[f"n_{entity}"]))
            values[f"{entity[:-3]}_is_new"] = float(bool(ent_val) and ent_val not in state[f"n_{entity}"])
        rows.append(values)
        state["times"].append((pd.Timestamp(row.event_time), float(row.amount)))
        state["categories"].add(row.merchant_category)
        for entity in ENTITY_COLUMNS:
            ent_val = _clean_entity(getattr(row, entity, ""))
            if ent_val:
                state[f"n_{entity}"].add(ent_val)
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
            value = _clean_entity(getattr(row, entity, ""))
            if not value:
                features[f"{entity[:-3]}_shared_accounts"] = 0.0
                features[f"{entity[:-3]}_is_reused"] = 0.0
                features[f"{entity[:-3]}_customer_degree"] = 0.0
                continue
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
            value = _clean_entity(getattr(row, entity, ""))
            if value:
                customer_entities[row.customer_id][entity].add(value)
                entity_customers[entity][value].add(row.customer_id)
    X = pd.DataFrame(rows).set_index("order_id")
    y = X.index.to_series().map(labels_by_id.is_abuse).astype(int) if labels_by_id is not None else pd.Series(index=X.index, dtype=int)
    return FeatureSet(X=X.drop(columns="customer_id"), y=y, ids=X.index.to_series(), manifest=_manifest(X.drop(columns="customer_id").columns, "historical customer-entity graph"))


def build_temporal_velocity_features(orders: pd.DataFrame, labels: pd.DataFrame | None = None) -> FeatureSet:
    """Build streaming sliding-window temporal connection velocity and burst features for entities."""
    data = orders.sort_values(["event_time", "order_id"], kind="mergesort").reset_index(drop=True)
    labels_by_id = labels.set_index("order_id") if labels is not None else None

    entity_history: dict[str, dict[str, deque[tuple[pd.Timestamp, str]]]] = {
        col: defaultdict(deque) for col in ENTITY_COLUMNS
    }
    entity_first_seen_cust: dict[str, dict[str, set[str]]] = {
        col: defaultdict(set) for col in ENTITY_COLUMNS
    }

    rows = []

    for row in data.itertuples(index=False):
        now = pd.Timestamp(row.event_time)
        cutoff_24h = now - pd.Timedelta(hours=24)
        cutoff_1h = now - pd.Timedelta(hours=1)

        feature_dict = {
            "order_id": row.order_id,
            "customer_id": row.customer_id,
        }

        distinct_1h_all = []
        distinct_24h_all = []
        order_count_1h_all = []
        order_count_24h_all = []
        burst_ratio_all = []
        burst_events_1h = 0

        for entity_col in ENTITY_COLUMNS:
            entity_val = str(getattr(row, entity_col))
            prefix = entity_col[:-3]
            if not entity_val or entity_val in ("nan", "None"):
                feature_dict[f"{prefix}_distinct_customers_1h"] = 0.0
                feature_dict[f"{prefix}_distinct_customers_24h"] = 0.0
                feature_dict[f"{prefix}_order_count_1h"] = 0.0
                feature_dict[f"{prefix}_order_count_24h"] = 0.0
                feature_dict[f"{prefix}_new_customers_1h"] = 0.0
                feature_dict[f"{prefix}_burst_ratio_1h_24h"] = 0.0
                distinct_1h_all.append(0.0)
                distinct_24h_all.append(0.0)
                order_count_1h_all.append(0.0)
                order_count_24h_all.append(0.0)
                burst_ratio_all.append(0.0)
                continue

            hist = entity_history[entity_col][entity_val]
            first_seen_set = entity_first_seen_cust[entity_col][entity_val]

            # Evict events older than 24 hours
            while hist and hist[0][0] <= cutoff_24h:
                hist.popleft()

            # 24-hour window stats
            orders_24h = len(hist)
            custs_24h = set(c for _, c in hist)
            distinct_custs_24h = len(custs_24h)

            # 1-hour window stats
            orders_1h = sum(1 for t, _ in hist if t > cutoff_1h)
            custs_1h = set(c for t, c in hist if t > cutoff_1h)
            distinct_custs_1h = len(custs_1h)

            # Customers new to this entity in the 1h window (not seen prior to 1h cutoff)
            new_custs_1h = sum(1 for c in custs_1h if c not in (first_seen_set - custs_1h))

            # 1h burst ratio relative to 24h baseline hourly rate
            hourly_avg_24h = max(1.0, orders_24h / 24.0)
            burst_ratio = orders_1h / hourly_avg_24h

            feature_dict[f"{prefix}_distinct_customers_1h"] = float(distinct_custs_1h)
            feature_dict[f"{prefix}_distinct_customers_24h"] = float(distinct_custs_24h)
            feature_dict[f"{prefix}_order_count_1h"] = float(orders_1h)
            feature_dict[f"{prefix}_order_count_24h"] = float(orders_24h)
            feature_dict[f"{prefix}_new_customers_1h"] = float(new_custs_1h)
            feature_dict[f"{prefix}_burst_ratio_1h_24h"] = float(burst_ratio)

            distinct_1h_all.append(distinct_custs_1h)
            distinct_24h_all.append(distinct_custs_24h)
            order_count_1h_all.append(orders_1h)
            order_count_24h_all.append(orders_24h)
            burst_ratio_all.append(burst_ratio)
            if distinct_custs_1h >= 2:
                burst_events_1h += 1

        # Cross-entity aggregate burst features
        feature_dict["max_entity_distinct_customers_1h"] = float(max(distinct_1h_all))
        feature_dict["max_entity_distinct_customers_24h"] = float(max(distinct_24h_all))
        feature_dict["max_entity_order_count_1h"] = float(max(order_count_1h_all))
        feature_dict["max_entity_order_count_24h"] = float(max(order_count_24h_all))
        feature_dict["max_entity_burst_ratio"] = float(max(burst_ratio_all))
        feature_dict["total_entity_burst_events_1h"] = float(burst_events_1h)

        rows.append(feature_dict)

        # Update streaming state strictly after extracting features for current event
        for entity_col in ENTITY_COLUMNS:
            entity_val = str(getattr(row, entity_col))
            if entity_val and entity_val not in ("nan", "None"):
                entity_history[entity_col][entity_val].append((now, row.customer_id))
                entity_first_seen_cust[entity_col][entity_val].add(row.customer_id)

    X = pd.DataFrame(rows).set_index("order_id")
    y = X.index.to_series().map(labels_by_id.is_abuse).astype(int) if labels_by_id is not None else pd.Series(index=X.index, dtype=int)
    return FeatureSet(X=X.drop(columns="customer_id"), y=y, ids=X.index.to_series(),
                      manifest=_manifest(X.drop(columns="customer_id").columns, "streaming entity connection velocity"))


def build_customer_relative_features(orders: pd.DataFrame, labels: pd.DataFrame | None = None) -> FeatureSet:
    """Build streaming sliding-window customer-relative velocity and burst features."""
    data = orders.sort_values(["event_time", "order_id"], kind="mergesort").reset_index(drop=True)
    labels_by_id = labels.set_index("order_id") if labels is not None else None

    # State per customer: deque of (Timestamp, dict of entity_col -> entity_val)
    # 30-day rolling window
    customer_history: dict[str, deque[tuple[pd.Timestamp, dict[str, str]]]] = defaultdict(deque)

    rows = []

    for row in data.itertuples(index=False):
        now = pd.Timestamp(row.event_time)
        cutoff_30d = now - pd.Timedelta(days=30)
        cutoff_7d = now - pd.Timedelta(days=7)
        cutoff_24h = now - pd.Timedelta(hours=24)
        cutoff_1h = now - pd.Timedelta(hours=1)

        hist = customer_history[row.customer_id]
        while hist and hist[0][0] <= cutoff_30d:
            hist.popleft()

        # Overall customer counts
        c_orders_30d = len(hist)
        c_orders_7d = sum(1 for t, _ in hist if t > cutoff_7d)
        c_orders_24h = sum(1 for t, _ in hist if t > cutoff_24h)
        c_orders_1h = sum(1 for t, _ in hist if t > cutoff_1h)

        # Activity burst ratios
        c_rate_1h_vs_24h = c_orders_1h / max(1.0, c_orders_24h / 24.0)
        c_rate_24h_vs_7d = c_orders_24h / max(1.0, c_orders_7d / 7.0)
        c_burst_score_24h_30d = c_orders_24h / max(0.1, c_orders_30d / 30.0)

        feature_dict = {
            "order_id": row.order_id,
            "customer_id": row.customer_id,
            "customer_order_count_1h": float(c_orders_1h),
            "customer_order_count_24h": float(c_orders_24h),
            "customer_order_count_7d": float(c_orders_7d),
            "customer_order_count_30d": float(c_orders_30d),
            "customer_orders_1h_vs_24h_rate": float(c_rate_1h_vs_24h),
            "customer_orders_24h_vs_7d_rate": float(c_rate_24h_vs_7d),
            "customer_activity_burst_score": float(c_burst_score_24h_30d),
        }

        entity_orders_1h_list = []
        entity_orders_24h_list = []
        entity_velocity_ratio_list = []

        curr_entities = {col: str(getattr(row, col)) for col in ENTITY_COLUMNS}

        for entity_col in ENTITY_COLUMNS:
            entity_val = curr_entities[entity_col]
            prefix = entity_col[:-3]

            if not entity_val or entity_val in ("nan", "None"):
                feature_dict[f"customer_{prefix}_orders_1h"] = 0.0
                feature_dict[f"customer_{prefix}_orders_24h"] = 0.0
                feature_dict[f"customer_{prefix}_orders_7d"] = 0.0
                feature_dict[f"customer_{prefix}_velocity_ratio"] = 0.0
                feature_dict[f"customer_{prefix}_share_of_activity_24h"] = 0.0
                entity_orders_1h_list.append(0.0)
                entity_orders_24h_list.append(0.0)
                entity_velocity_ratio_list.append(0.0)
                continue

            # Count customer's orders using this specific entity
            e_7d = sum(1 for t, e_map in hist if t > cutoff_7d and e_map.get(entity_col) == entity_val)
            e_24h = sum(1 for t, e_map in hist if t > cutoff_24h and e_map.get(entity_col) == entity_val)
            e_1h = sum(1 for t, e_map in hist if t > cutoff_1h and e_map.get(entity_col) == entity_val)

            e_vel_ratio = e_24h / max(1.0, e_7d / 7.0)
            e_share_24h = e_24h / max(1.0, float(c_orders_24h))

            feature_dict[f"customer_{prefix}_orders_1h"] = float(e_1h)
            feature_dict[f"customer_{prefix}_orders_24h"] = float(e_24h)
            feature_dict[f"customer_{prefix}_orders_7d"] = float(e_7d)
            feature_dict[f"customer_{prefix}_velocity_ratio"] = float(e_vel_ratio)
            feature_dict[f"customer_{prefix}_share_of_activity_24h"] = float(e_share_24h)

            entity_orders_1h_list.append(e_1h)
            entity_orders_24h_list.append(e_24h)
            entity_velocity_ratio_list.append(e_vel_ratio)

        feature_dict["max_customer_entity_orders_1h"] = float(max(entity_orders_1h_list))
        feature_dict["max_customer_entity_orders_24h"] = float(max(entity_orders_24h_list))
        feature_dict["max_customer_entity_velocity_ratio"] = float(max(entity_velocity_ratio_list))

        rows.append(feature_dict)

        # Update streaming state strictly after recording feature row
        hist.append((now, curr_entities))

    X = pd.DataFrame(rows).set_index("order_id")
    y = X.index.to_series().map(labels_by_id.is_abuse).astype(int) if labels_by_id is not None else pd.Series(index=X.index, dtype=int)
    return FeatureSet(X=X.drop(columns="customer_id"), y=y, ids=X.index.to_series(),
                      manifest=_manifest(X.drop(columns="customer_id").columns, "streaming customer-relative entity velocity"))


def build_combined_features(orders: pd.DataFrame, labels: pd.DataFrame | None = None, history_days: int = 30) -> FeatureSet:
    """Build combined baseline and graph features."""
    baseline = build_baseline_features(orders, labels, history_days)
    graph = build_graph_features(orders, labels, history_days)
    X = baseline.X.join(graph.X.drop(columns=[c for c in graph.X if c in baseline.X], errors="ignore"), how="inner")
    return FeatureSet(X=X, y=baseline.y.loc[X.index], ids=X.index.to_series(), manifest=pd.concat([baseline.manifest, graph.manifest], ignore_index=True))


def build_full_features(orders: pd.DataFrame, labels: pd.DataFrame | None = None, history_days: int = 30) -> FeatureSet:
    """Build full baseline, graph, and temporal velocity features (67 features)."""
    combined = build_combined_features(orders, labels, history_days)
    temporal = build_temporal_velocity_features(orders, labels)
    X = combined.X.join(temporal.X.drop(columns=[c for c in temporal.X if c in combined.X], errors="ignore"), how="inner")
    return FeatureSet(X=X, y=combined.y.loc[X.index], ids=X.index.to_series(), manifest=pd.concat([combined.manifest, temporal.manifest], ignore_index=True))


def build_extended_features(orders: pd.DataFrame, labels: pd.DataFrame | None = None, history_days: int = 30) -> FeatureSet:
    """Build extended features: baseline, graph, temporal velocity, and customer-relative features (97 features)."""
    full = build_full_features(orders, labels, history_days)
    cust_rel = build_customer_relative_features(orders, labels)
    X = full.X.join(cust_rel.X.drop(columns=[c for c in cust_rel.X if c in full.X], errors="ignore"), how="inner")
    return FeatureSet(X=X, y=full.y.loc[X.index], ids=X.index.to_series(), manifest=pd.concat([full.manifest, cust_rel.manifest], ignore_index=True))


def build_two_hop_features(orders: pd.DataFrame, labels: pd.DataFrame | None = None) -> FeatureSet:
    """Build streaming sliding-window 2-hop graph features (20 features)."""
    data = orders.sort_values(["event_time", "order_id"], kind="mergesort").reset_index(drop=True)
    labels_by_id = labels.set_index("order_id") if labels is not None else None

    # Inverted index: entity_col -> entity_val -> deque of (timestamp, customer_id)
    entity_history: dict[str, dict[str, deque[tuple[pd.Timestamp, str]]]] = {
        col: defaultdict(deque) for col in ENTITY_COLUMNS
    }
    # Customer history: customer_id -> deque of (timestamp, dict of entities)
    customer_history: dict[str, deque[tuple[pd.Timestamp, dict[str, str]]]] = defaultdict(deque)

    rows = []

    for row in data.itertuples(index=False):
        now = pd.Timestamp(row.event_time)
        cutoff_30d = now - pd.Timedelta(days=30)
        cutoff_7d = now - pd.Timedelta(days=7)
        cutoff_24h = now - pd.Timedelta(hours=24)

        curr_cust = row.customer_id
        curr_entities = {col: _clean_entity(getattr(row, col, "")) for col in ENTITY_COLUMNS}
        curr_d = curr_entities["device_id"]
        curr_a = curr_entities["address_id"]
        curr_ip = curr_entities["ip_id"]
        curr_p = curr_entities["payment_id"]

        # Prune entity history
        for col in ENTITY_COLUMNS:
            val = curr_entities[col]
            dq = entity_history[col][val]
            while dq and dq[0][0] <= cutoff_30d:
                dq.popleft()

        # Step 1: 1-hop connected customers (peers) per entity within 7d and 30d
        peers_by_entity_7d: dict[str, set[str]] = {}
        peers_by_entity_30d: dict[str, set[str]] = {}

        for col in ENTITY_COLUMNS:
            val = curr_entities[col]
            if not val:
                peers_by_entity_7d[col] = set()
                peers_by_entity_30d[col] = set()
                continue
            dq = entity_history[col][val]
            p_7d = {c for t, c in dq if t > cutoff_7d and c != curr_cust}
            p_30d = {c for t, c in dq if t > cutoff_30d and c != curr_cust}
            peers_by_entity_7d[col] = p_7d
            peers_by_entity_30d[col] = p_30d

        dev_peers_7d = peers_by_entity_7d["device_id"]
        addr_peers_7d = peers_by_entity_7d["address_id"]
        ip_peers_7d = peers_by_entity_7d["ip_id"]
        pay_peers_7d = peers_by_entity_7d["payment_id"]

        all_peers_7d = dev_peers_7d | addr_peers_7d | ip_peers_7d | pay_peers_7d
        all_peers_30d = (
            peers_by_entity_30d["device_id"]
            | peers_by_entity_30d["address_id"]
            | peers_by_entity_30d["ip_id"]
            | peers_by_entity_30d["payment_id"]
        )

        # Step 2: 2-hop entity expansions via specific peer sets
        conn_addr_via_dev = set()
        for c in dev_peers_7d:
            for t, e_map in customer_history[c]:
                if t > cutoff_7d:
                    a_val = e_map.get("address_id")
                    if a_val and a_val != curr_a:
                        conn_addr_via_dev.add(a_val)

        conn_dev_via_addr = set()
        for c in addr_peers_7d:
            for t, e_map in customer_history[c]:
                if t > cutoff_7d:
                    d_val = e_map.get("device_id")
                    if d_val and d_val != curr_d:
                        conn_dev_via_addr.add(d_val)

        conn_pay_via_dev = set()
        for c in dev_peers_7d:
            for t, e_map in customer_history[c]:
                if t > cutoff_7d:
                    p_val = e_map.get("payment_id")
                    if p_val and p_val != curr_p:
                        conn_pay_via_dev.add(p_val)

        conn_pay_via_addr = set()
        for c in addr_peers_7d:
            for t, e_map in customer_history[c]:
                if t > cutoff_7d:
                    p_val = e_map.get("payment_id")
                    if p_val and p_val != curr_p:
                        conn_pay_via_addr.add(p_val)

        conn_dev_via_ip = set()
        for c in ip_peers_7d:
            for t, e_map in customer_history[c]:
                if t > cutoff_7d:
                    d_val = e_map.get("device_id")
                    if d_val and d_val != curr_d:
                        conn_dev_via_ip.add(d_val)

        conn_addr_via_ip = set()
        for c in ip_peers_7d:
            for t, e_map in customer_history[c]:
                if t > cutoff_7d:
                    a_val = e_map.get("address_id")
                    if a_val and a_val != curr_a:
                        conn_addr_via_ip.add(a_val)

        # Step 3: Total 2-hop peer entities and peer order volume
        all_peer_devices_7d = set()
        all_peer_addresses_7d = set()
        all_peer_payments_7d = set()

        peer_orders_7d_total = 0
        peer_orders_24h_total = 0
        peer_max_orders_7d = 0

        for c in all_peers_7d:
            c_hist = customer_history[c]
            c_cnt_7d = 0
            for t, e_map in c_hist:
                if t > cutoff_7d:
                    c_cnt_7d += 1
                    peer_orders_7d_total += 1
                    d_v = e_map.get("device_id")
                    a_v = e_map.get("address_id")
                    p_v = e_map.get("payment_id")
                    if d_v and d_v != curr_d:
                        all_peer_devices_7d.add(d_v)
                    if a_v and a_v != curr_a:
                        all_peer_addresses_7d.add(a_v)
                    if p_v and p_v != curr_p:
                        all_peer_payments_7d.add(p_v)
                if t > cutoff_24h:
                    peer_orders_24h_total += 1
            if c_cnt_7d > peer_max_orders_7d:
                peer_max_orders_7d = c_cnt_7d

        peer_cluster_size_7d = (
            len(all_peers_7d)
            + len(all_peer_devices_7d)
            + len(all_peer_addresses_7d)
            + len(all_peer_payments_7d)
        )

        peer_vel_ratio = float(peer_orders_24h_total) / max(1.0, float(peer_orders_7d_total) / 7.0)

        # Step 4: Cross-entity shared customer triangles
        pairs_7d = 0
        pairs = [
            ("device_id", "address_id"),
            ("device_id", "ip_id"),
            ("device_id", "payment_id"),
            ("address_id", "ip_id"),
            ("address_id", "payment_id"),
            ("ip_id", "payment_id"),
        ]
        for col1, col2 in pairs:
            if peers_by_entity_7d[col1] & peers_by_entity_7d[col2]:
                pairs_7d += 1

        feature_dict = {
            "order_id": row.order_id,
            "customer_id": row.customer_id,
            "two_hop_shared_device_customers_7d": float(len(dev_peers_7d)),
            "two_hop_shared_address_customers_7d": float(len(addr_peers_7d)),
            "two_hop_shared_ip_customers_7d": float(len(ip_peers_7d)),
            "two_hop_shared_payment_customers_7d": float(len(pay_peers_7d)),
            "two_hop_distinct_connected_customers_7d": float(len(all_peers_7d)),
            "two_hop_distinct_connected_customers_30d": float(len(all_peers_30d)),
            "two_hop_connected_addresses_via_device_7d": float(len(conn_addr_via_dev)),
            "two_hop_connected_devices_via_address_7d": float(len(conn_dev_via_addr)),
            "two_hop_connected_payments_via_device_7d": float(len(conn_pay_via_dev)),
            "two_hop_connected_payments_via_address_7d": float(len(conn_pay_via_addr)),
            "two_hop_connected_devices_via_ip_7d": float(len(conn_dev_via_ip)),
            "two_hop_connected_addresses_via_ip_7d": float(len(conn_addr_via_ip)),
            "two_hop_total_peer_devices_7d": float(len(all_peer_devices_7d)),
            "two_hop_total_peer_addresses_7d": float(len(all_peer_addresses_7d)),
            "two_hop_total_peer_payments_7d": float(len(all_peer_payments_7d)),
            "two_hop_peer_cluster_size_7d": float(peer_cluster_size_7d),
            "two_hop_total_peer_orders_7d": float(peer_orders_7d_total),
            "two_hop_max_peer_orders_7d": float(peer_max_orders_7d),
            "two_hop_peer_velocity_ratio_24h_vs_7d": float(peer_vel_ratio),
            "two_hop_cross_entity_shared_cust_count_7d": float(pairs_7d),
        }

        rows.append(feature_dict)

        for col in ENTITY_COLUMNS:
            val = curr_entities[col]
            if val:
                entity_history[col][val].append((now, curr_cust))
        customer_history[curr_cust].append((now, curr_entities))

    X = pd.DataFrame(rows).set_index("order_id")
    y = X.index.to_series().map(labels_by_id.is_abuse).astype(int) if labels_by_id is not None else pd.Series(index=X.index, dtype=int)
    return FeatureSet(X=X.drop(columns="customer_id"), y=y, ids=X.index.to_series(),
                      manifest=_manifest(X.drop(columns="customer_id").columns, "streaming 2-hop graph features"))


def build_two_hop_extended_features(orders: pd.DataFrame, labels: pd.DataFrame | None = None, history_days: int = 30) -> FeatureSet:
    """Build Model E features (117 features): baseline (19) + graph (18) + entity temporal (30) + cust rel (30) + 2-hop (20)."""
    d_fs = build_extended_features(orders, labels, history_days)
    two_hop_fs = build_two_hop_features(orders, labels)
    X = d_fs.X.join(two_hop_fs.X.drop(columns=[c for c in two_hop_fs.X if c in d_fs.X], errors="ignore"), how="inner")
    return FeatureSet(X=X, y=d_fs.y.loc[X.index], ids=X.index.to_series(), manifest=pd.concat([d_fs.manifest, two_hop_fs.manifest], ignore_index=True))


def build_subgraph_features(orders: pd.DataFrame, labels: pd.DataFrame | None = None) -> FeatureSet:
    """Build streaming sliding-window suspicious subgraph features (20 features)."""
    data = orders.sort_values(["event_time", "order_id"], kind="mergesort").reset_index(drop=True)
    labels_by_id = labels.set_index("order_id") if labels is not None else None

    # Inverted indices for bipartite graph:
    # entity_col -> entity_val -> deque of (Timestamp, customer_id, order_id)
    entity_events: dict[str, dict[str, deque[tuple[pd.Timestamp, str, str]]]] = {
        col: defaultdict(deque) for col in ENTITY_COLUMNS
    }
    # customer_id -> deque of (Timestamp, dict of entity_col -> entity_val, order_id)
    customer_events: dict[str, deque[tuple[pd.Timestamp, dict[str, str], str]]] = defaultdict(deque)

    # Node first appearance tracker in local state: node_id -> first_seen_time
    node_first_seen: dict[str, pd.Timestamp] = {}
    edge_first_seen: dict[tuple[str, str], pd.Timestamp] = {}

    rows = []

    for row in data.itertuples(index=False):
        now = pd.Timestamp(row.event_time)
        cutoff_7d = now - pd.Timedelta(days=7)
        cutoff_24h = now - pd.Timedelta(hours=24)
        cutoff_1h = now - pd.Timedelta(hours=1)

        curr_cust = row.customer_id
        curr_entities = {col: _clean_entity(getattr(row, col, "")) for col in ENTITY_COLUMNS}

        # 1-Hop Customers & Entities in 7d and 24h
        custs_7d: set[str] = {curr_cust}
        custs_24h: set[str] = {curr_cust}

        ents_7d: set[str] = set(v for v in curr_entities.values() if v and v not in ("nan", "None", "null"))
        ents_24h: set[str] = set(v for v in curr_entities.values() if v and v not in ("nan", "None", "null"))

        edges_7d: set[tuple[str, str]] = set()
        edges_24h: set[tuple[str, str]] = set()

        # 1-Hop direct peers per entity
        direct_peers_7d: dict[str, set[str]] = {col: set() for col in ENTITY_COLUMNS}
        for col in ENTITY_COLUMNS:
            val = curr_entities[col]
            if not val or val in ("nan", "None", "null"):
                continue
            dq = entity_events[col][val]
            while dq and dq[0][0] <= cutoff_7d:
                dq.popleft()
            for t, c, _ in dq:
                if c != curr_cust:
                    direct_peers_7d[col].add(c)
                    custs_7d.add(c)
                    ents_7d.add(val)
                    edges_7d.add((c, val))
                    if t > cutoff_24h:
                        custs_24h.add(c)
                        ents_24h.add(val)
                        edges_24h.add((c, val))

        # 2-Hop expansion: gather all entities used by direct peers and their connecting customers
        all_1hop_peers_7d = set.union(*direct_peers_7d.values()) if direct_peers_7d else set()

        for peer in all_1hop_peers_7d:
            p_dq = customer_events[peer]
            for t, e_dict, _ in p_dq:
                if t > cutoff_7d:
                    for col, e_val in e_dict.items():
                        if not e_val or e_val in ("nan", "None", "null"):
                            continue
                        ents_7d.add(e_val)
                        edges_7d.add((peer, e_val))
                        # Find 2-hop peers using e_val
                        for t_e, c_2hop, _ in entity_events[col][e_val]:
                            if t_e > cutoff_7d and c_2hop != curr_cust:
                                custs_7d.add(c_2hop)
                                edges_7d.add((c_2hop, e_val))
                        if t > cutoff_24h:
                            ents_24h.add(e_val)
                            edges_24h.add((peer, e_val))
                            for t_e, c_2hop, _ in entity_events[col][e_val]:
                                if t_e > cutoff_24h and c_2hop != curr_cust:
                                    custs_24h.add(c_2hop)
                                    edges_24h.add((c_2hop, e_val))

        # Add past edges for current customer as well
        for t, e_dict, _ in customer_events[curr_cust]:
            if t > cutoff_7d:
                for col, e_val in e_dict.items():
                    if not e_val or e_val in ("nan", "None", "null"):
                        continue
                    ents_7d.add(e_val)
                    edges_7d.add((curr_cust, e_val))
                    if t > cutoff_24h:
                        ents_24h.add(e_val)
                        edges_24h.add((curr_cust, e_val))

        # Component Nodes & Edges
        total_nodes_24h = len(custs_24h) + len(ents_24h)
        total_nodes_7d = len(custs_7d) + len(ents_7d)

        edge_cnt_24h = len(edges_24h)
        edge_cnt_7d = len(edges_7d)

        # Density
        c_24 = max(1, len(custs_24h))
        e_24 = max(1, len(ents_24h))
        density_24h = float(edge_cnt_24h) / (float(c_24 * e_24) + 1e-5)

        c_7 = max(1, len(custs_7d))
        e_7 = max(1, len(ents_7d))
        density_7d = float(edge_cnt_7d) / (float(c_7 * e_7) + 1e-5)

        avg_ents_per_cust_7d = float(edge_cnt_7d) / float(c_7)
        avg_custs_per_ent_7d = float(edge_cnt_7d) / float(e_7)

        # Shared Modalities & Multi-entity conspirators
        shared_modalities = 0
        for col in ENTITY_COLUMNS:
            if len(direct_peers_7d[col]) > 0:
                shared_modalities += 1

        multi_modal_peers = 0
        all_candidate_peers = all_1hop_peers_7d
        for p in all_candidate_peers:
            shared_cnt = sum(1 for col in ENTITY_COLUMNS if p in direct_peers_7d[col])
            if shared_cnt >= 2:
                multi_modal_peers += 1

        max_overlap = 0
        for p in all_candidate_peers:
            shared_cnt = sum(1 for col in ENTITY_COLUMNS if p in direct_peers_7d[col])
            if shared_cnt > max_overlap:
                max_overlap = shared_cnt

        # 1-Hour expansion dynamics
        new_nodes_1h = 0
        all_comp_nodes = custs_7d | ents_7d
        for n in all_comp_nodes:
            first_t = node_first_seen.get(n)
            if first_t is not None and first_t > cutoff_1h:
                new_nodes_1h += 1

        new_edges_1h = 0
        for edge in edges_7d:
            first_t = edge_first_seen.get(edge)
            if first_t is not None and first_t > cutoff_1h:
                new_edges_1h += 1

        growth_ratio = float(new_nodes_1h) / (float(total_nodes_24h) / 24.0 + 1e-5)

        # Bridge behavior: number of disconnected peer components connected by current transaction
        peer_modalities = [direct_peers_7d[col] for col in ENTITY_COLUMNS if len(direct_peers_7d[col]) > 0]
        if len(peer_modalities) <= 1:
            bridge_merges = 0.0
        else:
            merged_sets = []
            for p_set in peer_modalities:
                new_set = set(p_set)
                remaining = []
                for existing in merged_sets:
                    if existing & new_set:
                        new_set |= existing
                    else:
                        remaining.append(existing)
                remaining.append(new_set)
                merged_sets = remaining
            bridge_merges = float(len(merged_sets))

        # 1-Hour order burst across component customers
        orders_1h = 0
        for c in custs_7d:
            for t, _, _ in customer_events[c]:
                if t > cutoff_1h:
                    orders_1h += 1

        feature_row = {
            "order_id": row.order_id,
            "customer_id": row.customer_id,
            "subgraph_node_count_24h": float(total_nodes_24h),
            "subgraph_customer_count_24h": float(len(custs_24h)),
            "subgraph_entity_count_24h": float(len(ents_24h)),
            "subgraph_edge_count_24h": float(edge_cnt_24h),
            "subgraph_node_count_7d": float(total_nodes_7d),
            "subgraph_customer_count_7d": float(len(custs_7d)),
            "subgraph_entity_count_7d": float(len(ents_7d)),
            "subgraph_edge_count_7d": float(edge_cnt_7d),
            "subgraph_edge_density_24h": float(density_24h),
            "subgraph_edge_density_7d": float(density_7d),
            "subgraph_avg_entities_per_cust_7d": float(avg_ents_per_cust_7d),
            "subgraph_avg_custs_per_entity_7d": float(avg_custs_per_ent_7d),
            "subgraph_shared_modality_count_7d": float(shared_modalities),
            "subgraph_multi_entity_conspirator_count_7d": float(multi_modal_peers),
            "subgraph_max_entity_overlap_degree_7d": float(max_overlap),
            "subgraph_new_nodes_1h": float(new_nodes_1h),
            "subgraph_new_edges_1h": float(new_edges_1h),
            "subgraph_growth_ratio_1h_vs_24h": float(growth_ratio),
            "subgraph_bridge_disjoint_components_7d": float(bridge_merges),
            "subgraph_order_burst_velocity_1h": float(orders_1h),
        }
        rows.append(feature_row)

        if curr_cust not in node_first_seen:
            node_first_seen[curr_cust] = now
        customer_events[curr_cust].append((now, curr_entities, row.order_id))

        for col in ENTITY_COLUMNS:
            val = curr_entities[col]
            if not val or val in ("nan", "None", "null"):
                continue
            if val not in node_first_seen:
                node_first_seen[val] = now
            edge = (curr_cust, val)
            if edge not in edge_first_seen:
                edge_first_seen[edge] = now
            entity_events[col][val].append((now, curr_cust, row.order_id))

    X = pd.DataFrame(rows).set_index("order_id")
    y = X.index.to_series().map(labels_by_id.is_abuse).astype(int) if labels_by_id is not None else pd.Series(index=X.index, dtype=int)
    return FeatureSet(X=X.drop(columns="customer_id"), y=y, ids=X.index.to_series(),
                      manifest=_manifest(X.drop(columns="customer_id").columns, "streaming suspicious subgraph features"))


def build_subgraph_extended_features(orders: pd.DataFrame, labels: pd.DataFrame | None = None, history_days: int = 30) -> FeatureSet:
    """Build Model F features (137 features): Model E (117) + suspicious subgraph features (20)."""
    e_fs = build_two_hop_extended_features(orders, labels, history_days)
    subgraph_fs = build_subgraph_features(orders, labels)
    X = e_fs.X.join(subgraph_fs.X.drop(columns=[c for c in subgraph_fs.X if c in e_fs.X], errors="ignore"), how="inner")
    return FeatureSet(X=X, y=e_fs.y.loc[X.index], ids=X.index.to_series(), manifest=pd.concat([e_fs.manifest, subgraph_fs.manifest], ignore_index=True))

