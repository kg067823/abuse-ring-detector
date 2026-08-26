"""Configurable synthetic merchant ecosystem with realistic hard negatives."""
from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from .config import Config
from .schemas import SyntheticDataset

ENTITY_COLUMNS = ["device_id", "ip_id", "address_id", "payment_id"]


def _choice(rng: np.random.Generator, values: list[str], n: int) -> np.ndarray:
    return rng.choice(values, size=n)


def generate_ecosystem(config: Config) -> SyntheticDataset:
    rng = np.random.default_rng(config.seed)
    start = pd.Timestamp("2025-01-01")
    n_customers, n_orders = config.customers, config.orders
    customer_ids = np.array([f"C{i:06d}" for i in range(n_customers)])
    signup = start + pd.to_timedelta(rng.integers(0, config.date_range_days - 20, n_customers), unit="D")
    household = rng.integers(0, max(1, int(n_customers * config.normal_sharing.get("household_fraction", .1))), n_customers)
    workgroup = rng.integers(0, max(1, int(n_customers * config.normal_sharing.get("business_fraction", .03))), n_customers)
    segments = rng.choice(["regular", "high_value", "business", "student"], n_customers, p=[.72, .10, .06, .12])
    customers = pd.DataFrame({
        "customer_id": customer_ids, "signup_time": signup,
        "country": rng.choice(["IN", "SG", "AE", "US"], n_customers, p=[.70, .08, .08, .14]),
        "segment": segments, "baseline_risk": np.clip(rng.beta(2, 18, n_customers), .01, .4),
        "household_id": household, "workgroup_id": workgroup,
    })

    # Normal entity pools intentionally include shared households/businesses.
    devices = np.array([f"D{i:05d}" for i in range(max(100, n_customers // 4))])
    ips = np.array([f"IP{i:05d}" for i in range(max(200, n_customers // 2))])
    addresses = np.array([f"A{i:05d}" for i in range(max(200, n_customers // 2))])
    payments = np.array([f"P{i:05d}" for i in range(max(300, n_customers // 2))])
    hh_device = {h: devices[h % len(devices)] for h in np.unique(household)}
    hh_address = {h: addresses[h % len(addresses)] for h in np.unique(household)}
    wg_ip = {w: ips[w % len(ips)] for w in np.unique(workgroup)}
    profile = pd.DataFrame({"customer_id": customer_ids, "household_id": household, "workgroup_id": workgroup})
    profile["activity"] = np.where(segments == "business", 4.0, np.where(segments == "high_value", 2.0, .8)) * rng.lognormal(0, .35, n_customers)
    weights = profile.activity.to_numpy() / profile.activity.sum()
    order_customer_idx = rng.choice(n_customers, n_orders, p=weights)
    order_customer = customer_ids[order_customer_idx]
    event_day = rng.integers(0, config.date_range_days, n_orders)
    event_time = start + pd.to_timedelta(event_day, unit="D") + pd.to_timedelta(rng.integers(0, 86400, n_orders), unit="s")
    amount = np.clip(rng.lognormal(7.5, .75, n_orders), 80, 200_000)
    amount *= np.where(segments[order_customer_idx] == "business", 1.8, 1.0)
    device = np.array([hh_device[household[i]] if rng.random() < .38 else rng.choice(devices) for i in order_customer_idx])
    address = np.array([hh_address[household[i]] if rng.random() < .52 else rng.choice(addresses) for i in order_customer_idx])
    ip = np.array([wg_ip[workgroup[i]] if rng.random() < .25 else rng.choice(ips) for i in order_customer_idx])
    payment = rng.choice(payments, n_orders)
    orders = pd.DataFrame({
        "order_id": [f"O{i:07d}" for i in range(n_orders)], "customer_id": order_customer,
        "event_time": event_time, "amount": amount.round(2), "currency": "INR",
        "device_id": device, "ip_id": ip, "address_id": address, "payment_id": payment,
        "merchant_category": rng.choice(["electronics", "fashion", "grocery", "travel", "home"], n_orders, p=[.2,.27,.25,.1,.18]),
        "status": "completed", "retry_count": rng.poisson(.08, n_orders),
    })

    labels = pd.DataFrame({"order_id": orders.order_id, "is_abuse": False, "ring_id": pd.NA,
                           "abuse_type": pd.NA, "loss_amount": 0.0, "reason_codes": ""})
    ring_rows, membership_rows = [], []
    selected = set()
    types = list(config.rings.types)
    type_probs = np.array([config.rings.types[t] for t in types])
    for ring_num in range(config.rings.count):
        ring_type = types[rng.choice(len(types), p=type_probs)]
        size = int(rng.integers(config.rings.min_size, config.rings.max_size + 1))
        members = rng.choice(n_customers, size=min(size, n_customers), replace=False)
        # Keep ring injections mostly distinct while permitting realistic overlap.
        selected.update(members.tolist())
        ring_id = f"R{ring_num:04d}"
        ring_start_day = int(rng.integers(20, max(21, config.date_range_days - 35)))
        ring_end_day = min(config.date_range_days - 1, ring_start_day + int(rng.integers(8, 35)))
        ring_start = start + pd.Timedelta(days=ring_start_day)
        ring_end = start + pd.Timedelta(days=ring_end_day + 1)
        intensity = float(rng.uniform(.7, 1.8))
        ring_rows.append({"ring_id": ring_id, "ring_type": ring_type, "start_time": ring_start,
                          "end_time": ring_end, "customer_count": len(members), "intensity": intensity})
        for member in members:
            membership_rows.append({"ring_id": ring_id, "customer_id": customer_ids[member],
                                     "joined_at": ring_start, "left_at": ring_end, "ring_type": ring_type})
        member_mask = orders.customer_id.isin(customer_ids[members]) & orders.event_time.between(ring_start, ring_end, inclusive="left")
        member_orders = orders.index[member_mask]
        if len(member_orders) == 0:
            # Ensure a ring has observable activity by adding no synthetic rows; later report flags this.
            continue
        idx = np.asarray(member_orders)
        if ring_type in {"shared_device", "mixed"}:
            shared = f"RD{ring_num:04d}"
            orders.loc[idx, "device_id"] = shared
        if ring_type in {"shared_address", "mixed"}:
            orders.loc[idx, "address_id"] = f"RA{ring_num:04d}"
        if ring_type == "mixed":
            orders.loc[idx, "payment_id"] = f"RP{ring_num:04d}"
            orders.loc[idx, "ip_id"] = f"RIP{ring_num:04d}"
        if ring_type in {"behavioral", "mixed"}:
            orders.loc[idx, "amount"] = np.clip(rng.lognormal(8.2, .35, len(idx)), 100, 120_000).round(2)
            # Coordinated but not identical category/timing patterns.
            orders.loc[idx, "merchant_category"] = rng.choice(["electronics", "fashion", "travel"], len(idx), p=[.5,.3,.2])
        reason = {"shared_device": "shared_device_velocity", "shared_address": "shared_address_returns",
                  "behavioral": "coordinated_behavior", "mixed": "multi_entity_coordination"}[ring_type]
        label_cols = ["is_abuse", "ring_id", "abuse_type", "loss_amount", "reason_codes"]
        labels.loc[idx, "is_abuse"] = True
        labels.loc[idx, "ring_id"] = ring_id
        labels.loc[idx, "abuse_type"] = ring_type
        labels.loc[idx, "loss_amount"] = orders.loc[idx, "amount"].to_numpy() * rng.uniform(.25, .8)
        labels.loc[idx, "reason_codes"] = reason

    orders = orders.sort_values(["event_time", "order_id"], kind="mergesort").reset_index(drop=True)
    labels = labels.set_index("order_id").loc[orders.order_id].reset_index()
    ground_truth = (customers[["customer_id"]].copy())
    membership_groups = pd.DataFrame(membership_rows).groupby("customer_id", as_index=False).agg(
        ring_id=("ring_id", "first"), ring_type=("ring_type", "first")) if membership_rows else pd.DataFrame(columns=["customer_id", "ring_id", "ring_type"])
    ground_truth = ground_truth.merge(membership_groups, on="customer_id", how="left")
    ground_truth["is_abusive"] = ground_truth.ring_id.notna()
    return _with_returns(SyntheticDataset(
        customers=customers, orders=orders, returns=pd.DataFrame(), labels=labels, ground_truth=ground_truth,
        rings=pd.DataFrame(ring_rows), ring_memberships=pd.DataFrame(membership_rows),
        metadata={"seed": config.seed, "start": str(start), "selected_abusive_customers": len(selected),
                  "entity_columns": ENTITY_COLUMNS},
    ), rng)


def _with_returns(dataset: SyntheticDataset, rng: np.random.Generator) -> SyntheticDataset:
    orders, labels = dataset.orders, dataset.labels
    return_mask = rng.random(len(orders)) < np.where(labels.is_abuse, .62, .10)
    chosen = orders.loc[return_mask]
    if chosen.empty:
        dataset.returns = pd.DataFrame(columns=["return_id", "order_id", "customer_id", "event_time", "reason", "refund_amount"])
        return dataset
    returns = pd.DataFrame({
        "return_id": [f"RET{i:07d}" for i in range(len(chosen))], "order_id": chosen.order_id.to_numpy(),
        "customer_id": chosen.customer_id.to_numpy(),
        "event_time": chosen.event_time.to_numpy() + pd.to_timedelta(rng.integers(1, 12, len(chosen)), unit="D"),
        "reason": rng.choice(["changed_mind", "not_as_described", "duplicate", "damaged"], len(chosen)),
        "refund_amount": chosen.amount.to_numpy().round(2),
    })
    dataset.returns = returns.sort_values("event_time").reset_index(drop=True)
    return dataset
