"""
Streaming Community & Ring Dossier Generation.

Provides a causal, temporal (t < T) operational ring dossier for trust and safety
investigators, synthesizing 1-hop and 2-hop connections, entity sharing, velocity,
exposure, and explanatory risk paths without future leakage.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Any
import numpy as np
import pandas as pd

ENTITY_COLUMNS = ["device_id", "address_id", "ip_id", "payment_id"]


@dataclass
class ExplanatoryPath:
    path_type: str  # e.g., '1-hop', '2-hop'
    source_customer: str
    intermediate_entity: str
    target_customer: str | None = None
    target_entity: str | None = None
    time_window: str = "7d"
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RingDossier:
    dossier_id: str
    target_customer_id: str
    target_order_id: str
    as_of_time: str
    model_score: float
    risk_tier: str
    participating_customers: list[str]
    total_customers_count: int
    connecting_entities: dict[str, list[str]]
    total_entities_count: int
    shared_devices: list[str]
    shared_addresses: list[str]
    shared_ips: list[str]
    shared_payments: list[str]
    explanatory_paths: list[dict[str, Any]]
    peer_orders_7d: int
    peer_orders_30d: int
    peer_velocity_ratio_24h_vs_7d: float
    total_cluster_exposure_inr: float
    first_cluster_activity_time: str | None
    last_cluster_activity_time: str | None
    cluster_lifespan_days: float
    narrative_summary: str
    subgraph_topology_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StreamingDossierExtractor:
    """
    Streaming causal extractor that builds an operational dossier for any order
    using only state strictly prior to the order's event_time (t < T).
    """

    def __init__(self, history_days: int = 30):
        self.history_days = history_days

    def extract_dossier(
        self,
        target_order_id: str,
        orders: pd.DataFrame,
        model_scores: dict[str, float] | None = None,
        loss_amounts: dict[str, float] | None = None,
    ) -> RingDossier:
        # Sort chronologically
        data = orders.sort_values(["event_time", "order_id"], kind="mergesort").reset_index(drop=True)
        target_rows = data[data["order_id"] == target_order_id]
        if target_rows.empty:
            raise ValueError(f"Target order {target_order_id} not found in orders.")

        target_row = target_rows.iloc[0]
        as_of_time = pd.Timestamp(target_row["event_time"])
        target_cust = target_row["customer_id"]
        score = float(model_scores.get(target_order_id, 0.0)) if model_scores else 0.0

        if score >= 0.80:
            risk_tier = "CRITICAL_RISK"
        elif score >= 0.50:
            risk_tier = "HIGH_RISK"
        elif score >= 0.30:
            risk_tier = "MEDIUM_RISK"
        else:
            risk_tier = "LOW_RISK"

        # Filter strictly prior orders (t < as_of_time) within 30 days
        cutoff_30d = as_of_time - pd.Timedelta(days=self.history_days)
        cutoff_7d = as_of_time - pd.Timedelta(days=7)
        cutoff_24h = as_of_time - pd.Timedelta(hours=24)

        prior_orders = data[(data["event_time"] < as_of_time) & (data["event_time"] >= cutoff_30d)]

        # Find 1-hop connected customers
        current_entities = {
            "device_id": target_row["device_id"],
            "address_id": target_row["address_id"],
            "ip_id": target_row["ip_id"],
            "payment_id": target_row["payment_id"],
        }

        # Build index of entity -> (timestamp, customer_id, order_id, amount)
        entity_history: dict[str, dict[str, list[dict]]] = {c: defaultdict(list) for c in ENTITY_COLUMNS}
        customer_history: dict[str, list[dict]] = defaultdict(list)

        for row in prior_orders.itertuples(index=False):
            t = pd.Timestamp(row.event_time)
            amt = float(row.amount)
            loss = float(loss_amounts.get(row.order_id, amt)) if loss_amounts else amt
            order_info = {
                "order_id": row.order_id,
                "customer_id": row.customer_id,
                "event_time": t,
                "amount": amt,
                "loss": loss,
                "device_id": row.device_id,
                "address_id": row.address_id,
                "ip_id": row.ip_id,
                "payment_id": row.payment_id,
            }
            customer_history[row.customer_id].append(order_info)
            for c in ENTITY_COLUMNS:
                e_val = getattr(row, c)
                entity_history[c][e_val].append(order_info)

        # 1-Hop Peers
        one_hop_peers: dict[str, set[str]] = defaultdict(set)
        for c in ENTITY_COLUMNS:
            e_val = current_entities[c]
            for o in entity_history[c].get(e_val, []):
                if o["customer_id"] != target_cust:
                    one_hop_peers[c].add(o["customer_id"])

        all_1hop_customers = set().union(*one_hop_peers.values()) if one_hop_peers else set()

        # 2-Hop Expansions (from 1-hop peers to their entities and other peers)
        two_hop_entities: dict[str, set[str]] = defaultdict(set)
        two_hop_customers: set[str] = set()
        explanatory_paths: list[ExplanatoryPath] = []

        # Document 1-hop paths
        for c, peers in one_hop_peers.items():
            e_val = current_entities[c]
            for p_cust in sorted(peers):
                explanatory_paths.append(ExplanatoryPath(
                    path_type="1-hop",
                    source_customer=target_cust,
                    intermediate_entity=f"{c}:{e_val}",
                    target_customer=p_cust,
                    time_window="30d",
                    description=f"Customer {target_cust} directly shares {c} ({e_val}) with peer {p_cust}"
                ))

        # Document 2-hop paths
        for p_cust in all_1hop_customers:
            for o in customer_history.get(p_cust, []):
                for c in ENTITY_COLUMNS:
                    e_val = o[c]
                    if e_val != current_entities[c]:
                        two_hop_entities[c].add(e_val)
                        # Find other peers touching this 2-hop entity
                        for o2 in entity_history[c].get(e_val, []):
                            if o2["customer_id"] != target_cust and o2["customer_id"] != p_cust:
                                two_hop_customers.add(o2["customer_id"])
                                explanatory_paths.append(ExplanatoryPath(
                                    path_type="2-hop",
                                    source_customer=target_cust,
                                    intermediate_entity=f"peer:{p_cust}",
                                    target_customer=o2["customer_id"],
                                    target_entity=f"{c}:{e_val}",
                                    time_window="30d",
                                    description=f"Customer {target_cust} -> Peer {p_cust} -> 2-Hop {c} ({e_val}) -> Peer {o2['customer_id']}"
                                ))

        all_cluster_customers = sorted(list({target_cust} | all_1hop_customers | two_hop_customers))

        # Entities in cluster
        shared_devices = sorted(list({current_entities["device_id"]} | two_hop_entities["device_id"]))
        shared_addresses = sorted(list({current_entities["address_id"]} | two_hop_entities["address_id"]))
        shared_ips = sorted(list({current_entities["ip_id"]} | two_hop_entities["ip_id"]))
        shared_payments = sorted(list({current_entities["payment_id"]} | two_hop_entities["payment_id"]))

        connecting_entities = {
            "devices": shared_devices,
            "addresses": shared_addresses,
            "ips": shared_ips,
            "payments": shared_payments,
        }
        total_entities_count = sum(len(v) for v in connecting_entities.values())

        # Cluster activity & exposure
        cluster_orders = []
        for cust in all_cluster_customers:
            cluster_orders.extend(customer_history.get(cust, []))

        # Add target order itself to cluster total exposure
        target_amt = float(target_row["amount"])
        target_loss = float(loss_amounts.get(target_order_id, target_amt)) if loss_amounts else target_amt

        orders_7d = [o for o in cluster_orders if o["event_time"] >= cutoff_7d]
        orders_24h = [o for o in cluster_orders if o["event_time"] >= cutoff_24h]

        peer_orders_7d = len(orders_7d)
        peer_orders_30d = len(cluster_orders)
        daily_7d_rate = peer_orders_7d / 7.0
        peer_velocity_ratio = len(orders_24h) / max(1.0, daily_7d_rate)

        total_exposure = sum(o["loss"] for o in cluster_orders) + target_loss

        all_times = [o["event_time"] for o in cluster_orders] + [as_of_time]
        first_time = min(all_times) if all_times else as_of_time
        last_time = max(all_times) if all_times else as_of_time
        lifespan_days = (last_time - first_time).total_seconds() / 86400.0

        # Subgraph topology metrics
        total_subgraph_nodes = len(all_cluster_customers) + total_entities_count
        subgraph_edge_count = len(explanatory_paths)
        subgraph_density_7d = (2.0 * subgraph_edge_count) / (total_subgraph_nodes * (total_subgraph_nodes - 1)) if total_subgraph_nodes > 1 else 0.0
        shared_modalities = sum(1 for k in ["devices", "addresses", "ips", "payments"] if len(connecting_entities[k]) > 0)
        
        subgraph_metrics = {
            "subgraph_node_count_24h": float(len(orders_24h) + len(all_1hop_customers)),
            "subgraph_customer_count_24h": float(len(all_1hop_customers) + 1),
            "subgraph_entity_count_24h": float(len(current_entities)),
            "subgraph_edge_density_7d": round(float(subgraph_density_7d), 4),
            "subgraph_shared_modality_count_7d": float(shared_modalities),
            "subgraph_multi_entity_conspirator_count_7d": float(len(all_1hop_customers)),
            "subgraph_order_burst_velocity_1h": float(len([o for o in cluster_orders if o["event_time"] >= as_of_time - pd.Timedelta(hours=1)])),
            "subgraph_growth_ratio_1h_vs_24h": round(float(peer_velocity_ratio), 2),
        }

        # Generate narrative summary
        narrative = (
            f"Alerted Order {target_order_id} (Customer {target_cust}, Score: {score:.4f}, Tier: {risk_tier}) "
            f"is connected to a {len(all_cluster_customers)}-member cluster spanning {total_entities_count} distinct entities. "
            f"Direct 1-hop connections identified {len(all_1hop_customers)} conspirator(s) via shared "
            f"{', '.join([k for k, v in one_hop_peers.items() if len(v) > 0]) or 'no direct entities'}. "
            f"2-hop expansions uncovered {len(two_hop_customers)} additional coordinated customer(s). "
            f"Cluster velocity: {peer_orders_7d} orders in past 7d ({len(orders_24h)} in last 24h, burst ratio: {peer_velocity_ratio:.2f}). "
            f"Cumulative historical cluster fraud exposure is INR {total_exposure:,.2f} over {lifespan_days:.1f} active days."
        )

        return RingDossier(
            dossier_id=f"DOSSIER-{target_order_id}",
            target_customer_id=target_cust,
            target_order_id=target_order_id,
            as_of_time=as_of_time.isoformat(),
            model_score=score,
            risk_tier=risk_tier,
            participating_customers=all_cluster_customers,
            total_customers_count=len(all_cluster_customers),
            connecting_entities=connecting_entities,
            total_entities_count=total_entities_count,
            shared_devices=shared_devices,
            shared_addresses=shared_addresses,
            shared_ips=shared_ips,
            shared_payments=shared_payments,
            explanatory_paths=[p.to_dict() for p in explanatory_paths[:25]],  # Top 25 paths
            peer_orders_7d=peer_orders_7d,
            peer_orders_30d=peer_orders_30d,
            peer_velocity_ratio_24h_vs_7d=round(peer_velocity_ratio, 2),
            total_cluster_exposure_inr=round(total_exposure, 2),
            first_cluster_activity_time=first_time.isoformat(),
            last_cluster_activity_time=last_time.isoformat(),
            cluster_lifespan_days=round(lifespan_days, 2),
            narrative_summary=narrative,
            subgraph_topology_metrics=subgraph_metrics,
        )
