"""NetworkX graph construction and deterministic community summaries."""
from __future__ import annotations

import networkx as nx
import pandas as pd

from .schemas import GraphSnapshot

ENTITY_COLUMNS = ["device_id", "ip_id", "address_id", "payment_id"]


def build_graph_snapshot(prior_orders: pd.DataFrame, as_of: pd.Timestamp, algorithm: str = "greedy_modularity") -> GraphSnapshot:
    """Build a customer/entity graph from events strictly before ``as_of``."""
    history = prior_orders[prior_orders.event_time < as_of]
    graph = nx.Graph()
    for row in history.itertuples(index=False):
        customer = f"customer:{row.customer_id}"
        graph.add_node(customer, kind="customer")
        for col in ENTITY_COLUMNS:
            entity = f"{col}:{getattr(row, col)}"
            graph.add_node(entity, kind=col)
            graph.add_edge(customer, entity, event_time=row.event_time, amount=float(row.amount))
    communities = detect_communities(graph, algorithm)
    return GraphSnapshot(as_of=pd.Timestamp(as_of), graph=graph, communities=communities)


def detect_communities(graph: nx.Graph, algorithm: str = "greedy_modularity") -> dict[str, int]:
    if graph.number_of_nodes() == 0:
        return {}
    if algorithm not in {"greedy_modularity", "connected_components"}:
        raise ValueError(f"unsupported community algorithm: {algorithm}")
    if algorithm == "connected_components":
        groups = list(nx.connected_components(graph))
    else:
        groups = list(nx.community.greedy_modularity_communities(graph))
    return {node: i for i, group in enumerate(groups) for node in group}


def graph_summary(snapshot: GraphSnapshot) -> pd.DataFrame:
    graph = snapshot.graph
    rows = []
    for node, attrs in graph.nodes(data=True):
        rows.append({"node": node, "kind": attrs.get("kind", "unknown"),
                     "degree": graph.degree(node), "community_id": snapshot.communities.get(node, -1),
                     "component_size": len(nx.node_connected_component(graph, node))})
    return pd.DataFrame(rows)
