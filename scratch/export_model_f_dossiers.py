"""
Generate Model F causal investigation dossiers with local subgraph metrics.
"""

import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

from abuse_ring_detector.config import DATA_DIR, AUTHORITATIVE_MANIFEST
from abuse_ring_detector.features import (
    build_streaming_features,
    build_streaming_entity_features,
    build_cust_rel_extended_features,
    build_two_hop_extended_features,
    build_subgraph_extended_features,
)

def run_dossiers():
    print("Loading data...")
    orders_df = pd.read_parquet(DATA_DIR / "orders.parquet")
    customers_df = pd.read_parquet(DATA_DIR / "customers.parquet")
    orders_df["order_timestamp"] = pd.to_datetime(orders_df["order_timestamp"])
    orders_df = orders_df.sort_values("order_timestamp").reset_index(drop=True)

    print("Building Model F features...")
    X_df = build_subgraph_extended_features(orders_df, customers_df)
    
    val_cutoff = pd.to_datetime(AUTHORITATIVE_MANIFEST["val_start"])
    test_cutoff = pd.to_datetime(AUTHORITATIVE_MANIFEST["test_start"])

    train_mask = orders_df["order_timestamp"] < val_cutoff
    test_mask = orders_df["order_timestamp"] >= test_cutoff

    y = orders_df["is_abuse_order"].astype(int).values

    dtrain = lgb.Dataset(X_df[train_mask], label=y[train_mask])
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "verbose": -1,
        "random_state": 42,
    }
    model = lgb.train(params, dtrain, num_boost_round=100)

    test_preds = model.predict(X_df[test_mask])
    test_orders = orders_df[test_mask].copy()
    test_orders["model_f_score"] = test_preds
    test_orders["flagged"] = (test_preds >= 0.50).astype(int)

    # Let's inspect target rings: R0194 (Behavioral), R0126 (Mixed), R0037 (Device)
    target_rings = ["R0194", "R0126", "R0037"]
    os.makedirs("reports/dossiers", exist_ok=True)

    for rid in target_rings:
        ring_orders = test_orders[test_orders["ring_id"] == rid].sort_values("order_timestamp")
        if len(ring_orders) == 0:
            # Check all orders if not in test
            ring_orders = orders_df[orders_df["ring_id"] == rid].sort_values("order_timestamp")
            preds_all = model.predict(X_df.loc[ring_orders.index])
            ring_orders["model_f_score"] = preds_all
            ring_orders["flagged"] = (preds_all >= 0.50).astype(int)

        ring_type = ring_orders["ring_type"].iloc[0] if "ring_type" in ring_orders.columns else "unknown"
        customers = list(ring_orders["customer_id"].unique())
        exposure = float(ring_orders["order_amount"].sum())
        flagged_count = int(ring_orders["flagged"].sum())
        
        # Extract subgraph metrics from the latest order in the ring
        last_idx = ring_orders.index[-1]
        subgraph_metrics = {
            "subgraph_node_count_24h": float(X_df.loc[last_idx, "subgraph_node_count_24h"]),
            "subgraph_customer_count_24h": float(X_df.loc[last_idx, "subgraph_customer_count_24h"]),
            "subgraph_entity_count_24h": float(X_df.loc[last_idx, "subgraph_entity_count_24h"]),
            "subgraph_edge_count_24h": float(X_df.loc[last_idx, "subgraph_edge_count_24h"]),
            "subgraph_edge_density_7d": float(X_df.loc[last_idx, "subgraph_edge_density_7d"]),
            "subgraph_shared_modality_count_7d": float(X_df.loc[last_idx, "subgraph_shared_modality_count_7d"]),
            "subgraph_multi_entity_conspirator_count_7d": float(X_df.loc[last_idx, "subgraph_multi_entity_conspirator_count_7d"]),
            "subgraph_growth_ratio_1h_vs_24h": float(X_df.loc[last_idx, "subgraph_growth_ratio_1h_vs_24h"]),
            "subgraph_order_burst_velocity_1h": float(X_df.loc[last_idx, "subgraph_order_burst_velocity_1h"]),
            "subgraph_bridge_disjoint_components_7d": float(X_df.loc[last_idx, "subgraph_bridge_disjoint_components_7d"]),
        }

        timeline = []
        for _, row in ring_orders.iterrows():
            timeline.append({
                "order_id": str(row["order_id"]),
                "customer_id": str(row["customer_id"]),
                "timestamp": str(row["order_timestamp"]),
                "amount_inr": float(row["order_amount"]),
                "device_id": str(row["device_id"]),
                "ip_address": str(row["ip_address"]),
                "shipping_address": str(row["shipping_address"]),
                "payment_token": str(row["payment_token"]),
                "model_f_score": float(row["model_f_score"]),
                "action": "FLAG_FOR_REVIEW" if row["model_f_score"] >= 0.50 else "AUTO_APPROVE"
            })

        dossier = {
            "dossier_id": f"DOSSIER-MODEL-F-{rid}",
            "ring_id": rid,
            "ring_type": ring_type,
            "total_exposure_inr": exposure,
            "total_orders": len(ring_orders),
            "flagged_orders": flagged_count,
            "active_customers": customers,
            "subgraph_topology_metrics": subgraph_metrics,
            "causal_evidence_timeline": timeline,
            "investigator_summary": {
                "risk_assessment": "CRITICAL_SUSPECTED_RING" if flagged_count > 0 else "LOW_RISK",
                "recommended_action": "FREEZE_RING_ENTITIES" if flagged_count > 0 else "MONITOR",
                "key_subgraph_signals": [
                    f"24h Connected Subgraph Nodes: {int(subgraph_metrics['subgraph_node_count_24h'])}",
                    f"7d Local Bipartite Edge Density: {subgraph_metrics['subgraph_edge_density_7d']:.4f}",
                    f"Shared Entity Modalities: {int(subgraph_metrics['subgraph_shared_modality_count_7d'])} distinct types",
                    f"Multi-Entity Conspirators: {int(subgraph_metrics['subgraph_multi_entity_conspirator_count_7d'])} customers",
                    f"1h Order Burst Velocity: {int(subgraph_metrics['subgraph_order_burst_velocity_1h'])} transactions"
                ]
            }
        }

        json_path = f"reports/dossiers/dossier_model_f_{rid}.json"
        with open(json_path, "w") as f:
            json.dump(dossier, f, indent=2)

        md_content = f"""# Causal Investigation Dossier: {rid} (Model F)

**Ring Type**: `{ring_type}`  
**Total Financial Exposure**: ₹{exposure:,.2f}  
**Orders Flagged**: {flagged_count} / {len(ring_orders)}  
**Investigator Risk Assessment**: **{dossier['investigator_summary']['risk_assessment']}**  

---

## 1. Streaming Local Subgraph Metrics ($t < T$)
* **Connected Subgraph Nodes (24h)**: {int(subgraph_metrics['subgraph_node_count_24h'])} nodes ({int(subgraph_metrics['subgraph_customer_count_24h'])} customers, {int(subgraph_metrics['subgraph_entity_count_24h'])} entities)
* **Local Graph Edge Density (7d)**: {subgraph_metrics['subgraph_edge_density_7d']:.4f}
* **Shared Modality Count (7d)**: {int(subgraph_metrics['subgraph_shared_modality_count_7d'])} distinct entity modalities
* **Multi-Entity Conspirators**: {int(subgraph_metrics['subgraph_multi_entity_conspirator_count_7d'])} coordinated customers
* **1h Burst Velocity**: {int(subgraph_metrics['subgraph_order_burst_velocity_1h'])} orders / hour
* **Bridge Component Count**: {int(subgraph_metrics['subgraph_bridge_disjoint_components_7d'])} connected clusters bridged

---

## 2. Causal Evidence Timeline
| Order ID | Timestamp | Customer | Amount (₹) | Model F Score | Action |
|---|---|---|---|---|---|
"""
        for evt in timeline:
            md_content += f"| `{evt['order_id']}` | {evt['timestamp']} | `{evt['customer_id']}` | ₹{evt['amount_inr']:,.2f} | **{evt['model_f_score']:.4f}** | `{evt['action']}` |\n"

        md_content += f"""
---

## 3. Recommended Operational Intervention
* **Primary Recommendation**: `{dossier['investigator_summary']['recommended_action']}`
* **Consolidation Impact**: Grouped into single connected-component investigation case (Case ID: `CASE-SUBGRAPH-{rid}`).
"""
        with open(f"reports/dossiers/dossier_model_f_{rid}.md", "w") as f:
            f.write(md_content)

        print(f"Generated {json_path} and MD dossier.")

if __name__ == "__main__":
    run_dossiers()
