"""Unit and property tests for streaming causal ring dossiers."""
import pandas as pd
import pytest
from abuse_ring_detector.dossier import StreamingDossierExtractor, RingDossier, ExplanatoryPath


def test_dossier_causal_temporal_isolation():
    """Verify that transactions occurring at or after target event time are strictly excluded."""
    t0 = pd.Timestamp("2025-01-01 10:00:00")
    t1 = pd.Timestamp("2025-01-01 12:00:00")
    t2 = pd.Timestamp("2025-01-01 14:00:00")

    orders = pd.DataFrame([
        # Past order: C1 uses D1 and A1
        {"order_id": "O001", "customer_id": "C1", "event_time": t0, "amount": 1000.0,
         "device_id": "D1", "ip_id": "IP1", "address_id": "A1", "payment_id": "P1"},
        # Target order: C2 uses D1 (1-hop connection to C1)
        {"order_id": "O002", "customer_id": "C2", "event_time": t1, "amount": 1500.0,
         "device_id": "D1", "ip_id": "IP2", "address_id": "A2", "payment_id": "P2"},
        # Future order: C3 uses D1 and P3 at t2
        {"order_id": "O003", "customer_id": "C3", "event_time": t2, "amount": 2000.0,
         "device_id": "D1", "ip_id": "IP3", "address_id": "A3", "payment_id": "P3"},
    ])

    extractor = StreamingDossierExtractor(history_days=30)
    dossier = extractor.extract_dossier("O002", orders)

    # C1 should be included as prior conspirator
    assert "C1" in dossier.participating_customers
    # C3 (future order) MUST NOT be included
    assert "C3" not in dossier.participating_customers
    assert "P3" not in dossier.shared_payments
    assert dossier.total_customers_count == 2
    assert dossier.target_order_id == "O002"
    assert dossier.target_customer_id == "C2"


def test_dossier_two_hop_conspirator_extraction():
    """Verify that 2-hop conspirators are correctly identified with explanation paths."""
    t0 = pd.Timestamp("2025-01-01 10:00:00")
    t1 = pd.Timestamp("2025-01-01 11:00:00")
    t2 = pd.Timestamp("2025-01-01 12:00:00")

    orders = pd.DataFrame([
        # C1 has D1 and A1
        {"order_id": "O001", "customer_id": "C1", "event_time": t0, "amount": 500.0,
         "device_id": "D1", "ip_id": "IP1", "address_id": "A1", "payment_id": "P1"},
        # C2 shares D1 with C1, and uses A2
        {"order_id": "O002", "customer_id": "C2", "event_time": t1, "amount": 600.0,
         "device_id": "D1", "ip_id": "IP2", "address_id": "A2", "payment_id": "P2"},
        # C3 shares A2 with C2 (2 hops from C1)
        {"order_id": "O003", "customer_id": "C3", "event_time": t1 + pd.Timedelta(minutes=30), "amount": 700.0,
         "device_id": "D3", "ip_id": "IP3", "address_id": "A2", "payment_id": "P3"},
        # Target order: C1 makes new purchase at t2
        {"order_id": "O004", "customer_id": "C1", "event_time": t2, "amount": 800.0,
         "device_id": "D1", "ip_id": "IP1", "address_id": "A1", "payment_id": "P1"},
    ])

    extractor = StreamingDossierExtractor(history_days=30)
    dossier = extractor.extract_dossier("O004", orders, model_scores={"O004": 0.88})

    assert dossier.risk_tier == "CRITICAL_RISK"
    assert "C2" in dossier.participating_customers
    assert "C3" in dossier.participating_customers
    assert len(dossier.explanatory_paths) > 0


def test_dossier_cold_start_zero_history():
    """Verify clean graceful behavior for brand new customer with zero network connections."""
    t0 = pd.Timestamp("2025-01-01 10:00:00")
    orders = pd.DataFrame([
        {"order_id": "O001", "customer_id": "C_NEW", "event_time": t0, "amount": 250.0,
         "device_id": "D_NEW", "ip_id": "IP_NEW", "address_id": "A_NEW", "payment_id": "P_NEW"},
    ])

    extractor = StreamingDossierExtractor(history_days=30)
    dossier = extractor.extract_dossier("O001", orders)

    assert dossier.total_customers_count == 1
    assert dossier.participating_customers == ["C_NEW"]
    assert dossier.total_entities_count == 4
    assert dossier.total_cluster_exposure_inr == 250.0
    assert dossier.peer_orders_7d == 0
    assert dossier.peer_orders_30d == 0
    assert isinstance(dossier.to_dict(), dict)
    assert "C_NEW" in dossier.narrative_summary


def test_dossier_serialization_and_narrative():
    """Verify that dossier produces valid serializable dict and descriptive narrative."""
    t0 = pd.Timestamp("2025-01-01 10:00:00")
    t1 = pd.Timestamp("2025-01-02 10:00:00")
    orders = pd.DataFrame([
        {"order_id": "O001", "customer_id": "C1", "event_time": t0, "amount": 1000.0,
         "device_id": "D1", "ip_id": "IP1", "address_id": "A1", "payment_id": "P1"},
        {"order_id": "O002", "customer_id": "C2", "event_time": t1, "amount": 2000.0,
         "device_id": "D1", "ip_id": "IP1", "address_id": "A2", "payment_id": "P2"},
    ])

    extractor = StreamingDossierExtractor(history_days=30)
    dossier = extractor.extract_dossier(
        "O002",
        orders,
        model_scores={"O002": 0.95},
        loss_amounts={"O001": 1000.0, "O002": 2000.0}
    )

    d_dict = dossier.to_dict()
    assert d_dict["model_score"] == 0.95
    assert d_dict["risk_tier"] == "CRITICAL_RISK"
    assert "D1" in d_dict["shared_devices"]
    assert "IP1" in d_dict["shared_ips"]
    assert d_dict["total_cluster_exposure_inr"] == 3000.0
    assert "O002" in dossier.narrative_summary
    assert "CRITICAL_RISK" in dossier.narrative_summary
