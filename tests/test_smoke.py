import pandas as pd

from abuse_ring_detector.config import Config, RingConfig
from abuse_ring_detector.features import build_baseline_features, build_graph_features
from abuse_ring_detector.splits import split_by_time
from abuse_ring_detector.synthetic import generate_ecosystem


def small_config():
    return Config(customers=200, orders=500, date_range_days=60,
                  rings=RingConfig(count=100, min_size=3, max_size=6))


def test_generation_schema_and_determinism():
    first = generate_ecosystem(small_config())
    second = generate_ecosystem(small_config())
    assert first.orders.equals(second.orders)
    assert first.labels.is_abuse.any()
    assert len(first.rings) == 100
    assert set(first.rings.ring_type) == {"shared_device", "shared_address", "behavioral", "mixed"}
    assert set(first.ground_truth.columns) >= {"customer_id", "is_abusive"}


def test_split_is_chronological():
    data = generate_ecosystem(small_config())
    split = split_by_time(data.orders)
    assert split.train.event_time.max() <= split.validation.event_time.min()
    assert split.validation.event_time.max() <= split.test.event_time.min()


def test_features_exclude_current_event_entities():
    data = generate_ecosystem(small_config())
    base = build_baseline_features(data.orders, data.labels)
    graph = build_graph_features(data.orders, data.labels)
    assert "order_id" not in base.X.columns
    assert "customer_id" not in graph.X.columns
    first = data.orders.sort_values(["event_time", "order_id"]).iloc[0].order_id
    assert graph.X.loc[first, "graph_prior_orders"] == 0
    assert graph.X.loc[first, "device_shared_accounts"] == 0
