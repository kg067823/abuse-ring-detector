"""Automated data quality, distribution balance, and leakage-prevention tests."""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
import pytest

from abuse_ring_detector.config import Config, RingConfig
from abuse_ring_detector.dataset_quality import compute_dataset_quality
from abuse_ring_detector.features import build_baseline_features, build_combined_features
from abuse_ring_detector.splits import split_by_time
from abuse_ring_detector.synthetic import generate_ecosystem


@pytest.fixture(scope="module")
def sample_dataset():
    cfg = Config(
        seed=42,
        customers=10_000,
        orders=25_000,
        date_range_days=180,
        rings=RingConfig(
            count=160,
            min_size=4,
            max_size=16,
            min_duration_days=7,
            max_duration_days=24,
            activity_rate=0.85,
            min_orders_per_member=1,
            max_orders_per_member=3,
            minimum_test_active_rings=20,
            minimum_test_rings_per_type=4,
        ),
    )
    return generate_ecosystem(cfg), cfg


def test_sufficient_active_rings_in_test(sample_dataset):
    dataset, cfg = sample_dataset
    split = split_by_time(dataset.orders, 0.70, 0.15)
    quality = compute_dataset_quality(dataset, split)

    test_active = quality.metrics["test_active_rings"]
    assert test_active >= cfg.rings.minimum_test_active_rings, (
        f"Expected at least {cfg.rings.minimum_test_active_rings} active rings in test, got {test_active}"
    )


def test_all_configured_ring_types_in_test(sample_dataset):
    dataset, cfg = sample_dataset
    split = split_by_time(dataset.orders, 0.70, 0.15)
    quality = compute_dataset_quality(dataset, split)

    test_by_type = quality.metrics["test_active_by_type"]
    for r_type in ["shared_device", "shared_address", "behavioral", "mixed"]:
        count = test_by_type.get(r_type, 0)
        assert count >= cfg.rings.minimum_test_rings_per_type, (
            f"Expected at least {cfg.rings.minimum_test_rings_per_type} test rings for type '{r_type}', got {count}"
        )


def test_chronological_split_no_future_leakage(sample_dataset):
    dataset, _ = sample_dataset
    split = split_by_time(dataset.orders, 0.70, 0.15)

    assert split.train.event_time.max() <= split.validation.event_time.min()
    assert split.validation.event_time.max() <= split.test.event_time.min()


def test_no_label_in_features(sample_dataset):
    dataset, _ = sample_dataset
    fs_base = build_baseline_features(dataset.orders, dataset.labels, 30)
    fs_graph = build_combined_features(dataset.orders, dataset.labels, 30)

    forbidden = {"is_abuse", "ring_id", "abuse_type", "loss_amount", "reason_codes", "is_abusive"}
    assert forbidden.isdisjoint(set(fs_base.X.columns)), "Baseline features leak target labels!"
    assert forbidden.isdisjoint(set(fs_graph.X.columns)), "Graph features leak target labels!"


def test_no_entity_id_prefix_leaks(sample_dataset):
    dataset, _ = sample_dataset
    orders = dataset.orders

    # Verify standard entity regex patterns across ALL rows (abusive and normal)
    dev_pattern = re.compile(r"^D\d{5}$")
    addr_pattern = re.compile(r"^A\d{5}$")
    ip_pattern = re.compile(r"^IP\d{5}$")
    pmt_pattern = re.compile(r"^P\d{5}$")

    assert orders.device_id.str.match(dev_pattern).all(), "device_id contains non-standard prefix leaks!"
    assert orders.address_id.str.match(addr_pattern).all(), "address_id contains non-standard prefix leaks!"
    assert orders.ip_id.str.match(ip_pattern).all(), "ip_id contains non-standard prefix leaks!"
    assert orders.payment_id.str.match(pmt_pattern).all(), "payment_id contains non-standard prefix leaks!"


def test_legitimate_sharing_present(sample_dataset):
    dataset, _ = sample_dataset
    merged = dataset.orders.merge(dataset.labels[["order_id", "is_abuse"]], on="order_id")
    legit = merged[~merged.is_abuse]

    # Check that legitimate accounts share devices and addresses
    dev_users = legit.groupby("device_id")["customer_id"].nunique()
    addr_users = legit.groupby("address_id")["customer_id"].nunique()

    assert (dev_users > 1).sum() > 0, "No legitimate multi-account device sharing found!"
    assert (addr_users > 1).sum() > 0, "No legitimate multi-account address sharing found!"


def test_abusive_legitimate_overlap(sample_dataset):
    dataset, _ = sample_dataset
    split = split_by_time(dataset.orders, 0.70, 0.15)
    quality = compute_dataset_quality(dataset, split)

    overlap = quality.entity_overlap
    assert overlap["device_overlap_count"] > 0, "Abuse devices must overlap with legitimate pool"
    assert overlap["address_overlap_count"] > 0, "Abuse addresses must overlap with legitimate pool"
    assert overlap["ip_overlap_count"] > 0, "Abuse IPs must overlap with legitimate pool"
    assert overlap["payment_overlap_count"] > 0, "Abuse payments must overlap with legitimate pool"


def test_deterministic_dataset_generation():
    cfg = Config(seed=1234, customers=1000, orders=2500, date_range_days=60, rings=RingConfig(count=20))
    d1 = generate_ecosystem(cfg)
    d2 = generate_ecosystem(cfg)

    pd.testing.assert_frame_equal(d1.orders, d2.orders)
    pd.testing.assert_frame_equal(d1.labels, d2.labels)
    pd.testing.assert_frame_equal(d1.customers, d2.customers)


def test_dataset_quality_reporting(sample_dataset):
    dataset, _ = sample_dataset
    split = split_by_time(dataset.orders, 0.70, 0.15)
    report = compute_dataset_quality(dataset, split)

    assert not report.split_summary.empty
    assert set(report.split_summary["Split"]) == {"Train", "Validation", "Test"}
    assert not report.rings_by_type.empty
    assert "Total" in set(report.rings_by_type["Ring Type"])
    assert report.ring_statistics["total_rings"] > 0
    assert report.ring_statistics["mean_ring_size"] > 0
