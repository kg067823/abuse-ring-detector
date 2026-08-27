"""Configuration loading and validation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RingConfig:
    count: int = 260
    min_size: int = 4
    max_size: int = 16
    min_duration_days: int = 7
    max_duration_days: int = 24
    activity_rate: float = 0.85
    min_orders_per_member: int = 1
    max_orders_per_member: int = 3
    minimum_test_active_rings: int = 40
    minimum_test_rings_per_type: int = 8
    types: dict[str, float] = field(default_factory=lambda: {
        "shared_device": 0.25, "shared_address": 0.25,
        "behavioral": 0.25, "mixed": 0.25,
    })


@dataclass
class Config:
    seed: int = 42
    customers: int = 20_000
    orders: int = 50_000
    date_range_days: int = 180
    rings: RingConfig = field(default_factory=RingConfig)
    normal_sharing: dict[str, float] = field(default_factory=dict)
    split: dict[str, float] = field(default_factory=lambda: {"train": .7, "validation": .15, "test": .15})
    graph: dict[str, Any] = field(default_factory=lambda: {"history_days": 30, "community_algorithm": "greedy_modularity"})
    model: dict[str, Any] = field(default_factory=lambda: {"backend": "hist_gradient_boosting"})
    costs: dict[str, float] = field(default_factory=lambda: {"review_cost": 2.0, "false_positive_block_cost": 10.0, "alert_budget": .05})
    outputs: dict[str, Any] = field(default_factory=lambda: {"format": "csv.gz"})


def validate_config(config: Config) -> None:
    if config.customers < 10 or config.orders < 20 or config.date_range_days < 2:
        raise ValueError("customers/orders/date_range_days are too small")
    if not 10 <= config.rings.count <= 1000:
        raise ValueError("rings.count must be between 10 and 1000")
    if config.rings.min_size < 2 or config.rings.max_size < config.rings.min_size:
        raise ValueError("invalid ring size range")
    if config.rings.min_duration_days < 1 or config.rings.max_duration_days < config.rings.min_duration_days:
        raise ValueError("invalid ring duration range")
    if not 0.0 < config.rings.activity_rate <= 1.0:
        raise ValueError("ring activity_rate must be in (0, 1]")
    if set(config.rings.types) != {"shared_device", "shared_address", "behavioral", "mixed"}:
        raise ValueError("all four ring types must be configured")
    if abs(sum(config.rings.types.values()) - 1) > 1e-6 or any(v < 0 for v in config.rings.types.values()):
        raise ValueError("ring type weights must be nonnegative and sum to one")
    if abs(sum(config.split.values()) - 1) > 1e-6 or any(v <= 0 for v in config.split.values()):
        raise ValueError("split ratios must be positive and sum to one")
    if config.model.get("backend") not in {"hist_gradient_boosting", "xgboost", "auto"}:
        raise ValueError("unsupported model backend")


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    if overrides:
        raw.update(overrides)
    rings = RingConfig(**raw.pop("rings", {}))
    config = Config(rings=rings, **raw)
    validate_config(config)
    return config
