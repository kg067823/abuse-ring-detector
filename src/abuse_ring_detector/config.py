"""Configuration loading and validation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RingConfig:
    count: int = 180
    min_size: int = 5
    max_size: int = 24
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
    if not 100 <= config.rings.count <= 300:
        raise ValueError("rings.count must be between 100 and 300")
    if config.rings.min_size < 3 or config.rings.max_size < config.rings.min_size:
        raise ValueError("invalid ring size range")
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
