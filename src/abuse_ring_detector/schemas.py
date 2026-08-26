"""Typed boundaries between POC pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import pandas as pd


@dataclass
class SyntheticDataset:
    customers: pd.DataFrame
    orders: pd.DataFrame
    returns: pd.DataFrame
    labels: pd.DataFrame
    ground_truth: pd.DataFrame
    rings: pd.DataFrame
    ring_memberships: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TimeSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_end: pd.Timestamp
    validation_end: pd.Timestamp


@dataclass
class FeatureSet:
    X: pd.DataFrame
    y: pd.Series
    ids: pd.Series
    manifest: pd.DataFrame


@dataclass
class GraphSnapshot:
    as_of: pd.Timestamp
    graph: nx.Graph
    communities: dict[str, int]


@dataclass
class ModelBundle:
    estimator: Any
    feature_columns: list[str]
    backend: str
    feature_manifest: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)
