"""Chronological dataset splitting."""
from __future__ import annotations

import pandas as pd

from .schemas import TimeSplit


def split_by_time(orders: pd.DataFrame, train_ratio=.70, validation_ratio=.15) -> TimeSplit:
    if not 0 < train_ratio < 1 or not 0 < validation_ratio < 1 or train_ratio + validation_ratio >= 1:
        raise ValueError("invalid split ratios")
    data = orders.sort_values(["event_time", "order_id"], kind="mergesort").reset_index(drop=True)
    n = len(data)
    train_end = data.iloc[max(0, int(n * train_ratio) - 1)].event_time
    val_end = data.iloc[max(0, int(n * (train_ratio + validation_ratio)) - 1)].event_time
    train = data[data.event_time <= train_end].copy()
    validation = data[(data.event_time > train_end) & (data.event_time <= val_end)].copy()
    test = data[data.event_time > val_end].copy()
    return TimeSplit(train, validation, test, pd.Timestamp(train_end), pd.Timestamp(val_end))
