import pandas as pd
from abuse_ring_detector.config import load_config
from abuse_ring_detector.synthetic import generate_ecosystem
from abuse_ring_detector.splits import split_by_time

config_180 = load_config("configs/default.yaml")
dataset_180 = generate_ecosystem(config_180)
split_180 = split_by_time(dataset_180.orders, config_180.split["train"], config_180.split["validation"])

print("--- 180-Day Dataset Split Audit ---")
print(f"Train     : {split_180.train.event_time.min()} to {split_180.train.event_time.max()} ({len(split_180.train)} orders)")
print(f"Validation: {split_180.train_end} to {split_180.validation_end} ({len(split_180.validation)} orders)")
print(f"Test      : {split_180.validation_end} to {split_180.test.event_time.max()} ({len(split_180.test)} orders)")

config_210 = load_config("configs/default.yaml")
config_210.date_range_days = 210
config_210.orders = 58333 # Scaled proportionally (50000 * 210 / 180)
dataset_210 = generate_ecosystem(config_210)
orders_210 = dataset_210.orders

holdout_210 = orders_210[orders_210["event_time"] > pd.Timestamp("2025-06-30")] # Days 180+
print("\n--- 210-Day Extended Ecosystem Holdout Audit ---")
print(f"Total Orders: {len(orders_210)}")
print(f"Untouched Holdout (Days 180+): {holdout_210.event_time.min()} to {holdout_210.event_time.max()} ({len(holdout_210)} orders)")
