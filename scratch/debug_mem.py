from abuse_ring_detector.config import load_config
from abuse_ring_detector.synthetic import generate_ecosystem
from abuse_ring_detector.features import build_subgraph_extended_features
from abuse_ring_detector.inference import TransactionPayload, ProductionInferenceService
from abuse_ring_detector.models import fit_model
from abuse_ring_detector.splits import split_by_time
import pandas as pd
import traceback

try:
    config = load_config("configs/default.yaml")
    dataset = generate_ecosystem(config)
    orders = dataset.orders
    labels = dataset.labels

    split = split_by_time(orders, config.split["train"], config.split["validation"])
    fs_all = build_subgraph_extended_features(orders.head(100), labels, config.graph["history_days"])
    feature_names = fs_all.X.columns.tolist()

    train_ids = pd.Index(split.train["order_id"])
    train_ids = [tid for tid in train_ids if tid in fs_all.X.index]
    model_f = fit_model(fs_all.X.loc[train_ids], fs_all.y.loc[train_ids], config.model["backend"], config.seed)
    model_f.feature_columns = feature_names

    service = ProductionInferenceService(
        model=model_f,
        feature_names=feature_names,
        threshold=0.50,
        model_version="v1.0.0-ModelF",
        schema_version="v1.0.0"
    )

    p1 = TransactionPayload(order_id="TEST_01", customer_id="C_1", event_time="2025-06-01T10:00:00", amount=100.0)
    print("Scoring p1...")
    r1 = service.score_transaction(p1)
    print("r1:", r1)

    p2 = TransactionPayload(order_id="TIME_002", customer_id="C_TIME", event_time="2025-06-27T15:00:00", amount=200.0)
    print("Scoring p2...")
    r2 = service.score_transaction(p2)
    print("r2:", r2)

except Exception as e:
    traceback.print_exc()
