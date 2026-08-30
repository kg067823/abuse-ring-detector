"""Generate and save frozen Model F model bundle to artifacts/model_f_bundle.pkl."""
import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import pandas as pd

from abuse_ring_detector.config import load_config
from abuse_ring_detector.synthetic import generate_ecosystem
from abuse_ring_detector.splits import split_by_time
from abuse_ring_detector.features import build_subgraph_extended_features
from abuse_ring_detector.models import fit_model
from abuse_ring_detector.inference import save_model_artifact

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generate_artifact")

def main():
    config_path = "configs/default.yaml"
    out_path = Path("artifacts/model_f_bundle.pkl")
    
    logger.info("Generating ecosystem and fitting Model F...")
    config = load_config(config_path)
    dataset = generate_ecosystem(config)
    split = split_by_time(dataset.orders, config.split["train"], config.split["validation"])
    fs_all = build_subgraph_extended_features(dataset.orders, dataset.labels, config.graph["history_days"])
    
    train_ids = pd.Index(split.train["order_id"]) if hasattr(split.train, "order_id") else split.train.index
    model = fit_model(fs_all.X.loc[train_ids], fs_all.y.loc[train_ids], config.model["backend"], config.seed)
    model.feature_columns = fs_all.X.columns.tolist()
    
    checksum = save_model_artifact(model, out_path)
    logger.info(f"Successfully saved Model F bundle to {out_path} with SHA-256 checksum={checksum}")

if __name__ == "__main__":
    main()
