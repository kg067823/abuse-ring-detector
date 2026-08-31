"""Build the new, explicitly labeled Model F-R1 reconstruction."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from abuse_ring_detector.config import load_config
from abuse_ring_detector.features import build_subgraph_extended_features
from abuse_ring_detector.models import fit_model, predict_scores
from abuse_ring_detector.r1 import R1Calibration
from abuse_ring_detector.schemas import ModelBundle
from abuse_ring_detector.splits import split_by_time
from abuse_ring_detector.synthetic import generate_ecosystem

R1_VERSION = "model_f_r1"
R1_IDENTITY = "graph_temporal_custrel_subgraph"
R1_THRESHOLD = 0.50
OLD_CHECKSUM = "82e77daac0762a04"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ece(y_true: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    if total == 0:
        return 0.0
    value = 0.0
    for i in range(bins):
        mask = (scores >= edges[i]) & (scores <= edges[i + 1] if i == bins - 1 else scores < edges[i + 1])
        if mask.any():
            value += float(mask.mean()) * abs(float(y_true[mask].mean()) - float(scores[mask].mean()))
    return value


def metrics(y_true: np.ndarray, raw: np.ndarray, calibrated: np.ndarray) -> dict[str, float | int]:
    pred = calibrated >= R1_THRESHOLD
    return {
        "orders": int(len(y_true)),
        "positive_events": int(y_true.sum()),
        "raw_pr_auc": float(average_precision_score(y_true, raw)),
        "raw_roc_auc": float(roc_auc_score(y_true, raw)),
        "calibrated_brier": float(brier_score_loss(y_true, calibrated)),
        "calibrated_ece": float(ece(y_true, calibrated)),
        "precision_at_0_50": float(precision_score(y_true, pred, zero_division=0)),
        "recall_at_0_50": float(recall_score(y_true, pred, zero_division=0)),
        "f1_at_0_50": float(f1_score(y_true, pred, zero_division=0)),
        "true_positives": int(((y_true == 1) & pred).sum()),
        "false_positives": int(((y_true == 0) & pred).sum()),
        "false_negatives": int(((y_true == 1) & ~pred).sum()),
    }


def main() -> None:
    config = load_config(ROOT / "configs/default.yaml")
    if config.seed != 42:
        raise RuntimeError("R1 requires seed 42")

    dataset = generate_ecosystem(config)
    split = split_by_time(dataset.orders, config.split["train"], config.split["validation"])
    features = build_subgraph_extended_features(
        dataset.orders, dataset.labels, config.graph["history_days"]
    )
    feature_names = list(features.X.columns)
    if len(feature_names) != 137 or len(set(feature_names)) != 137:
        raise RuntimeError(f"R1 requires exactly 137 unique features, got {len(feature_names)}")
    if not np.isfinite(features.X.to_numpy(dtype=float)).all():
        raise RuntimeError("R1 feature matrix contains NaN or infinity")

    train_ids = pd.Index(split.train["order_id"])
    validation_ids = pd.Index(split.validation["order_id"])
    test_ids = pd.Index(split.test["order_id"])
    params = {
        "max_iter": int(config.model["max_iter"]),
        "learning_rate": float(config.model["learning_rate"]),
        "max_leaf_nodes": int(config.model["max_leaf_nodes"]),
    }
    bundle = fit_model(
        features.X.loc[train_ids], features.y.loc[train_ids],
        backend="hist_gradient_boosting", seed=42, params=params,
    )
    bundle.feature_columns = feature_names
    bundle.feature_manifest = features.manifest
    bundle.metadata.update({
        "model_version": R1_VERSION,
        "model_identity": R1_IDENTITY,
        "model_name": R1_IDENTITY,
        "architecture": "HistGradientBoostingClassifier",
        "seed": 42,
        "hyperparameters": params,
        "threshold": R1_THRESHOLD,
        "reconstruction": True,
        "original_artifact_recovered": False,
    })

    val_y = features.y.loc[validation_ids].to_numpy(dtype=int)
    val_raw = predict_scores(bundle, features.X.loc[validation_ids])
    clipped = np.clip(val_raw, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    platt = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
    platt.fit(logits, val_y)
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(val_raw, val_y)
    platt_adapter = R1Calibration("platt_scaling", platt)
    isotonic_adapter = R1Calibration("isotonic_regression", isotonic)
    platt_scores = platt_adapter.predict(val_raw)
    isotonic_scores = isotonic_adapter.predict(val_raw)
    platt_brier = float(brier_score_loss(val_y, platt_scores))
    isotonic_brier = float(brier_score_loss(val_y, isotonic_scores))
    if platt_brier <= isotonic_brier:
        calibration = platt_adapter
        calibration_method = "platt_scaling"
    else:
        calibration = isotonic_adapter
        calibration_method = "isotonic_regression"
    bundle.calibrator = calibration
    bundle.metadata.update({
        "calibration_method": calibration_method,
        "calibration_is_new_r1_fit": True,
        "calibration_validation_brier": {
            "platt_scaling": platt_brier,
            "isotonic_regression": isotonic_brier,
        },
    })

    test_y = features.y.loc[test_ids].to_numpy(dtype=int)
    test_raw = predict_scores(bundle, features.X.loc[test_ids])
    test_calibrated = calibration.predict(test_raw)
    measured = {
        "validation": metrics(val_y, val_raw, calibration.predict(val_raw)),
        "test": metrics(test_y, test_raw, test_calibrated),
    }

    artifact_path = ROOT / "artifacts/model_f_r1_bundle.pkl"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    import pickle
    with artifact_path.open("wb") as handle:
        pickle.dump(bundle, handle, protocol=4)
    checksum = sha256_file(artifact_path)
    if checksum == OLD_CHECKSUM or checksum.startswith(OLD_CHECKSUM):
        raise RuntimeError("R1 artifact unexpectedly reused the historical checksum")

    timestamp = datetime.now(timezone.utc).isoformat()
    feature_names_json = feature_names
    manifest = {
        "model_version": R1_VERSION,
        "model_identity": R1_IDENTITY,
        "architecture": "HistGradientBoostingClassifier",
        "feature_count": 137,
        "feature_names": feature_names_json,
        "feature_composition": {"baseline": 19, "graph": 18, "temporal": 30, "customer_relative": 30, "two_hop": 20, "subgraph": 20},
        "seed": 42,
        "hyperparameters": params,
        "threshold": R1_THRESHOLD,
        "calibration_method": calibration_method,
        "calibration_is_new_r1_fit": True,
        "training_data": "Reconstructed deterministic synthetic ecosystem from repository configs/default.yaml and generate_ecosystem(), seed 42.",
        "split": "Chronological split_by_time(): 70% train, 15% validation, 15% test.",
        "split_counts": {"train": int(len(train_ids)), "validation": int(len(validation_ids)), "test": int(len(test_ids))},
        "measured_metrics": measured,
        "artifact_path": "artifacts/model_f_r1_bundle.pkl",
        "artifact_sha256": checksum,
        "created_at": timestamp,
        "python_version": platform.python_version(),
        "dependencies": {"numpy": np.__version__, "pandas": pd.__version__},
        "reconstruction_statement": "This artifact is a reconstruction/new freeze and is not the original historical Model F artifact.",
        "original_checksum_not_reused": OLD_CHECKSUM,
    }
    (ROOT / "model_f_r1_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    contract = {
        "contract_version": "inference_contract_r1.v1",
        "model_version": R1_VERSION,
        "model_identity": R1_IDENTITY,
        "artifact_path": "artifacts/model_f_r1_bundle.pkl",
        "artifact_sha256": checksum,
        "feature_count": 137,
        "feature_names": feature_names_json,
        "threshold": R1_THRESHOLD,
        "calibration_method": calibration_method,
        "shadow_mode": True,
        "enforce_decisions": False,
        "reconstruction_statement": manifest["reconstruction_statement"],
    }
    (ROOT / "inference_contract_r1.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(json.dumps({"artifact": str(artifact_path), "sha256": checksum, "feature_count": 137, "calibration_method": calibration_method, "metrics": measured}, indent=2))


if __name__ == "__main__":
    main()
