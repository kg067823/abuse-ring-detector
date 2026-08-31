"""Validation tests for the newly reconstructed Model F-R1 artifact."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from abuse_ring_detector.models import predict_scores

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/model_f_r1_bundle.pkl"
MANIFEST = ROOT / "model_f_r1_manifest.json"
CONTRACT = ROOT / "inference_contract_r1.json"
OLD_CHECKSUM = "82e77daac0762a04"


def _metadata():
    return json.loads(MANIFEST.read_text())


def test_r1_artifact_manifest_and_contract_exist():
    assert ARTIFACT.exists()
    assert MANIFEST.exists()
    assert CONTRACT.exists()


def test_r1_byte_checksum_and_identity():
    manifest = _metadata()
    contract = json.loads(CONTRACT.read_text())
    checksum = hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
    assert len(checksum) == 64
    assert checksum == manifest["artifact_sha256"] == contract["artifact_sha256"]
    assert checksum != OLD_CHECKSUM
    with ARTIFACT.open("rb") as handle:
        bundle = pickle.load(handle)
    assert bundle.metadata["model_version"] == "model_f_r1"
    assert bundle.metadata["model_identity"] == "graph_temporal_custrel_subgraph"


def test_r1_feature_contract_and_configuration():
    manifest = _metadata()
    contract = json.loads(CONTRACT.read_text())
    assert manifest["feature_count"] == contract["feature_count"] == 137
    assert len(manifest["feature_names"]) == 137
    assert manifest["feature_names"] == contract["feature_names"]
    assert manifest["seed"] == 42
    assert manifest["threshold"] == contract["threshold"] == 0.50
    assert manifest["hyperparameters"] == {"max_iter": 160, "learning_rate": 0.08, "max_leaf_nodes": 15}
    assert manifest["reconstruction_statement"].startswith("This artifact is a reconstruction")


def test_r1_single_batch_parity_and_score_safety():
    with ARTIFACT.open("rb") as handle:
        bundle = pickle.load(handle)
    feature_names = bundle.feature_columns
    row = np.zeros((1, len(feature_names)), dtype=float)
    batch = np.zeros((3, len(feature_names)), dtype=float)
    single = predict_scores(bundle, __import__("pandas").DataFrame(row, columns=feature_names))
    scores = predict_scores(bundle, __import__("pandas").DataFrame(batch, columns=feature_names))
    assert single.shape == (1,)
    assert scores.shape == (3,)
    np.testing.assert_allclose(single[0], scores[0])
    assert np.isfinite(scores).all()
    assert ((scores >= 0.0) & (scores <= 1.0)).all()


def test_r1_calibration_is_persisted_and_safe():
    with ARTIFACT.open("rb") as handle:
        bundle = pickle.load(handle)
    assert bundle.calibrator is not None
    raw = np.array([0.0, 0.2, 0.5, 0.99])
    calibrated = bundle.calibrator.predict(raw)
    assert calibrated.shape == raw.shape
    assert np.isfinite(calibrated).all()
    assert ((calibrated >= 0.0) & (calibrated <= 1.0)).all()


def test_r1_measured_metrics_are_present_and_finite():
    metrics = _metadata()["measured_metrics"]
    for split in ("validation", "test"):
        assert metrics[split]["orders"] > 0
        for key, value in metrics[split].items():
            if isinstance(value, float):
                assert np.isfinite(value), (split, key, value)
