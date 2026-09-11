import shutil
import joblib
import numpy as np
import pandas as pd
import pytest
from backend.config import Settings
from backend.services.model_service import ModelService
from backend.services.prediction_service import PredictionService
from src.evaluate import evaluate, optimize_threshold


def test_saved_preprocessor_matches_model(web_artifact, feature_payload):
    model = ModelService(web_artifact)
    frame = pd.DataFrame([feature_payload])
    preprocessor = joblib.load(web_artifact / "preprocessor.joblib")
    direct = model.pipeline.named_steps["classifier"].predict_proba(preprocessor.transform(frame))[:, 1]
    np.testing.assert_allclose(direct, model.predict_proba(frame), rtol=1e-12)


def test_threshold_changes_decision_not_score(web_artifact, feature_payload):
    scorer = PredictionService(ModelService(web_artifact), Settings(_env_file=None), None)
    low = scorer.score(pd.DataFrame([feature_payload]), 0, explain=True, record=False)[0]
    high = scorer.score(pd.DataFrame([feature_payload]), 1, record=False)[0]
    assert low["prediction"] == "Fraud" and high["prediction"] == "Legitimate"
    assert 0 <= low["fraud_probability"] <= 1
    assert low["fraud_probability"] == high["fraud_probability"]
    assert low["risk_level"] == high["risk_level"]
    assert len(low["signals"]) == 3


def test_artifact_corruption_is_rejected_before_loading(web_artifact, tmp_path, monkeypatch):
    copy = tmp_path / "model"
    shutil.copytree(web_artifact, copy)
    (copy / "fraud_model.joblib").write_bytes(b"untrusted bytes")
    monkeypatch.setattr(joblib, "load", lambda *args: pytest.fail("corrupt model must never be deserialized"))
    with pytest.raises(ValueError, match="integrity"):
        ModelService(copy)


def test_threshold_is_validation_f2_optimum():
    labels = [0, 0, 0, 1, 1, 1]
    scores = [.01, .05, .4, .2, .7, .9]
    threshold = optimize_threshold(labels, scores)
    chosen = evaluate(labels, scores, threshold)["f2"]
    assert threshold != .5
    assert chosen >= max(evaluate(labels, scores, t)["f2"] for t in scores) - 1e-12


def test_threshold_preserves_endpoint_optimum():
    assert optimize_threshold([0, 0, 1, 1], [0., 0., 0., 0.]) == 0
    assert optimize_threshold([0, 0, 1, 1], [0., 0., 1., 1.]) == 1


def test_risk_configuration_validation():
    with pytest.raises(ValueError):
        Settings(_env_file=None, risk_medium=.8, risk_high=.3)


def test_real_artifact_when_available():
    from pathlib import Path
    path = Path("models")
    if not (path / "manifest.json").exists():
        pytest.skip("Train the ULB model to run artifact integration verification.")
    model = ModelService(path)
    assert model.metadata["dataset"]["splits"]["test"]["fraud"] > 0
    scores = model.predict_proba(pd.DataFrame(model.examples.values()))
    assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
