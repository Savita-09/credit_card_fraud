import pandas as pd
import pytest
from fastapi.testclient import TestClient
from backend.config import Settings
from backend.csv_utils import results_csv
from backend.main import create_app


def csv_file(features, count=2):
    return pd.DataFrame([features] * count).to_csv(index=False).encode()


def test_health_and_info(web_client):
    assert web_client.get("/health").json()["status"] == "healthy"
    assert web_client.get("/model-info").json()["active_threshold"] == .4
    assert web_client.get("/statistics?scope=live").json()["summary"]["total"] == 0
    assert web_client.get("/statistics?scope=wrong").status_code == 422


def test_missing_model_readiness(tmp_path):
    with TestClient(create_app(Settings(_env_file=None, model_dir=tmp_path / "absent", history_db=tmp_path / "db"))) as client:
        assert client.get("/health").status_code == 503
        assert client.get("/model-info").status_code == 503


def test_single_updates_live_statistics(web_client, feature_payload):
    response = web_client.post("/predict", json={"features": feature_payload, "threshold": 0})
    assert response.status_code == 200, response.text
    assert response.json()["prediction"] == "Fraud"
    summary = web_client.get("/statistics?scope=live").json()["summary"]
    assert summary == {"total": 1, "fraud": 1, "legitimate": 0, "fraud_percentage": 100.0}


def test_null_feature_uses_training_imputation(web_client, feature_payload):
    feature_payload["Amount"] = None
    assert web_client.post("/predict", json={"features": feature_payload}).status_code == 200


@pytest.mark.parametrize("mutation", [lambda f: f.pop("V1"), lambda f: f.update(extra=1), lambda f: f.update(Amount="invalid"), lambda f: f.update(Amount=True), lambda f: f.update(Amount=-1)])
def test_invalid_features(web_client, feature_payload, mutation):
    mutation(feature_payload)
    assert web_client.post("/predict", json={"features": feature_payload}).status_code == 422


def test_threshold_and_payload_limits(web_client, feature_payload):
    assert web_client.post("/predict", json={"features": feature_payload, "threshold": 2}).status_code == 422
    assert web_client.post("/predict", content=b"x" * 70000).status_code == 413
    assert web_client.post("/predict", content='{"features":{"Amount":NaN,"Time":0,"V1":0}}', headers={"Content-Type": "application/json"}).status_code == 422


def test_batch_validation_is_read_only_and_prediction_export(web_client, feature_payload):
    content = csv_file(feature_payload)
    response = web_client.post("/predict/batch/validate", files={"file": ("sample.csv", content, "text/csv")})
    assert response.status_code == 200, response.text
    assert response.json()["rows"] == 2
    assert web_client.get("/statistics?scope=live").json()["summary"]["total"] == 0
    result = web_client.post("/predict/batch?threshold=0", files={"file": ("sample.csv", content, "text/csv")})
    assert result.status_code == 200, result.text
    assert result.json()["fraud_count"] == 2
    assert "fraud_probability" in result.json()["csv"]
    assert web_client.get("/statistics?scope=live").json()["summary"]["total"] == 2
    exported = web_client.post("/predict/batch?format=csv", files={"file": ("sample.csv", content, "text/csv")})
    assert exported.status_code == 200 and "text/csv" in exported.headers["content-type"]
    assert "attachment" in exported.headers["content-disposition"]


@pytest.mark.parametrize("name,content,status", [("x.txt", b"Amount,Time,V1\n1,1,1", 422), ("x.csv", b"", 422), ("x.csv", b"Amount,Time,V1,Class\n1,1,1,0", 422), ("x.csv", b"Amount,Time,V1\nno,0,0", 422), ("x.csv", b"Amount,Time,V1\n" + b"1,1,1\n" * 11, 422), ("x.csv", b"x" * 2500, 413)])
def test_bad_uploads(web_client, name, content, status):
    assert web_client.post("/predict/batch", files={"file": (name, content)}).status_code == status


def test_unknown_request_keys_rejected(web_client, feature_payload):
    assert web_client.post("/predict", json={"features": feature_payload, "surprise": 1}).status_code == 422


def test_cors_allowlist(web_client):
    allowed = web_client.options("/predict", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:5173"
    denied = web_client.get("/health", headers={"Origin": "https://untrusted.example"})
    assert "access-control-allow-origin" not in denied.headers


def test_csv_formula_neutralization():
    frame = pd.DataFrame({"merchant": ["=CMD()", "  @evil", "safe"], "Amount": [-1, 2, 3]})
    results = [{"prediction": "Legitimate", "fraud_probability": .1, "risk_level": "Low", "threshold": .4, "model_version": "v1"}] * 3
    csv = results_csv(frame, results)
    assert "'=CMD()" in csv and "'  @evil" in csv and ",-1," in csv
