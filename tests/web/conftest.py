"""Synthetic fixtures test contracts; benchmark reporting uses ULB only."""
import hashlib
import importlib.metadata
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SkPipeline
from backend.config import Settings
from backend.main import create_app
from src.data_processing import make_schema
from src.feature_engineering import preprocessing


@pytest.fixture(scope="session")
def web_artifact(tmp_path_factory):
    folder = tmp_path_factory.mktemp("web-model")
    rng = np.random.default_rng(17)
    frame = pd.DataFrame({"Amount": rng.uniform(0, 500, 240), "Time": rng.uniform(0, 86400, 240), "V1": rng.normal(0, 1, 240)})
    frame["Class"] = (frame.V1 > 1).astype(int)
    schema = make_schema(frame, "Class")
    pipeline = Pipeline(preprocessing(schema).steps + [("classifier", LogisticRegression(class_weight="balanced"))])
    pipeline.fit(frame.drop(columns="Class"), frame.Class)
    joblib.dump(pipeline, folder / "fraud_model.joblib")
    joblib.dump(SkPipeline(pipeline.steps[:2]), folder / "preprocessor.joblib")
    metadata = {"model_name": "Test logistic", "model_version": "test-v1", "features": schema,
                "selected_threshold": .4, "packages": {p: importlib.metadata.version(p) for p in ["scikit-learn", "numpy"]}, "dataset": {"source": "test fixture only"}}
    example = frame.drop(columns="Class").iloc[0].to_dict()
    for filename, obj in [("model_metadata.json", metadata), ("evaluation_snapshot.json", {"summary": {"total": 10}}), ("examples.json", {"legitimate": example, "fraud": example})]:
        (folder / filename).write_text(json.dumps(obj), encoding="utf-8")
    frame.drop(columns="Class").head(3).to_csv(folder / "sample_transactions.csv", index=False)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}
    (folder / "manifest.json").write_text(json.dumps({"model_version": "test-v1", "sha256": hashes}), encoding="utf-8")
    return folder


@pytest.fixture
def web_client(web_artifact, tmp_path):
    settings = Settings(_env_file=None, model_dir=web_artifact, history_db=tmp_path / "history.db", max_upload_bytes=2048, max_batch_rows=10)
    with TestClient(create_app(settings)) as client:
        yield client


@pytest.fixture
def feature_payload(web_artifact):
    return json.loads((web_artifact / "examples.json").read_text())["legitimate"]
