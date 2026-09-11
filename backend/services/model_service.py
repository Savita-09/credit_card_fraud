"""Load only operator-provided, integrity-checked artifacts; never user-uploaded pickle files."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
from src.data_processing import validate_features


class ModelService:
    def __init__(self, directory: Path):
        self.directory = directory.resolve(strict=True)
        manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        for filename in ["fraud_model.joblib", "preprocessor.joblib", "model_metadata.json", "evaluation_snapshot.json", "examples.json", "sample_transactions.csv"]:
            content = (self.directory / filename).read_bytes()
            if hashlib.sha256(content).hexdigest() != manifest["sha256"].get(filename):
                raise ValueError(f"Model artifact integrity check failed: {filename}")
        self.metadata = json.loads((self.directory / "model_metadata.json").read_text(encoding="utf-8"))
        if manifest["model_version"] != self.metadata["model_version"]:
            raise ValueError("Artifact and metadata versions disagree.")
        for package, version in self.metadata["packages"].items():
            if importlib.metadata.version(package) != version:
                raise ValueError(f"Package version mismatch for {package}; install the artifact's pinned environment or retrain.")
        # A checksum detects corruption, not a malicious operator replacing model + manifest.
        self.pipeline = joblib.load(self.directory / "fraud_model.joblib")
        self.schema = self.metadata["features"]
        self.snapshot = json.loads((self.directory / "evaluation_snapshot.json").read_text(encoding="utf-8"))
        self.examples = json.loads((self.directory / "examples.json").read_text(encoding="utf-8"))

    def predict_proba(self, frame):
        validated = validate_features(frame, self.schema)
        return self.pipeline.predict_proba(validated)[:, 1]
