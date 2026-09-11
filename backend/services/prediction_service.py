from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pandas as pd
from src.data_processing import validate_features


class PredictionService:
    def __init__(self, model, settings, history):
        self.model = model
        self.settings = settings
        self.history = history

    @property
    def default_threshold(self):
        override = self.settings.default_threshold
        return override if override is not None else self.model.metadata["selected_threshold"]

    def score(self, frame, threshold=None, explain=False, record=True, source="single"):
        frame = validate_features(frame, self.model.schema)
        threshold = self.default_threshold if threshold is None else threshold
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1.")
        probabilities = self.model.predict_proba(frame)
        if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
            raise RuntimeError("Model returned invalid scores.")
        results = []
        for index, probability in enumerate(probabilities):
            prediction = "Fraud" if probability >= threshold else "Legitimate"
            risk = "High" if probability >= self.settings.risk_high else "Medium" if probability >= self.settings.risk_medium else "Low"
            amount = frame.iloc[index].get("Amount")
            result = {"id": str(uuid4()), "prediction": prediction, "fraud_probability": float(probability),
                      "risk_level": risk, "threshold": threshold, "model_name": self.model.metadata["model_name"],
                      "model_version": self.model.metadata["model_version"], "signals": [],
                      "explanation_method": "Single-feature replacement with the training median/mode. Score deltas are sensitivity signals, non-additive and non-causal.",
                      "recommendation": "Flag for analyst review and verify transaction context." if prediction == "Fraud" else "No model flag. Continue routine transaction checks.",
                      "amount": float(amount) if amount is not None and pd.notna(amount) else None,
                      "created_at": datetime.now(timezone.utc).isoformat()}
            if explain:
                result["signals"] = self.explain(frame.iloc[[index]], float(probability))
            results.append(result)
        if record:
            self.history.record(results, source)
        return results

    def explain(self, row, probability):
        variants = pd.concat([row] * len(self.model.schema), ignore_index=True)
        for index, feature in enumerate(self.model.schema):
            variants.loc[index, feature["name"]] = feature["default"]
        reference_probabilities = self.model.predict_proba(variants)
        signals = []
        for feature, reference_probability in zip(self.model.schema, reference_probabilities):
            value = row.iloc[0][feature["name"]]
            signals.append({"feature": feature["name"], "value": None if pd.isna(value) else value,
                            "reference": feature["default"], "score_delta": float(probability - reference_probability)})
        return sorted(signals, key=lambda item: abs(item["score_delta"]), reverse=True)[:6]
