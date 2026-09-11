"""Train/inference-shared transformations; no label- or dataset-fitted heuristics."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler


def float32_array(values):
    return np.asarray(values, dtype=np.float32)


class TransactionFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.numeric_names_ = set(X.select_dtypes(include="number").columns)
        self.geo_features_ = {"lat", "long", "merch_lat", "merch_long"}.issubset(self.numeric_names_)
        reserved = ({"Amount_log1p"} if "Amount" in self.numeric_names_ else set()) | ({"Time_phase_sin", "Time_phase_cos"} if "Time" in self.numeric_names_ else set())
        if self.geo_features_:
            reserved |= {"merchant_distance_km"}
            if "Time" in self.numeric_names_:
                reserved |= {"Time_week_sin", "Time_week_cos"}
        if reserved.intersection(X.columns):
            raise ValueError("Input columns collide with reserved derived-feature names. Rename or remove those columns.")
        return self

    def transform(self, X):
        result = X.copy()
        if "Amount" in result and np.issubdtype(result.Amount.dtype, np.number):
            result["Amount_log1p"] = np.log1p(result.Amount)
        if "Time" in result and np.issubdtype(result.Time.dtype, np.number):
            # Time is elapsed seconds; these are relative daily phases, not a known local clock hour.
            phase = 2 * np.pi * result.Time / 86400
            result["Time_phase_sin"] = np.sin(phase)
            result["Time_phase_cos"] = np.cos(phase)
        if getattr(self, "geo_features_", False):
            lat1, lon1, lat2, lon2 = [np.radians(result[name]) for name in ["lat", "long", "merch_lat", "merch_long"]]
            haversine = np.sin((lat2 - lat1) / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2)**2
            result["merchant_distance_km"] = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))
            if "Time" in self.numeric_names_:
                week = 2 * np.pi * result.Time / (7 * 86400)
                result["Time_week_sin"], result["Time_week_cos"] = np.sin(week), np.cos(week)
        return result

    def get_feature_names_out(self, input_features=None):
        names = list(self.feature_names_in_)
        numeric_names = getattr(self, "numeric_names_", set(self.feature_names_in_))
        if "Amount" in numeric_names:
            names.append("Amount_log1p")
        if "Time" in numeric_names:
            names.extend(["Time_phase_sin", "Time_phase_cos"])
        if getattr(self, "geo_features_", False):
            names.append("merchant_distance_km")
            if "Time" in numeric_names:
                names.extend(["Time_week_sin", "Time_week_cos"])
        return np.asarray(names, dtype=object)


def preprocessing(schema):
    numeric = [f["name"] for f in schema if f["type"] == "number"]
    categorical = [f["name"] for f in schema if f["type"] == "category"]
    if "Amount" in numeric:
        numeric.append("Amount_log1p")
    if "Time" in numeric:
        numeric.extend(["Time_phase_sin", "Time_phase_cos"])
    if {"lat", "long", "merch_lat", "merch_long"}.issubset(numeric):
        numeric.append("merchant_distance_km")
        if "Time" in numeric:
            numeric.extend(["Time_week_sin", "Time_week_cos"])
    transformer = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                              ("scaler", RobustScaler()),
                              ("float32", FunctionTransformer(float32_array, feature_names_out="one-to-one"))]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                                  ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32))]), categorical),
    ], verbose_feature_names_out=False)
    return Pipeline([("features", TransactionFeatures()), ("columns", transformer)])
