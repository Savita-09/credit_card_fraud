"""Schema inspection, strict validation and reproducible, stratified data splits."""
from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def read_csv(source: Path | bytes) -> pd.DataFrame:
    raw = source.read_bytes() if isinstance(source, Path) else source
    try:
        text = raw.decode("utf-8-sig")
        rows = csv.reader(io.StringIO(text), strict=True)
        header = next(rows)
        if not header or any(not name.strip() for name in header):
            raise ValueError("CSV must have a nonempty header for every column.")
        if len(set(header)) != len(header):
            raise ValueError("CSV contains duplicate column names.")
        count = 0
        for line, row in enumerate(rows, 2):
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError(f"CSV row {line} has {len(row)} fields; expected {len(header)}.")
            count += 1
        if not count:
            raise ValueError("CSV has no transaction rows.")
        return pd.read_csv(io.StringIO(text))
    except (UnicodeDecodeError, StopIteration, csv.Error, pd.errors.ParserError) as exc:
        raise ValueError("Invalid CSV. Use a UTF-8, comma-separated file with a header.") from exc


def inspect_dataset(frame: pd.DataFrame, target: str = "Class") -> dict:
    if target not in frame:
        raise ValueError(f"Target '{target}' was not found. Available columns: {list(frame.columns)}")
    if frame[target].isna().any() or set(frame[target].unique()) != {0, 1}:
        raise ValueError("Target must contain both binary labels 0 (legitimate) and 1 (fraud), without missing values.")
    if frame[target].value_counts().min() < 30:
        raise ValueError("At least 30 examples of each class are required for splitting and cross-validation.")
    numeric = frame.drop(columns=target).select_dtypes(include="number").columns.tolist()
    categorical = [c for c in frame if c not in numeric and c != target]
    if not numeric and not categorical:
        raise ValueError("Dataset has no input features.")
    return {
        "rows": len(frame), "columns": len(frame.columns), "target": target,
        "numeric_features": numeric, "categorical_features": categorical,
        "dtypes": frame.dtypes.astype(str).to_dict(),
        "missing_values": frame.isna().sum().astype(int).to_dict(),
        "duplicate_rows": int(frame.duplicated().sum()),
        "class_counts": {str(k): int(v) for k, v in frame[target].value_counts().items()},
        "fraud_percentage": float(frame[target].mean() * 100),
        "legitimate_percentage": float((1 - frame[target].mean()) * 100),
    }


def split_dataset(frame: pd.DataFrame, target: str = "Class", seed: int = 42, strategy: str = "stratified"):
    # Feature-identical rows are grouped by keeping one before any split; conflicting labels are rejected.
    features = [c for c in frame if c != target]
    if frame.groupby(features, dropna=False)[target].nunique().gt(1).any():
        raise ValueError("Identical feature rows have conflicting labels. Resolve them before training.")
    clean = frame.drop_duplicates(subset=features).copy()
    if strategy == "chronological":
        if "Time" not in clean or clean.Time.isna().any():
            raise ValueError("Chronological splitting requires nonmissing numeric Time values.")
        clean = clean.sort_values("Time", kind="stable")
        first_time, second_time = clean.Time.iloc[int(.6 * len(clean))], clean.Time.iloc[int(.8 * len(clean))]
        partitions = (clean[clean.Time < first_time], clean[(clean.Time >= first_time) & (clean.Time < second_time)], clean[clean.Time >= second_time])
        for part in partitions:
            if set(part[target].unique()) != {0, 1} or part[target].value_counts().min() < 30:
                raise ValueError("Each chronological partition needs at least 30 examples of both classes.")
        return partitions
    if strategy != "stratified":
        raise ValueError("Split strategy must be stratified or chronological.")
    train, holdout = train_test_split(clean, test_size=0.4, stratify=clean[target], random_state=seed)
    validation, test = train_test_split(holdout, test_size=0.5, stratify=holdout[target], random_state=seed)
    return train, validation, test


def make_schema(train: pd.DataFrame, target: str) -> list[dict]:
    schema = []
    for name in train.drop(columns=target):
        series = train[name]
        numeric = pd.api.types.is_numeric_dtype(series)
        if numeric:
            values = pd.to_numeric(series).replace([np.inf, -np.inf], np.nan)
            item = {"name": name, "type": "number", "default": float(values.median()) if values.notna().any() else 0.0,
                    "minimum_observed": float(values.min()) if values.notna().any() else None,
                    "maximum_observed": float(values.max()) if values.notna().any() else None}
        else:
            values = series.dropna().astype(str)
            if values.nunique() > 100:
                raise ValueError(f"'{name}' has more than 100 categories. Remove identifiers or engineer this feature explicitly.")
            item = {"name": name, "type": "category", "default": str(values.mode().iloc[0]) if len(values) else "missing",
                    "categories": sorted(values.unique().tolist())}
        item["nullable"] = True
        schema.append(item)
    return schema


def validate_features(frame: pd.DataFrame, schema: list[dict]) -> pd.DataFrame:
    expected = [f["name"] for f in schema]
    missing = sorted(set(expected) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(expected))
    if missing or extra:
        raise ValueError(f"Feature mismatch. Missing columns: {missing}. Unexpected columns: {extra}.")
    if frame.empty:
        raise ValueError("No transaction rows supplied.")
    result = frame.loc[:, expected].copy()
    for feature in schema:
        name = feature["name"]
        column = result[name]
        if feature["type"] == "number":
            if column.map(lambda x: isinstance(x, (bool, list, dict))).any():
                raise ValueError(f"'{name}' must contain numbers or empty values.")
            converted = pd.to_numeric(column, errors="coerce")
            invalid = column.notna() & converted.isna()
            if invalid.any() or np.isinf(converted.to_numpy(dtype=float, na_value=np.nan)).any():
                raise ValueError(f"'{name}' contains invalid or non-finite numbers.")
            if name in {"Amount", "Time"} and converted.lt(0).any():
                raise ValueError(f"'{name}' cannot be negative.")
            if name == "city_pop" and converted.lt(0).any():
                raise ValueError("'city_pop' cannot be negative.")
            if name in {"lat", "merch_lat", "long", "merch_long"}:
                limit = 90 if name in {"lat", "merch_lat"} else 180
                if converted.abs().gt(limit).any():
                    raise ValueError(f"'{name}' must be between {-limit} and {limit} degrees.")
            if converted.abs().gt(1e12).any():
                raise ValueError(f"'{name}' exceeds the supported numeric range (+/- 1e12).")
            result[name] = converted.astype(float)
        else:
            if column.dropna().map(lambda x: not isinstance(x, str) or len(x) > 200).any():
                raise ValueError(f"'{name}' must contain strings of at most 200 characters or empty values.")
            result[name] = column.where(column.notna(), np.nan)
    return result
