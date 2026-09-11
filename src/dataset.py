"""Pinned-source loading and explicit adaptation of Hugging Face transaction columns."""
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.data_processing import read_csv

HF_DATASET_ID = "santosh3110/credit_card_fraud_transactions"
TIME_ORIGIN = "2019-01-01T00:00:00"
HF_INPUTS = ["trans_date_trans_time", "amt", "category", "state", "city_pop", "lat", "long", "merch_lat", "merch_long"]
EXCLUDED_COLUMNS = ["Unnamed: 0", "cc_num", "merchant", "first", "last", "gender", "street", "city", "zip", "job", "dob", "trans_num", "unix_time"]


def prepare_transactions(raw):
    """Only transaction time, amount, category and coarse/location signals enter the model."""
    missing = set(HF_INPUTS) - set(raw.columns)
    if missing:
        raise ValueError(f"Hugging Face transaction columns missing: {sorted(missing)}")
    text = raw["trans_date_trans_time"]
    dates = pd.to_datetime(text, format="%m/%d/%y %H:%M", errors="coerce")
    for date_format in ["%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M", "%m/%d/%y %H:%M:%S"]:
        missing_dates = dates.isna()
        if not missing_dates.any():
            break
        dates.loc[missing_dates] = pd.to_datetime(text.loc[missing_dates], format=date_format, errors="coerce")
    if dates.isna().any():
        raise ValueError("Transaction dates must be valid month/day/year timestamps or ISO timestamps.")
    features = raw.loc[:, HF_INPUTS].drop(columns="trans_date_trans_time").rename(columns={"amt": "Amount"}).copy()
    features.insert(1, "Time", (dates - pd.Timestamp(TIME_ORIGIN)).dt.total_seconds())
    if features.Time.lt(0).any():
        raise ValueError("Transaction dates precede the supported time origin, 2019-01-01.")
    for name in ["category", "state"]:
        features[name] = features[name].astype(object)
    return features


def load_training_frame(path: Path, target=None):
    source_path = path.with_suffix(".source.json")
    source = json.loads(source_path.read_text(encoding="utf-8")) if source_path.exists() else {}
    if source.get("sha256"):
        with path.open("rb") as stream:
            actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual_hash != source["sha256"]:
            raise ValueError("Training CSV does not match its source manifest checksum.")
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    if set(HF_INPUTS).issubset(columns):
        target = target or "is_fraud"
        if target not in columns:
            raise ValueError(f"Target '{target}' is missing from the source CSV.")
        # Avoid loading names, card numbers, addresses and identifiers into training memory.
        raw = pd.read_csv(path, usecols=HF_INPUTS + [target])
        frame = prepare_transactions(raw)
        frame[target] = raw[target]
        source.update(raw_columns=columns, excluded_columns=[name for name in columns if name not in HF_INPUTS + [target]],
                      raw_target=target, time_origin=TIME_ORIGIN, adapter="huggingface-transactions-v1",
                      amount_unit=source.get("amount_unit", "dataset units"), currency=source.get("currency"),
                      default_split="chronological",
                      feature_note="Amount and elapsed seconds since 2019-01-01, transaction category, state, population and location coordinates. Names, card numbers and identifiers are excluded.")
    else:
        frame = read_csv(path)
        target = target or "Class"
        source.setdefault("default_split", "stratified")
        source.setdefault("name", "ULB credit-card benchmark" if "V28" in frame else "Transaction benchmark")
        source.setdefault("currency", "EUR" if "V28" in frame else None)
        source.setdefault("amount_unit", source.get("currency") or "dataset units")
        source.setdefault("feature_note", "Use the numeric/categorical feature schema saved with the model.")
    source.setdefault("name", "Hugging Face credit-card transactions" if source.get("dataset_id") == HF_DATASET_ID else "Transaction benchmark")
    source.setdefault("source", source.get("url", "user-provided CSV"))
    return frame, target, source
