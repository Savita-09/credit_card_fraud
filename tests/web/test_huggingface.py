import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from src.dataset import load_training_frame, prepare_transactions
from src.data_processing import make_schema, split_dataset, validate_features
from src.feature_engineering import preprocessing
from src.train import training_folds


def source_rows():
    return pd.DataFrame({"trans_date_trans_time": ["1/1/19 0:00", "1/2/19 9:30"], "amt": [4.97, 100.0],
                         "category": ["misc_net", "grocery_pos"], "state": ["NC", "WA"], "city_pop": [3495, 149],
                         "lat": [0., 45.], "long": [0., -120.], "merch_lat": [0., 45.], "merch_long": [1., -120.],
                         "cc_num": ["1234567890123456789", "9876543210987654321"], "first": ["Example", "Example"],
                         "trans_num": ["id-a", "id-b"], "unix_time": [999, 999], "is_fraud": [0, 1]})


def test_huggingface_loader_uses_label_and_excludes_identifiers(tmp_path):
    path = tmp_path / "transactions.csv"
    source_rows().to_csv(path, index=False)
    frame, target, source = load_training_frame(path)
    assert target == "is_fraud" and frame[target].tolist() == [0, 1]
    assert frame.Time.tolist() == [0., 120600.]
    assert frame.Amount.tolist() == [4.97, 100.]
    assert not {"cc_num", "first", "trans_num", "unix_time"}.intersection(frame.columns)
    assert source["default_split"] == "chronological"
    assert source["time_origin"] == "2019-01-01T00:00:00"


def test_bad_dates_and_tampered_source_are_rejected(tmp_path):
    raw = source_rows()
    raw.loc[0, "trans_date_trans_time"] = "not a date"
    with pytest.raises(ValueError, match="dates"):
        prepare_transactions(raw)
    path = tmp_path / "transactions.csv"
    source_rows().to_csv(path, index=False)
    path.with_suffix('.source.json').write_text(json.dumps({"sha256": hashlib.sha256(b"different data").hexdigest()}))
    with pytest.raises(ValueError, match="checksum"):
        load_training_frame(path)


def test_chronological_partitions_and_cv_do_not_share_timestamp_boundaries():
    frame = pd.DataFrame({"Time": np.repeat(np.arange(301), 3), "Amount": np.arange(903) + 1,
                          "is_fraud": (np.arange(903) % 5 == 0).astype(int)})
    train, validation, test = split_dataset(frame.sample(frac=1, random_state=9), "is_fraud", strategy="chronological")
    assert train.Time.max() < validation.Time.min() <= validation.Time.max() < test.Time.min()
    assert len(train) + len(validation) + len(test) == len(frame)
    assert set(train.index).isdisjoint(validation.index) and set(validation.index).isdisjoint(test.index)
    for earlier, later in training_folds(train, "is_fraud", "chronological"):
        assert train.Time.iloc[earlier].max() < train.Time.iloc[later].min()
        assert set(earlier).isdisjoint(later)


def test_location_features_and_unknown_categories_share_inference_pipeline():
    frame = prepare_transactions(source_rows())
    labeled = frame.assign(is_fraud=[0, 1])
    schema = make_schema(labeled, "is_fraud")
    pipe = preprocessing(schema).fit(frame)
    engineered = pipe.named_steps["features"].transform(frame)
    assert engineered.merchant_distance_km.iloc[0] == pytest.approx(111.19508, rel=1e-5)
    assert engineered.merchant_distance_km.iloc[1] == pytest.approx(0.)
    assert {"Time_week_sin", "Time_week_cos"}.issubset(engineered.columns)
    unknown = frame.copy()
    unknown.loc[0, "category"] = "new-category"
    transformed = pipe.transform(unknown)
    assert transformed.dtype == np.float32
    assert np.isfinite(transformed).all() and len(transformed) == len(unknown)
    assert len(pipe.get_feature_names_out()) == transformed.shape[1]
    unknown.loc[0, "merch_lat"] = 100
    with pytest.raises(ValueError, match="degrees"):
        validate_features(unknown, schema)
