import numpy as np
import pandas as pd
import pytest
from src.data_processing import inspect_dataset, make_schema, read_csv, split_dataset, validate_features
from src.feature_engineering import preprocessing


def test_missing_numeric_and_unseen_category():
    train = pd.DataFrame({"Amount": [2., 4., None, 8.], "merchant": ["shop", "shop", "cafe", None], "Class": [0, 0, 1, 0]})
    schema = make_schema(train, "Class")
    X = validate_features(train.drop(columns="Class"), schema)
    preprocessor = preprocessing(schema).fit(X)
    unseen = validate_features(pd.DataFrame({"Amount": [None], "merchant": ["new merchant"]}), schema)
    assert np.isfinite(preprocessor.transform(unseen)).all()
    imputer = preprocessor.named_steps["columns"].named_transformers_["numeric"].named_steps["imputer"]
    assert imputer.statistics_[0] == 4
    preprocessor.transform(pd.DataFrame({"Amount": [1000000.], "merchant": ["shop"]}))
    assert imputer.statistics_[0] == 4


def test_duplicate_rows_cannot_cross_partitions():
    frame = pd.DataFrame({"V1": range(300), "Class": [int(i % 3 == 0) for i in range(300)]})
    frame = pd.concat([frame, frame.iloc[:20]], ignore_index=True)
    train, validation, test = split_dataset(frame)
    sets = [set(part.V1) for part in [train, validation, test]]
    assert sum(len(part) for part in [train, validation, test]) == 300
    assert not sets[0] & sets[1] and not sets[1] & sets[2] and not sets[0] & sets[2]
    assert all(part.Class.nunique() == 2 for part in [train, validation, test])


def test_conflicting_duplicate_labels_rejected():
    with pytest.raises(ValueError, match="conflicting"):
        split_dataset(pd.DataFrame({"V1": [1, 1], "Class": [0, 1]}))


@pytest.mark.parametrize("content", [b"", b"V1,V1\n1,2", b"V1,Amount\n1,2,3", b"V1,Amount\n1", b"V1,Amount\n", b'V1\n"unclosed', b"\xff\xff"])
def test_bad_csv_rejected(content):
    with pytest.raises(ValueError):
        read_csv(content)


def test_schema_rejects_extra_missing_and_invalid():
    schema = [{"name": "Amount", "type": "number"}]
    for frame in [pd.DataFrame({"unknown": [1]}), pd.DataFrame({"Amount": ["bad"]}), pd.DataFrame({"Amount": [-1]}), pd.DataFrame({"Amount": [float("inf")]}), pd.DataFrame({"Amount": [True]})]:
        with pytest.raises(ValueError):
            validate_features(frame, schema)


def test_wrong_dataset_cannot_silently_use_default_target():
    with pytest.raises(ValueError, match="Target 'Class'"):
        inspect_dataset(pd.DataFrame({"TARGET": [0, 1], "AMT_CREDIT": [5, 10]}))


def test_categorical_named_amount_does_not_invent_numeric_features():
    train = pd.DataFrame({"Amount": ["small", "large"], "V1": [0., 1.], "Class": [0, 1]})
    schema = make_schema(train, "Class")
    pipeline = preprocessing(schema).fit(train.drop(columns="Class"))
    assert len(pipeline.get_feature_names_out()) == pipeline.transform(train.drop(columns="Class")).shape[1]


def test_derived_feature_name_collision_is_rejected():
    frame = pd.DataFrame({"Amount": [1.], "Amount_log1p": [5.], "Class": [0]})
    with pytest.raises(ValueError, match="reserved"):
        preprocessing(make_schema(frame, "Class")).fit(frame.drop(columns="Class"))
