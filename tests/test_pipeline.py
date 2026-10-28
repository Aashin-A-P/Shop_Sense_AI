"""End-to-end checks for ShopSense AI data and saved pipelines."""

import numpy as np
import pandas as pd
import pytest

from utils.data_loader import (
    FEATURE_COLUMNS,
    REQUIRED_COLUMNS,
    get_test_sample,
    load_dataset,
    split_dataset,
)
from utils.predictions import MODEL_FILES, MODELS_DIR, load_metadata, load_model, predict_session


@pytest.fixture(scope="session")
def dataset():
    return load_dataset()


@pytest.fixture(scope="session")
def split(dataset):
    return split_dataset(dataset)


@pytest.mark.parametrize("model_key", list(MODEL_FILES))
def test_all_model_pipelines_load_successfully(model_key):
    model = load_model(model_key)
    assert hasattr(model, "predict")
    assert set(model.named_steps) == {"preprocessor", "classifier"}


def test_dataset_contains_required_columns(dataset):
    assert set(REQUIRED_COLUMNS).issubset(dataset.columns)


def test_split_preserves_class_distribution(dataset, split):
    _, _, y_train, y_test = split
    full_rate = dataset["Revenue"].astype(int).mean()
    assert abs(y_train.mean() - full_rate) < 0.002
    assert abs(y_test.mean() - full_rate) < 0.002


def test_train_and_test_indices_do_not_overlap(split):
    X_train, X_test, _, _ = split
    assert set(X_train.index).isdisjoint(X_test.index)
    assert len(X_train) + len(X_test) == 12330


@pytest.mark.parametrize("model_key", list(MODEL_FILES))
def test_models_generate_valid_predictions_and_probabilities(model_key, split):
    _, X_test, _, _ = split
    model = load_model(model_key)
    sample = X_test.iloc[[0]]
    prediction = model.predict(sample)
    probability = model.predict_proba(sample)
    assert prediction.shape == (1,)
    assert int(prediction[0]) in {0, 1}
    assert probability.shape == (1, 2)
    assert np.all((probability >= 0) & (probability <= 1))
    assert probability[0].sum() == pytest.approx(1.0)


@pytest.mark.parametrize("purchase", [True, False])
def test_sample_loading_uses_real_held_out_record(dataset, split, purchase):
    sample, actual = get_test_sample(dataset, purchase)
    _, X_test, _, y_test = split
    assert list(sample.index) == FEATURE_COLUMNS
    assert actual == int(purchase)
    assert sample.name in X_test.index
    assert int(y_test.loc[sample.name]) == actual


def test_sample_records_contain_all_17_features(dataset):
    sample, _ = get_test_sample(dataset, True)
    assert len(sample) == 17
    assert list(sample.index) == FEATURE_COLUMNS


def test_manual_prediction_schema_matches_training_schema(dataset):
    values = {}
    for column in FEATURE_COLUMNS:
        values[column] = dataset[column].iloc[0]
    frame = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    prediction, probability = predict_session(load_model("logistic_regression"), frame)
    assert prediction in {0, 1}
    assert 0 <= probability <= 1


def test_metadata_contains_evaluation_metrics():
    metadata = load_metadata()
    assert set(metadata["evaluation_metrics"]) == set(MODEL_FILES)
    for values in metadata["evaluation_metrics"].values():
        assert {"accuracy", "precision", "recall", "f1_score", "roc_auc", "confusion_matrix"}.issubset(values)


def test_confusion_matrices_contain_valid_counts():
    metadata = load_metadata()
    test_size = metadata["testing_samples"]
    for values in metadata["evaluation_metrics"].values():
        matrix = np.asarray(values["confusion_matrix"])
        assert matrix.shape == (2, 2)
        assert np.issubdtype(matrix.dtype, np.integer)
        assert np.all(matrix >= 0)
        assert matrix.sum() == test_size


def test_decision_tree_feature_importance_is_accessible():
    model = load_model("decision_tree")
    names = model.named_steps["preprocessor"].get_feature_names_out()
    importance = model.named_steps["classifier"].feature_importances_
    assert len(names) == len(importance)
    assert len(importance) > len(FEATURE_COLUMNS)
    assert importance.sum() == pytest.approx(1.0)
    assert np.all(importance >= 0)


def test_model_artifacts_are_real_files():
    for filename in MODEL_FILES.values():
        assert (MODELS_DIR / filename).stat().st_size > 1000

