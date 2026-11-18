
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    save_model,
    train_model,
)


@pytest.fixture
def sample_data():
    """Creates a small dataset for unit testing."""
    X = np.array([
        [25, 30],
        [30, 35],
        [35, 40],
        [45, 50],
        [50, 55],
        [55, 60],
    ])

    y = np.array([0, 0, 0, 1, 1, 1])

    return X, y


@pytest.fixture
def trained_model(sample_data):
    """Creates a trained model for unit testing."""
    X, y = sample_data
    return train_model(X, y)


def test_train_model(sample_data):
    """Tests whether the training function returns a fitted model."""
    X, y = sample_data
    model = train_model(X, y)

    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "classes_")
    assert model.n_features_in_ == X.shape[1]


def test_compute_model_metrics():
    """Tests whether model evaluation metrics are calculated correctly."""
    y = np.array([0, 0, 1, 1])
    preds = np.array([0, 1, 1, 1])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert precision == pytest.approx(2 / 3)
    assert recall == pytest.approx(1.0)
    assert fbeta == pytest.approx(0.8)


def test_inference(trained_model, sample_data):
    """Tests whether inference returns valid binary predictions."""
    X, _ = sample_data
    preds = inference(trained_model, X)

    assert len(preds) == len(X)
    assert np.all(np.isin(preds, [0, 1]))


def test_save_model(trained_model, tmp_path):
    """Tests whether the trained model is saved successfully."""
    model_path = tmp_path / "test_model.pkl"

    save_model(trained_model, model_path)

    assert model_path.exists()
    assert model_path.stat().st_size > 0


def test_load_model(trained_model, sample_data, tmp_path):
    """Tests whether a saved model can be loaded correctly."""
    X, _ = sample_data
    model_path = tmp_path / "test_model.pkl"

    save_model(trained_model, model_path)
    loaded_model = load_model(model_path)

    original_preds = inference(trained_model, X)
    loaded_preds = inference(loaded_model, X)

    assert isinstance(loaded_model, RandomForestClassifier)
    assert np.array_equal(original_preds, loaded_preds)
