import numpy as np
import pandas as pd
import pytest

from models import ModelTrainer, mape


def test_mape_known_value():
    y_true = pd.Series([100.0, 200.0, 300.0])
    y_pred = pd.Series([110.0, 180.0, 300.0])
    # errors: 10%, 10%, 0% -> mean 6.667%
    assert mape(y_true, y_pred) == pytest.approx(6.6667, abs=1e-3)


@pytest.fixture
def tiny_regression_data():
    rng = np.random.default_rng(1)
    n = 200
    X = pd.DataFrame(
        {"x1": rng.normal(size=n), "x2": rng.normal(size=n), "x3": rng.normal(size=n)},
        index=pd.date_range("2023-01-01", periods=n, freq="h"),
    )
    y = pd.Series(50_000 + 1000 * X["x1"] - 500 * X["x2"] + rng.normal(0, 50, n), index=X.index)
    return X, y


def test_train_baseline_and_evaluate(tiny_regression_data):
    X, y = tiny_regression_data
    trainer = ModelTrainer(config_path="config/config.yaml")
    model = trainer.train_baseline(X, y)
    metrics = trainer.evaluate(model, X, y, "linear_regression")

    assert set(metrics) == {"MAE", "RMSE", "MAPE", "R2"}
    assert metrics["R2"] > 0.9  # near-linear synthetic signal should fit well


def test_cross_validate_returns_one_score_per_fold(tiny_regression_data):
    X, y = tiny_regression_data
    trainer = ModelTrainer(config_path="config/config.yaml")
    model = trainer.train_baseline(X, y)
    scores = trainer.cross_validate(model, X, y, "linear_regression")

    assert len(scores) == trainer.cv_folds
    assert (scores >= 0).all()


def test_svm_subsamples_large_training_sets(tiny_regression_data):
    X, y = tiny_regression_data
    trainer = ModelTrainer(config_path="config/config.yaml")
    model = trainer.train_svm(X, y, max_samples=50)

    assert model.support_vectors_.shape[0] <= 50
