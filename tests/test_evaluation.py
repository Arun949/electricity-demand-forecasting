import numpy as np
import pandas as pd
import pytest

from evaluation import detect_drift, ensemble_predict, get_prediction_intervals


class _ConstantModel:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


def test_ensemble_predict_averages_fitted_models():
    models = {"a": _ConstantModel(100.0), "b": _ConstantModel(200.0)}
    X = pd.DataFrame({"f": range(5)})

    preds = ensemble_predict(models, X, names=["a", "b"])

    assert np.allclose(preds, 150.0)


def test_ensemble_predict_respects_weights():
    models = {"a": _ConstantModel(0.0), "b": _ConstantModel(100.0)}
    X = pd.DataFrame({"f": range(3)})

    preds = ensemble_predict(models, X, names=["a", "b"], weights=[3, 1])

    assert np.allclose(preds, 25.0)  # (3*0 + 1*100) / 4


def test_prediction_intervals_contain_point_forecast():
    y_val = pd.Series([100.0, 105.0, 95.0, 110.0])
    y_val_pred = pd.Series([100.0, 100.0, 100.0, 100.0])
    y_test_pred = np.array([200.0, 300.0])

    lower, upper = get_prediction_intervals(y_val, y_val_pred, y_test_pred)

    assert (lower < y_test_pred).all()
    assert (upper > y_test_pred).all()


def test_detect_drift_flags_clearly_shifted_distributions():
    reference = pd.Series(np.random.default_rng(42).normal(50_000, 1000, 500))
    identical = pd.Series(np.random.default_rng(42).normal(50_000, 1000, 500))
    shifted = pd.Series(np.random.default_rng(0).normal(70_000, 1000, 500))

    assert detect_drift(reference, shifted) is True
    assert detect_drift(reference, identical) is False
