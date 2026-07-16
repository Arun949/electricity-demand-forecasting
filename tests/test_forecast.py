import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from feature_engineering import FeatureEngineer
from forecast import ForecastEngine


@pytest.fixture
def forecast_engine(small_combined_df):
    """Builds a ForecastEngine through the exact same feature pipeline as
    production (create_features -> split -> fit_outlier_bounds -> cap ->
    scale), just with the small fixture + a fast LinearRegression instead of
    two years of data + GBDT. Entirely in-memory: scale_features(fit=True)
    has no disk side effect (see feature_engineering.py), so this never
    touches the real project's models/ directory.
    """
    fe = FeatureEngineer(config_path="config/config.yaml")
    featured = fe.create_features(small_combined_df.copy())
    train_df, _, _ = fe.create_splits(featured)
    fe.fit_outlier_bounds(train_df)
    train_df = fe.cap_outliers(train_df)
    train_scaled = fe.scale_features(train_df, fit=True)

    X_train = train_scaled.drop(columns=["demand"])
    y_train = train_scaled["demand"]
    model = LinearRegression().fit(X_train, y_train)

    return ForecastEngine(
        config=fe.config,
        history=small_combined_df,
        model=model,
        model_name="linear_regression",
        feature_cols=list(X_train.columns),
        scaler=fe.scaler,
        scaler_feature_cols=fe._fitted_feature_cols,
    )


def test_min_max_datetime_bounds(forecast_engine):
    max_depth = max(forecast_engine.lags + forecast_engine.windows)
    assert forecast_engine.min_datetime == forecast_engine.history_start + pd.Timedelta(hours=max_depth)
    assert forecast_engine.max_datetime > forecast_engine.history_end


def test_predict_at_historical_point_is_reasonably_accurate(forecast_engine):
    target = forecast_engine.min_datetime + pd.Timedelta(hours=200)
    result = forecast_engine.predict_at(target)

    assert result["is_forecast"] is False
    assert result["weather_source"] == "historical (real)"
    assert np.isfinite(result["demand_predicted"])
    assert not np.isnan(result["demand_actual"])
    # Fixture signal is smooth + noisy; a linear fit should land within ~15%.
    error_pct = abs(result["demand_predicted"] - result["demand_actual"]) / result["demand_actual"] * 100
    assert error_pct < 15


def test_predict_at_future_point_triggers_recursive_rollout(forecast_engine):
    future_target = forecast_engine.history_end + pd.Timedelta(hours=10)
    result = forecast_engine.predict_at(future_target)

    assert result["is_forecast"] is True
    assert np.isnan(result["demand_actual"])
    assert result["weather_source"] in ("live forecast", "seasonal average")
    assert np.isfinite(result["demand_predicted"])
    assert result["demand_predicted"] > 0


def test_predict_range_spans_boundary_chronologically(forecast_engine):
    start = forecast_engine.history_end - pd.Timedelta(hours=3)
    end = forecast_engine.history_end + pd.Timedelta(hours=3)
    df = forecast_engine.predict_range(start, end)

    assert list(df.index) == sorted(df.index)
    assert not df.loc[df.index <= forecast_engine.history_end, "is_forecast"].any()
    assert df.loc[df.index > forecast_engine.history_end, "is_forecast"].all()


def test_feature_row_lags_and_rolling_match_manual_pandas_computation(forecast_engine):
    """Regression test for the exact bug class that once made the scaler see
    the wrong statistics: lag/rolling values must match a plain pandas
    shift+rolling computation on the same raw series."""
    ts = forecast_engine.min_datetime + pd.Timedelta(hours=50)
    row_df, _ = forecast_engine._build_feature_row(ts, forecast_engine.demand_history)

    demand = forecast_engine.demand_history
    for lag in forecast_engine.lags:
        expected = demand.loc[ts - pd.Timedelta(hours=lag)]
        assert row_df.iloc[0][f"demand_lag_{lag}h"] == pytest.approx(expected)
    for window in forecast_engine.windows:
        expected_mean = demand.shift(1).rolling(window).mean().loc[ts]
        assert row_df.iloc[0][f"demand_rolling_mean_{window}h"] == pytest.approx(expected_mean)


def test_history_is_reindexed_to_a_gap_free_hourly_grid(forecast_engine):
    """The fast rollout path does positional array arithmetic ("168 slots
    back" = "168 hours back"), which silently breaks if the source history
    has missing timestamps (real ENTSO-E/join data does). Regression test
    for that exact bug: the engine must present a perfectly regular grid."""
    expected = pd.date_range(
        forecast_engine.history_start, forecast_engine.history_end, freq="h",
        tz=forecast_engine.history_start.tz,
    )
    assert forecast_engine.demand_history.index.equals(expected)


def test_gappy_history_is_healed_and_predictions_stay_correct(small_combined_df):
    """`small_combined_df` is naturally gap-free, so it can't reproduce the
    original bug on its own -- drop a row here (mimicking a real ENTSO-E/
    join gap) and confirm the engine still builds a full grid and the fast
    path still matches the reference implementation across the gap."""
    gappy = small_combined_df.drop(small_combined_df.index[500])

    fe = FeatureEngineer(config_path="config/config.yaml")
    featured = fe.create_features(small_combined_df.copy())  # feature/model setup uses the intact df
    train_df, _, _ = fe.create_splits(featured)
    fe.fit_outlier_bounds(train_df)
    train_df = fe.cap_outliers(train_df)
    train_scaled = fe.scale_features(train_df, fit=True)
    model = LinearRegression().fit(train_scaled.drop(columns=["demand"]), train_scaled["demand"])

    engine = ForecastEngine(
        config=fe.config,
        history=gappy,  # the gappy one, exercising the reindex/interpolate path
        model=model,
        model_name="linear_regression",
        feature_cols=list(train_scaled.drop(columns=["demand"]).columns),
        scaler=fe.scaler,
        scaler_feature_cols=fe._fitted_feature_cols,
    )

    expected_index = pd.date_range(engine.history_start, engine.history_end, freq="h", tz=engine.history_start.tz)
    assert engine.demand_history.index.equals(expected_index)
    assert not engine.demand_history.isnull().any()

    demand_arr = engine.demand_history.to_numpy()
    for offset in [510, 520, 700]:  # spans across the healed gap at position 500
        ts = engine.history_start + pd.Timedelta(hours=offset)
        ref_pred, _ = engine._predict_one(ts, engine.demand_history)
        fast_pred, _ = engine._predict_one_fast(ts, demand_arr, offset)
        assert fast_pred == pytest.approx(ref_pred, abs=1e-6)


def test_fast_path_matches_reference_implementation_exactly(forecast_engine):
    """The recursive rollout's numpy fast path must agree with the slower,
    more obviously-correct DataFrame-based reference implementation --
    caught a real bug (positional misalignment from history gaps) before
    this test existed."""
    demand_series = forecast_engine.demand_history
    demand_arr = demand_series.to_numpy()
    min_offset = max(forecast_engine.lags + forecast_engine.windows)  # earliest point with full lag history

    for offset in [min_offset, min_offset + 240, len(demand_arr) - 1]:
        ts = forecast_engine.history_start + pd.Timedelta(hours=offset)
        ref_pred, ref_src = forecast_engine._predict_one(ts, demand_series)
        fast_pred, fast_src = forecast_engine._predict_one_fast(ts, demand_arr, offset)
        assert fast_pred == pytest.approx(ref_pred, abs=1e-6)
        assert fast_src == ref_src
