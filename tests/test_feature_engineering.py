import numpy as np

from feature_engineering import FeatureEngineer


def make_engineer():
    return FeatureEngineer(config_path="config/config.yaml")


def test_create_features_has_no_nans_and_expected_columns(small_combined_df):
    fe = make_engineer()
    featured = fe.create_features(small_combined_df)

    assert not featured.isnull().any().any()
    for col in ["hour_sin", "hour_cos", "demand_lag_1h", "demand_lag_168h", "demand_rolling_mean_24h"]:
        assert col in featured.columns


def test_splits_are_chronological_and_non_overlapping(small_combined_df):
    fe = make_engineer()
    featured = fe.create_features(small_combined_df)
    train, val, test = fe.create_splits(featured)

    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()
    assert len(train) + len(val) + len(test) == len(featured)


def test_outlier_bounds_fit_only_on_train_then_applied_elsewhere(small_combined_df):
    fe = make_engineer()
    featured = fe.create_features(small_combined_df)
    train, val, test = fe.create_splits(featured)

    fe.fit_outlier_bounds(train)
    bounds_from_train = fe._outlier_bounds

    train_capped = fe.cap_outliers(train)
    val_capped = fe.cap_outliers(val)

    assert fe._outlier_bounds == bounds_from_train  # capping val must not refit bounds
    assert train_capped["demand"].between(*bounds_from_train).all()
    assert val_capped["demand"].between(*bounds_from_train).all()


def test_scaler_fit_on_train_leaves_target_unscaled(small_combined_df):
    fe = make_engineer()
    featured = fe.create_features(small_combined_df)
    train, val, _ = fe.create_splits(featured)
    fe.fit_outlier_bounds(train)
    train = fe.cap_outliers(train)
    val = fe.cap_outliers(val)

    train_scaled = fe.scale_features(train, fit=True)
    val_scaled = fe.scale_features(val, fit=False)

    # Target must stay in original MW units (not standardized).
    assert np.isclose(train_scaled["demand"].mean(), train["demand"].mean())
    # Scaled train features should be ~standardized (mean ~0, std ~1).
    feature_cols = [c for c in train_scaled.columns if c != "demand"]
    assert abs(train_scaled[feature_cols].mean().mean()) < 0.5
    # Val was transformed with train's scaler, not refit -> not guaranteed mean 0.
    assert val_scaled.shape[1] == train_scaled.shape[1]
