"""
Phase 2 (part 2): Feature engineering, temporal split, outlier capping, scaling.

Order matters for avoiding leakage:
  1. Create temporal/lag/rolling features from the full cleaned series
     (lags/rolling windows only look backward, so this step alone is safe).
  2. Split train/val/test temporally (no shuffling -- this is a time series).
  3. Fit outlier bounds AND the feature scaler on the TRAIN split only, then
     apply (never re-fit) them to val/test.
  4. The target column (`demand`) is left unscaled so downstream MAE/RMSE/MAPE
     stay in interpretable MW units.
"""
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils import get_logger, load_config, resolve_path

logger = get_logger(__name__)


class FeatureEngineer:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.processed_dir = resolve_path(self.config["data"]["processed_dir"])
        self.models_dir = resolve_path("models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.target = self.config["features"]["target"]
        self.lags = self.config["features"]["lags"]
        self.windows = self.config["features"]["rolling_windows"]
        self.scaler = StandardScaler()
        self._outlier_bounds = None

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["hour"] = df.index.hour
        df["dayofweek"] = df.index.dayofweek
        df["month"] = df.index.month
        df["quarter"] = df.index.quarter
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

        for lag in self.lags:
            df[f"demand_lag_{lag}h"] = df[self.target].shift(lag)

        for window in self.windows:
            df[f"demand_rolling_mean_{window}h"] = df[self.target].shift(1).rolling(window).mean()
            df[f"demand_rolling_std_{window}h"] = df[self.target].shift(1).rolling(window).std()

        n_before = len(df)
        df = df.dropna()
        logger.info(
            "Created %d features; dropped %d rows with NaN lags/rolling stats (%d remain)",
            df.shape[1], n_before - len(df), len(df),
        )
        return df

    def create_splits(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_ratio = self.config["split"]["train_ratio"]
        val_ratio = self.config["split"]["val_ratio"]

        n = len(df)
        train_idx = int(n * train_ratio)
        val_idx = int(n * (train_ratio + val_ratio))

        train_df = df.iloc[:train_idx].copy()
        val_df = df.iloc[train_idx:val_idx].copy()
        test_df = df.iloc[val_idx:].copy()

        for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            logger.info(
                "%-5s set: %5d samples (%s to %s)",
                name, len(split), split.index.min(), split.index.max(),
            )
        return train_df, val_df, test_df

    def fit_outlier_bounds(self, train_df: pd.DataFrame) -> None:
        """IQR bounds fit on TRAIN target only."""
        q1, q3 = train_df[self.target].quantile([0.25, 0.75])
        iqr = q3 - q1
        self._outlier_bounds = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        logger.info("Outlier bounds (fit on train): [%.0f, %.0f] MW", *self._outlier_bounds)

    def cap_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        lower, upper = self._outlier_bounds
        n_capped = ((df[self.target] < lower) | (df[self.target] > upper)).sum()
        df[self.target] = df[self.target].clip(lower=lower, upper=upper)
        if n_capped:
            logger.info("Capped %d outlier rows in demand", n_capped)
        return df

    def scale_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Scale everything except the target so MAE/RMSE/MAPE stay in MW.

        Deliberately has no disk side effect -- fitting is pure so it's safe
        to call from tests. `save_scaler()` is the explicit persist step,
        called only from `run()`.
        """
        df = df.copy()
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != self.target]

        if fit:
            df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
            self._fitted_feature_cols = feature_cols
        else:
            df[feature_cols] = self.scaler.transform(df[feature_cols])
        return df

    def save_scaler(self) -> None:
        with open(self.models_dir / "scaler.pkl", "wb") as f:
            pickle.dump({"scaler": self.scaler, "feature_cols": self._fitted_feature_cols}, f)

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = pd.read_csv(self.processed_dir / "combined_data.csv", index_col=0, parse_dates=True)
        df = self.create_features(df)

        train_df, val_df, test_df = self.create_splits(df)

        self.fit_outlier_bounds(train_df)
        train_df = self.cap_outliers(train_df)
        val_df = self.cap_outliers(val_df)
        test_df = self.cap_outliers(test_df)

        train_df = self.scale_features(train_df, fit=True)
        self.save_scaler()
        val_df = self.scale_features(val_df, fit=False)
        test_df = self.scale_features(test_df, fit=False)

        train_df.to_csv(self.processed_dir / "train_data.csv")
        val_df.to_csv(self.processed_dir / "val_data.csv")
        test_df.to_csv(self.processed_dir / "test_data.csv")

        logger.info(
            "Data leakage checks: scaler+outlier bounds fit on TRAIN only; "
            "features built from backward-looking lags/rolling windows only; "
            "target left unscaled for interpretable metrics."
        )
        return train_df, val_df, test_df


if __name__ == "__main__":
    FeatureEngineer().run()
