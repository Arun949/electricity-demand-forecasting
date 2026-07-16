"""
Phase 2 (part 1): Combine raw sources and clean missing values.

Deliberately does NOT do outlier capping or scaling here -- those are
statistics "fit" on data, and fitting them before the train/val/test split
would leak test-period information into training (a bug in the original
project guide, which scaled -- and even capped the target -- on the full
dataset before splitting). See feature_engineering.py, where those steps
are fit on the training split only and then applied to val/test.
"""
import pandas as pd

from utils import get_logger, load_config, resolve_path

logger = get_logger(__name__)


class DataPreprocessor:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.raw_dir = resolve_path(self.config["data"]["raw_dir"])
        self.processed_dir = resolve_path(self.config["data"]["processed_dir"])
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_raw(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        demand = pd.read_csv(self.raw_dir / "demand_fr.csv", index_col=0, parse_dates=True)
        weather = pd.read_csv(self.raw_dir / "weather_openmeteo_paris.csv", index_col=0, parse_dates=True)
        holidays_df = pd.read_csv(self.raw_dir / "french_holidays.csv", parse_dates=["date"])
        return demand, weather, holidays_df

    def combine_datasets(
        self, demand: pd.DataFrame, weather: pd.DataFrame, holidays_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Inner-join demand and weather on timestamp, flag holidays."""
        df = demand[["demand"]].join(
            weather.drop(columns=["is_synthetic"], errors="ignore"), how="inner"
        )
        holiday_dates = set(holidays_df["date"].dt.date)
        df["is_holiday"] = df.index.date
        df["is_holiday"] = df["is_holiday"].isin(holiday_dates).astype(int)

        logger.info("Combined dataset shape: %s", df.shape)
        logger.info("Date range: %s to %s", df.index.min(), df.index.max())
        return df

    def clean_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill short gaps (<=6h, causal), backward-fill any remainder."""
        df = df.copy()
        missing_before = df.isnull().sum()
        if missing_before.any():
            logger.info("Missing values before cleaning:\n%s", missing_before[missing_before > 0])

        df = df.ffill(limit=6).bfill()

        missing_after = df.isnull().sum().sum()
        if missing_after:
            logger.warning("%d missing values remain after cleaning", missing_after)
        return df

    def data_quality_report(self, df: pd.DataFrame) -> None:
        """Course-required bias/quality checks: missingness, temporal coverage, weekday/weekend skew."""
        missing_pct = df.isnull().mean() * 100
        logger.info("Missing data %%:\n%s", missing_pct[missing_pct > 0] if missing_pct.any() else "none")
        logger.info(
            "Temporal coverage: %s to %s (%d unique years)",
            df.index.min(), df.index.max(), df.index.year.nunique(),
        )
        weekend_mean = df.loc[df.index.dayofweek >= 5, "demand"].mean()
        weekday_mean = df.loc[df.index.dayofweek < 5, "demand"].mean()
        logger.info(
            "Weekday mean demand: %.0f MW | Weekend mean demand: %.0f MW (%.1f%% lower)",
            weekday_mean, weekend_mean, (weekday_mean - weekend_mean) / weekday_mean * 100,
        )

    def run(self) -> pd.DataFrame:
        demand, weather, holidays_df = self.load_raw()
        df = self.combine_datasets(demand, weather, holidays_df)
        df = self.clean_missing(df)
        self.data_quality_report(df)

        out_path = self.processed_dir / "combined_data.csv"
        df.to_csv(out_path)
        logger.info("Saved combined, cleaned dataset -> %s", out_path)
        return df


if __name__ == "__main__":
    DataPreprocessor().run()
