import numpy as np
import pandas as pd

from preprocessing import DataPreprocessor


def make_preprocessor():
    return DataPreprocessor(config_path="config/config.yaml")


def test_combine_datasets_flags_holidays(small_combined_df):
    pre = make_preprocessor()
    demand = small_combined_df[["demand"]].copy()
    weather = small_combined_df.drop(columns=["demand", "is_holiday"]).copy()
    holiday_date = small_combined_df.index[100].date()
    holidays_df = pd.DataFrame({"date": [pd.Timestamp(holiday_date)], "holiday": ["Test Holiday"]})

    combined = pre.combine_datasets(demand, weather, holidays_df)

    assert "is_holiday" in combined.columns
    assert combined.loc[combined.index.date == holiday_date, "is_holiday"].eq(1).all()
    assert combined["is_holiday"].sum() >= 24  # the whole flagged day


def test_clean_missing_fills_short_gaps(small_combined_df):
    pre = make_preprocessor()
    df = small_combined_df.copy()
    df.iloc[5:8, df.columns.get_loc("demand")] = np.nan  # 3h gap, well under the 6h ffill limit

    cleaned = pre.clean_missing(df)

    assert not cleaned["demand"].isnull().any()
    # forward-filled value should equal the last valid observation before the gap
    assert cleaned["demand"].iloc[5] == df["demand"].iloc[4]
