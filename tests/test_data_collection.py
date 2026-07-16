import pandas as pd

from data_collection import DataCollector


def make_collector():
    return DataCollector(config_path="config/config.yaml")


def test_synthetic_demand_has_no_nans_and_plausible_range():
    collector = make_collector()
    demand = collector._generate_synthetic_demand(weather=None)

    assert not demand["demand"].isnull().any()
    assert (demand["demand"] > 0).all()
    assert demand["demand"].between(20_000, 110_000).all()
    assert (demand["is_synthetic"] == 1).all()


def test_synthetic_weather_has_expected_columns():
    collector = make_collector()
    weather = collector._generate_synthetic_weather()

    for col in ["temperature", "humidity", "precipitation", "cloud_cover", "wind_speed"]:
        assert col in weather.columns
    assert not weather["temperature"].isnull().any()


def test_collect_holidays_returns_known_french_holiday():
    collector = make_collector()
    holidays_df = collector.collect_holidays()

    assert not holidays_df.empty
    assert {"date", "holiday"}.issubset(holidays_df.columns)
    bastille_days = holidays_df[holidays_df["date"].dt.strftime("%m-%d") == "07-14"]
    assert not bastille_days.empty
