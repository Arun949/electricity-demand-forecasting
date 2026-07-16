import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def small_combined_df() -> pd.DataFrame:
    """~90 days of hourly data with a realistic-ish signal, for fast unit tests."""
    rng = np.random.default_rng(0)
    index = pd.date_range("2023-01-01", periods=24 * 90, freq="h", tz="UTC")
    hour = index.hour.values
    dow = index.dayofweek.values

    demand = (
        50_000
        + 5000 * np.sin((hour - 6) / 24 * 2 * np.pi)
        + np.where(dow >= 5, -3000, 0)
        + rng.normal(0, 500, len(index))
    )
    temperature = 10 + 5 * np.sin((hour - 6) / 24 * 2 * np.pi) + rng.normal(0, 1, len(index))

    return pd.DataFrame(
        {
            "demand": demand,
            "temperature": temperature,
            "humidity": rng.uniform(40, 90, len(index)),
            "precipitation": rng.exponential(0.2, len(index)),
            "cloud_cover": rng.uniform(0, 100, len(index)),
            "wind_speed": rng.uniform(0, 30, len(index)),
            "is_holiday": 0,
        },
        index=index,
    )
