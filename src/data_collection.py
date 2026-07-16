"""
Phase 1: Data Collection.

Pulls three sources and writes them to data/raw/:
  1. Electricity demand   -> ENTSO-E Transparency Platform (needs a free API key)
  2. Weather               -> Open-Meteo historical archive (free, no key needed)
  3. Public holidays       -> `holidays` Python package (offline, always real)

Weather and holidays are always real data. Demand requires an ENTSO-E API key
(register at https://transparency.entsoe.eu/, free). Without a key -- or if
the ENTSO-E call fails for any reason -- this module falls back to a clearly
labeled SYNTHETIC demand series (diurnal/weekly/seasonal pattern + temperature
dependence + noise) so the rest of the pipeline can still be built and tested
end to end. Every synthetic row is flagged in an `is_synthetic` column and a
warning file is written next to the CSV -- swap in a real ENTSOE_API_KEY and
rerun to replace it with real grid data before drawing any real conclusions.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from utils import get_logger, load_config, resolve_path

logger = get_logger(__name__)


class DataCollector:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.raw_dir = resolve_path(self.config["data"]["raw_dir"])
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        self.start = pd.Timestamp(self.config["data"]["start_date"], tz="UTC")
        self.end = pd.Timestamp(self.config["data"]["end_date"], tz="UTC")
        self.country_code = self.config["project"]["country_code"]

    # ------------------------------------------------------------------ #
    # 1. Electricity demand (ENTSO-E, with synthetic fallback)
    # ------------------------------------------------------------------ #
    def collect_demand(self, weather: pd.DataFrame | None = None) -> pd.DataFrame:
        api_key = os.getenv("ENTSOE_API_KEY", "").strip()

        if api_key:
            try:
                demand = self._fetch_entsoe_demand(api_key)
                demand["is_synthetic"] = 0
                logger.info("Collected %d hours of REAL ENTSO-E demand data", len(demand))
                return demand
            except Exception as exc:  # noqa: BLE001 - any failure -> fallback
                logger.warning("ENTSO-E fetch failed (%s); falling back to synthetic demand", exc)
        else:
            logger.warning(
                "ENTSOE_API_KEY not set (see .env.example) - generating SYNTHETIC demand "
                "so the pipeline is runnable. Add a free key and rerun for real grid data."
            )

        demand = self._generate_synthetic_demand(weather)
        warning_path = self.raw_dir / "SYNTHETIC_DATA_WARNING.txt"
        warning_path.write_text(
            "demand_fr.csv contains SYNTHETIC data (see is_synthetic column).\n"
            "Set ENTSOE_API_KEY in .env and rerun `python src/data_collection.py` "
            "to replace it with real ENTSO-E grid data before drawing conclusions.\n"
        )
        return demand

    def _fetch_entsoe_demand(self, api_key: str) -> pd.DataFrame:
        from entsoe import EntsoePandasClient

        client = EntsoePandasClient(api_key=api_key)
        # query_load returns a 1-column DataFrame named "Actual Load" (not a Series).
        raw = client.query_load(country_code=self.country_code, start=self.start, end=self.end)
        df = raw.iloc[:, [0]].copy()
        df.columns = ["demand"]
        df.index.name = "datetime"
        df.index = df.index.tz_convert("UTC")
        return df

    def _generate_synthetic_demand(self, weather: pd.DataFrame | None) -> pd.DataFrame:
        """Physically-plausible synthetic demand: diurnal + weekly + seasonal + temperature + noise."""
        index = pd.date_range(self.start, self.end, freq="h", inclusive="left")
        rng = np.random.default_rng(42)

        hour = index.hour.values
        dow = index.dayofweek.values
        doy = index.dayofyear.values

        base = 55_000.0
        # Two daily peaks (~8h and ~19h), trough overnight.
        daily = 5500 * np.exp(-((hour - 8) ** 2) / 10) + 6500 * np.exp(-((hour - 19) ** 2) / 8)
        daily -= 4000 * np.exp(-((hour - 4) ** 2) / 8)
        # Weekend demand runs lower (less industrial/commercial activity).
        weekend_effect = np.where(dow >= 5, -4500, 0)
        # Winter-peaking seasonality (France is heating-dominated).
        seasonal = 8000 * np.cos(2 * np.pi * (doy - 15) / 365.25)

        if weather is not None and "temperature" in weather:
            temp = weather["temperature"].reindex(index).interpolate().bfill().ffill().values
        else:
            temp = 12 - 8 * np.cos(2 * np.pi * (doy - 15) / 365.25)

        heating = 450 * np.clip(18 - temp, 0, None) ** 0.8
        cooling = 300 * np.clip(temp - 24, 0, None) ** 0.8

        # Slow-moving autocorrelated residual (random walk, mean-reverting) + white noise.
        walk = np.zeros(len(index))
        for i in range(1, len(walk)):
            walk[i] = 0.98 * walk[i - 1] + rng.normal(0, 150)
        white_noise = rng.normal(0, 600, len(index))

        demand = base + daily + weekend_effect + seasonal + heating + cooling + walk + white_noise
        demand = np.clip(demand, 25_000, 100_000)

        df = pd.DataFrame({"demand": demand, "is_synthetic": 1}, index=index)
        df.index.name = "datetime"
        logger.info("Generated %d hours of SYNTHETIC demand data", len(df))
        return df

    # ------------------------------------------------------------------ #
    # 2. Weather (Open-Meteo historical archive - free, no key, real)
    # ------------------------------------------------------------------ #
    def collect_weather(self) -> pd.DataFrame:
        lat = self.config["data"]["weather"]["latitude"]
        lon = self.config["data"]["weather"]["longitude"]
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": self.start.date().isoformat(),
            "end_date": self.end.date().isoformat(),
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,cloud_cover,wind_speed_10m",
            "timezone": "UTC",
        }
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()["hourly"]
            df = pd.DataFrame(
                {
                    "temperature": payload["temperature_2m"],
                    "humidity": payload["relative_humidity_2m"],
                    "precipitation": payload["precipitation"],
                    "cloud_cover": payload["cloud_cover"],
                    "wind_speed": payload["wind_speed_10m"],
                },
                index=pd.to_datetime(payload["time"], utc=True),
            )
            df.index.name = "datetime"
            # Archive API excludes the final partial day; trim to requested end.
            df = df[df.index < self.end]
            logger.info("Collected %d hours of REAL Open-Meteo weather data", len(df))
            return df
        except Exception as exc:  # noqa: BLE001
            logger.warning("Open-Meteo fetch failed (%s); falling back to synthetic weather", exc)
            return self._generate_synthetic_weather()

    def _generate_synthetic_weather(self) -> pd.DataFrame:
        index = pd.date_range(self.start, self.end, freq="h", inclusive="left")
        doy = index.dayofyear.values
        hour = index.hour.values
        rng = np.random.default_rng(7)

        temp = 12 - 8 * np.cos(2 * np.pi * (doy - 15) / 365.25)
        temp += 4 * np.sin(2 * np.pi * (hour - 6) / 24)  # warmer in afternoon
        temp += rng.normal(0, 1.5, len(index))

        df = pd.DataFrame(
            {
                "temperature": temp,
                "humidity": np.clip(70 - 0.8 * temp + rng.normal(0, 8, len(index)), 20, 100),
                "precipitation": np.clip(rng.exponential(0.3, len(index)) - 0.2, 0, None),
                "cloud_cover": np.clip(rng.normal(50, 25, len(index)), 0, 100),
                "wind_speed": np.clip(rng.normal(12, 5, len(index)), 0, None),
                "is_synthetic": 1,
            },
            index=index,
        )
        df.index.name = "datetime"
        logger.info("Generated %d hours of SYNTHETIC weather data", len(df))
        return df

    # ------------------------------------------------------------------ #
    # 3. Holidays (real, offline)
    # ------------------------------------------------------------------ #
    def collect_holidays(self) -> pd.DataFrame:
        import holidays as holidays_lib

        years = range(self.start.year, self.end.year + 1)
        fr_holidays = holidays_lib.country_holidays(self.country_code, years=years)
        # Sort (date, name) tuples -- dates compare fine; dicts don't support "<".
        sorted_items = sorted(fr_holidays.items())
        df = pd.DataFrame([{"date": date, "holiday": name} for date, name in sorted_items])
        df["date"] = pd.to_datetime(df["date"])
        logger.info("Collected %d French holidays (%s)", len(df), self.country_code)
        return df

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def run(self) -> dict[str, pd.DataFrame]:
        weather = self.collect_weather()
        demand = self.collect_demand(weather)
        holidays_df = self.collect_holidays()

        demand.to_csv(self.raw_dir / "demand_fr.csv")
        weather.to_csv(self.raw_dir / "weather_openmeteo_paris.csv")
        holidays_df.to_csv(self.raw_dir / "french_holidays.csv", index=False)

        logger.info("Raw data written to %s", self.raw_dir)
        return {"demand": demand, "weather": weather, "holidays": holidays_df}


if __name__ == "__main__":
    collector = DataCollector()
    collector.run()
