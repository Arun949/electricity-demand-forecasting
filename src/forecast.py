"""
Interactive forecasting engine: given an arbitrary date & time, return a
demand prediction on request -- the "ask the model" feature.

Two regimes, both going through the same feature pipeline as training:
  1. Timestamp already in the historical window -> predicted straight from
     real observed inputs (weather, lags, rolling stats all real).
  2. Timestamp beyond the last known data point -> recursive (autoregressive)
     rollout: each hour's prediction is appended to a working demand series
     so the next hour's lag/rolling features can see it, exactly like a real
     production forecaster would when the true future value isn't in yet.
     Weather for that gap comes from a live Open-Meteo forecast when the
     timestamp falls in its ~16-day horizon, else a seasonal (month, hour)
     climatology computed from history -- always labeled so the UI can show
     provenance rather than presenting a guess as fact.
"""
from __future__ import annotations

import pickle
import warnings
from functools import lru_cache

import holidays as holidays_lib
import numpy as np
import pandas as pd
import requests

from utils import get_logger, load_config, resolve_path

# The fast rollout path predicts on a plain numpy array (see
# _predict_one_fast) rather than the DataFrame the model was fit on, which
# is intentional -- DataFrame validation is what makes scaler.transform() +
# a DataFrame-based predict() ~7x slower per call, and that matters when a
# multi-year forecast runs thousands of these in a row. sklearn/xgboost
# warn about the missing feature names on every such call; the values are
# still passed in the correct (asserted) order, so the warning is silenced
# rather than left to spam thousands of times per request.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

logger = get_logger(__name__)

MAX_HOURS_BEYOND_HISTORY = 24 * 365 * 3  # ~3 years beyond history_end; error compounds with horizon


@lru_cache(maxsize=8)
def _fetch_openmeteo_forecast_payload(lat: float, lon: float) -> dict | None:
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "hourly": "temperature_2m,relative_humidity_2m,precipitation,cloud_cover,wind_speed_10m",
                "timezone": "UTC",
                "forecast_days": 16,
            },
            timeout=10,
        )
        resp.raise_for_status()
        payload = resp.json()["hourly"]
        payload["_times"] = pd.to_datetime(payload["time"], utc=True)
        return payload
    except Exception as exc:  # noqa: BLE001
        logger.warning("Open-Meteo forecast fetch failed (%s); will fall back to climatology", exc)
        return None


class ForecastEngine:
    def __init__(
        self,
        config: dict,
        history: pd.DataFrame,
        model,
        model_name: str,
        feature_cols: list[str],
        scaler,
        scaler_feature_cols: list[str],
    ):
        """Pure constructor -- takes already-loaded data/model objects, no
        disk I/O, so it's cheap to build with small in-memory fixtures in
        tests. Real usage goes through `ForecastEngine.from_project()`."""
        self.config = config
        self.target = config["features"]["target"]
        self.lags = config["features"]["lags"]
        self.windows = config["features"]["rolling_windows"]
        self.country_code = config["project"]["country_code"]
        self.lat = config["data"]["weather"]["latitude"]
        self.lon = config["data"]["weather"]["longitude"]
        self.weather_cols = ["temperature", "humidity", "precipitation", "cloud_cover", "wind_speed"]

        # Real source data has occasional gaps (e.g. ENTSO-E publication
        # gaps, or a demand/weather timestamp that didn't survive the join)
        # -- missing *rows*, which preprocessing.py's ffill/bfill never sees
        # since it only fills NaNs in existing rows. The recursive rollout
        # below does fast positional array arithmetic ("168 hours before
        # this row" = "168 array slots before this row"), which is only
        # valid on a perfectly regular hourly grid, so reindex + interpolate
        # the handful of gaps here.
        full_index = pd.date_range(history.index.min(), history.index.max(), freq="h", tz=history.index.tz)
        history = history.reindex(full_index)
        history[self.target] = history[self.target].interpolate()
        history[self.weather_cols] = history[self.weather_cols].interpolate()

        self.demand_history = history[self.target]
        self.weather_history = history[self.weather_cols]
        self.history_start = history.index.min()
        self.history_end = history.index.max()

        max_depth = max(self.lags + self.windows)
        self.min_datetime = self.history_start + pd.Timedelta(hours=max_depth)
        self.max_datetime = self.history_end + pd.Timedelta(hours=MAX_HOURS_BEYOND_HISTORY)

        # Three-tier climatology fallback for weather when neither real
        # history nor a live forecast is available: (month, hour) is the
        # most specific and always present given >=1 full year of history,
        # but degrade gracefully to hour-only, then the overall mean, rather
        # than raising on a combination the history happens not to cover
        # (e.g. a short/partial-year dataset).
        clim = self.weather_history.copy()
        clim["month"], clim["hour"] = clim.index.month, clim.index.hour
        self._climatology = clim.groupby(["month", "hour"])[self.weather_cols].mean()
        self._climatology_by_hour = clim.groupby("hour")[self.weather_cols].mean()
        self._climatology_overall = self.weather_history[self.weather_cols].mean()

        self.model = model
        self.model_name = model_name
        self.feature_cols = feature_cols
        self.scaler = scaler
        self.scaler_feature_cols = scaler_feature_cols

        # The recursive rollout's hot path bypasses sklearn's transform()
        # (validation overhead dominates at this scale: ~1.4ms/call, versus
        # ~0.001ms for the equivalent raw numpy arithmetic) and predicts on
        # a plain array instead of a 1-row DataFrame. Both require
        # `feature_cols` and `scaler_feature_cols` to be the exact same
        # order -- assert it once, loudly, rather than silently mis-scale
        # (the exact bug class that once made a historical prediction 12.7%
        # off instead of ~0.03%; see README design decisions).
        assert feature_cols == scaler_feature_cols, (
            "model feature_cols and scaler feature_cols must match exactly, order included"
        )
        self._scaler_mean = scaler.mean_
        self._scaler_scale = scaler.scale_

        self._fr_holidays = holidays_lib.country_holidays(self.country_code)

    @classmethod
    def from_project(cls, config_path: str = "config/config.yaml") -> "ForecastEngine":
        """Real entrypoint: loads history/model/scaler from the project's
        data/ and models/ directories on disk."""
        config = load_config(config_path)
        processed_dir = resolve_path(config["data"]["processed_dir"])
        history = pd.read_csv(processed_dir / "combined_data.csv", index_col=0, parse_dates=True)

        models_dir = resolve_path("models")
        with open(models_dir / "best_model.pkl", "rb") as f:
            bundle = pickle.load(f)
        with open(models_dir / "scaler.pkl", "rb") as f:
            scaler_bundle = pickle.load(f)

        return cls(
            config=config,
            history=history,
            model=bundle["model"],
            model_name=bundle["name"],
            feature_cols=bundle["feature_cols"],
            scaler=scaler_bundle["scaler"],
            scaler_feature_cols=scaler_bundle["feature_cols"],
        )

    # ------------------------------------------------------------------ #
    # Weather provenance
    # ------------------------------------------------------------------ #
    def _climatology_weather(self, ts: pd.Timestamp) -> dict:
        key = (ts.month, ts.hour)
        if key in self._climatology.index:
            return self._climatology.loc[key].to_dict()
        if ts.hour in self._climatology_by_hour.index:
            return self._climatology_by_hour.loc[ts.hour].to_dict()
        return self._climatology_overall.to_dict()

    def _live_forecast_weather(self, ts: pd.Timestamp) -> dict | None:
        now = pd.Timestamp.now(tz="UTC")
        if not (now - pd.Timedelta(hours=1) <= ts <= now + pd.Timedelta(days=16)):
            return None
        payload = _fetch_openmeteo_forecast_payload(self.lat, self.lon)
        if payload is None:
            return None
        idx = payload["_times"].get_indexer([ts], method="nearest")[0]
        return {
            "temperature": payload["temperature_2m"][idx],
            "humidity": payload["relative_humidity_2m"][idx],
            "precipitation": payload["precipitation"][idx],
            "cloud_cover": payload["cloud_cover"][idx],
            "wind_speed": payload["wind_speed_10m"][idx],
        }

    def _get_weather(self, ts: pd.Timestamp) -> tuple[dict, str]:
        if ts in self.weather_history.index:
            return self.weather_history.loc[ts].to_dict(), "historical (real)"
        live = self._live_forecast_weather(ts)
        if live is not None:
            return live, "live forecast"
        return self._climatology_weather(ts), "seasonal average"

    # ------------------------------------------------------------------ #
    # Feature construction + single-point prediction
    # ------------------------------------------------------------------ #
    def _build_feature_row(self, ts: pd.Timestamp, demand_series: pd.Series) -> tuple[pd.DataFrame, str]:
        weather, weather_source = self._get_weather(ts)
        is_holiday = int(ts.date() in self._fr_holidays)

        row = {
            **weather,
            "is_holiday": is_holiday,
            "hour": ts.hour,
            "dayofweek": ts.dayofweek,
            "month": ts.month,
            "quarter": ts.quarter,
            "is_weekend": int(ts.dayofweek >= 5),
            "hour_sin": np.sin(2 * np.pi * ts.hour / 24),
            "hour_cos": np.cos(2 * np.pi * ts.hour / 24),
            "month_sin": np.sin(2 * np.pi * ts.month / 12),
            "month_cos": np.cos(2 * np.pi * ts.month / 12),
            "dow_sin": np.sin(2 * np.pi * ts.dayofweek / 7),
            "dow_cos": np.cos(2 * np.pi * ts.dayofweek / 7),
        }
        for lag in self.lags:
            row[f"demand_lag_{lag}h"] = demand_series.loc[ts - pd.Timedelta(hours=lag)]
        for window in self.windows:
            window_slice = demand_series.loc[ts - pd.Timedelta(hours=window): ts - pd.Timedelta(hours=1)]
            row[f"demand_rolling_mean_{window}h"] = window_slice.mean()
            row[f"demand_rolling_std_{window}h"] = window_slice.std()

        return pd.DataFrame([row])[self.feature_cols], weather_source

    def _predict_one(self, ts: pd.Timestamp, demand_series: pd.Series) -> tuple[float, str]:
        """Reference (DataFrame-based) single-point predictor. Correct and
        readable, but ~7x slower than `_predict_one_fast` at scale -- used
        for tests and any one-off caller that doesn't need the fast path."""
        row_df, weather_source = self._build_feature_row(ts, demand_series)
        scaled = row_df.copy()
        scaled[self.scaler_feature_cols] = self.scaler.transform(scaled[self.scaler_feature_cols])
        pred = float(self.model.predict(scaled)[0])
        return pred, weather_source

    def _build_feature_vector_fast(self, ts: pd.Timestamp, demand_arr: np.ndarray, i: int) -> tuple[np.ndarray, str]:
        """Same features/values as `_build_feature_row`, assembled directly
        into an already-scaled numpy vector: no per-call DataFrame
        construction and no sklearn `transform()` validation overhead. Used
        by the recursive rollout, which can run thousands of times per
        request. `i` is `ts`'s integer hour-offset from `history_start`, and
        `demand_arr` is a contiguous hourly array (real history, then
        recursively-appended predictions) indexed the same way.
        """
        weather, weather_source = self._get_weather(ts)
        is_holiday = float(ts.date() in self._fr_holidays)

        values = {
            **weather,
            "is_holiday": is_holiday,
            "hour": ts.hour,
            "dayofweek": ts.dayofweek,
            "month": ts.month,
            "quarter": ts.quarter,
            "is_weekend": float(ts.dayofweek >= 5),
            "hour_sin": np.sin(2 * np.pi * ts.hour / 24),
            "hour_cos": np.cos(2 * np.pi * ts.hour / 24),
            "month_sin": np.sin(2 * np.pi * ts.month / 12),
            "month_cos": np.cos(2 * np.pi * ts.month / 12),
            "dow_sin": np.sin(2 * np.pi * ts.dayofweek / 7),
            "dow_cos": np.cos(2 * np.pi * ts.dayofweek / 7),
        }
        for lag in self.lags:
            values[f"demand_lag_{lag}h"] = demand_arr[i - lag]
        for window in self.windows:
            window_vals = demand_arr[i - window:i]
            values[f"demand_rolling_mean_{window}h"] = window_vals.mean()
            values[f"demand_rolling_std_{window}h"] = window_vals.std(ddof=1)  # match pandas' default ddof=1

        raw = np.array([values[c] for c in self.feature_cols], dtype=float)
        scaled = (raw - self._scaler_mean) / self._scaler_scale
        return scaled, weather_source

    def _predict_one_fast(self, ts: pd.Timestamp, demand_arr: np.ndarray, i: int) -> tuple[float, str]:
        scaled_vec, weather_source = self._build_feature_vector_fast(ts, demand_arr, i)
        pred = float(self.model.predict(scaled_vec.reshape(1, -1))[0])
        return pred, weather_source

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def predict_range(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Hourly predictions for [start, end]. Never mutates engine state --
        each call works on a private numpy array seeded from history, so
        repeated or concurrent calls stay independent and reproducible."""
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        if end.tzinfo is None:
            end = end.tz_localize("UTC")

        hour = pd.Timedelta(hours=1)
        known_len = len(self.demand_history)
        total_len = max(int((end - self.history_start) / hour) + 1, known_len)
        demand_arr = np.full(total_len, np.nan)
        demand_arr[:known_len] = self.demand_history.to_numpy()

        rows = []

        # Recursive rollout beyond the last known data point.
        for i in range(known_len, total_len):
            ts = self.history_start + i * hour
            pred, weather_source = self._predict_one_fast(ts, demand_arr, i)
            demand_arr[i] = pred
            if ts >= start:
                rows.append((ts, pred, True, weather_source))

        # Historical portion, for chart context / actual-vs-predicted.
        safe_start = max(start, self.min_datetime)
        hist_end = min(end, self.history_end)
        if safe_start <= hist_end:
            start_i = int((safe_start - self.history_start) / hour)
            end_i = int((hist_end - self.history_start) / hour)
            for i in range(start_i, end_i + 1):
                ts = self.history_start + i * hour
                pred, weather_source = self._predict_one_fast(ts, demand_arr, i)
                rows.append((ts, pred, False, weather_source))

        if not rows:
            return pd.DataFrame(columns=["demand_predicted", "is_forecast", "weather_source", "demand_actual"])

        result = pd.DataFrame(rows, columns=["datetime", "demand_predicted", "is_forecast", "weather_source"])
        result = result.drop_duplicates(subset="datetime").set_index("datetime").sort_index()
        result["demand_actual"] = self.demand_history.reindex(result.index)
        result.loc[result["is_forecast"], "demand_actual"] = np.nan
        return result

    def predict_at(self, ts: pd.Timestamp) -> dict:
        df = self.predict_range(ts, ts)
        row = df.loc[pd.Timestamp(ts).tz_localize("UTC") if pd.Timestamp(ts).tzinfo is None else pd.Timestamp(ts)]
        return row.to_dict()
