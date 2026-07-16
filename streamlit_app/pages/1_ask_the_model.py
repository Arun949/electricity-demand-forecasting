import sys
from datetime import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (  # noqa: E402
    PALETTE,
    format_mw,
    format_pct,
    inject_base_css,
    pipeline_ready,
    plotly_layout,
    render_footer,
    render_header,
    status_badge,
)
from forecast import ForecastEngine  # noqa: E402

st.set_page_config(page_title="Ask the Model", page_icon="💬", layout="wide")
inject_base_css()
render_header("Ask the Model", "Pick any date & time — get a live prediction", icon="💬")

if not pipeline_ready():
    st.warning("Run the pipeline first (see Home page).")
    st.stop()


@st.cache_resource
def get_engine() -> ForecastEngine:
    return ForecastEngine.from_project()


@st.cache_data(show_spinner=False)
def cached_predict_range(_engine: ForecastEngine, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return _engine.predict_range(start, end)


engine = get_engine()
blue = PALETTE["categorical"][0]

st.markdown(
    "This isn't a fixed chart — it's a live query against the trained model. Pick a date and "
    "hour and the model computes a fresh prediction for that exact moment, built from the same "
    "features it was trained on (recent demand, weather, calendar)."
)

col_date, col_hour, col_btn = st.columns([2, 2, 1])
with col_date:
    picked_date = st.date_input(
        "Date",
        value=min(engine.history_end.date(), engine.max_datetime.date()),
        min_value=engine.min_datetime.date(),
        max_value=engine.max_datetime.date(),
    )
with col_hour:
    picked_time = st.selectbox(
        "Hour (UTC)", options=[time(h) for h in range(24)],
        format_func=lambda t: f"{t.hour:02d}:00", index=12,
    )
with col_btn:
    st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
    ask = st.button("Ask the model", type="primary", width="stretch")

target = pd.Timestamp.combine(picked_date, picked_time).tz_localize("UTC")
horizon_beyond_history = (target - engine.history_end).total_seconds() / 3600

if ask or "last_target" in st.session_state:
    st.session_state["last_target"] = target
    window_start = min(target, engine.history_end) - pd.Timedelta(hours=47)
    window_end = target
    with st.spinner("Predicting..."):
        result_df = cached_predict_range(engine, window_start, window_end)

    if target not in result_df.index:
        st.error("Couldn't compute a prediction for that moment — try a different date/time.")
        st.stop()

    point = result_df.loc[target]
    is_forecast = bool(point["is_forecast"])

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    hcol1, hcol2 = st.columns([2, 1])
    with hcol1:
        st.markdown(
            f'<div style="font-size:0.85rem;color:#52514e;font-weight:600;'
            f'text-transform:uppercase;letter-spacing:.04em;">Predicted demand · '
            f'{target:%A, %b %d %Y — %H:%M} UTC</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="font-size:3rem;font-weight:700;line-height:1.15;color:#0b0b0b;">'
            f'{format_mw(point["demand_predicted"])}</div>',
            unsafe_allow_html=True,
        )
        if is_forecast:
            st.markdown(status_badge("Forecast — beyond available data", "warning"), unsafe_allow_html=True)
        else:
            st.markdown(status_badge("Historical — verified against real data", "good"), unsafe_allow_html=True)
    with hcol2:
        st.markdown("<br>", unsafe_allow_html=True)
        if not is_forecast and not pd.isna(point["demand_actual"]):
            err = abs(point["demand_predicted"] - point["demand_actual"]) / point["demand_actual"] * 100
            st.metric("Actual demand", format_mw(point["demand_actual"]))
            st.metric("Error", format_pct(err))
        else:
            st.metric("Hours beyond last known data", f"{max(horizon_beyond_history, 0):.0f}h")
            st.metric("Weather input", point["weather_source"].title())

    if is_forecast and horizon_beyond_history > 48:
        st.info(
            "This project is scoped (and validated) for **24-48h-ahead** forecasts — see the "
            "Predictions and Model Performance pages for real accuracy in that window. Beyond "
            "48h, each hour's prediction feeds the next hour's lag features, so error compounds "
            "and this should be read as illustrative, not production-grade.",
            icon="⚠️",
        )

    # --- Context chart -----------------------------------------------------
    fig = go.Figure()
    actual = result_df["demand_actual"].dropna()
    if len(actual):
        fig.add_trace(go.Scatter(
            x=actual.index, y=actual, mode="lines", name="Actual demand",
            line=dict(color=PALETTE["ink"], width=2),
        ))
    fig.add_trace(go.Scatter(
        x=result_df.index, y=result_df["demand_predicted"], mode="lines", name="Predicted demand",
        line=dict(color=blue, width=2, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=[target], y=[point["demand_predicted"]], mode="markers", name="Your query",
        marker=dict(color=blue, size=11, line=dict(color="#ffffff", width=2)),
    ))
    fig.update_layout(**plotly_layout(
        title=dict(text=f"{engine.model_name.replace('_', ' ').title()} — 48h context around your query", x=0),
        xaxis_title="Time", yaxis_title="Demand (MW)", height=460, hovermode="x unified",
    ))
    st.plotly_chart(fig, width="stretch")

    with st.expander("What the model saw (inputs for this prediction)"):
        st.write(
            {
                "weather_source": point["weather_source"],
                "is_forecast": is_forecast,
                "target_utc": str(target),
                "last_known_data_point": str(engine.history_end),
            }
        )
else:
    st.caption("Pick a date and hour, then click **Ask the model**.")

render_footer()
