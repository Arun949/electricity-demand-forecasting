import sys
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
    load_best_model,
    load_eval_metrics,
    load_test_data,
    pipeline_ready,
    plotly_layout,
    render_footer,
    render_header,
    render_sidebar_brand,
)

inject_base_css()
render_sidebar_brand()
render_header("Predictions", "Forecast vs. actual · most recent test-set window", icon="📈")

if not pipeline_ready():
    st.warning("Run the pipeline first (see Home page).")
    st.stop()

model_bundle = load_best_model()
model = model_bundle["model"]
feature_cols = model_bundle["feature_cols"]
metrics = load_eval_metrics()
test_df = load_test_data()

horizon = st.slider("Forecast window (hours)", min_value=24, max_value=168, value=48, step=24)
window = test_df.tail(horizon)

X_window = window[feature_cols]
y_actual = window["demand"]
y_pred = model.predict(X_window)

coverage_pct = metrics.get("prediction_interval_coverage_pct")
residual_std = (y_actual - y_pred).std()
upper = y_pred + 1.96 * residual_std
lower = y_pred - 1.96 * residual_std

blue = PALETTE["categorical"][0]

fig = go.Figure()
fig.add_trace(go.Scatter(x=window.index, y=upper, line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig.add_trace(
    go.Scatter(
        x=window.index, y=lower, fill="tonexty", fillcolor="rgba(42,120,214,0.12)",
        line=dict(width=0), name="95% interval",
    )
)
fig.add_trace(go.Scatter(
    x=window.index, y=y_actual, mode="lines", name="Actual demand",
    line=dict(color=PALETTE["ink"], width=2),
))
fig.add_trace(go.Scatter(
    x=window.index, y=y_pred, mode="lines", name="Predicted demand",
    line=dict(color=blue, width=2, dash="dot"),
))
fig.update_layout(**plotly_layout(
    title=dict(text=f"{model_bundle['name'].replace('_', ' ').title()} — Actual vs Predicted", x=0),
    xaxis_title="Time", yaxis_title="Demand (MW)", height=480, hovermode="x unified",
))
st.plotly_chart(fig, width="stretch")

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
col1.metric("Window MAPE", format_pct((abs((y_actual - y_pred) / y_actual)).mean() * 100))
col2.metric("Window mean demand", format_mw(y_actual.mean()))
if coverage_pct is not None:
    col3.metric("Interval coverage (full test set)", format_pct(coverage_pct, 1))

st.caption(
    "Predictions are computed on real held-out test-set rows the model never trained on. "
    "The shaded band is a 95% Gaussian interval sized from validation-set residual spread."
)

render_footer()
