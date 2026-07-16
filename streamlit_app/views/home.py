import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (  # noqa: E402
    format_mw,
    format_pct,
    inject_base_css,
    is_synthetic_run,
    load_eval_metrics,
    load_test_data,
    pipeline_ready,
    render_footer,
    render_header,
    render_sidebar_brand,
    status_badge,
)

inject_base_css()
render_sidebar_brand()
render_header("Electricity Demand Forecasting", "French grid · 24-48h ahead")

if not pipeline_ready():
    st.warning(
        "No trained model / test data found yet. Run the pipeline first:\n\n"
        "```bash\n"
        "python src/data_collection.py\n"
        "python src/preprocessing.py\n"
        "python src/feature_engineering.py\n"
        "python src/models.py\n"
        "python src/evaluation.py\n"
        "```"
    )
    st.stop()

metrics = load_eval_metrics()
test_df = load_test_data()
best = metrics["test_metrics"]
target_met = best["MAPE"] < 5.0

st.markdown('<div class="hero-band">', unsafe_allow_html=True)
with st.container(key="hero_kpi"):
    hcol1, hcol2 = st.columns([2, 1])
    with hcol1:
        st.markdown(
            '<div style="font-size:0.85rem;color:#52514e;font-weight:600;'
            'text-transform:uppercase;letter-spacing:.04em;">Test-set MAPE &middot; best model</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="font-size:3.2rem;font-weight:700;line-height:1.1;color:#0b0b0b;">'
            f'{format_pct(best["MAPE"])}</div>',
            unsafe_allow_html=True,
        )
        badge_html = (
            status_badge("Target met (&lt; 5% MAPE)", "good")
            if target_met
            else status_badge("Above 5% MAPE target", "critical")
        )
        st.markdown(badge_html, unsafe_allow_html=True)
    with hcol2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric("Winning model", metrics["best_model"].replace("_", " ").title())
st.markdown("</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
col1.metric("Test MAE", format_mw(best["MAE"]))
col2.metric("Test RMSE", format_mw(best["RMSE"]))
col3.metric("Test R²", f"{best['R2']:.3f}")

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
st.markdown("##### Dataset")
st.markdown(
    f"**{test_df.index.min():%b %d, %Y}** &rarr; **{test_df.index.max():%b %d, %Y}** "
    f"&middot; {len(test_df):,} hourly test-set records"
)

if is_synthetic_run():
    st.info(
        "This run used **synthetic demand data** (no ENTSOE_API_KEY was set during "
        "data collection). Add a free key to `.env` and rerun the pipeline for real "
        "ENTSO-E grid data.",
        icon="ℹ️",
    )

st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
st.markdown("##### Explore")
nc1, nc2, nc3 = st.columns(3)
with nc1:
    st.page_link("pages/1_ask_the_model.py", label="**Ask the Model**\n\nQuery any date live", icon="💬")
    st.page_link("pages/2_predictions.py", label="**Predictions**\n\nActual vs. predicted on held-out data", icon="📈")
with nc2:
    st.page_link("pages/3_model_performance.py", label="**Model Performance**\n\nCompare all 5 models", icon="📊")
    st.page_link("pages/4_feature_analysis.py", label="**Feature Analysis**\n\nWhat drives the forecast", icon="🔍")
with nc3:
    st.page_link("pages/5_data_drift.py", label="**Data Drift**\n\nStatistical drift monitoring", icon="🚨")

render_footer(f"Test window {test_df.index.min():%Y-%m-%d} to {test_df.index.max():%Y-%m-%d}")
