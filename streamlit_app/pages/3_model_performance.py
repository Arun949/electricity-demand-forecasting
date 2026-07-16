import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (  # noqa: E402
    PALETTE,
    PROJECT_ROOT,
    inject_base_css,
    load_best_model,
    load_test_data,
    load_test_results,
    pipeline_ready,
    plotly_layout,
    render_footer,
    render_header,
    render_sidebar_brand,
)

inject_base_css()
render_sidebar_brand()
render_header("Model Performance", "Test-set comparison across all five models", icon="📊")

if not pipeline_ready():
    st.warning("Run the pipeline first (see Home page).")
    st.stop()

blue = PALETTE["categorical"][0]
muted = PALETTE["baseline"]

results_df = load_test_results()
if results_df is not None:
    results_df = results_df.sort_values("MAPE")
    best_model_name = results_df.index[0]

    st.dataframe(
        results_df,
        width="stretch",
        column_config={
            "MAPE": st.column_config.ProgressColumn(
                "MAPE (%)", format="%.2f%%", min_value=0, max_value=float(results_df["MAPE"].max()) * 1.15,
            ),
            "MAE": st.column_config.NumberColumn("MAE (MW)", format="%.0f"),
            "RMSE": st.column_config.NumberColumn("RMSE (MW)", format="%.0f"),
            "R2": st.column_config.NumberColumn("R²", format="%.3f"),
        },
    )

    bar_colors = [blue if m == best_model_name else muted for m in results_df.index]
    fig = go.Figure(go.Bar(
        x=results_df.index, y=results_df["MAPE"], marker_color=bar_colors,
        text=[f"{v:.2f}%" for v in results_df["MAPE"]], textposition="outside",
        cliponaxis=False,
    ))
    fig.update_layout(**plotly_layout(
        title=dict(text="Test MAPE by model (lower is better)", x=0),
        yaxis_title="MAPE (%)", height=420, showlegend=False,
    ))
    st.plotly_chart(fig, width="stretch")
else:
    st.warning("outputs/test_results.csv not found - run `python src/evaluation.py`.")

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
st.markdown("##### Predicted vs. actual — best model, full test set")

model_bundle = load_best_model()
test_df = load_test_data()
model = model_bundle["model"]
y_actual = test_df["demand"]
y_pred = model.predict(test_df[model_bundle["feature_cols"]])

fig2 = go.Figure()
lims = [min(y_actual.min(), y_pred.min()), max(y_actual.max(), y_pred.max())]
fig2.add_trace(go.Scatter(
    x=lims, y=lims, mode="lines", line=dict(color=muted, width=2, dash="dash"), name="Perfect fit",
))
fig2.add_trace(go.Scatter(
    x=y_actual, y=y_pred, mode="markers",
    marker=dict(size=4, opacity=0.35, color=blue), name="Predictions",
))
fig2.update_layout(**plotly_layout(
    title=dict(text=f"{model_bundle['name'].replace('_', ' ').title()} — predicted vs. actual demand", x=0),
    xaxis_title="Actual demand (MW)", yaxis_title="Predicted demand (MW)", height=480,
))
st.plotly_chart(fig2, width="stretch")

bias_variance_png = PROJECT_ROOT / "outputs" / "04_bias_variance_tradeoff.png"
if bias_variance_png.exists():
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown("##### Bias-variance tradeoff")
    st.image(str(bias_variance_png))
    st.caption(
        "Large gap between train and validation MAPE = overfitting (high variance). "
        "High error on both = underfitting (high bias)."
    )

render_footer()
