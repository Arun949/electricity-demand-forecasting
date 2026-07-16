import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (  # noqa: E402
    PALETTE,
    PROJECT_ROOT,
    inject_base_css,
    load_best_model,
    pipeline_ready,
    plotly_layout,
    render_footer,
    render_header,
)

st.set_page_config(page_title="Feature Analysis", page_icon="🔍", layout="wide")
inject_base_css()
render_header("Feature Analysis", "What drives the model's predictions", icon="🔍")

if not pipeline_ready():
    st.warning("Run the pipeline first (see Home page).")
    st.stop()

model_bundle = load_best_model()
model, feature_cols = model_bundle["model"], model_bundle["feature_cols"]
blue = PALETTE["categorical"][0]

if hasattr(model, "feature_importances_"):
    imp_df = (
        pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
        .sort_values("importance", ascending=True)
        .tail(15)
        .reset_index(drop=True)
    )
    fig = go.Figure(go.Bar(
        x=imp_df["importance"], y=imp_df["feature"], orientation="h",
        marker_color=blue,
        text=[f"{v:.3f}" for v in imp_df["importance"]], textposition="outside", cliponaxis=False,
    ))
    fig.update_layout(**plotly_layout(
        title=dict(text=f"Top 15 features — {model_bundle['name'].replace('_', ' ').title()}", x=0),
        xaxis_title="Importance", height=520, showlegend=False,
    ))
    st.plotly_chart(fig, width="stretch")

    top5 = imp_df.sort_values("importance", ascending=False).head(5)
    st.caption(
        "Top driver: **" + top5.iloc[0]["feature"] + f"** ({top5.iloc[0]['importance']:.3f}) — "
        "recent lagged demand and time-of-week features dominate, matching how grid operators "
        "reason about load."
    )
else:
    st.info(f"{model_bundle['name']} does not expose feature_importances_ (e.g. linear/SVM models).")

importance_png = PROJECT_ROOT / "outputs" / "05_feature_importance.png"
if importance_png.exists():
    with st.expander("Static export (outputs/05_feature_importance.png)"):
        st.image(str(importance_png))

render_footer()
