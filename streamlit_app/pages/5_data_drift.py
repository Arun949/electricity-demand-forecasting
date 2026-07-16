import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import inject_base_css, load_test_data, pipeline_ready, render_footer, render_header  # noqa: E402

st.set_page_config(page_title="Data Drift", page_icon="🚨", layout="wide")
inject_base_css()
render_header("Data Drift Monitoring", "Kolmogorov–Smirnov test, earlier vs. later test window", icon="🚨")

if not pipeline_ready():
    st.warning("Run the pipeline first (see Home page).")
    st.stop()

test_df = load_test_data()
st.markdown(
    "Splits the test set into an earlier and later half and runs a "
    "**Kolmogorov-Smirnov test** on each numeric column, comparing distributions. "
    "This mirrors the check that would run in production between a reference "
    "training window and a recent scoring window."
)

midpoint = len(test_df) // 2
earlier, later = test_df.iloc[:midpoint], test_df.iloc[midpoint:]

alpha = st.slider("Significance level (alpha)", 0.01, 0.10, 0.05, 0.01)

rows = []
for col in test_df.select_dtypes(include="number").columns:
    stat, p_value = ks_2samp(earlier[col], later[col])
    rows.append({"feature": col, "ks_statistic": stat, "p_value": p_value, "drift": p_value < alpha})

drift_df = pd.DataFrame(rows).sort_values("ks_statistic", ascending=False)
drift_df["status"] = drift_df["drift"].map({True: "⚠ Drift detected", False: "✓ OK"})

n_drifted = int(drift_df["drift"].sum())
if n_drifted:
    st.error(
        f"{n_drifted} of {len(drift_df)} features show significant drift (p < {alpha}). "
        "Expected here: the test window spans Aug→Dec, a genuine seasonal shift in demand/weather."
    )
else:
    st.success(f"No significant drift detected across {len(drift_df)} features (p ≥ {alpha}).")

st.dataframe(
    drift_df[["feature", "ks_statistic", "p_value", "status"]],
    width="stretch",
    hide_index=True,
    column_config={
        "feature": st.column_config.TextColumn("Feature"),
        "ks_statistic": st.column_config.ProgressColumn(
            "KS statistic", format="%.3f", min_value=0.0, max_value=1.0,
        ),
        "p_value": st.column_config.NumberColumn("p-value", format="%.3e"),
        "status": st.column_config.TextColumn("Status"),
    },
)

st.caption(
    "Recommendation: if `demand` itself drifts, or several features drift together, "
    "retrain the model on a more recent window before trusting new predictions."
)

render_footer()
