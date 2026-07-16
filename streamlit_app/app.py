import streamlit as st

st.set_page_config(
    page_title="Electricity Demand Forecasting",
    page_icon="⚡",
    layout="wide",
    menu_items={"About": "Electricity Demand Forecasting -- MSc AI, Statistical Machine Learning."},
)

home = st.Page("views/home.py", title="Home", icon="⚡", default=True, url_path="")
ask_the_model = st.Page("pages/1_ask_the_model.py", title="Ask the Model", icon="💬", url_path="ask_the_model")
predictions = st.Page("pages/2_predictions.py", title="Predictions", icon="📈", url_path="predictions")
model_performance = st.Page(
    "pages/3_model_performance.py", title="Model Performance", icon="📊", url_path="model_performance"
)
feature_analysis = st.Page(
    "pages/4_feature_analysis.py", title="Feature Analysis", icon="🔍", url_path="feature_analysis"
)
data_drift = st.Page("pages/5_data_drift.py", title="Data Drift", icon="🚨", url_path="data_drift")

pg = st.navigation([home, ask_the_model, predictions, model_performance, feature_analysis, data_drift])
pg.run()
