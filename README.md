# Electricity Demand Forecasting

MSc AI - Statistical Machine Learning | EPITA International Programs

End-to-end ML pipeline that predicts hourly French electricity demand from
grid, weather, and calendar data: collection -> cleaning -> feature
engineering -> five compared model families -> evaluation -> MLflow tracking
-> a Streamlit dashboard.

> **Deadline check:** the project docs state 27/06/2026. If you're reading
> this after that date, double check the real deadline (it may have been a
> typo for 2025) before planning the remaining weeks.

## 1. What this is

- **Business problem:** forecast electricity demand 24-48h ahead to help
  grid operators balance supply/demand as renewable generation grows.
- **Target metric:** MAPE < 5% on a held-out, chronologically-final test set.
- **Models compared:** Linear Regression, SVM (RBF), Decision Tree, Random
  Forest, XGBoost (Gradient Boosting).
- **Deliverables:** reproducible pipeline (`src/`), EDA notebook
  (`notebooks/01_eda.ipynb`), MLflow experiment tracking (`mlruns.db`), a DVC
  pipeline definition (`dvc.yaml`), and a Streamlit dashboard
  (`streamlit_app/`).

## 2. Data sources

| Source | What | Free? | Notes |
|---|---|---|---|
| [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) | Hourly electricity demand (France) | Yes, needs a free API key | See setup below. Without a key, `data_collection.py` **automatically falls back to a labeled synthetic demand series** so the pipeline still runs end to end. |
| [Open-Meteo](https://open-meteo.com/) historical archive | Hourly temperature, humidity, precipitation, cloud cover, wind speed | Yes, no key needed | Swapped in for the guide's original NOAA suggestion: NOAA's CDO API needs a separate token and only gives daily data, while Open-Meteo is free, keyless, and already hourly. |
| [`holidays`](https://pypi.org/project/holidays/) Python package | French public holidays | Yes, offline | Always real - no network call. |

**If you want real grades from real data:** register at
https://transparency.entsoe.eu/ (Account Settings -> Web Api Security
Token, usually instant), copy `.env.example` to `.env`, and paste the key in
as `ENTSOE_API_KEY`. Rerun `python src/data_collection.py` afterwards - it
will overwrite the synthetic fallback with real grid data, and the
`SYNTHETIC_DATA_WARNING.txt` marker file simply won't be regenerated.

Until then, everything (including the Streamlit dashboard) runs on clearly
flagged synthetic demand, which is realistic enough (diurnal/weekly/seasonal
pattern + temperature dependence + noise) to validate that the whole
pipeline works, but **is not real grid data** - don't draw business
conclusions or report its MAPE as a real result.

## 3. Setup

```bash
cd electricity-demand-forecasting
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env              # then fill in ENTSOE_API_KEY (optional but recommended)
```

## 4. Run the pipeline

```bash
python src/data_collection.py      # -> data/raw/
python src/preprocessing.py        # -> data/processed/combined_data.csv
python src/feature_engineering.py  # -> data/processed/{train,val,test}_data.csv, models/scaler.pkl
python src/models.py               # -> models/*.pkl, outputs/validation_results.csv
python src/evaluation.py           # -> models/best_model.pkl, outputs/*.png, outputs/eval_metrics.json
```

Or reproduce the whole thing with DVC (tracks which stages are stale and
re-runs only what changed). `dvc` is deliberately left out of
`requirements.txt` -- its dependency resolution can make `pip`'s resolver
hang for a very long time (observed 8+ minutes doing effectively nothing);
install it on its own, ideally with a fast/stable connection:

```bash
pip install dvc
dvc init      # first time only
dvc repro
```

Explore the statistical analysis (normality tests, Pearson vs. Spearman
correlation, Central Limit Theorem demonstration, bias detection) in:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

Track experiments with MLflow:

```bash
python src/mlops_utils.py          # logs every trained model + metrics
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

(SQLite, not a plain `./mlruns` folder, because MLflow 3+ deprecated the
filesystem-only tracking store.)

Launch the dashboard (needs `models/best_model.pkl` and
`data/processed/test_data.csv`, i.e. run the pipeline first):

```bash
streamlit run streamlit_app/app.py
```

The dashboard is a native Streamlit multipage app: `app.py` is the home/
overview page, and `streamlit_app/pages/` holds:

1. **Ask the Model** - the interactive centerpiece. Pick any date and hour and
   the model predicts it live, on request - not a precomputed chart. Points
   inside the historical window are answered from real inputs (and compared
   against the real outcome); points beyond the last known data point trigger
   a genuine recursive (autoregressive) forecast, with live Open-Meteo
   forecast weather when the date is within its ~16-day horizon and a
   seasonal climatology fallback otherwise. Backed by `src/forecast.py`.
2. **Predictions** - actual vs. predicted over a fixed recent test-set window.
3. **Model Performance** - full test-set comparison across all five models.
4. **Feature Analysis** - live feature-importance ranking.
5. **Data Drift** - Kolmogorov-Smirnov drift check, earlier vs. later test window.

All five are computed live from your actual trained model and real held-out
data (not the placeholder `np.random` mock data from earlier drafts of this
guide).

## 5. Run the tests

```bash
pytest                             # config in pytest.ini, uses tests/ + a small synthetic fixture
```

Tests are unit-level and fast (no network, no full pipeline run): they check
feature engineering has no leakage (scaler/outlier bounds fit on train only,
splits stay chronological), metric math, and the synthetic-data fallbacks.

## 6. Project structure

```
electricity-demand-forecasting/
├── README.md
├── requirements.txt / .env.example / .gitignore / pytest.ini
├── config/config.yaml            # single source of truth for paths & hyperparameters
├── data/{raw,processed,external}
├── notebooks/01_eda.ipynb
├── src/
│   ├── utils.py                  # config loading, logging, path resolution
│   ├── data_collection.py        # ENTSO-E + Open-Meteo + holidays (+ synthetic fallback)
│   ├── preprocessing.py          # combine sources, clean missing values
│   ├── feature_engineering.py    # lags/rolling/cyclical features, temporal split, scaling
│   ├── models.py                 # 5 model families, TimeSeriesSplit CV, tuning
│   ├── evaluation.py             # test-set metrics, bias-variance, importance, drift
│   ├── mlops_utils.py            # MLflow logging
│   └── forecast.py               # interactive/recursive forecasting engine ("ask the model")
├── models/                       # trained model .pkl files (gitignored)
├── outputs/                      # plots + metrics.json (gitignored)
├── streamlit_app/
│   ├── app.py                    # home page
│   ├── common.py                 # shared cached loaders, design system
│   └── pages/1_ask_the_model.py, 2_predictions.py, 3_model_performance.py,
│             4_feature_analysis.py, 5_data_drift.py
├── tests/                        # pytest
├── dvc.yaml                      # DVC pipeline definition
└── .github/workflows/model-training.yml
```

## 7. Deploy to Streamlit Community Cloud

Free, made by Streamlit, deploys straight from GitHub. Two things make this
repo deploy-ready out of the box:

- **Pre-computed artifacts are committed**, not just gitignored data. Six
  files are deliberately carved out of the otherwise-broad `data/`/`models/`/
  `outputs/` ignore rules: `models/best_model.pkl`, `models/scaler.pkl`,
  `data/processed/{combined_data,test_data}.csv`, and the two `outputs/*.png`
  plots. That's everything the dashboard and "Ask the Model" read at
  runtime (~4.6 MB total) - the deployed app serves real results immediately
  instead of showing "run the pipeline first," and it needs **no secrets**:
  `ENTSOE_API_KEY` is only used by `data_collection.py` (a local/CI step),
  never by the Streamlit app itself. Open-Meteo's live forecast weather is
  free and keyless.
- **`packages.txt`** installs `libgomp1` - XGBoost's compiled extension
  needs it and Streamlit Cloud's base image doesn't include it by default;
  without this, `import xgboost` fails on first load.

### Steps

1. Push this directory (`electricity-demand-forecasting/`, not its parent)
   as the **root** of a GitHub repo - that's how Streamlit Cloud finds
   `requirements.txt` and `packages.txt` without extra path config.
2. Go to [share.streamlit.io](https://share.streamlit.io) -> **New app** ->
   pick the repo/branch -> set **Main file path** to `streamlit_app/app.py`.
3. Deploy. No secrets needed for the steps above. First build takes a few
   minutes (installing xgboost/mlflow/etc.); subsequent wakes from sleep are
   faster.

### Keeping the deployed app's data current

The committed artifacts are a **snapshot** - the deployed app won't
re-collect ENTSO-E data or retrain on its own. To update it: rerun the
pipeline locally (`python src/data_collection.py` through
`python src/evaluation.py`) with a fresh date range in `config/config.yaml`,
then commit and push the same six files again.

### If you'd rather not commit model/data files

Alternatives, roughly in order of effort: (a) point `data.start_date` /
`end_date` in `config/config.yaml` at a short recent window so the
artifacts stay small enough to feel fine committing; (b) add a startup
check in `streamlit_app/app.py` that runs the pipeline on first launch if
`models/best_model.pkl` is missing (adds `ENTSOE_API_KEY` as a Streamlit
Cloud secret, and a slow, network-dependent cold start); (c) host the
artifacts externally (e.g. a cloud storage bucket) and download them at
startup instead of committing them. None of these are wired up here - the
current setup optimizes for "clone, deploy, it just works."

## 8. Design decisions that differ from the original project brief

The original planning docs (`Electricity_Demand_Forecasting_Complete_Guide.md`,
`Advanced_Multi_Source_Data_Strategy.md`, `QUICK START.md`,
`Project_Overview.docx`) contain illustrative copy-paste code with a few bugs
that this implementation fixes:

- **No data leakage:** the guide's example fit the `StandardScaler` (and
  capped target outliers) on the *full* dataset before the train/val/test
  split, and even scaled the target column itself (which would make MAPE
  meaningless). Here, outlier bounds and the scaler are fit on the train
  split only, then applied - never refit - to val/test, and the target stays
  in raw MW units throughout.
- **Time-series-aware cross-validation:** `KFold(shuffle=False,
  random_state=42)` (from the guide) is both invalid in scikit-learn (a
  `random_state` requires `shuffle=True`) and not ideal for autocorrelated,
  lag-featured time series. This project uses `TimeSeriesSplit` instead, so
  every validation fold is strictly after its training fold.
- **Real, keyless weather data:** Open-Meteo replaces the guide's NOAA
  snippet, whose URL/params don't match NOAA's real API and would need a
  separate CDO token anyway.
- **SVM training is subsampled** (default 3,000 rows) because `SVR(kernel=
  "rbf")` scales roughly quadratically with sample count and the full
  training split has ~10,000+ hourly rows.
- **Ensembling is a manual prediction average**, not
  `sklearn.ensemble.VotingRegressor` - `VotingRegressor.fit()` re-fits fresh
  clones of every estimator (including the already-slow SVR) rather than
  reusing already-trained models.
- **The dashboard uses your real trained model and real test-set data**
  instead of `np.random.normal(...)` mock forecasts.
- **Fitting and persisting are separate steps.** `feature_engineering.py`'s
  `scale_features()` used to write `models/scaler.pkl` as a side effect of
  fitting, unconditionally, on every call. That's exactly what bit us:
  `tests/test_feature_engineering.py` calls it (correctly, in-memory) on a
  small synthetic fixture, and because the write wasn't gated, running
  `pytest` silently overwrote the *real* scaler with one fit on fake data.
  `scale_features()` is now pure (no disk I/O); `save_scaler()` is the
  explicit persist step, called only from `run()`. Caught by building
  `src/forecast.py` and noticing a historical prediction was off by 12.7%
  when the model's real test MAPE is 1.4% - a good reminder to sanity-check
  a pipeline's *outputs*, not just that each stage runs without erroring.

## 9. Presentation & talking points

See `Electricity_Demand_Forecasting_Complete_Guide.md` (sections
"Presentation Strategy" and "Resume & Interview Talking Points") for a
10-minute slide outline and interview-ready explanations - those sections
don't have the code bugs above and are still a good reference once you have
real results to plug in.
