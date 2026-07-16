# Team Split (4 members)

The codebase is fully built and tested. This splits ownership across 4 people
so each can explain, defend, and demo their part — and so your GitHub commit
history reflects real individual contribution, not one giant initial commit.

## Member 1 — Data & EDA

**Owns:** `src/data_collection.py`, `src/preprocessing.py`, `src/utils.py`,
`config/config.yaml`, `notebooks/01_eda.ipynb`, `.env.example`, README
sections 1-3.

**Know cold for Q&A:**
- Why ENTSO-E (real, needs a free key) + Open-Meteo (real, keyless, swapped
  in for the original guide's broken NOAA snippet) + `holidays` (offline)
- The synthetic-fallback design: what happens with no `ENTSOE_API_KEY`, and
  how `SYNTHETIC_DATA_WARNING.txt` flags it end to end (including in the
  dashboard header badge)
- EDA statistical requirements: Shapiro-Wilk normality test, Pearson vs.
  Spearman correlation, the Central Limit Theorem demo (hourly vs. daily
  demand), and the weekday/weekend bias check

## Member 2 — Feature Engineering & Models

**Owns:** `src/feature_engineering.py`, `src/models.py`,
`tests/test_feature_engineering.py`, `tests/test_models.py`.

**Know cold for Q&A:**
- Lag features (1h/7h/24h/168h) and rolling stats (24h/168h) — and why
  they're `shift(1)`-based (a row's own rolling window must never include
  its own value)
- Why the scaler and outlier bounds are fit on the train split only, then
  applied (never refit) to val/test — this is the #1 data-leakage question
  a grader will ask
- The 5 models (Linear Regression, SVM, Decision Tree, Random Forest,
  GBDT/XGBoost), why SVM is subsampled to 3,000 rows, and why
  `TimeSeriesSplit` replaces plain `KFold` for cross-validation

## Member 3 — Evaluation, MLOps & the Interactive Forecaster

**Owns:** `src/evaluation.py`, `src/mlops_utils.py`, `src/forecast.py`,
`tests/test_evaluation.py`, `tests/test_forecast.py`.

**Know cold for Q&A:**
- Test-set results table, bias-variance plot, feature importance, the
  manual ensemble (and why not `VotingRegressor` — it force-refits every
  estimator including the slow SVR), prediction intervals, KS drift test
- MLflow: SQLite-backed tracking (`mlruns.db`), why (MLflow 3+ dropped the
  plain-folder store)
- **`src/forecast.py` is the most defensible/impressive piece** — the "Ask
  the Model" engine. Be ready to explain: historical vs. recursive
  (autoregressive) prediction, the three-tier weather fallback (real
  history → live Open-Meteo forecast → seasonal climatology), and the two
  real bugs found while building it (a test that silently corrupted the
  real scaler by writing to disk on fake data; 6 missing hourly timestamps
  in the real data breaking a positional-array optimization) — great
  "how we found and fixed a real bug" story for the presentation.

## Member 4 — Dashboard, Deployment & Presentation

**Owns:** `streamlit_app/` (all pages, `common.py`, design system),
`.streamlit/config.toml`, `packages.txt`, `.python-version`, `dvc.yaml`,
`.github/workflows/model-training.yml`, README sections 4-7, the slide deck.

**Know cold for Q&A:**
- The 6-page dashboard: Home, **Ask the Model** (the flagship interactive
  feature), Predictions, Model Performance, Feature Analysis, Data Drift
- The design system in `common.py` (`PALETTE`, `plotly_layout()`,
  `render_header()`) — why it exists (consistent charts/branding instead of
  default Streamlit look)
- Deployment: Streamlit Community Cloud, why 6 specific artifacts are
  committed despite the broad `.gitignore` (so the deployed app works with
  zero secrets and no pipeline run), and the `libgomp1`/XGBoost gotcha
- Owns final end-to-end verification before the demo: run the pipeline
  once, launch the dashboard, click through all 6 pages live

## Suggested GitHub workflow

1. One person (any of you) does the initial commit of the full repo as-is —
   note in the commit message that it's the baseline everyone builds on.
2. Each member then makes at least one real follow-up commit *in their
   area* — a docstring improvement, an extra test, a comment, a config
   tweak — so the commit history shows genuine per-person activity, not
   just one bulk push. Small and honest beats padded and fake.
3. Suggested repo root: `electricity-demand-forecasting/` itself (not the
   parent folder with the planning docs) — see README section 7 for why
   that matters for deployment.
