# Electricity Demand Forecasting — Full Project Report

MSc AI · Statistical Machine Learning · EPITA International Programs

This document explains the entire project end to end: where the data comes
from, how it's used, which models were trained and why, how the dashboard
works page by page, and the questions a professor is likely to ask — with
answers you can actually defend.

---

## 1. The problem

**Business problem.** As France adds more renewable generation (solar,
wind), grid operators need accurate short-term demand forecasts to keep
supply and demand balanced — renewables are weather-dependent and harder to
schedule than gas/nuclear baseload, so the demand side needs to be
predictable to compensate.

**Technical problem.** Predict hourly electricity demand for the French
grid **24–48 hours ahead**, using historical demand, weather, and calendar
data.

**Success metric.** MAPE (Mean Absolute Percentage Error) **< 5%** on a
held-out test set. Achieved: **1.39%** (best model).

---

## 2. Data collection — where every number comes from

Three sources, combined on an hourly grid:

| Source | What | Real or fallback? |
|---|---|---|
| [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) | Hourly electricity demand (load) for the French bidding zone (`FR`) | **Real**, via a free API key. Falls back to a clearly-labeled synthetic series (diurnal + weekly + seasonal pattern + temperature dependence + noise) if no key is configured, so the pipeline is still runnable without one. |
| [Open-Meteo](https://open-meteo.com/) historical archive | Hourly temperature, humidity, precipitation, cloud cover, wind speed for Paris (48.8566°N, 2.3522°E) | **Real**, free, no API key required at all. |
| [`holidays`](https://pypi.org/project/holidays/) Python package | French public holidays | **Real**, computed offline from calendar rules (no network call). |

**Why Open-Meteo instead of NOAA** (the source in the original project
brief): NOAA's Climate Data Online API needs a separate registration token
and only gives *daily* data; Open-Meteo needs no key and is already hourly
— a strict upgrade for the same job.

**Collection window:** 2023-01-01 to 2025-01-01 (24 months), configured in
`config/config.yaml`. This run used the **real ENTSO-E key** — not the
synthetic fallback (verified: no `SYNTHETIC_DATA_WARNING.txt` is present,
and demand values match real French grid magnitudes, ~40–70 GW).

**Script:** `src/data_collection.py`. Run standalone:
```bash
python src/data_collection.py
```
Output: `data/raw/demand_fr.csv`, `data/raw/weather_openmeteo_paris.csv`,
`data/raw/french_holidays.csv`.

### A real data-quality issue found and fixed

The raw combined data had **6 missing hourly timestamps** — not NaN values
in existing rows, but rows that didn't exist at all (a normal artifact of
joining two independently-collected time series; ENTSO-E and Open-Meteo
don't guarantee identical timestamp coverage). `preprocessing.py`'s
forward/backward fill only patches missing *values*, not missing *rows*, so
this gap was invisible until it broke a later optimization (see §11). Fixed
by reindexing to a perfectly regular hourly grid with interpolation at the
point where it mattered (the forecasting engine).

---

## 3. How the data is used — pipeline stages

```
data_collection.py → preprocessing.py → feature_engineering.py → models.py → evaluation.py
```

### 3.1 Preprocessing (`src/preprocessing.py`)
- Inner-joins demand + weather on timestamp, flags holidays (`is_holiday`)
- Forward-fills gaps ≤6 hours (causal — only looks backward), backward-fills
  any remainder
- Reports data-quality stats: missingness, temporal coverage, weekday vs.
  weekend demand skew (weekday mean **50,225 MW** vs. weekend **45,090 MW**,
  10.2% lower — the model needs to learn this, hence `is_weekend`/`dayofweek`
  features)

### 3.2 Feature engineering (`src/feature_engineering.py`)
Creates 25 features from the cleaned series (26 columns total including the
target):

- **Temporal:** `hour`, `dayofweek`, `month`, `quarter`, `is_weekend`
- **Cyclical encodings** (`sin`/`cos` pairs for `hour`, `month`,
  `dayofweek`): so the model sees hour 23 and hour 0 as adjacent, not
  maximally far apart
- **Lag features:** demand 1h, 7h, 24h, 168h (1 week) ago
- **Rolling statistics:** 24h and 168h rolling mean/std of demand, computed
  as `shift(1).rolling(window)` — critically, shifted by 1 first so a row's
  own rolling window never includes its own value (a leakage bug in the
  original project guide, which computed unshifted rolling stats)
- **Weather:** temperature, humidity, precipitation, cloud cover, wind speed
- **Calendar:** `is_holiday`

**Temporal train/val/test split (60/20/20), never shuffled** — this is time
series, not i.i.d. data:

| Split | Rows | Date range |
|---|---|---|
| Train | 10,422 | 2023-01-08 → 2024-03-17 |
| Validation | 3,474 | 2024-03-17 → 2024-08-09 |
| Test | 3,474 | 2024-08-09 → 2024-12-31 |

(The first 168 hours of the raw series are dropped — not enough history yet
for the 168h lag/rolling features.)

**No data leakage** (this is the single most important design decision in
the whole pipeline, and the most likely professor question):
- Outlier bounds (IQR-based, `[20,093, 79,892]` MW, 44 rows capped) are
  fit on the **train split only**, then applied — never refit — to
  val/test.
- The `StandardScaler` is fit on **train only**, then applied to val/test.
- The **target column (`demand`) is never scaled** — MAE/RMSE/MAPE stay in
  real MW units, not standardized units (the original guide scaled the
  target too, which would make MAPE numerically meaningless).
- Lag/rolling features only ever look **backward** in time, so no row's
  features can see its own or a future row's target value.

---

## 4. Models — what was used and why

Five model families, matching the course syllabus (Linear Regression → SVM
→ Decision Tree → Random Forest → Gradient Boosting, i.e. increasing
capacity to capture non-linearity):

| Model | Test MAPE | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|
| **GBDT (XGBoost)** ← **winner** | **1.39%** | 667 MW | 883 MW | 0.991 |
| Random Forest | 1.51% | 740 MW | 995 MW | 0.989 |
| Decision Tree | 1.97% | 955 MW | 1,305 MW | 0.981 |
| Linear Regression | 2.32% | 1,122 MW | 1,414 MW | 0.978 |
| SVM (RBF kernel) | 4.89% | 2,390 MW | 3,252 MW | 0.884 |

**Why each model, and why GBDT wins:**
- **Linear Regression** — the baseline. Simple, interpretable, fast, but
  can't capture the non-linear interaction between e.g. temperature and
  hour-of-day (heating load isn't linear in temperature).
- **SVM (RBF)** — can model non-linearity via the kernel trick, but scales
  poorly (~quadratic) with sample count. Trained on a **3,000-row subsample**
  of the 10,422-row train set for tractability — the tradeoff shows: it's
  the weakest model here (4.89% MAPE), partly *because* it sees less data,
  not because SVMs are inherently worse at this task.
- **Decision Tree** — learns hierarchical splits (e.g. "is it a weekday AND
  is the hour between 8–20 AND is temperature < 10°C"), interpretable, but
  a single tree overfits easily.
- **Random Forest** — bagging ensemble of trees, much more robust than a
  single tree, close second place.
- **Gradient Boosting (XGBoost)** — sequential ensemble where each tree
  corrects the previous ensemble's residual errors. Best performer because
  electricity demand has exactly the kind of structured, non-linear,
  interacting pattern (time-of-day × day-of-week × season × weather) that
  boosted trees are built for, and lag features give it an enormous amount
  of signal to exploit (see feature importance below).

**Top predictive features** (GBDT):

| Rank | Feature | Importance |
|---|---|---|
| 1 | `demand_lag_1h` | 57.7% |
| 2 | `demand_lag_24h` | 19.5% |
| 3 | `demand_rolling_mean_24h` | 9.1% |
| 4 | `dow_cos` | 2.9% |
| 5 | `demand_lag_168h` | 2.6% |
| 6 | `is_weekend` | 2.2% |
| 7–10 | `dow_sin`, `is_holiday`, `hour_cos`, `hour_sin` | 0.7–1.1% each |

Interpretation: **"what was demand doing very recently" dominates**
(lag_1h alone is 58% of the signal) — intuitive, since electricity demand
is highly autocorrelated hour to hour. Yesterday-same-hour (`lag_24h`) and
weekly patterns (`lag_168h`, `dow_*`) carry most of the rest. Weather
matters (it's in the model) but is a smaller direct contributor than the
autoregressive signal — which is exactly why the "Ask the Model" feature's
recursive forecast (§10) needs real or estimated recent demand to work at
all.

**Cross-validation:** `TimeSeriesSplit` (5 folds), not plain `KFold` — for
autocorrelated time series with lag features, a shuffled or blocked
`KFold` fold can leak information (a "validation" row sitting right next to
a training row shares almost all its lag features with it).
`TimeSeriesSplit` always validates on data strictly *after* what a fold
trained on.

**Hyperparameters** (`src/models.py`): Random Forest (200 trees, max depth
15), XGBoost (200 trees, max depth 7, learning rate 0.05, 80% row/column
subsampling), Decision Tree (max depth 15), SVR (RBF, C=100). A
`RandomizedSearchCV` tuner for Random Forest is implemented but not run by
default (keeps the standard pipeline fast; available as a documented
enhancement).

---

## 5. Evaluation methodology (course requirements)

- **Statistical analysis** (`notebooks/01_eda.ipynb`): Shapiro-Wilk
  normality test on hourly demand (rejects normality — demand is
  bimodal/multimodal from day/night and weekday/weekend effects), Pearson
  vs. Spearman correlation (weather vs. demand), and a Central Limit
  Theorem demonstration (daily-averaged demand is visibly closer to
  Gaussian than raw hourly demand).
- **Bias-variance tradeoff**: train vs. validation MAPE plotted per model
  (`outputs/04_bias_variance_tradeoff.png`) — the gap between train and
  validation error is the standard diagnostic for over/underfitting.
- **Ensemble**: manual average of Random Forest + GBDT predictions (not
  `sklearn.ensemble.VotingRegressor`, which force-refits every estimator
  including the already-slow SVR from scratch). Test MAPE: **1.39%** —
  essentially identical to GBDT alone, since GBDT dominates the average.
- **Prediction intervals**: 95% Gaussian interval sized from validation
  residual spread. Empirical coverage on the test set: **91.7%** (a bit
  under the nominal 95%, expected since residual variance isn't perfectly
  constant across the year — a legitimate limitation to mention).
- **Data drift detection**: Kolmogorov-Smirnov test comparing the earlier
  half vs. later half of the test set. Result: **statistic 0.665, p ≈ 0 —
  drift detected.** This is *expected and correct*: the test window spans
  August → December, a genuine seasonal shift in both demand and weather.
  It's a good demonstration that the drift monitor actually works, not a
  bug.

---

## 6. MLOps

- **MLflow** (`src/mlops_utils.py`): every trained model, its
  hyperparameters, and its validation metrics are logged to a local SQLite
  tracking store (`mlruns.db` — MLflow 3+ deprecated the plain-folder
  store). `mlflow ui --backend-store-uri sqlite:///mlruns.db` to browse.
- **DVC** (`dvc.yaml`): defines the five pipeline stages
  (data_collection → preprocessing → feature_engineering → training →
  evaluation) with explicit dependencies/outputs, so `dvc repro` only
  re-runs stages whose inputs actually changed.
- **CI** (`.github/workflows/model-training.yml`): runs the test suite (and
  can run the full pipeline) on every push.
- **Tests** (`tests/`, 25 tests, `pytest`): cover feature engineering
  (no-leakage guarantees, chronological splits), model metrics, the
  synthetic-data fallback, and the forecasting engine (see §11 for the bugs
  these caught).

---

## 7. The website — architecture

Built with **Streamlit**, using its explicit `st.navigation()` /
`st.Page()` API (`streamlit_app/app.py` is the router; each page is a
separate script under `streamlit_app/pages/`, plus `streamlit_app/views/
home.py`). Shared helpers (`streamlit_app/common.py`) provide: cached
data/model loaders, a small design system (colors, chart theme, KPI-card
and header CSS), and formatting utilities.

**Everything the dashboard shows is computed from the real trained model
and real held-out data on disk** — nothing is mocked. It needs no secrets
to run: the ENTSO-E key is only used by the one-time data collection step,
never by the deployed app.

---

## 8. The website — page by page

### 🏠 Home
The landing page. Shows the winning model's test MAPE as a large headline
number with a "target met" badge, three supporting KPI cards (MAE, RMSE,
R²), the dataset's date range, and a row of navigation cards linking to
every other page. If the underlying data collection fell back to synthetic
demand, a banner says so here.

### 💬 Ask the Model — the interactive centerpiece
Pick **any date and hour**, click **Ask the model**, and it computes a
fresh prediction live — not a precomputed chart. Two regimes:

1. **Historical dates** (within the collected data): predicted from real
   inputs, shown against what actually happened, with the error %.
2. **Dates beyond the last known data point**: triggers a genuine
   **recursive (autoregressive) forecast** — see §10 for how this actually
   works under the hood.

Quick-pick buttons ("Latest data", "+1 week", "+1 month", "+6 months",
"+1 year") set the date without triggering a prediction; only the explicit
button click computes one. A 48-hour context chart shows actual vs.
predicted around the query point, and an expander shows exactly what
inputs (weather source, holiday flag) fed the prediction.

### 📈 Predictions
A fixed backtest: actual vs. predicted demand over an adjustable recent
window (24–168h) of the **held-out test set** — data the model never
trained on. This is the classic "does it generalize" check, distinct from
Ask the Model's arbitrary-point query.

### 📊 Model Performance
The full comparison table across all five models (the table in §4), with
MAPE shown as an in-table progress bar and a bar chart highlighting the
winner. Below that: a predicted-vs-actual scatter plot for the best model,
and the bias-variance tradeoff chart.

### 🔍 Feature Analysis
An interactive bar chart of the top 15 most important features for the
winning model (the ranking in §4), computed live from the model's
`feature_importances_`.

### 🚨 Data Drift
Runs the Kolmogorov-Smirnov drift test (§5) live, with an adjustable
significance threshold, and shows which features have shifted between the
earlier and later half of the test set.

---

## 9. Deployment

Two platforms are supported from the same repo (`README.md` §7 has full
steps):

- **Streamlit Community Cloud** (primary) — deploys straight from GitHub,
  free, no Docker/config needed.
- **Hugging Face Spaces** — the README's YAML frontmatter (`sdk: streamlit`,
  `app_file: streamlit_app/app.py`) configures this automatically.

**Six specific artifacts are deliberately committed to git**, carved out of
an otherwise-broad `.gitignore`: `models/best_model.pkl`,
`models/scaler.pkl`, `data/processed/{combined_data,test_data}.csv`, and
two `outputs/*.png` plots. That's everything the dashboard and "Ask the
Model" read at runtime (~4.6 MB total) — the deployed app serves real
results immediately instead of needing to run the pipeline in the cloud.

---

## 10. Deep dive: how "Ask the Model" actually works (`src/forecast.py`)

This is the most technically involved part of the project and worth
understanding in detail, since it's the most likely thing a professor
digs into.

**Historical queries** are straightforward: build the same 25 features the
model was trained on (real weather, real lags, real rolling stats from
history), scale them with the saved `StandardScaler`, predict.

**Future queries** (beyond the last known data point) use **recursive
rollout**: predict hour *t+1*, append that prediction to the working demand
series, then use it as input (for lag/rolling features) to predict hour
*t+2*, and so on. This mirrors how a real production forecaster works when
the true future value isn't in yet — it's inherently **sequential** (each
step depends on the previous prediction), which has real consequences:

- **Error compounds with horizon.** The dashboard explicitly warns beyond
  48h that predictions should be read as illustrative, not
  production-grade — the project is scoped and validated for 24–48h ahead,
  matching the actual brief.
- **Weather for the gap** comes from a three-tier fallback: real historical
  weather if the date is in the past, a **live Open-Meteo forecast** if the
  date is within its ~16-day forecast horizon from today, or a **seasonal
  (month, hour) climatology** average otherwise — always labeled in the UI
  so a guess is never presented as fact.
- **Performance**: a naive implementation costs ~7ms per recursive step
  (dominated by `sklearn`'s per-call `DataFrame` validation overhead). The
  hot path was rewritten to operate on raw NumPy arrays with precomputed
  scaler statistics, cutting this to **~0.57ms/step** (~12×) — a 3-year-out
  query now takes ~15 seconds locally instead of several minutes, with a
  real progress bar (not just a spinner) for anything beyond ~200 hours of
  rollout.

---

## 11. Real bugs found during development (good "how we validated this"
material)

Three genuine bugs were found by actually running the system end to end and
cross-checking results — not just by code review. All three are exactly the
kind of thing a professor might probe for ("how do you know your pipeline
doesn't have a subtle leak?"):

1. **A test polluted production state.** `feature_engineering.py`'s
   `scale_features()` used to write `models/scaler.pkl` as a side effect of
   fitting, unconditionally. A unit test called it (correctly, in-memory)
   on a small synthetic fixture — and because the write wasn't gated,
   running `pytest` silently overwrote the *real* scaler with one fit on
   fake data. Found because a historical "Ask the Model" prediction was off
   by **12.7%** when the real test MAPE is 1.4% — a good example of why you
   sanity-check a pipeline's outputs, not just that each stage runs without
   erroring. Fixed by separating "fit" (pure, no disk I/O) from "persist"
   (an explicit step called only by the real pipeline entrypoint).
2. **Six missing hourly timestamps** (§2) silently broke a performance
   optimization that assumed the demand history was a perfectly regular
   hourly grid — positional array arithmetic ("168 hours ago" = "168 array
   slots ago") drifted out of alignment after each gap. Fixed by
   reindexing + interpolating to a gap-free grid, with a regression test
   that deliberately reintroduces a gap to guard against recurrence.
3. **A folder committed as `Outputs/` instead of `outputs/`.** Invisible on
   macOS (case-insensitive filesystem — "it worked on my machine"), but
   Streamlit Cloud runs on case-sensitive Linux, where `outputs/
   eval_metrics.json` (what the code asks for) genuinely didn't exist —
   only `Outputs/eval_metrics.json` did. Caused a `TypeError` crash on the
   deployed app. Fixed with a content-preserving git rename; also hardened
   `pipeline_ready()` to check for this specific file so a similar future
   gap fails with a clear message instead of a crash.

---

## 12. Known limitations & future work

- **Prediction intervals are Gaussian and static-width** (sized from
  validation residual std), not quantile-regression-based — coverage is
  91.7% against a nominal 95%, so intervals are slightly too narrow.
- **Recursive forecasting accuracy degrades with horizon** — by design,
  and clearly communicated in the UI, but a genuine limitation of any
  autoregressive approach without a dedicated multi-step model.
- **Single geographic point for weather** (Paris) as a proxy for the whole
  French grid — reasonable for a course project, a real system would use a
  demand-weighted average across multiple weather stations.
- **No live retraining loop** — `detect_drift()` exists and works, but
  nothing automatically triggers retraining when drift is detected; that's
  the natural next MLOps step (a scheduled job that reruns the pipeline and
  promotes a new model only if it beats the current one on held-out data).
- **SVM is trained on a subsample** for tractability — a fairer comparison
  would need a faster SVM formulation (e.g. linear SVR + explicit kernel
  approximation) to use the full training set.

---

## 13. Expected questions from the professor (with answers)

**Q: How do you know there's no data leakage?**
> Outlier bounds and the feature scaler are fit on the train split only,
> then applied — never refit — to validation/test. Lag and rolling features
> only look backward (`shift(1)` before `rolling()`). The split itself is
> temporal, not random, so no future information ever appears in training.
> Two things in the original project guide *did* leak (scaling before the
> split, and unshifted rolling stats) — I fixed both and can show the diff.

**Q: Why MAPE as the primary metric instead of RMSE?**
> MAPE is scale-independent and business-interpretable ("predictions are
> off by 1.4% on average") in a way raw MW error isn't to a non-technical
> stakeholder. I still report MAE/RMSE/R² alongside it since MAPE alone can
> be misleading near-zero values (not a concern here — demand never
> approaches zero).

**Q: Why does Gradient Boosting outperform the others?**
> Electricity demand has strongly non-linear, interacting structure (hour
> × day-of-week × season × weather), which boosted trees capture natively
> through sequential residual correction. The feature importances confirm
> it's mostly exploiting recent-lag autocorrelation (lag_1h is 58% of the
> signal) — demand right now is the single best predictor of demand very
> soon, and GBDT weights that most effectively of the five models tried.

**Q: Why is SVM the weakest model here — doesn't the RBF kernel handle
non-linearity too?**
> It does, but SVR training cost scales roughly quadratically with sample
> count, so I subsampled to 3,000 of the 10,422 training rows for
> tractability. It's seeing ~30% of the data the other models get — the
> comparison isn't perfectly apples-to-apples, and I say so explicitly
> rather than hiding it.

**Q: How would this behave in a real production deployment?**
> It already demonstrates the core pieces: MLflow experiment tracking, a
> DVC-defined reproducible pipeline, a CI workflow, and a live drift monitor.
> What's missing for true production is a scheduled retraining trigger tied
> to the drift detector, and quantile-regression-based prediction intervals
> instead of a static Gaussian width.

**Q: What happens if you ask it to predict a date 3 years from now — is
that a real forecast?**
> It's a genuine recursive/autoregressive forecast (each hour's prediction
> feeds the next hour's lag features), not a canned answer, but it's
> explicitly labeled as such and the UI warns that accuracy is only
> validated for the project's actual scope, 24–48 hours ahead. Weather for
> far-future dates falls back to a seasonal climatology average since no
> real forecast exists that far out — also labeled, not hidden.

**Q: Why Open-Meteo instead of the NOAA source the brief mentioned?**
> NOAA's actual API needs a separate registration token the brief didn't
> account for and only provides daily granularity; Open-Meteo needs no key
> and is already hourly — I substituted it because it's a strict
> improvement for the same job, not because the original source is
> unusable.

**Q: How did you validate the pipeline actually works, beyond "the tests
pass"?**
> By running it end to end on real data and checking outputs made physical
> sense (§11) — that's how the scaler-corruption and casing bugs were
> caught, neither of which any unit test alone would have surfaced. Tests
> were then added for both to prevent regression.

**Q: What was each team member's contribution?**
> See `TEAM.md` — ownership is split by pipeline stage (data/EDA; feature
> engineering & models; evaluation/MLOps/the interactive forecaster;
> dashboard & deployment), with a note on what each person should be able
> to defend in Q&A.

---

## 14. Quick reference: running everything yourself

```bash
# Pipeline (produces everything in data/, models/, outputs/)
python src/data_collection.py
python src/preprocessing.py
python src/feature_engineering.py
python src/models.py
python src/evaluation.py

# Dashboard
streamlit run streamlit_app/app.py

# Tests
pytest

# Experiment tracking UI
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

Full setup instructions: `README.md`. Team ownership breakdown: `TEAM.md`.
