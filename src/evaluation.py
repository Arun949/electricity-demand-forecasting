"""
Phase 4 (final) + Phase 5 enhancements: test-set evaluation, model selection,
bias-variance analysis, feature importance, a manual ensemble, prediction
intervals, and data-drift detection.

The test set is touched exactly once, here, for final reporting only -- it is
never used for tuning (tuning uses TimeSeriesSplit CV on the train split, see
models.py).
"""
import json
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from models import ModelTrainer, load_split, mape
from utils import get_logger, load_config, resolve_path

logger = get_logger(__name__)


def load_models(models_dir) -> dict:
    models = {}
    for path in sorted(models_dir.glob("*.pkl")):
        # "scaler" and "best_model" aren't bare estimators (they're dicts /
        # metadata bundles written by feature_engineering.py and this module's
        # own run()), so a rerun mustn't try to .predict() with them.
        if path.stem in ("scaler", "best_model"):
            continue
        with open(path, "rb") as f:
            models[path.stem] = pickle.load(f)
    return models


def select_best_model(models: dict, X_test, y_test, trainer: ModelTrainer) -> tuple[str, object, pd.DataFrame]:
    test_results = {name: trainer.evaluate(model, X_test, y_test, f"[TEST] {name}") for name, model in models.items()}
    results_df = pd.DataFrame(test_results).T.sort_values("MAPE")
    best_name = results_df["MAPE"].idxmin()
    logger.info("Best model on TEST set: %s (MAPE=%.2f%%)", best_name, results_df.loc[best_name, "MAPE"])
    return best_name, models[best_name], results_df


def analyze_bias_variance(models: dict, X_train, y_train, X_val, y_val, outputs_dir) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        train_mape = mape(y_train, model.predict(X_train))
        val_mape = mape(y_val, model.predict(X_val))
        rows.append({"model": name, "train_mape": train_mape, "val_mape": val_mape, "gap": val_mape - train_mape})

    df = pd.DataFrame(rows).sort_values("gap")

    plt.figure(figsize=(10, 5))
    x = np.arange(len(df))
    plt.plot(x, df["train_mape"], "o-", label="Train MAPE")
    plt.plot(x, df["val_mape"], "s-", label="Validation MAPE")
    plt.xticks(x, df["model"], rotation=30, ha="right")
    plt.ylabel("MAPE (%)")
    plt.title("Bias-Variance: Train vs Validation Error by Model")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outputs_dir / "04_bias_variance_tradeoff.png", dpi=200)
    plt.close()

    logger.info("Bias-variance summary:\n%s", df.to_string(index=False))
    return df


def analyze_feature_importance(model, feature_names, outputs_dir, top_n: int = 20) -> None:
    if not hasattr(model, "feature_importances_"):
        logger.info("Model %s has no feature_importances_; skipping", type(model).__name__)
        return

    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(order)), importances[order])
    plt.xticks(range(len(order)), [feature_names[i] for i in order], rotation=60, ha="right")
    plt.ylabel("Importance")
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.savefig(outputs_dir / "05_feature_importance.png", dpi=200)
    plt.close()

    logger.info("Top 10 features:")
    for i, idx in enumerate(order[:10], 1):
        logger.info("  %2d. %-30s %.4f", i, feature_names[idx], importances[idx])


def ensemble_predict(models: dict, X, names: list[str], weights: list[float] | None = None) -> np.ndarray:
    """Average predictions of already-fitted models (no refitting -- avoids
    re-running the slow SVR fit that sklearn's VotingRegressor would trigger)."""
    preds = np.column_stack([models[name].predict(X) for name in names])
    return np.average(preds, axis=1, weights=weights)


def get_prediction_intervals(y_val, y_val_pred, y_test_pred, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian interval around point predictions, width from validation residual std."""
    residual_std = np.std(y_val - y_val_pred)
    return y_test_pred - z * residual_std, y_test_pred + z * residual_std


def detect_drift(y_reference: pd.Series, y_recent: pd.Series, alpha: float = 0.05) -> bool:
    stat, p_value = ks_2samp(y_reference, y_recent)
    drifted = bool(p_value < alpha)
    logger.info(
        "KS drift test: statistic=%.4f p=%.4e -> %s",
        stat, p_value, "DRIFT DETECTED" if drifted else "no significant drift",
    )
    return drifted


def run():
    config = load_config()
    processed_dir = resolve_path(config["data"]["processed_dir"])
    models_dir = resolve_path("models")
    outputs_dir = resolve_path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    target = config["features"]["target"]

    X_train, y_train = load_split(processed_dir, "train", target)
    X_val, y_val = load_split(processed_dir, "val", target)
    X_test, y_test = load_split(processed_dir, "test", target)

    trainer = ModelTrainer(config_path="config/config.yaml")
    models = load_models(models_dir)
    if not models:
        raise FileNotFoundError("No trained models found in models/. Run `python src/models.py` first.")

    # --- Final test-set comparison & model selection ---------------------
    best_name, best_model, test_results_df = select_best_model(models, X_test, y_test, trainer)
    test_results_df.to_csv(outputs_dir / "test_results.csv")

    # --- Bias-variance ------------------------------------------------------
    analyze_bias_variance(models, X_train, y_train, X_val, y_val, outputs_dir)

    # --- Feature importance --------------------------------------------------
    analyze_feature_importance(best_model, list(X_train.columns), outputs_dir)

    # --- Manual ensemble (tree models only -- avoids slow SVR refit) --------
    ensemble_names = [n for n in ("random_forest", "gbdt") if n in models]
    if len(ensemble_names) >= 2:
        ens_pred_test = ensemble_predict(models, X_test, ensemble_names)
        ens_mape = mape(y_test, ens_pred_test)
        logger.info("Ensemble(%s) TEST MAPE: %.2f%%", "+".join(ensemble_names), ens_mape)
    else:
        ens_mape = None

    # --- Prediction intervals -------------------------------------------------
    y_val_pred_best = best_model.predict(X_val)
    y_test_pred_best = best_model.predict(X_test)
    lower, upper = get_prediction_intervals(y_val, y_val_pred_best, y_test_pred_best)
    coverage = float(np.mean((y_test >= lower) & (y_test <= upper)) * 100)
    logger.info("95%% prediction interval empirical coverage on test set: %.1f%%", coverage)

    # --- Data drift: first half of test vs second half -----------------------
    midpoint = len(y_test) // 2
    detect_drift(y_test.iloc[:midpoint], y_test.iloc[midpoint:])

    # --- Persist best model + metrics for the dashboard / DVC pipeline -------
    with open(models_dir / "best_model.pkl", "wb") as f:
        pickle.dump({"name": best_name, "model": best_model, "feature_cols": list(X_train.columns)}, f)

    eval_metrics = {
        "best_model": best_name,
        "test_metrics": test_results_df.loc[best_name].to_dict(),
        "ensemble_mape": ens_mape,
        "prediction_interval_coverage_pct": coverage,
        "all_models_test": test_results_df.to_dict(orient="index"),
    }
    with open(outputs_dir / "eval_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=2, default=float)

    logger.info("Evaluation complete. Best model = %s. Artifacts in %s", best_name, outputs_dir)
    return eval_metrics


if __name__ == "__main__":
    run()
