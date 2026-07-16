"""
Phase 3 + 4: Train and cross-validate the five course-required model families
(Linear Regression, SVM, Decision Tree, Random Forest, Gradient Boosting),
then evaluate on the validation split.

Cross-validation uses TimeSeriesSplit rather than plain shuffled/blocked
KFold: this is time-ordered, autocorrelated data (lag/rolling features), so a
fold whose "validation" rows sit immediately before or after training rows
would leak information through those lags. TimeSeriesSplit always validates
on data strictly after what the fold trained on.
"""
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from utils import get_logger, load_config, resolve_path

logger = get_logger(__name__)


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


class ModelTrainer:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.random_state = self.config["models"]["random_state"]
        self.cv_folds = self.config["models"]["cv_folds"]
        self.models_dir = resolve_path("models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.models: dict[str, object] = {}

    # ------------------------------------------------------------------ #
    # Model definitions
    # ------------------------------------------------------------------ #
    def train_baseline(self, X_train, y_train):
        model = LinearRegression()
        model.fit(X_train, y_train)
        self.models["linear_regression"] = model
        return model

    def train_svm(self, X_train, y_train, max_samples: int = 3000):
        """SVR training cost grows ~quadratically with sample count; subsample
        for tractability on hourly-resolution data (thousands of rows)."""
        if len(X_train) > max_samples:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(X_train), size=max_samples, replace=False)
            idx.sort()
            X_sub, y_sub = X_train.iloc[idx], y_train.iloc[idx]
            logger.info("SVM: subsampled %d/%d training rows for tractability", max_samples, len(X_train))
        else:
            X_sub, y_sub = X_train, y_train

        model = SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.1)
        model.fit(X_sub, y_sub)
        self.models["svm"] = model
        return model

    def train_decision_tree(self, X_train, y_train):
        model = DecisionTreeRegressor(
            max_depth=15, min_samples_split=10, min_samples_leaf=5, random_state=self.random_state
        )
        model.fit(X_train, y_train)
        self.models["decision_tree"] = model
        return model

    def train_random_forest(self, X_train, y_train):
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=self.random_state,
        )
        model.fit(X_train, y_train)
        self.models["random_forest"] = model
        return model

    def train_gradient_boosting(self, X_train, y_train):
        import xgboost as xgb

        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=self.random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        self.models["gbdt"] = model
        return model

    def train_all(self, X_train, y_train) -> dict:
        steps = [
            ("linear_regression", self.train_baseline),
            ("svm", self.train_svm),
            ("decision_tree", self.train_decision_tree),
            ("random_forest", self.train_random_forest),
            ("gbdt", self.train_gradient_boosting),
        ]
        for name, fn in steps:
            logger.info("Training %s...", name)
            fn(X_train, y_train)
        return self.models

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    def evaluate(self, model, X, y, model_name: str) -> dict:
        y_pred = model.predict(X)
        metrics = {
            "MAE": mean_absolute_error(y, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y, y_pred))),
            "MAPE": mape(y, y_pred),
            "R2": r2_score(y, y_pred),
        }
        logger.info(
            "%-16s MAE=%.1f  RMSE=%.1f  MAPE=%.2f%%  R2=%.3f",
            model_name, metrics["MAE"], metrics["RMSE"], metrics["MAPE"], metrics["R2"],
        )
        return metrics

    def cross_validate(self, model, X_train, y_train, model_name: str) -> np.ndarray:
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        scores = cross_val_score(
            model, X_train, y_train, cv=tscv, scoring="neg_mean_absolute_percentage_error", n_jobs=-1
        )
        mape_scores = -scores * 100
        logger.info(
            "%-16s CV MAPE: mean=%.2f%% std=%.2f%% folds=%s",
            model_name, mape_scores.mean(), mape_scores.std(), np.round(mape_scores, 2),
        )
        return mape_scores

    def tune_random_forest(self, X_train, y_train, n_iter: int = 15) -> RandomForestRegressor:
        param_dist = {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 15, 20, None],
            "min_samples_split": [5, 10, 20],
            "min_samples_leaf": [2, 5, 10],
        }
        base_model = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        search = RandomizedSearchCV(
            base_model,
            param_dist,
            n_iter=n_iter,
            cv=TimeSeriesSplit(n_splits=3),
            scoring="neg_mean_absolute_percentage_error",
            n_jobs=-1,
            random_state=self.random_state,
        )
        search.fit(X_train, y_train)
        logger.info("Best RF params: %s (CV MAPE=%.2f%%)", search.best_params_, -search.best_score_ * 100)
        self.models["random_forest_tuned"] = search.best_estimator_
        return search.best_estimator_

    def save_model(self, name: str) -> None:
        with open(self.models_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(self.models[name], f)


def load_split(processed_dir, split_name: str, target: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(processed_dir / f"{split_name}_data.csv", index_col=0, parse_dates=True)
    return df.drop(columns=[target]), df[target]


if __name__ == "__main__":
    config = load_config()
    processed_dir = resolve_path(config["data"]["processed_dir"])
    target = config["features"]["target"]

    X_train, y_train = load_split(processed_dir, "train", target)
    X_val, y_val = load_split(processed_dir, "val", target)

    trainer = ModelTrainer()
    trainer.train_all(X_train, y_train)

    results = {}
    cv_means = {}
    for name, model in trainer.models.items():
        results[name] = trainer.evaluate(model, X_val, y_val, name)
        trainer.save_model(name)

        if name == "svm":
            # SVR's O(n^2-n^3) fit cost makes 5x refits on the full train set
            # impractical here; train_svm() already documents the tradeoff.
            logger.info("Skipping CV for svm (see train_svm docstring); reported metrics are val-set only")
            continue
        cv_scores = trainer.cross_validate(model, X_train, y_train, name)
        cv_means[name] = cv_scores.mean()

    results_df = pd.DataFrame(results).T.sort_values("MAPE")
    results_df["CV_MAPE_mean"] = pd.Series(cv_means)
    logger.info("\nValidation set comparison:\n%s", results_df.to_string())

    outputs_dir = resolve_path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    results_df.to_csv(outputs_dir / "validation_results.csv")
