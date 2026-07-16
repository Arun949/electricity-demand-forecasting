"""MLOps: log trained models, params, and metrics to a local MLflow tracking store."""
import pickle

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from mlflow.models.signature import infer_signature

from utils import get_logger, load_config, resolve_path

logger = get_logger(__name__)


class MLOpsManager:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        # MLflow >=3 deprecated the plain filesystem store ("./mlruns") in
        # favor of a database backend; SQLite is the zero-setup equivalent.
        db_path = resolve_path(self.config["mlflow"]["tracking_uri"])
        db_path.parent.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])

    def log_model(self, model, model_name: str, X_train, y_train, metrics: dict) -> None:
        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(
                {
                    "model_type": model_name,
                    "n_features": X_train.shape[1],
                    "n_training_samples": X_train.shape[0],
                    **{k: v for k, v in getattr(model, "get_params", dict)().items()
                       if isinstance(v, (int, float, str, bool)) or v is None},
                }
            )
            # NaN metrics (e.g. svm's skipped CV_MAPE_mean) aren't meaningful
            # to log and trip a NaN-handling constraint bug in some MLflow/
            # SQLAlchemy backend combinations -- drop them before logging.
            clean_metrics = {k: float(v) for k, v in metrics.items() if pd.notna(v)}
            mlflow.log_metrics(clean_metrics)

            signature = infer_signature(X_train, y_train)
            try:
                if type(model).__module__.startswith("xgboost"):
                    mlflow.xgboost.log_model(model, name="model", signature=signature)
                else:
                    mlflow.sklearn.log_model(model, name="model", signature=signature)
            except Exception as exc:  # noqa: BLE001 - logging must not break the pipeline
                logger.warning("Could not log model artifact for %s: %s", model_name, exc)

            logger.info("Logged %s to MLflow", model_name)

    def log_all_from_disk(self) -> None:
        """Convenience entrypoint: log every models/*.pkl using outputs/validation_results.csv."""
        models_dir = resolve_path("models")
        outputs_dir = resolve_path("outputs")
        processed_dir = resolve_path(self.config["data"]["processed_dir"])
        target = self.config["features"]["target"]

        results_path = outputs_dir / "validation_results.csv"
        if not results_path.exists():
            raise FileNotFoundError("Run `python src/models.py` first to produce validation_results.csv")
        results_df = pd.read_csv(results_path, index_col=0)

        train_df = pd.read_csv(processed_dir / "train_data.csv", index_col=0, parse_dates=True)
        X_train = train_df.drop(columns=[target])
        y_train = train_df[target]

        for model_name in results_df.index:
            model_path = models_dir / f"{model_name}.pkl"
            if not model_path.exists():
                continue
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            self.log_model(model, model_name, X_train, y_train, results_df.loc[model_name].to_dict())


if __name__ == "__main__":
    MLOpsManager().log_all_from_disk()
