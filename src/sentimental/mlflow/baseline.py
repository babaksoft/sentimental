import os
from typing import Any
from datetime import datetime

import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from ..config import config
from ..pipeline import build_pipeline
from ..utils import plot_confusion_matrix, get_data


def evaluate(df: pd.DataFrame, run_name: str, clean: bool=False):
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("run_id", run.info.run_id)
        params: dict[str, Any] = {}

        # Setup vectorizer
        tfidf_params: dict[str, Any] = {
            "strip_accents": None,
            "lowercase": False,
            "preprocessor": None
        }
        tfidf = TfidfVectorizer(**tfidf_params)
        params.update(
            { f"tfidf_{key}": val for key, val in tfidf_params.items() }
        )

        # Setup cross-validation
        cv_params: dict[str, Any] = {
            "n_splits": 10,
            "shuffle": True,
            "random_state": config.RANDOM_STATE
        }
        cv = StratifiedKFold(**cv_params)
        params.update(
            { f"cv_{key}": val for key, val in cv_params.items() }
        )

        # Setup model
        model_params: dict[str, Any] = {
            "class_weight": "balanced",
            "solver": "saga",
            "max_iter": 1500,
            "random_state": config.RANDOM_STATE
        }
        model = LogisticRegression(**model_params)
        params["model_type"] = type(model).__name__
        params.update(
            { f"model_{key}": val for key, val in model_params.items() }
        )

        y = df[config.TARGET]
        if clean:
            x = df.drop(config.TARGET, axis=1)
            pipeline = Pipeline([
                ("transformer", build_pipeline()),
                ("classifier", model)
            ])
        else:
            x = df[config.FEATURE]
            pipeline = Pipeline([
                ("vectorizer", tfidf),
                ("classifier", model)
            ])

        scoring = ["accuracy", "f1_macro"]
        start = datetime.now()
        cv_result = cross_validate(
            pipeline, x, y, scoring=scoring,
            cv=cv, n_jobs=-1
        )
        end = datetime.now()
        params.update({ "duration": str(end - start) })

        metrics = {
            "cv_accuracy_mean": round(cv_result["test_accuracy"].mean(), 4),
            "cv_accuracy_std": round(cv_result["test_accuracy"].std(), 4),
            "cv_f1_macro_mean": round(cv_result["test_f1_macro"].mean(), 4),
            "cv_f1_macro_std": round(cv_result["test_f1_macro"].std(), 4),
        }

        mlflow.log_metrics(metrics)
        mlflow.log_params(params)
        if clean:
            pipeline.fit(x, y)
            df_val = get_data("validation")
            x_val = df_val.drop(config.TARGET, axis=1)
            y_val = df_val[config.TARGET]
            plot_confusion_matrix(pipeline, x_val, y_val)
            path = config.METRICS_DIR / f"cm_baseline.png"
            mlflow.log_artifact(path)
        mlflow.end_run()


def evaluate_baseline():
    experiment_name = "Baseline Model"
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name=experiment_name)
    train_path = config.DATA_DIR / "prepared" / config.TRAIN_FILE
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            "Train data not found. Please run ingest.py first."
        )

    df = pd.read_csv(train_path, names=[config.FEATURE, config.TARGET])
    evaluate(df, run_name="Raw", clean=False)
    evaluate(df, run_name="Preprocessed", clean=True)
    print(f"[INFO] Experiment '{experiment_name}' successfully completed.")


if __name__ == "__main__":
    evaluate_baseline()
