import os
from typing import Any
from datetime import datetime

import pandas as pd
import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer

from ..config import config
from ..pipeline import clean_text


def vectorize(x, run_name, clean=False):
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("run_id", run.info.run_id)
        params: dict[str, Any] = {
            "strip_accents": None,
            "lowercase": False,
            "preprocessor": None
        }

        vectorizer = TfidfVectorizer(**params)
        start = datetime.now()
        if clean:
            x = x.map(lambda text: clean_text(text))
        x_trans = vectorizer.fit_transform(x)
        end = datetime.now()
        params = { f"tfidf_{key}": val for key, val in params.items() }
        params.update({
            "pipe_feature_count": x_trans.shape[1],
            "pipe_duration": str(end - start)
        })

        mlflow.log_params(params)
        mlflow.end_run()


def run_pipeline():
    experiment_name = "Data Pipeline"
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name=experiment_name)
    train_path = config.DATA_DIR / "prepared" / config.TRAIN_FILE
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            "Train data not found. Please run ingest.py first."
        )

    df = pd.read_csv(train_path, names=[config.FEATURE, config.TARGET])
    x_raw = df[config.FEATURE]
    vectorize(x_raw, run_name="Raw", clean=False)

    vectorize(x_raw, run_name="Preprocessed", clean=True)
    print(f"Experiment '{experiment_name}' successfully completed.")


if __name__ == "__main__":
    run_pipeline()
