import os

import pandas as pd
import mlflow
import joblib

from ..config import config
from ..pipeline import build_pipeline
from ..utils import get_data


def set_version_tags():
    tags = {
        "stage": config.PIPELINE_STAGE,
        "pipeline_version": config.PIPELINE_VERSION,
        "data_version": config.DATA_VERSION,
        "data_commit_hash": config.DATA_COMMIT_HASH,
        "code_commit_hash": config.CODE_COMMIT_HASH,
        "status": config.PIPELINE_STATUS,
    }
    mlflow.set_tags(tags)


def set_version_params():
    df = pd.read_csv(config.DATA_DIR / "raw" / config.RAW_FILE)
    df_train = get_data("train")
    df_val = get_data("validation")
    df_test = get_data("test")
    params = {
        "raw_size": len(df),
        "cleaned_size": len(df_train) + len(df_val) + len(df_test),
        "train_size": len(df_train),
        "val_size": len(df_val),
        "test_size": len(df_test),
    }
    mlflow.log_params(params)


def set_version_artifacts(pipeline):
    root_dir = config.ARTIFACTS_DIR / f"pipeline_{config.PIPELINE_VERSION}"
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    path = root_dir / f"pipeline_{config.PIPELINE_VERSION}.joblib"
    joblib.dump(pipeline, path)
    mlflow.log_artifact(path)

    path = root_dir / f"pipeline_{config.PIPELINE_VERSION}.md"  # Manually created
    mlflow.log_artifact(path)


def mlflow_register():
    experiment_name = "Preprocessing"
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=f"Pipeline {config.PIPELINE_VERSION}") as run:
        mlflow.set_tag("run_id", run.info.run_id)
        set_version_tags()
        set_version_params()

        train_path = config.DATA_DIR / "prepared" / config.TRAIN_FILE
        df_train = pd.read_csv(train_path, names=[config.FEATURE, config.TARGET])
        x_train = df_train.drop(config.TARGET, axis=1)
        pipeline = build_pipeline().fit(x_train)
        set_version_artifacts(pipeline)
        print(f"[INFO] Experiment '{experiment_name}' successfully completed.")


if __name__ == "__main__":
    mlflow_register()
