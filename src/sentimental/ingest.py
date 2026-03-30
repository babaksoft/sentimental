import os

import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split

from .config import config


def drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    with mlflow.start_run(run_name="Drop missing") as run:
        mlflow.set_tag("run_id", run.info.run_id)
        mlflow.set_tag("drop_reason", "missing")

        df_no_missing = df.dropna(axis=0)
        drop_count = len(df) - len(df_no_missing)
        drop_percent = round(100 * drop_count / len(df), 2)

        metrics = {
            "dataset_size": len(df),
            "missing_count": drop_count,
            "missing_percent": drop_percent,
            "cleaned_size": len(df_no_missing),
        }
        mlflow.log_metrics(metrics)
        mlflow.end_run()
        return df_no_missing


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    with mlflow.start_run(run_name="Drop duplicates") as run:
        mlflow.set_tag("run_id", run.info.run_id)
        mlflow.set_tag("drop_reason", "duplicate")

        df_no_dupes = df.drop_duplicates()
        drop_count = len(df) - len(df_no_dupes)
        drop_percent = round(100 * drop_count / len(df), 2)

        metrics = {
            "dataset_size": len(df),
            "duplicate_count": drop_count,
            "duplicate_percent": drop_percent,
            "cleaned_size": len(df_no_dupes),
        }
        mlflow.log_metrics(metrics)
        mlflow.end_run()
        return df_no_dupes


def drop_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    data: pd.DataFrame = df.copy()
    features = list(data.columns.drop(config.TARGET))

    # Step 1 : Find feature groups with >1 unique label
    grouped = data.groupby(features)[config.TARGET].nunique()
    conflicting_groups = grouped[grouped > 1]

    # Step 2 : Extract all conflicting rows
    conflict_keys = conflicting_groups.index
    df_conflict = pd.DataFrame(list(conflict_keys), columns=features)
    data.merge(df_conflict, on=features, how="inner")

    # Step 3 : Remove conflicting rows
    df_no_conflict = data.merge(df_conflict, on=features, how="left", indicator=True)
    df_no_conflict = df_no_conflict[df_no_conflict["_merge"] == "left_only"].drop(
        columns="_merge"
    )

    with mlflow.start_run(run_name="Fix label noise") as run:
        mlflow.set_tag("run_id", run.info.run_id)
        mlflow.set_tag("drop_reason", "label_conflict")

        noise_count = len(df) - len(df_no_conflict)
        noise_percent = round(100 * noise_count / len(df), 2)
        metrics = {
            "dataset_size": len(df),
            "noisy_label_count": noise_count,
            "noisy_label_percent": noise_percent,
            "cleaned_size": len(df_no_conflict),
        }
        mlflow.log_metrics(metrics)
        mlflow.end_run()

    return df_no_conflict


def ingest(raw_path, to_dir):
    rs = config.RANDOM_STATE
    df = pd.read_csv(raw_path)
    df = df.drop(labels=["ID"], axis=1)

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name="Data Ingestion")
    df = drop_missing(df)
    df = drop_duplicates(df)
    df = drop_conflicts(df)

    df_train, df_test = train_test_split(
        df,
        test_size=config.TRAIN_TEST_SPLIT,
        stratify=df[config.TARGET],
        random_state=rs,
    )
    df_val, df_test = train_test_split(
        df_test,
        test_size=config.TEST_VAL_SPLIT,
        stratify=df_test[config.TARGET],
        random_state=rs,
    )

    df_train.to_csv(to_dir / config.TRAIN_FILE, header=False, index=False)
    df_val.to_csv(to_dir / config.VALIDATION_FILE, header=False, index=False)
    df_test.to_csv(to_dir / config.TEST_FILE, header=False, index=False)

    with mlflow.start_run(run_name="Split dataset") as run:
        mlflow.set_tag("run_id", run.info.run_id)
        mlflow.set_tag("split_strategy", "stratified")
        metrics = {
            "dataset_size": len(df),
            "train_test_split": config.TRAIN_TEST_SPLIT,
            "test_val_split": config.TEST_VAL_SPLIT,
            "random_state": rs,
            "train_size": len(df_train),
            "val_size": len(df_val),
            "test_size": len(df_test),
        }
        mlflow.log_metrics(metrics)
        mlflow.end_run()


def main():
    to_dir = config.DATA_DIR / "prepared"
    if os.path.exists(to_dir / config.TRAIN_FILE):
        print("[INFO] Dataset is already ingested.")
        return

    raw_path = config.DATA_DIR / "raw" / config.RAW_FILE
    ingest(raw_path, to_dir)
    print("[INFO] Raw dataset was successfully ingested.")


if __name__ == "__main__":
    main()
