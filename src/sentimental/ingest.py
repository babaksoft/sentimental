import os

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import config
from .utils import feature_target_split


def ingest(raw_path, to_dir):
    rs = config.RANDOM_STATE

    x, y = feature_target_split(raw_path)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=config.TEST_SPLIT, stratify=y, random_state=rs
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=config.TRAIN_SPLIT*config.VAL_SPLIT,
        stratify=y_train, random_state=rs)

    df_train = pd.concat([x_train, y_train], axis=1)
    df_val = pd.concat([x_val, y_val], axis=1)
    df_test = pd.concat([x_test, y_test], axis=1)

    df_train.to_csv(to_dir / config.TRAIN_FILE, header=True, index=False)
    df_val.to_csv(to_dir / config.VALIDATION_FILE, header=True, index=False)
    df_test.to_csv(to_dir / config.TEST_FILE, header=True, index=False)


def main():
    to_dir = config.DATA_PATH / "prepared"
    if os.path.exists(to_dir / config.TRAIN_FILE):
        print("[INFO] Dataset is already ingested.")
        return

    raw_path = config.DATA_PATH / "raw" / config.RAW_FILE
    ingest(raw_path, to_dir)
    print("[INFO] Raw dataset was successfully ingested.")


if __name__ == '__main__':
    main()
