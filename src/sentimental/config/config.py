from pathlib import Path

from sklearn.linear_model import LogisticRegression


RANDOM_STATE = 147

PROJECT_NAME = "sentimental"
PACKAGE_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PACKAGE_ROOT / "data"
MODEL_PATH = PACKAGE_ROOT / "model"
METRICS_PATH = PACKAGE_ROOT / "metrics"

RAW_FILE = "" # Name of raw CSV dataset
TRAIN_FILE = "train.csv"
VALIDATION_FILE = "validation.csv"
TEST_FILE = "test.csv"
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

TARGET = "" # Used for supervised learning tasks

BASELINE_MODEL = LogisticRegression()
