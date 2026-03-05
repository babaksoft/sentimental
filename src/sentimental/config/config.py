from pathlib import Path

from sklearn.linear_model import LogisticRegression

# Global config
RANDOM_STATE = 147
PROJECT_NAME = "sentimental"
MLFLOW_TRACKING_URI = "http://localhost:5000/"
PACKAGE_ROOT = Path(__file__).resolve().parent.parent

# Path config
DATA_PATH = PACKAGE_ROOT / "data"
MODEL_PATH = PACKAGE_ROOT / "model"
METRICS_PATH = PACKAGE_ROOT / "metrics"

# Data ingestion config
TARGET = "status" # Needs stratification
RAW_FILE = "mental_health.csv"
TRAIN_FILE = "train.csv"
VALIDATION_FILE = "validation.csv"
TEST_FILE = "test.csv"
TRAIN_TEST_SPLIT = 0.3 # Train : 70%
TEST_VAL_SPLIT = 0.5   # Val : 15% , Test : 15%

# Preprocessing pipeline config
FEATURE = "statement"

BASELINE_MODEL = LogisticRegression()
