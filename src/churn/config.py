"""Paths, feature lists and training settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "telco_churn.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATHS = {
    "xgb": MODEL_DIR / "churn_model.joblib",       # best model, served by default
    "baseline": MODEL_DIR / "baseline_lr.joblib",  # logistic regression baseline
}
DEFAULT_MODEL = "xgb"
REPORT_DIR = ROOT / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

ID_COL = "customerID"
TARGET = "Churn"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_NUMERIC = ["SeniorCitizen"]  # already 0/1
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + BINARY_NUMERIC + CATEGORICAL_FEATURES
