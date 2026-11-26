"""Central config file — paths, seeds, clinical thresholds."""

from pathlib import Path

# paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_PATH = DATA_DIR / "raw" / "healthcare-dataset-stroke-data.csv"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
CLEANED_EDA_PATH = DATA_PROCESSED_DIR / "stroke_cleaned_eda.csv"

MODELS_DIR = PROJECT_ROOT / "models"
LR_MODEL_PATH = MODELS_DIR / "logistic_regression_pipeline.joblib"
BEST_MODEL_PATH = MODELS_DIR / "best_model_pipeline.joblib"
METRICS_PATH = MODELS_DIR / "model_metrics.json"

DOCS_DIR = PROJECT_ROOT / "docs"
FIGURES_DIR = DOCS_DIR / "figures"

# reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
ALPHA = 0.05

# cleaning constants
BMI_CAP = 65.0
PEDIATRIC_SMOKING_AGE_CUTOFF = 12.0

# clinical bins (WHO / ADA / AHA)
AGE_BINS = [0.0, 18.0, 45.0, 65.0, 120.0]
AGE_LABELS = ["<18", "18-44", "45-64", ">=65"]

WHO_BMI_BINS = [0.0, 18.5, 25.0, 30.0, 200.0]
WHO_BMI_LABELS = ["Underweight", "Normal", "Overweight", "Obese"]

ADA_GLUCOSE_BINS = [0.0, 100.0, 126.0, 200.0, 500.0]
ADA_GLUCOSE_LABELS = [
    "Normal (<100)",
    "Prediabetes (100-125)",
    "Hyperglycemia (126-199)",
    "Severe (>=200)",
]

# default triage thresholds (refined after modeling)
DEFAULT_LOW_RISK_THRESHOLD = 0.25
DEFAULT_HIGH_RISK_THRESHOLD = 0.55

# feature column groups
NUMERIC_FEATURES = ["age", "avg_glucose_level", "bmi"]
BINARY_FEATURES = [
    "hypertension",
    "heart_disease",
    "bmi_was_missing",
    "smoking_was_unknown",
]
CATEGORICAL_FEATURES = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]
