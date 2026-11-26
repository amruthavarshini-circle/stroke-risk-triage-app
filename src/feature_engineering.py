"""Feature engineering and preprocessing pipelines."""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    ADA_GLUCOSE_BINS,
    ADA_GLUCOSE_LABELS,
    AGE_BINS,
    AGE_LABELS,
    DATA_RAW_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    WHO_BMI_BINS,
    WHO_BMI_LABELS,
)
from src.data_cleaning import (
    ClinicalDataCleaner,
    drop_other_gender_rows,
    load_raw_data,
)

ENGINEERED_NUMERIC_COLS = [
    "age", "avg_glucose_level", "bmi",
    "cardiometabolic_risk_count", "age_x_glucose", "hypertension_x_age",
]

ENGINEERED_BINARY_COLS = [
    "hypertension", "heart_disease",
    "bmi_was_missing", "smoking_was_unknown",
]

ENGINEERED_CATEGORICAL_COLS = [
    "gender", "ever_married", "work_type", "Residence_type",
    "smoking_status", "bmi_category", "glucose_risk_tier", "age_group",
]


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add clinical bins (WHO BMI, ADA glucose, age group) and interaction features."""

    def fit(self, X, y=None):
        return self  # stateless

    def transform(self, X):
        df = X.copy()

        # WHO BMI categories
        df["bmi_category"] = pd.cut(
            df["bmi"], bins=WHO_BMI_BINS, labels=WHO_BMI_LABELS,
            right=False, include_lowest=True,
        ).astype(str)

        # ADA glucose risk tiers
        df["glucose_risk_tier"] = pd.cut(
            df["avg_glucose_level"], bins=ADA_GLUCOSE_BINS, labels=ADA_GLUCOSE_LABELS,
            right=False, include_lowest=True,
        ).astype(str)

        # age groups
        df["age_group"] = pd.cut(
            df["age"], bins=AGE_BINS, labels=AGE_LABELS,
            right=False, include_lowest=True,
        ).astype(str)

        # interaction features
        df["cardiometabolic_risk_count"] = (
            (df["hypertension"] == 1).astype(int)
            + (df["heart_disease"] == 1).astype(int)
            + (df["avg_glucose_level"] >= 140.0).astype(int)
            + (df["bmi"] >= 30.0).astype(int)
        ).astype(float)

        df["age_x_glucose"] = (df["age"] * df["avg_glucose_level"]) / 100.0
        df["hypertension_x_age"] = df["hypertension"] * df["age"]

        return df


def build_preprocessor():
    """Build the ColumnTransformer for scaling, passthrough, and one-hot encoding."""
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, ENGINEERED_NUMERIC_COLS),
            ("bin", "passthrough", ENGINEERED_BINARY_COLS),
            ("cat", categorical_pipe, ENGINEERED_CATEGORICAL_COLS),
        ],
        verbose_feature_names_out=False,
    )


def make_modeling_pipeline(estimator, use_smote=False, random_state=RANDOM_STATE):
    """Build the full pipeline: Cleaner -> FeatureEng -> Preprocessor -> [SMOTE] -> Model."""
    steps = [
        ("cleaner", ClinicalDataCleaner(adult_unknown_strategy="keep")),
        ("features", ClinicalFeatureEngineer()),
        ("preprocessor", build_preprocessor()),
    ]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state)))
        steps.append(("classifier", estimator))
        return ImbPipeline(steps=steps)

    steps.append(("classifier", estimator))
    return Pipeline(steps=steps)


def load_stratified_train_test(raw_path=DATA_RAW_PATH, test_size=TEST_SIZE,
                                random_state=RANDOM_STATE):
    """Load data, remove 'Other' gender, and do a stratified train/test split."""
    raw_df = load_raw_data(raw_path)
    filtered_df = drop_other_gender_rows(raw_df)
    X = filtered_df.drop(columns=["stroke"])
    y = filtered_df["stroke"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state,
    )
    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_stratified_train_test()
    print(f"Train shape: {X_train.shape} (stroke rate: {y_train.mean():.4%})")
    print(f"Test shape:  {X_test.shape} (stroke rate: {y_test.mean():.4%})")

    prep_pipe = Pipeline(steps=[
        ("cleaner", ClinicalDataCleaner()),
        ("features", ClinicalFeatureEngineer()),
        ("preprocessor", build_preprocessor()),
    ])
    X_tr_mat = prep_pipe.fit_transform(X_train, y_train)
    X_te_mat = prep_pipe.transform(X_test)
    feat_names = prep_pipe.named_steps["preprocessor"].get_feature_names_out()

    assert X_tr_mat.shape[0] == len(X_train)
    assert X_te_mat.shape[0] == len(X_test)
    assert X_tr_mat.shape[1] == X_te_mat.shape[1] == len(feat_names)
    assert not np.isnan(X_tr_mat).any(), "NaNs in transformed train data!"
    assert not np.isnan(X_te_mat).any(), "NaNs in transformed test data!"

    # edge case test: infant with missing BMI and Other gender
    edge_case = pd.DataFrame([{
        "id": 99999, "gender": "Other", "age": 8.0,
        "hypertension": 0, "heart_disease": 0,
        "ever_married": "No", "work_type": "children",
        "Residence_type": "Urban", "avg_glucose_level": 88.5,
        "bmi": np.nan, "smoking_status": "Unknown",
    }])
    edge_mat = prep_pipe.transform(edge_case)
    assert edge_mat.shape == (1, len(feat_names))
    assert not np.isnan(edge_mat).any()

    print(f"Transformed feature count: {len(feat_names)}")
    print("Feature engineering checks PASSED!")
