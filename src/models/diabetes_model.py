"""
src/models/diabetes_model.py
-----------------------------
Metabolic Vertical — Diabetes readmission / risk classifier.

Training data  : master_patient_table.parquet
                 rows where diabetes_label is not NaN AND
                 source_dataset in {"diabetes_130", "nhanes", "chronic"}
Features       : age, gender, bmi, glucose, hba1c,
                 blood_pressure_sys, blood_pressure_dia
Output artefacts:
    models/diabetes_xgb.pkl
    models/diabetes_features.pkl

Usage:
    python src/models/diabetes_model.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.base_model import BaseModel

# ------------------------------------------------------------------
# Feature set
# ------------------------------------------------------------------

DIABETES_FEATURES = [
    "age",
    "gender",
    "bmi",
    "glucose",
    "hba1c",
    "blood_pressure_sys",
    "blood_pressure_dia",
]

_MASTER_PATH = "data/processed/master_patient_table.parquet"
_MODEL_OUT = "models/diabetes_xgb.pkl"
_FEATURES_OUT = "models/diabetes_features.pkl"


def train_diabetes_model(
    master_path: str = _MASTER_PATH,
    test_size: float = 0.20,
    random_state: int = 42,
    shap_plot: bool = False,
) -> tuple[BaseModel, dict]:
    """
    Train the Diabetes risk classifier and persist model artifacts.

    Parameters
    ----------
    master_path  : path to master_patient_table.parquet
    test_size    : fraction of data held out for final evaluation
    random_state : reproducibility seed
    shap_plot    : if True, renders a SHAP summary plot (requires display)

    Returns (model: BaseModel, metrics: dict)
    """
    os.makedirs("models", exist_ok=True)

    master = pd.read_parquet(master_path)

    # Filter to rows that have a diabetes label and come from relevant sources
    df = master[
        master["diabetes_label"].notna()
        & master["source_dataset"].isin(["diabetes_130", "nhanes", "chronic"])
    ].copy()

    print(f"[DiabetesModel] Training rows : {len(df):,}")
    print(f"[DiabetesModel] Label balance : {df['diabetes_label'].value_counts().to_dict()}")

    X = df[DIABETES_FEATURES].copy()
    # Ensure no residual NaNs (median per column as safety net)
    X = X.fillna(X.median(numeric_only=True))
    y = df["diabetes_label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    model = BaseModel(
        name="DiabetesClassifier",
        features=DIABETES_FEATURES,
        label="diabetes_label",
    )

    cv_metrics = model.train(X_train, y_train, experiment_name="PULSE-Diabetes")
    test_metrics = model.evaluate(X_test, y_test)

    # Persist artifacts before SHAP so they are always written
    joblib.dump(model.model, _MODEL_OUT)
    joblib.dump(DIABETES_FEATURES, _FEATURES_OUT)
    print(f"[DiabetesModel] Saved → {_MODEL_OUT}")
    print(f"[DiabetesModel] Saved → {_FEATURES_OUT}")

    # SHAP on a representative sample
    sample_size = min(300, len(X_test))
    model.explain(
        X_test.sample(sample_size, random_state=random_state),
        plot=shap_plot,
    )

    return model, {**cv_metrics, **test_metrics}


if __name__ == "__main__":
    train_diabetes_model(shap_plot=False)
