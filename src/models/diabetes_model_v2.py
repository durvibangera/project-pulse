"""
src/models/diabetes_model_v2.py
--------------------------------
Improved Metabolic Vertical — three model variants trained and compared:

  V2  Extended+Balanced  : 16 features, all sources, scale_pos_weight, optimal threshold
  V2a diabetes_130-only  : hospital readmission sub-model, all 16 features
  V2b NHANES-only        : clinical diagnosis sub-model, 7 base features

Improvements vs diabetes_model.py (V1 baseline):
  1. Re-engineered features  : +9 clinical workflow columns (time_in_hospital,
                                num_medications, num_lab_procedures, num_procedures,
                                number_diagnoses, number_inpatient, number_emergency,
                                diabetes_med, glucose_serum_high)
  2. Separate sub-models     : diabetes_130 readmission task vs NHANES clinical task
  3. Optimal threshold       : PR-curve sweep on training set maximises F1
  5. scale_pos_weight        : neg/pos ratio per dataset corrects class imbalance

Artifacts saved:
    models/diabetes_v2_xgb.pkl        <- V2 combined
    models/diabetes_v2_features.pkl
    models/diabetes_130_xgb.pkl       <- diabetes_130 sub-model
    models/diabetes_130_features.pkl
    models/nhanes_diab_xgb.pkl        <- NHANES sub-model
    models/nhanes_diab_features.pkl

Usage:
    python src/models/diabetes_model_v2.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.models.base_model import BaseModel

# ------------------------------------------------------------------
# Feature sets
# ------------------------------------------------------------------

BASE_FEATURES = [
    "age",
    "gender",
    "bmi",
    "glucose",
    "hba1c",
    "blood_pressure_sys",
    "blood_pressure_dia",
]

# 7 base + 9 clinical workflow columns (populated only for diabetes_130 rows;
# NHANES / chronic rows have NaN which are median-filled at train time)
EXTENDED_FEATURES = BASE_FEATURES + [
    "time_in_hospital",
    "num_medications",
    "num_lab_procedures",
    "num_procedures",
    "number_diagnoses",
    "number_inpatient",
    "number_emergency",
    "diabetes_med",
    "glucose_serum_high",
]

_MASTER_PATH = "data/processed/master_patient_table.parquet"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _find_best_threshold(y_true: pd.Series, y_proba: np.ndarray) -> float:
    """Return the probability threshold that maximises F1 on the given set."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns one extra value at the end — align lengths
    f1 = np.where(
        (precision[:-1] + recall[:-1]) == 0,
        0.0,
        2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1]),
    )
    best_idx = int(np.argmax(f1))
    return float(thresholds[best_idx])


def _eval_at_threshold(
    label: str,
    y_test: pd.Series,
    y_proba: np.ndarray,
    threshold: float,
) -> dict:
    """Evaluate predictions at a custom probability threshold."""
    preds = (y_proba >= threshold).astype(int)
    auc = roc_auc_score(y_test, y_proba)
    f1  = f1_score(y_test, preds, zero_division=0)

    print(f"\n{'=' * 58}")
    print(f"  {label}  [threshold = {threshold:.3f}]")
    print(f"{'=' * 58}")
    print(classification_report(y_test, preds, zero_division=0))
    print(f"  ROC-AUC : {auc:.4f}")
    print(f"  F1      : {f1:.4f}")
    print(f"{'=' * 58}\n")

    return {"auc": auc, "f1": f1, "threshold": threshold}


# ------------------------------------------------------------------
# Sub-model trainers
# ------------------------------------------------------------------

def _train_v2_combined(master: pd.DataFrame, random_state: int = 42) -> dict:
    """
    V2: Extended 16-feature model over all relevant sources.
    Uses scale_pos_weight and an optimised probability threshold.
    """
    df = master[
        master["diabetes_label"].notna()
        & master["source_dataset"].isin(["diabetes_130", "nhanes", "chronic"])
    ].copy()

    print(f"\n[V2-Combined] Rows           : {len(df):,}")
    print(f"[V2-Combined] Label balance  : {df['diabetes_label'].value_counts().to_dict()}")

    vc  = df["diabetes_label"].value_counts()
    spw = float(vc.get(0, 1)) / float(vc.get(1, 1))
    print(f"[V2-Combined] scale_pos_weight: {spw:.3f}")

    X = df[EXTENDED_FEATURES].copy()
    X = X.fillna(X.median(numeric_only=True))
    y = df["diabetes_label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=random_state
    )

    model = BaseModel(
        name="DiabetesV2-Combined",
        features=EXTENDED_FEATURES,
        label="diabetes_label",
        xgb_params={"scale_pos_weight": spw},
    )

    cv_metrics = model.train(X_train, y_train, experiment_name="PULSE-Diabetes-V2")

    # Find best threshold on the training set then apply to held-out test set
    train_proba = model.model.predict_proba(X_train)[:, 1]
    best_thr = _find_best_threshold(y_train, train_proba)
    print(f"[V2-Combined] Optimal threshold (train PR): {best_thr:.3f}")

    test_proba = model.model.predict_proba(X_test)[:, 1]
    test_metrics = _eval_at_threshold("V2-Combined", y_test, test_proba, best_thr)

    joblib.dump(model.model, "models/diabetes_v2_xgb.pkl")
    joblib.dump(EXTENDED_FEATURES, "models/diabetes_v2_features.pkl")
    print("[V2-Combined] Saved → models/diabetes_v2_xgb.pkl")

    model.explain(
        X_test.sample(min(300, len(X_test)), random_state=random_state),
        plot=False,
    )

    return {**cv_metrics, **test_metrics}


def _train_diabetes130_submodel(master: pd.DataFrame, random_state: int = 42) -> dict:
    """
    V2a: diabetes_130 rows only.
    Task: 30-day hospital readmission (all-cause).
    Label: 0 = no early readmission, 1 = readmission within 30 days.
    """
    df = master[master["source_dataset"] == "diabetes_130"].copy()

    print(f"\n[V2a-D130] Rows           : {len(df):,}")
    print(f"[V2a-D130] Label balance  : {df['diabetes_label'].value_counts().to_dict()}")

    vc  = df["diabetes_label"].value_counts()
    spw = float(vc.get(0, 1)) / float(vc.get(1, 1))
    print(f"[V2a-D130] scale_pos_weight: {spw:.3f}")

    X = df[EXTENDED_FEATURES].copy()
    X = X.fillna(X.median(numeric_only=True))
    y = df["diabetes_label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=random_state
    )

    model = BaseModel(
        name="DiabetesV2a-D130-Readmission",
        features=EXTENDED_FEATURES,
        label="diabetes_label",
        xgb_params={"scale_pos_weight": spw},
    )

    cv_metrics = model.train(X_train, y_train, experiment_name="PULSE-Diabetes-V2")

    train_proba = model.model.predict_proba(X_train)[:, 1]
    best_thr = _find_best_threshold(y_train, train_proba)
    print(f"[V2a-D130] Optimal threshold: {best_thr:.3f}")

    test_proba = model.model.predict_proba(X_test)[:, 1]
    test_metrics = _eval_at_threshold("V2a-D130-Readmission", y_test, test_proba, best_thr)

    joblib.dump(model.model, "models/diabetes_130_xgb.pkl")
    joblib.dump(EXTENDED_FEATURES, "models/diabetes_130_features.pkl")
    print("[V2a-D130] Saved → models/diabetes_130_xgb.pkl")

    model.explain(
        X_test.sample(min(300, len(X_test)), random_state=random_state),
        plot=False,
    )

    return {**cv_metrics, **test_metrics}


def _train_nhanes_submodel(master: pd.DataFrame, random_state: int = 42) -> dict:
    """
    V2b: NHANES rows only.
    Task: Clinical diabetes diagnosis (HbA1c ≥ 6.5 OR fasting glucose ≥ 126).
    Uses BASE_FEATURES only — extended workflow columns are diabetes_130-specific.
    scale_pos_weight ≈ 4.5 (heavily skewed toward non-diabetic).
    """
    df = master[
        (master["source_dataset"] == "nhanes")
        & master["diabetes_label"].notna()
    ].copy()

    print(f"\n[V2b-NHANES] Rows           : {len(df):,}")
    print(f"[V2b-NHANES] Label balance  : {df['diabetes_label'].value_counts().to_dict()}")

    vc  = df["diabetes_label"].value_counts()
    spw = float(vc.get(0, 1)) / float(vc.get(1, 1))
    print(f"[V2b-NHANES] scale_pos_weight: {spw:.3f}")

    X = df[BASE_FEATURES].copy()
    X = X.fillna(X.median(numeric_only=True))
    y = df["diabetes_label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=random_state
    )

    model = BaseModel(
        name="DiabetesV2b-NHANES-Clinical",
        features=BASE_FEATURES,
        label="diabetes_label",
        xgb_params={
            "scale_pos_weight": spw,
            "n_estimators": 200,  # smaller dataset — fewer trees avoids overfitting
            "max_depth": 4,
        },
    )

    cv_metrics = model.train(X_train, y_train, experiment_name="PULSE-Diabetes-V2")

    train_proba = model.model.predict_proba(X_train)[:, 1]
    best_thr = _find_best_threshold(y_train, train_proba)
    print(f"[V2b-NHANES] Optimal threshold: {best_thr:.3f}")

    test_proba = model.model.predict_proba(X_test)[:, 1]
    test_metrics = _eval_at_threshold("V2b-NHANES-Clinical", y_test, test_proba, best_thr)

    joblib.dump(model.model, "models/nhanes_diab_xgb.pkl")
    joblib.dump(BASE_FEATURES, "models/nhanes_diab_features.pkl")
    print("[V2b-NHANES] Saved → models/nhanes_diab_xgb.pkl")

    model.explain(
        X_test.sample(min(50, len(X_test)), random_state=random_state),
        plot=False,
    )

    return {**cv_metrics, **test_metrics}


# ------------------------------------------------------------------
# Comparison table
# ------------------------------------------------------------------

def _print_comparison(results: dict) -> None:
    print("\n" + "=" * 72)
    print("  P.U.L.S.E.  DIABETES MODEL COMPARISON")
    print("=" * 72)
    header = (
        f"{'Model':<36} {'CV AUC':>8} {'Test AUC':>9} "
        f"{'Test F1':>8} {'Threshold':>10}"
    )
    print(header)
    print("-" * 72)
    for name, r in results.items():
        cv  = r.get("cv_auc_mean", float("nan"))
        auc = r.get("auc",         float("nan"))
        f1  = r.get("f1",          float("nan"))
        thr = r.get("threshold",   0.50)
        print(
            f"{name:<36} {cv:>8.4f} {auc:>9.4f} {f1:>8.4f} {thr:>10.3f}"
        )
    print("=" * 72 + "\n")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def train_all(master_path: str = _MASTER_PATH, random_state: int = 42) -> None:
    os.makedirs("models", exist_ok=True)
    master = pd.read_parquet(master_path)

    results = {}

    results["V2-Combined (16 feat+balanced)"] = _train_v2_combined(master, random_state)
    results["V2a-D130 (readmission, 16 feat)"] = _train_diabetes130_submodel(master, random_state)
    results["V2b-NHANES (clinical, 7 feat)"]   = _train_nhanes_submodel(master, random_state)

    # Reference entry from the original V1 baseline run
    results["V1-Baseline (7 feat, thr=0.50)"] = {
        "cv_auc_mean": 0.6205,
        "auc": 0.6207,
        "f1": 0.16,
        "threshold": 0.50,
    }

    _print_comparison(results)


if __name__ == "__main__":
    train_all()
