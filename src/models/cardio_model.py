"""
src/models/cardio_model.py
--------------------------
Cardiovascular Vertical — Heart disease classifier (Statlog dataset).

Training data  : loaded directly from raw Statlog file
                 (only 270 rows; no need to use master table)
Features       : age, gender, blood_pressure_sys, cholesterol,
                 max_heart_rate, chest_pain_type, exercise_angina,
                 st_depression, major_vessels
Output artefacts:
    models/cardio_xgb.pkl
    models/cardio_features.pkl

Usage:
    python src/models/cardio_model.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from src.data.loader import load_statlog
from src.data.cleaner import clean_statlog
from src.models.base_model import BaseModel

# ------------------------------------------------------------------
# Feature set
# ------------------------------------------------------------------

CARDIO_FEATURES = [
    "age",
    "gender",
    "blood_pressure_sys",
    "cholesterol",
    "max_heart_rate",
    "chest_pain_type",
    "exercise_angina",
    "st_depression",
    "major_vessels",
]

_STATLOG_PATH = "data/raw/statlog-heart/heart.dat"
_MODEL_OUT = "models/cardio_xgb.pkl"
_FEATURES_OUT = "models/cardio_features.pkl"


def train_cardio_model(
    statlog_path: str = _STATLOG_PATH,
    test_size: float = 0.20,
    random_state: int = 42,
    shap_plot: bool = False,
) -> tuple[BaseModel, dict]:
    """
    Train the Cardiovascular risk classifier and persist model artifacts.

    Parameters
    ----------
    statlog_path : path to heart.dat
    test_size    : fraction of data held out for final evaluation
    random_state : reproducibility seed
    shap_plot    : if True, renders a SHAP summary plot (requires display)

    Returns (model: BaseModel, metrics: dict)
    """
    os.makedirs("models", exist_ok=True)

    raw = load_statlog(statlog_path)
    df = clean_statlog(raw)

    print(f"[CardioModel] Training rows : {len(df):,}")
    print(f"[CardioModel] Label balance : {df['cardio_label'].value_counts().to_dict()}")

    # Ensure all required feature columns exist
    missing = [c for c in CARDIO_FEATURES if c not in df.columns]
    if missing:
        raise KeyError(
            f"[CardioModel] Missing feature columns after cleaning: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    X = df[CARDIO_FEATURES].copy()
    X = X.fillna(X.median(numeric_only=True))
    y = df["cardio_label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Mild class imbalance: 120 positive / 150 negative → scale_pos_weight ≈ 1.25
    vc  = y_train.value_counts()
    spw = float(vc.get(0, 1)) / float(vc.get(1, 1))
    print(f"[CardioModel] scale_pos_weight = {spw:.3f}")

    # Statlog is small — reduce estimators to avoid overfitting on ~216 train rows
    model = BaseModel(
        name="CardioClassifier",
        features=CARDIO_FEATURES,
        label="cardio_label",
        xgb_params={"n_estimators": 200, "max_depth": 4, "scale_pos_weight": spw},
    )

    cv_metrics = model.train(X_train, y_train, experiment_name="PULSE-Cardio")
    test_metrics = model.evaluate(X_test, y_test)

    # ------------------------------------------------------------------
    # Bootstrap 95 % CI on test-set ROC-AUC  (improvement 4)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(random_state)
    test_proba = model.model.predict_proba(X_test)[:, 1]
    boot_aucs: list[float] = []
    for _ in range(1_000):
        idx = rng.integers(0, len(y_test), size=len(y_test))
        ys, ps = y_test.iloc[idx], test_proba[idx]
        if len(np.unique(ys)) < 2:
            continue  # skip degenerate bootstrap samples
        boot_aucs.append(roc_auc_score(ys, ps))
    ci_lo, ci_hi = float(np.percentile(boot_aucs, 2.5)), float(np.percentile(boot_aucs, 97.5))
    print(
        f"[CardioModel] Bootstrap AUC 95% CI : "
        f"{ci_lo:.4f} – {ci_hi:.4f}  (n_boot=1000)"
    )
    test_metrics["auc_ci_lo"] = ci_lo
    test_metrics["auc_ci_hi"] = ci_hi

    # Persist artifacts before SHAP so they are always written
    joblib.dump(model.model, _MODEL_OUT)
    joblib.dump(CARDIO_FEATURES, _FEATURES_OUT)
    print(f"[CardioModel] Saved → {_MODEL_OUT}")
    print(f"[CardioModel] Saved → {_FEATURES_OUT}")

    sample_size = min(50, len(X_test))
    model.explain(
        X_test.sample(sample_size, random_state=random_state),
        plot=shap_plot,
    )

    return model, {**cv_metrics, **test_metrics}


if __name__ == "__main__":
    train_cardio_model(shap_plot=False)
