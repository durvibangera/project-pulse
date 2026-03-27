"""
src/monitoring/drift_detector.py
---------------------------------
Sentinel Suite — NHANES population drift detector.

Compares NHANES 2015-16 (pre-pandemic baseline) against NHANES 2021-23
(post-pandemic RWE) to detect data drift and model performance degradation.

Design notes:
  - Uses Evidently 0.7.x API (Dataset + DataDefinition, not legacy Report)
  - DRIFT_FEATURES are limited to columns with real NHANES data:
      age, gender, glucose, hba1c, serum_creatinine
    (bmi / blood_pressure_sys/dia are absent from NHANES XPT files)
  - Model predictions use nhanes_diab_xgb.pkl; missing features (bmi, bp)
    are filled with master-table global medians so the model receives
    the same constant it saw during training.
  - ClassificationPreset compares predicted risk scores between cycles,
    surfacing concept drift even without new ground-truth labels.

Usage:
    python src/monitoring/drift_detector.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import joblib
import numpy as np
import pandas as pd

from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.metrics import DriftedColumnsCount, ValueDrift
from evidently.presets import ClassificationPreset, DataDriftPreset

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

# Features actually present in the NHANES cycle parquets
DRIFT_FEATURES = ["age", "gender", "glucose", "hba1c", "serum_creatinine"]

# Features expected by the NHANES diabetes classifier
MODEL_FEATURES = [
    "age", "gender", "bmi", "glucose", "hba1c",
    "blood_pressure_sys", "blood_pressure_dia",
]

# Fill values for model features absent from NHANES parquets.
# These are the global medians computed from the master patient table
# (the same constant the model saw during training via median imputation).
_FILL_CONSTANTS = {
    "bmi": 27.11,
    "blood_pressure_sys": 120.5,
    "blood_pressure_dia": 75.0,  # standard clinical default (all NaN in master)
}

_REF_PATH    = "data/processed/nhanes_2015.parquet"
_CUR_PATH    = "data/processed/nhanes_2021.parquet"
_MODEL_PATH  = "models/nhanes_diab_xgb.pkl"
_OUTPUT_PATH = "reports/model_health_report.html"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _prepare_df(raw: pd.DataFrame, model) -> pd.DataFrame:
    """
    Filter to labelled rows, add model predictions, and return a
    clean DataFrame ready for Evidently.

    Columns in output:
        DRIFT_FEATURES + ['target', 'prediction', 'prediction_proba']
    """
    df = raw.dropna(subset=["diabetes_label"]).copy()

    # Build feature matrix for model (fill constants for missing cols)
    X = df[DRIFT_FEATURES].copy()
    for col, val in _FILL_CONSTANTS.items():
        X[col] = val

    X = X[MODEL_FEATURES]  # ensure correct column order

    df["prediction_proba"] = model.predict_proba(X)[:, 1]
    df["prediction"]       = (df["prediction_proba"] >= 0.5).astype(int)
    df["target"]           = df["diabetes_label"].astype(int)

    return df[DRIFT_FEATURES + ["target", "prediction", "prediction_proba"]].reset_index(drop=True)


def _build_evidently_dataset(df: pd.DataFrame) -> Dataset:
    """Wrap a prepared DataFrame as an Evidently 0.7 Dataset."""
    data_def = DataDefinition(
        numerical_columns=["age", "glucose", "hba1c", "serum_creatinine"],
        categorical_columns=["gender"],
        classification=[
            BinaryClassification(
                target="target",
                prediction_labels="prediction",
                prediction_probas="prediction_proba",
            )
        ],
    )
    return Dataset.from_pandas(df, data_definition=data_def)


# ------------------------------------------------------------------
# Main drift report
# ------------------------------------------------------------------

def run_drift_report(
    reference_path: str = _REF_PATH,
    current_path:   str = _CUR_PATH,
    model_path:     str = _MODEL_PATH,
    output_path:    str = _OUTPUT_PATH,
) -> dict:
    """
    Run the full Sentinel drift report and save an HTML artefact.

    Returns a structured result dict with drift flags and column-level scores.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("[SENTINEL] Loading NHANES cycle parquets ...")
    ref_raw = pd.read_parquet(reference_path)
    cur_raw = pd.read_parquet(current_path)

    print(f"[SENTINEL] 2015-16 total rows : {len(ref_raw):,}")
    print(f"[SENTINEL] 2021-23 total rows : {len(cur_raw):,}")

    model = joblib.load(model_path)

    ref_df = _prepare_df(ref_raw, model)
    cur_df = _prepare_df(cur_raw, model)

    print(f"[SENTINEL] 2015-16 labelled   : {len(ref_df):,} "
          f"(pos={int(ref_df.target.sum())})")
    print(f"[SENTINEL] 2021-23 labelled   : {len(cur_df):,} "
          f"(pos={int(cur_df.target.sum())})")

    print("[SENTINEL] Building Evidently datasets ...")
    ref_ds = _build_evidently_dataset(ref_df)
    cur_ds = _build_evidently_dataset(cur_df)

    print("[SENTINEL] Running drift report ...")
    report = Report(
        metrics=[
            DataDriftPreset(),
            ClassificationPreset(),
            DriftedColumnsCount(),
            ValueDrift(column="glucose"),
            ValueDrift(column="hba1c"),
            ValueDrift(column="age"),
            ValueDrift(column="serum_creatinine"),
        ]
    )

    snapshot = report.run(current_data=cur_ds, reference_data=ref_ds)
    snapshot.save_html(output_path)
    print(f"[SENTINEL] Health Report saved → {output_path}")

    # ------------------------------------------------------------------
    # Parse results into structured dict
    # ------------------------------------------------------------------
    result = _parse_snapshot(snapshot)
    _print_summary(result)
    return result


# ------------------------------------------------------------------
# Result parsing
# ------------------------------------------------------------------

def _parse_snapshot(snapshot) -> dict:
    """
    Extract key drift indicators from the Evidently 0.7 Snapshot.

    Evidently 0.7 dict structure per metric:
        { "metric_name": "<ClassName>(...)", "config": {...}, "value": <float|dict> }

    Returns a dict consumed by health_report.py.
    """
    result: dict = {
        "dataset_drift": None,
        "drifted_columns_count": None,
        "column_drift": {},
        "classification": {},
    }

    try:
        raw = snapshot.dict()

        for m in raw.get("metrics", []):
            name  = m.get("metric_name", "")
            value = m.get("value")
            cfg   = m.get("config", {})

            # ----------------------------------------------------------
            # DriftedColumnsCount — dataset-level drift summary
            # ----------------------------------------------------------
            if name.startswith("DriftedColumnsCount"):
                if isinstance(value, dict):
                    result["drifted_columns_count"] = int(value.get("count", 0))
                    share = float(value.get("share", 0.0))
                    # drift_share default = 0.5; dataset drift if share >= threshold
                    drift_share = float(cfg.get("drift_share", 0.5))
                    result["dataset_drift"] = share >= drift_share

            # ----------------------------------------------------------
            # ValueDrift(column=X, threshold=T) — per-column score
            # ----------------------------------------------------------
            elif name.startswith("ValueDrift"):
                col = cfg.get("column")
                threshold = float(cfg.get("threshold", 0.1))
                if col and isinstance(value, (int, float)):
                    score = float(value)
                    result["column_drift"][col] = {
                        "drift_score":    score,
                        "drift_detected": score > threshold,
                        "threshold":      threshold,
                        "method":         cfg.get("method", ""),
                    }

            # ----------------------------------------------------------
            # Classification metrics — Accuracy, RocAuc, F1Score, etc.
            # ----------------------------------------------------------
            elif name.startswith("Accuracy("):
                result["classification"]["accuracy"] = float(value)
            elif name.startswith("RocAuc("):
                result["classification"]["roc_auc"] = float(value)
            elif name.startswith("F1Score("):
                result["classification"]["f1"] = float(value)
            elif name.startswith("Precision("):
                result["classification"]["precision"] = float(value)
            elif name.startswith("Recall("):
                result["classification"]["recall"] = float(value)

    except Exception as exc:
        print(f"[SENTINEL] Warning: result parsing failed — {exc}")

    return result


def _print_summary(result: dict) -> None:
    drift_flag = result.get("dataset_drift")
    n_drift    = result.get("drifted_columns_count")

    print()
    print("=" * 58)
    print("  SENTINEL SUITE — DRIFT SUMMARY")
    print("=" * 58)
    print(f"  Dataset drift detected : {drift_flag}")
    if n_drift is not None:
        print(f"  Drifted columns        : {n_drift}")
    print()
    for col, info in result.get("column_drift", {}).items():
        flag  = "⚠ DRIFT" if info.get("drift_detected") else "OK"
        score = info.get("drift_score")
        score_str = f"{score:.4f}" if score is not None else "n/a"
        print(f"  {col:<22} score={score_str}  {flag}")
    print()
    if drift_flag:
        print("  [ALERT] Significant data drift detected — retraining recommended.")
    else:
        print("  [OK] No significant drift detected.")
    print("=" * 58 + "\n")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    run_drift_report()
