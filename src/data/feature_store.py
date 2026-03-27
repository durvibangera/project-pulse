"""
src/data/feature_store.py
--------------------------
Builds the P.U.L.S.E. Master Patient Table and cycle-specific NHANES parquets.

Outputs (written to data/processed/):
    master_patient_table.parquet  — all sources combined, master schema
    nhanes_2015.parquet           — NHANES 2015-16 cleaned (drift reference)
    nhanes_2021.parquet           — NHANES 2021-23 cleaned (drift current)

Usage:
    python src/data/feature_store.py
"""

import os
import sys

# Allow running as a script from the project root
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd

from src.data.loader import (
    load_diabetes_130,
    load_nhanes_cycle,
    load_chronic_disease,
    load_statlog,
)
from src.data.cleaner import (
    clean_diabetes_130,
    clean_nhanes,
    clean_chronic_disease,
    clean_statlog,
    handle_missing,
    remove_outliers_iqr,
)

# ---------------------------------------------------------------------------
# Master schema column order
# ---------------------------------------------------------------------------

MASTER_COLS = [
    "patient_id",
    "age",
    "gender",
    "bmi",
    "glucose",
    "hba1c",
    "blood_pressure_sys",
    "blood_pressure_dia",
    "cholesterol",
    "heart_rate",
    "serum_creatinine",
    "diabetes_label",
    "cardio_label",
    "source_dataset",
    "nhanes_cycle",
]

# Feature columns that will be winsorised before saving
_WINSORISE_COLS = [
    "age", "bmi", "glucose", "hba1c",
    "blood_pressure_sys", "blood_pressure_dia",
    "cholesterol", "heart_rate", "serum_creatinine",
    # Extended diabetes_130 clinical features
    "time_in_hospital", "num_medications", "num_lab_procedures",
    "num_procedures", "number_diagnoses", "number_inpatient", "number_emergency",
]

_PROCESSED_DIR = "data/processed"


def _ensure_dirs() -> None:
    os.makedirs(_PROCESSED_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)


def build_feature_store(
    diabetes_path: str = "data/raw/diabetes-uci/diabetic_data.csv",
    nhanes_2015_dir: str = "data/raw/nhanes/2015-16",
    nhanes_2021_dir: str = "data/raw/nhanes/2021-23",
    chronic_path: str = "data/raw/chronic-disease/chronic_disease_dataset.csv",
    statlog_path: str = "data/raw/statlog-heart/heart.dat",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load, clean, and unify all datasets into the master patient table.

    Returns the master DataFrame and saves three parquet files.
    """
    _ensure_dirs()
    frames = []

    # ------------------------------------------------------------------
    # D1 — Diabetes 130-US Hospitals
    # ------------------------------------------------------------------
    if verbose:
        print("[PULSE] Loading Diabetes 130-US Hospitals …")
    d1 = clean_diabetes_130(load_diabetes_130(diabetes_path))
    if verbose:
        print(f"        {len(d1):,} rows")
    frames.append(d1)

    # ------------------------------------------------------------------
    # D2a — NHANES 2015-16 (training / drift reference baseline)
    # ------------------------------------------------------------------
    if verbose:
        print("[PULSE] Loading NHANES 2015-16 …")
    nhanes_2015 = clean_nhanes(
        load_nhanes_cycle(nhanes_2015_dir), cycle="2015_16"
    )
    nhanes_2015_path = os.path.join(_PROCESSED_DIR, "nhanes_2015.parquet")
    nhanes_2015.to_parquet(nhanes_2015_path, index=False)
    if verbose:
        print(f"        {len(nhanes_2015):,} rows → saved {nhanes_2015_path}")
    frames.append(nhanes_2015)

    # ------------------------------------------------------------------
    # D2b — NHANES 2021-23 (production / drift current stream)
    # ------------------------------------------------------------------
    if verbose:
        print("[PULSE] Loading NHANES 2021-23 …")
    nhanes_2021 = clean_nhanes(
        load_nhanes_cycle(nhanes_2021_dir), cycle="2021_23"
    )
    nhanes_2021_path = os.path.join(_PROCESSED_DIR, "nhanes_2021.parquet")
    nhanes_2021.to_parquet(nhanes_2021_path, index=False)
    if verbose:
        print(f"        {len(nhanes_2021):,} rows → saved {nhanes_2021_path}")
    frames.append(nhanes_2021)

    # ------------------------------------------------------------------
    # D3 — Chronic Disease Prediction
    # ------------------------------------------------------------------
    if verbose:
        print("[PULSE] Loading Chronic Disease dataset …")
    d3 = clean_chronic_disease(load_chronic_disease(chronic_path))
    if verbose:
        print(f"        {len(d3):,} rows")
    frames.append(d3)

    # ------------------------------------------------------------------
    # D4 — Statlog Heart Disease
    # ------------------------------------------------------------------
    if verbose:
        print("[PULSE] Loading Statlog Heart …")
    d4 = clean_statlog(load_statlog(statlog_path))
    if verbose:
        print(f"        {len(d4):,} rows")
    frames.append(d4)

    # ------------------------------------------------------------------
    # Combine into Master Patient Table
    # ------------------------------------------------------------------
    if verbose:
        print("[PULSE] Combining into master patient table …")

    master = pd.concat(frames, ignore_index=True)

    # Reindex to master schema; extra columns (cardio features from statlog)
    # are preserved beyond MASTER_COLS via concat — we keep them for the ML layer.
    master = master.reindex(
        columns=MASTER_COLS + [
            c for c in master.columns if c not in MASTER_COLS
        ]
    )

    # Winsorise numeric features (IQR × 3) before median fill
    for col in _WINSORISE_COLS:
        if col in master.columns and master[col].notna().sum() > 0:
            master = remove_outliers_iqr(master, col)

    # Median-fill feature columns (labels and IDs are skipped inside handle_missing)
    master = handle_missing(master)

    # ------------------------------------------------------------------
    # Save master parquet
    # ------------------------------------------------------------------
    master_path = os.path.join(_PROCESSED_DIR, "master_patient_table.parquet")
    master.to_parquet(master_path, index=False)

    if verbose:
        print(f"\n[PULSE] ✓ Master Patient Table")
        print(f"        Shape   : {master.shape}")
        print(f"        Saved   : {master_path}")
        print(f"\n        Source breakdown:")
        print(master["source_dataset"].value_counts().to_string())
        print(f"\n        Diabetes label distribution (non-NaN):")
        print(master["diabetes_label"].value_counts(dropna=True).to_string())
        print(f"\n        Cardio label distribution (non-NaN):")
        print(master["cardio_label"].value_counts(dropna=True).to_string())

    return master


if __name__ == "__main__":
    build_feature_store()
