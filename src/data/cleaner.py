"""
src/data/cleaner.py
-------------------
Schema standardisation: maps each raw dataset to the P.U.L.S.E. master schema.

Master schema target columns
-----------------------------
patient_id          str
age                 float
gender              int        0=Female, 1=Male
bmi                 float
glucose             float      mg/dL
hba1c               float      %
blood_pressure_sys  float      mmHg
blood_pressure_dia  float      mmHg
cholesterol         float      mg/dL
heart_rate          float      bpm
serum_creatinine    float      mg/dL
diabetes_label      float      0/1  (float to allow NaN)
cardio_label        float      0/1  (float to allow NaN)
source_dataset      str
nhanes_cycle        str | None
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# D1 — Diabetes 130-US Hospitals
# ---------------------------------------------------------------------------

_HBA1C_MAP = {">8": 9.0, ">7": 7.5, "Norm": 5.5, "None": np.nan}


def clean_diabetes_130(df: pd.DataFrame) -> pd.DataFrame:
    """Map the UCI Diabetes 130 raw frame to the master schema."""
    out = pd.DataFrame()
    out["patient_id"] = df["encounter_id"].astype(str)

    # Gender
    out["gender"] = df["gender"].map({"Male": 1, "Female": 0})

    # Age: ranges like "[70-80)" → extract lower bound then add 5 for midpoint
    lower_bound = df["age"].str.extract(r"\[(\d+)")[0].astype(float)
    out["age"] = lower_bound + 5

    # BMI, glucose, creatinine — not present in this dataset
    out["bmi"] = np.nan
    out["glucose"] = np.nan
    out["serum_creatinine"] = np.nan

    # HbA1c from A1Cresult categorical
    out["hba1c"] = pd.to_numeric(
        df["A1Cresult"].map(_HBA1C_MAP), errors="coerce"
    )

    # Blood pressure / cholesterol / heart rate — not in this dataset
    out["blood_pressure_sys"] = np.nan
    out["blood_pressure_dia"] = np.nan
    out["cholesterol"] = np.nan
    out["heart_rate"] = np.nan

    # Labels
    out["diabetes_label"] = (df["readmitted"] != "NO").astype(float)
    out["cardio_label"] = np.nan

    # --- Extended clinical workflow features (diabetes_130 only) ---
    out["time_in_hospital"]    = pd.to_numeric(df["time_in_hospital"],    errors="coerce")
    out["num_medications"]     = pd.to_numeric(df["num_medications"],     errors="coerce")
    out["num_lab_procedures"]  = pd.to_numeric(df["num_lab_procedures"],  errors="coerce")
    out["num_procedures"]      = pd.to_numeric(df["num_procedures"],      errors="coerce")
    out["number_diagnoses"]    = pd.to_numeric(df["number_diagnoses"],    errors="coerce")
    out["number_inpatient"]    = pd.to_numeric(df["number_inpatient"],    errors="coerce")
    out["number_emergency"]    = pd.to_numeric(df["number_emergency"],    errors="coerce")
    # Diabetes medication flag
    out["diabetes_med"]        = (df.get("diabetesMed", pd.NA) == "Yes").astype(float)
    # Elevated serum glucose flag (max_glu_serum > 200 or >300)
    _glu = df.get("max_glu_serum", pd.NA)
    out["glucose_serum_high"]  = _glu.isin([">200", ">300"]).astype(float)

    out["source_dataset"] = "diabetes_130"
    out["nhanes_cycle"] = None
    return out


# ---------------------------------------------------------------------------
# D2 — NHANES
# ---------------------------------------------------------------------------

def clean_nhanes(df: pd.DataFrame, cycle: str) -> pd.DataFrame:
    """
    Map a merged NHANES frame (DEMO + BIOPRO + GHB + DIQ) to the master schema.

    NHANES column reference
    -----------------------
    SEQN       Respondent sequence number
    RIDAGEYR   Age at screening (years)
    RIAGENDR   Gender (1=Male, 2=Female)
    LBXSGL     Serum glucose mg/dL  (BIOPRO; falls back to LBXGLU if absent)
    LBXGH      Glycohemoglobin HbA1c %  (GHB)
    LBXSCR     Serum creatinine mg/dL   (BIOPRO)
    DIQ010     Doctor told you have diabetes (1=Yes, 2=No, 3=Borderline)
    # BMX / BPX files absent → bmi, blood_pressure_sys/dia left as NaN
    """
    out = pd.DataFrame()
    out["patient_id"] = df["SEQN"].astype(str) + f"_{cycle}"
    out["age"] = pd.to_numeric(df.get("RIDAGEYR"), errors="coerce")

    gender_raw = df.get("RIAGENDR")
    out["gender"] = pd.to_numeric(gender_raw, errors="coerce").map({1: 1, 2: 0})

    # Body measures absent → NaN
    out["bmi"] = pd.to_numeric(df.get("BMXBMI"), errors="coerce") if "BMXBMI" in df.columns else np.nan
    out["blood_pressure_sys"] = pd.to_numeric(df.get("BPXSY1"), errors="coerce") if "BPXSY1" in df.columns else np.nan
    out["blood_pressure_dia"] = pd.to_numeric(df.get("BPXDI1"), errors="coerce") if "BPXDI1" in df.columns else np.nan
    out["heart_rate"] = np.nan
    out["cholesterol"] = np.nan

    # Glucose: prefer LBXSGL (BIOPRO serum glucose), fall back to LBXGLU
    if "LBXSGL" in df.columns:
        out["glucose"] = pd.to_numeric(df["LBXSGL"], errors="coerce")
    elif "LBXGLU" in df.columns:
        out["glucose"] = pd.to_numeric(df["LBXGLU"], errors="coerce")
    else:
        out["glucose"] = np.nan

    # HbA1c
    out["hba1c"] = pd.to_numeric(df.get("LBXGH"), errors="coerce") if "LBXGH" in df.columns else np.nan

    # Serum creatinine
    out["serum_creatinine"] = pd.to_numeric(df.get("LBXSCR"), errors="coerce") if "LBXSCR" in df.columns else np.nan

    # Diabetes label: HbA1c ≥ 6.5 OR glucose ≥ 126, OR doctor diagnosis (DIQ010=1)
    hba1c_flag = out["hba1c"] >= 6.5
    glucose_flag = out["glucose"] >= 126
    diag_flag = pd.Series(False, index=df.index)
    if "DIQ010" in df.columns:
        diag_flag = pd.to_numeric(df["DIQ010"], errors="coerce") == 1

    out["diabetes_label"] = (hba1c_flag | glucose_flag | diag_flag).astype(float)
    # Keep NaN where all three source signals are missing
    all_missing = out["hba1c"].isna() & out["glucose"].isna() & (~diag_flag)
    out.loc[all_missing, "diabetes_label"] = np.nan

    out["cardio_label"] = np.nan
    out["source_dataset"] = "nhanes"
    out["nhanes_cycle"] = cycle
    return out


# ---------------------------------------------------------------------------
# D3 — Chronic Disease Prediction
# ---------------------------------------------------------------------------

def clean_chronic_disease(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map the Chronic Disease dataset to the master schema.

    Raw columns available:
        age, gender, bmi, blood_pressure, cholesterol_level, glucose_level,
        physical_activity, smoking_status, alcohol_intake, family_history,
        biomarker_A–D, target

    target values: 0 (no disease) and 4 (disease) — binarised to 0/1.
    gender: assumed 0=Female, 1=Male (matches master schema directly).
    """
    out = pd.DataFrame()
    out["patient_id"] = "chronic_" + df.index.astype(str)
    out["age"] = pd.to_numeric(df.get("age"), errors="coerce")
    out["gender"] = pd.to_numeric(df.get("gender"), errors="coerce")
    out["bmi"] = pd.to_numeric(df.get("bmi"), errors="coerce")
    out["glucose"] = pd.to_numeric(df.get("glucose_level"), errors="coerce")
    out["blood_pressure_sys"] = pd.to_numeric(df.get("blood_pressure"), errors="coerce")
    out["blood_pressure_dia"] = np.nan
    out["cholesterol"] = pd.to_numeric(df.get("cholesterol_level"), errors="coerce")
    out["heart_rate"] = np.nan
    out["hba1c"] = np.nan
    out["serum_creatinine"] = np.nan

    # Binarise: 0 → no disease, anything else → disease
    out["diabetes_label"] = (pd.to_numeric(df.get("target"), errors="coerce") > 0).astype(float)
    out["cardio_label"] = np.nan

    out["source_dataset"] = "chronic"
    out["nhanes_cycle"] = None
    return out


# ---------------------------------------------------------------------------
# D4 — Statlog Heart Disease
# ---------------------------------------------------------------------------

def clean_statlog(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map the Statlog (Heart) dataset to the master schema.

    Raw label: 1 = absence of disease, 2 = presence → binarised to 0/1.
    Column 'sex' (1=Male, 0=Female) maps directly to 'gender'.
    """
    out = pd.DataFrame()
    out["patient_id"] = "statlog_" + df.index.astype(str)
    out["age"] = pd.to_numeric(df["age"], errors="coerce")
    out["gender"] = pd.to_numeric(df["sex"], errors="coerce")
    out["bmi"] = np.nan
    out["glucose"] = np.nan
    out["hba1c"] = np.nan
    out["blood_pressure_sys"] = pd.to_numeric(df["blood_pressure_sys"], errors="coerce")
    out["blood_pressure_dia"] = np.nan
    out["cholesterol"] = pd.to_numeric(df["cholesterol"], errors="coerce")
    out["heart_rate"] = pd.to_numeric(df["max_heart_rate"], errors="coerce")
    out["serum_creatinine"] = np.nan
    out["diabetes_label"] = np.nan
    out["cardio_label"] = (pd.to_numeric(df["cardio_label"], errors="coerce") == 2).astype(float)
    out["source_dataset"] = "statlog"
    out["nhanes_cycle"] = None

    # Preserve extra cardio features needed by the ML model (kept as separate cols)
    for col in ("chest_pain_type", "exercise_angina", "st_depression",
                "major_vessels", "max_heart_rate", "thal", "st_slope",
                "fasting_blood_sugar", "resting_ecg"):
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")

    return out


# ---------------------------------------------------------------------------
# Shared post-processing utilities
# ---------------------------------------------------------------------------

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill numeric NaN values with per-column median.
    Applied column-wise so label columns (which should remain NaN for
    rows where the label is genuinely absent) are NOT filled by default.
    Operates only on feature columns — label and ID columns are left untouched.
    """
    label_cols = {"diabetes_label", "cardio_label"}
    id_cols = {"patient_id", "source_dataset", "nhanes_cycle"}
    skip = label_cols | id_cols

    num_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in skip
    ]
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    return df


def remove_outliers_iqr(
    df: pd.DataFrame, col: str, factor: float = 3.0
) -> pd.DataFrame:
    """Winsorise extreme values in *col* using factor × IQR."""
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    df[col] = df[col].clip(lower=q1 - factor * iqr, upper=q3 + factor * iqr)
    return df
