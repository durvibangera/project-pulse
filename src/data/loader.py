"""
src/data/loader.py
------------------
Raw dataset loaders for all five P.U.L.S.E. data sources.

Dataset paths (relative to project root):
  D1  data/raw/diabetes-uci/diabetic_data.csv
  D2  data/raw/nhanes/{2015-16,2021-23}/*.xpt
  D3  data/raw/chronic-disease/chronic_disease_dataset.csv
  D4  data/raw/statlog-heart/heart.dat
  D5  data/raw/drug-reviews/drugLibTrain_raw.tsv + drugLibTest_raw.tsv
"""

import os
import pandas as pd
import pyreadstat


def _read_xpt(fpath: str) -> pd.DataFrame:
    """
    Read a SAS XPT file, trying multiple strategies in order:
      1. pandas.read_sas  (handles most encoding variants natively)
      2. pyreadstat.read_xport  (fallback)
    """
    try:
        return pd.read_sas(fpath, format="xport", encoding="latin-1")
    except Exception:
        pass
    try:
        df, _ = pyreadstat.read_xport(fpath)
        return df
    except Exception:
        df, _ = pyreadstat.read_xport(fpath, encoding="latin-1")
        return df

# ---------------------------------------------------------------------------
# D1 — Diabetes 130-US Hospitals
# ---------------------------------------------------------------------------

def load_diabetes_130(
    path: str = "data/raw/diabetes-uci/diabetic_data.csv",
) -> pd.DataFrame:
    """Load the UCI Diabetes 130 dataset, replacing '?' sentinels with NA."""
    df = pd.read_csv(path)
    df = df.replace("?", pd.NA)
    return df


# ---------------------------------------------------------------------------
# D2 — NHANES (one call per cycle directory)
# ---------------------------------------------------------------------------

def load_nhanes_cycle(cycle_dir: str) -> pd.DataFrame:
    """
    Load and merge all XPT files from a single NHANES cycle directory.

    Files detected by keyword in filename (case-insensitive):
        demo   → DEMO_*.xpt   (demographics: SEQN, RIDAGEYR, RIAGENDR)
        ghb    → GHB_*.xpt    (HbA1c: LBXGH)
        biopro → BIOPRO_*.xpt (biochemistry: LBXSGL serum glucose, LBXSCR creatinine)
        diq    → DIQ_*.xpt    (diabetes questionnaire: DIQ010)

    All frames are left-joined to DEMO on SEQN (respondent sequence number).
    Returns the merged DataFrame.
    """
    buckets = {"demo": None, "ghb": None, "biopro": None, "diq": None}

    for fname in os.listdir(cycle_dir):
        lower = fname.lower()
        if not lower.endswith(".xpt"):
            continue
        if "demo" in lower:
            buckets["demo"] = fname
        elif "ghb" in lower:
            buckets["ghb"] = fname
        elif "biopro" in lower:
            buckets["biopro"] = fname
        elif "diq" in lower:
            buckets["diq"] = fname

    if buckets["demo"] is None:
        raise FileNotFoundError(
            f"No DEMO XPT file found in {cycle_dir}. "
            f"Files present: {os.listdir(cycle_dir)}"
        )

    frames: dict[str, pd.DataFrame] = {}
    for key, fname in buckets.items():
        if fname is not None:
            fpath = os.path.join(cycle_dir, fname)
            frames[key] = _read_xpt(fpath)

    merged = frames["demo"].copy()
    for key in ("biopro", "ghb", "diq"):
        if key in frames:
            # Drop any columns that already exist in merged (except SEQN)
            existing = [c for c in frames[key].columns if c in merged.columns and c != "SEQN"]
            sub = frames[key].drop(columns=existing)
            merged = merged.merge(sub, on="SEQN", how="left")

    return merged


# ---------------------------------------------------------------------------
# D3 — Chronic Disease Prediction (Kaggle)
# ---------------------------------------------------------------------------

def load_chronic_disease(
    path: str = "data/raw/chronic-disease/chronic_disease_dataset.csv",
) -> pd.DataFrame:
    """Load the Chronic Disease Prediction dataset."""
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# D4 — Statlog Heart Disease
# ---------------------------------------------------------------------------

_STATLOG_COLS = [
    "age", "sex", "chest_pain_type", "blood_pressure_sys",
    "cholesterol", "fasting_blood_sugar", "resting_ecg",
    "max_heart_rate", "exercise_angina", "st_depression",
    "st_slope", "major_vessels", "thal", "cardio_label",
]


def load_statlog(
    path: str = "data/raw/statlog-heart/heart.dat",
) -> pd.DataFrame:
    """
    Load the Statlog (Heart) dataset.
    Space-separated, no header, 270 rows × 14 columns.
    cardio_label: 1 = absence of disease, 2 = presence of disease.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, names=_STATLOG_COLS)
    return df


# ---------------------------------------------------------------------------
# D5 — Drug Reviews (Druglib.com)
# ---------------------------------------------------------------------------

def load_drug_reviews(
    train_path: str = "data/raw/drug-reviews/drugLibTrain_raw.tsv",
    test_path: str = "data/raw/drug-reviews/drugLibTest_raw.tsv",
) -> pd.DataFrame:
    """
    Load and concatenate train + test drug review TSV files.
    Encoding: latin-1 (contains special characters in review text).
    Adds a 'split' column to track provenance.
    """
    train = pd.read_csv(train_path, sep="\t", encoding="latin-1")
    train["split"] = "train"
    test = pd.read_csv(test_path, sep="\t", encoding="latin-1")
    test["split"] = "test"
    return pd.concat([train, test], ignore_index=True)
