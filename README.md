# P.U.L.S.E. — Predictive Unified Life-sciences Summarization Engine (v2.0)

An end-to-end Clinical Intelligence Platform demonstrating Real-World Evidence (RWE) analysis, multi-vertical predictive modeling, ML model monitoring, NLP-powered research summarization, **clinical trial simulation**, SQL feasibility analysis, and YAML-driven model training.

---

## Architecture (4 Core Layers + 6 Add-Ons)

| Layer | Component | Description |
|---|---|---|
| 1 | **Feature Store** | Unified patient table across 5 datasets (127k+ rows) |
| 2 | **ML Engine** | Diabetes + Cardiovascular XGBoost models (MLflow tracked) |
| 3 | **Sentinel Suite** | NHANES drift detection (2015-16 → 2021-23, Evidently) |
| 4 | **Evidence Summarizer** | Drug review NLP + RAG pipeline (Ollama/Llama 3) |
| A1 | **SQL Feasibility** | DuckDB pre-modelling checks (5 checks across 2 notebooks) |
| A2 | **PySpark ETL** | Distributed feature store mirror (optional, 455MB) |
| A3 | **Trial Simulator** | Propensity score matching → Kaplan-Meier curves (lifelines) |
| A4 | **SQL Analytics** | Advanced DuckDB queries (demographics, correlations, trend analysis) |
| A5 | **PDF Reports** | ReportLab clinical report generation (executives summary, model health) |
| A6 | **YAML Runner** | CLI model training: `python run.py --config configs/diabetes.yaml` |

---

## Datasets

| ID | Dataset | Source | Path |
|---|---|---|---|
| D1 | Diabetes 130-US Hospitals | UCI | `data/raw/diabetes-uci/` |
| D2 | NHANES (2015-16 + 2021-23) | CDC | `data/raw/nhanes/` |
| D3 | Chronic Disease Prediction | Kaggle | `data/raw/chronic-disease/` |
| D4 | Statlog Heart Disease | UCI | `data/raw/statlog-heart/` |
| D5 | Drug Reviews (Druglib.com) | UCI | `data/raw/drug-reviews/` |

---

## Setup

```bash
# 1. Create environment
conda create -n pulse python=3.11
conda activate pulse

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env

# 4. Install Ollama + pull Llama 3 (for NLP layer)
# https://ollama.com/download
ollama pull llama3
```

---

## Run Order

```bash
# Layer 1 — Build Feature Store (generates 3 parquet files in data/processed/)
python src/data/feature_store.py

# Layer 2 — Train Models (requires Feature Store)
python src/models/diabetes_model.py
python src/models/cardio_model.py

# Layer 3 — Drift Detection (requires both models + parquets)
python src/monitoring/drift_detector.py

# Layer 4 — Index Drug Reviews into ChromaDB
python src/nlp/rag_pipeline.py

# Add-On 6 — Train models via YAML config (alternative to Layer 2)
python run.py --config configs/diabetes.yaml
python run.py --config configs/cardio.yaml

# MLflow UI (view all experiment runs and metrics)
mlflow ui --port 5000

# Streamlit Dashboard (all 7 tabs, all layers + add-ons)
streamlit run app/dashboard.py
```

**Optional:**
- Notebooks 07, 08, 09 (Add-Ons 1, 3, 4) can be run independently in Jupyter for exploration
- PySpark ETL (`src/data/feature_store_spark.py`) requires `pip install pyspark` (455 MB, optional)

---

## Project Structure

```
pulse/
├── data/
│   ├── raw/                          # Original datasets (5 sources)
│   └── processed/                    # Parquet feature store outputs
├── src/
│   ├── data/
│   │   ├── loader.py                 # Dataset loaders
│   │   ├── cleaner.py                # Schema standardization
│   │   ├── feature_store.py          # Master patient table builder
│   │   ├── feature_store_spark.py    # PySpark distributed ETL (Add-On 2)
│   │   └── spark_eda.py              # Spark SQL exploratory analysis
│   ├── models/
│   │   ├── base_model.py             # XGBoost + MLflow + SHAP base class
│   │   ├── diabetes_model.py         # Metabolic vertical (7 features)
│   │   ├── diabetes_model_v2.py      # Improved variant
│   │   └── cardio_model.py           # Cardiovascular vertical (5 features)
│   ├── monitoring/
│   │   ├── drift_detector.py         # Evidently AI drift reports
│   │   └── health_report.py          # Model health alert parser
│   ├── nlp/
│   │   ├── rag_pipeline.py           # ChromaDB indexer (sentence-transformers)
│   │   └── summarizer.py             # LangChain + Ollama QA chain
│   └── reporting/
│       └── pdf_report.py             # ReportLab clinical PDF (Add-On 5)
├── app/
│   └── dashboard.py                  # Streamlit 7-tab dashboard (v2.0)
├── notebooks/
│   ├── 01_data_engineering.ipynb     # Feature store walkthrough
│   ├── 02_diabetes_modelling.ipynb   # XGBoost diabetes + SHAP
│   ├── 03_cardio_modelling.ipynb     # XGBoost cardio
│   ├── 04_sentinel_drift.ipynb       # Evidently drift analysis
│   ├── 05_nlp_rag_pipeline.ipynb     # ChromaDB + RAG QA
│   ├── 06_end_to_end_showcase.ipynb  # Full pipeline demo
│   ├── 07_sql_feasibility.ipynb      # DuckDB 5 feasibility checks (Add-On 1)
│   ├── 08_clinical_trial_simulator.ipynb  # PSM + Kaplan-Meier (Add-On 3)
│   └── 09_sql_analytics.ipynb        # DuckDB queries (Add-On 4)
├── configs/                          # YAML model configs (Add-On 6)
│   ├── diabetes.yaml                 # Diabetes training config
│   └── cardio.yaml                   # Cardio training config
├── models/                           # Saved model artifacts (.pkl)
├── reports/                          # Auto-generated drift HTML + trial PNG
├── mlflow/                           # MLflow experiment tracking store
├── chroma_db/                        # ChromaDB persistent vector store
├── run.py                            # YAML-driven CLI model runner (Add-On 6)
├── requirements.txt                  # Python dependencies
├── plan.txt                          # Implementation plan
├── prompt.txt                        # System prompt
└── README.md                         # This file
```

---

## Streamlit Dashboard (7 Tabs)

Run: `streamlit run app/dashboard.py`

| Tab | Name | Purpose |
|-----|------|---------|
| 1 | 🩺 Risk Prediction | XGBoost inference UI — 7 biomarker sliders → diabetes/cardio risk probability |
| 2 | 📊 Drift Monitor | Evidently HTML report embedded; feature-level drift alerts on NHANES 2015-16 vs 2021-23 |
| 3 | 💊 Drug Evidence | RAG search over 215k+ drug reviews; LLM summarization via Ollama |
| 4 | 📈 SQL Analytics | DuckDB analytics: demographics by dataset, ADA risk strata, biomarker correlations, NHANES trend, high-risk registry |
| 5 | 🔍 Feasibility | DuckDB pre-modelling triage: 5 checks (subgroup availability, feature completeness, class balance, drift, sample size) with traffic-light scoring |
| 6 | 🧪 Trial Sim | Propensity score matching → 1:1 cohort matching → outcome metrics (ARR/NNT) → Kaplan-Meier + log-rank test |
| 7 | ⚙️ Model Runner | YAML config selector → shows config parameters → trains model via `python run.py --config …` |

**Sidebar features:**
- Project branding & tech stack overview
- "Generate PDF Report" button → ReportLab clinical PDF (Executive Summary, Dataset Overview, Model Performance, RWE Drift, Next Steps)

---

## Add-Ons (6 Targeted Enhancements)

### Add-On 1: DuckDB SQL Feasibility Module
- **Files:** `notebooks/07_sql_feasibility.ipynb`, Dashboard Tab 5
- **Features:** 5 pre-modelling SQL checks (subgroup data, feature completeness, class balance, biomarker shift, sample size) run directly on Parquet with zero-server DuckDB
- **Integration:** Tab 5 dashboard with traffic-light metrics and expandable result sections

### Add-On 2: PySpark Distributed ETL
- **Files:** `src/data/feature_store_spark.py`, `src/data/spark_eda.py`
- **Features:** Mirrors Layer 1 feature store using PySpark DataFrames + Spark SQL for distributed pipeline demo
- **Status:** Syntax-validated; optional install (PySpark = 455 MB)

### Add-On 3: Clinical Trial Simulator
- **Files:** `notebooks/08_clinical_trial_simulator.ipynb`, Dashboard Tab 6
- **Features:** Propensity Score Matching (LogisticRegression) → 1:1 NearestNeighbors cohort matching → outcome metrics (ARR, NNT) → Kaplan-Meier curves + log-rank test (lifelines)
- **Output:** Saves `reports/kaplan_meier.png` for re-use; interactive Streamlit button to regenerate with fresh data

### Add-On 4: SQL Analytics Sprint
- **Files:** `notebooks/09_sql_analytics.ipynb`, Dashboard Tab 4
- **Features:** 5 DuckDB analytics queries (demographics, ADA risk tiers, biomarker correlation heatmap, NHANES cross-cycle trend, high-risk patient registry)
- **Integration:** Full Streamlit Tab 4 with CSV download button for high-risk cohort

### Add-On 5: PDF Report Generator
- **File:** `src/reporting/pdf_report.py`
- **Features:** ReportLab 5-section clinical report (Executive Summary, Dataset Overview, Model Performance, RWE Drift Analysis, Next Steps)
- **Integration:** Streamlit sidebar "Generate PDF Report" button (triggered on-demand)

### Add-On 6: YAML-Driven Model Runner
- **Files:** `run.py`, `configs/diabetes.yaml`, `configs/cardio.yaml`, Dashboard Tab 7
- **Features:** 
  - CLI: `python run.py --config configs/diabetes.yaml`
  - Argparse with `--config` flag; loads master Parquet, filters by source_datasets, trains XGBoost, saves `.pkl`
  - Streamlit Tab 7: Config selectbox → JSON display + parameters → "🚀 Train Model" button
  - Streams training output (stdout/stderr) back to Streamlit UI

---

## Key Technical Decisions

- **XGBoost + SHAP** — Non-linear biomarker interactions (glucose × HbA1c × BMI); SHAP for clinical interpretability
- **Evidently AI** — Production-grade drift library; generates shareable HTML reports
- **ChromaDB + all-MiniLM-L6-v2** — Zero-cost local embeddings for drug review indexing
- **Ollama / Llama 3** — Free local LLM for RAG summarization; no API key required
- **DuckDB** — Zero-server SQL analytics directly on Parquet files; enables feasibility + analytics tabs without extra databases
- **lifelines** — Clinical-grade survival analysis (Kaplan-Meier, log-rank test) for trial simulation
- **ReportLab** — Programmatic PDF generation for executive summaries (no template needed)
- **Parquet over CSV** — 5-10× faster read, preserves dtypes for the 100K+ row master table
- **Stratified splits** — Clinical datasets are class-imbalanced; `stratify=y` enforced throughout
