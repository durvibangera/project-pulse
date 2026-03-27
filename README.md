# P.U.L.S.E. — Predictive Unified Life-sciences Summarization Engine

An end-to-end Clinical Intelligence Platform demonstrating Real-World Evidence (RWE) analysis, multi-vertical predictive modeling, ML model monitoring, and NLP-powered research summarization.

---

## Architecture

| Layer | Component | Description |
|---|---|---|
| 1 | **Feature Store** | Unified patient table across 5 datasets |
| 2 | **ML Engine** | Diabetes + Cardiovascular XGBoost models |
| 3 | **Sentinel Suite** | NHANES drift detection (2015-16 → 2021-23) |
| 4 | **Evidence Summarizer** | Drug review NLP + RAG pipeline (Ollama/Llama 3) |

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

# MLflow UI
mlflow ui --port 5000

# Streamlit Dashboard (all layers)
streamlit run app/dashboard.py
```

---

## Project Structure

```
pulse/
├── data/
│   ├── raw/                    # Original datasets (5 sources)
│   └── processed/              # Parquet feature store outputs
├── src/
│   ├── data/
│   │   ├── loader.py           # Dataset loaders
│   │   ├── cleaner.py          # Schema standardization
│   │   └── feature_store.py    # Master patient table builder
│   ├── models/
│   │   ├── base_model.py       # XGBoost + MLflow + SHAP
│   │   ├── diabetes_model.py   # Metabolic vertical
│   │   └── cardio_model.py     # Cardiovascular vertical
│   ├── monitoring/
│   │   ├── drift_detector.py   # Evidently AI drift reports
│   │   └── health_report.py    # Model health alert parser
│   └── nlp/
│       ├── rag_pipeline.py     # ChromaDB indexer
│       └── summarizer.py       # LangChain + Ollama QA chain
├── app/
│   └── dashboard.py            # Streamlit multi-tab dashboard
├── notebooks/                  # Jupyter analysis notebooks (01–06)
├── models/                     # Saved model artifacts (.pkl)
├── reports/                    # Auto-generated drift HTML reports
├── mlflow/                     # MLflow experiment tracking store
└── chroma_db/                  # ChromaDB persistent vector store
```

---

## Key Technical Decisions

- **XGBoost + SHAP** — Non-linear biomarker interactions (glucose × HbA1c × BMI); SHAP for clinical interpretability
- **Evidently AI** — Production-grade drift library; generates shareable HTML reports
- **ChromaDB + all-MiniLM-L6-v2** — Zero-cost local embeddings for drug review indexing
- **Ollama / Llama 3** — Free local LLM for RAG summarization; no API key required
- **Parquet over CSV** — 5-10× faster read, preserves dtypes for the 100K+ row master table
- **Stratified splits** — Clinical datasets are class-imbalanced; `stratify=y` enforced throughout
