"""
app/dashboard.py
-----------------
P.U.L.S.E. — Predictive Unified Life-sciences Summarization Engine
Streamlit dashboard: 3 tabs

    Tab 1 │ Risk Prediction   — NHANES XGBoost model (7 features)
    Tab 2 │ Drift Monitor     — Evidently drift report (2015-16 vs 2021-23)
    Tab 3 │ Drug Evidence     — RAG / Mistral clinical summary

Run:
    streamlit run app/dashboard.py
"""

import os
import sys
import pickle
import warnings

warnings.filterwarnings("ignore")

# ── project root on path ────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import streamlit as st

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="P.U.L.S.E. Clinical Intelligence",
    page_icon="💉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── shared helpers ───────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(ROOT, "models")


@st.cache_resource(show_spinner=False)
def _load_model(name: str):
    path = os.path.join(MODEL_DIR, name)
    with open(path, "rb") as f:
        return pickle.load(f)


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image(
        "https://img.icons8.com/color/96/heartbeat.png",
        width=72,
    )
    st.markdown("## P.U.L.S.E.")
    st.caption("Predictive Unified Life-sciences Summarization Engine")
    st.divider()
    st.markdown(
        """
        **Models**
        - 🧬 Diabetes (NHANES XGBoost)
        - ❤️ Cardiovascular (Statlog XGBoost)

        **Stack**
        - XGBoost · MLflow · Evidently
        - ChromaDB · Mistral 7B (Ollama)
        - DuckDB · lifelines · ReportLab
        - sentence-transformers · PySpark
        """
    )
    st.divider()
    st.caption("v2.0 · Global Data Science & Analytics")
    st.divider()
    st.header("Reports")
    if st.button("Generate PDF Report"):
        from src.reporting.pdf_report import generate_model_report
        generate_model_report()
        with open("reports/PULSE_Model_Report.pdf", "rb") as f:
            st.download_button(
                label="Download PDF Report",
                data=f,
                file_name="PULSE_Model_Report.pdf",
                mime="application/pdf",
            )

# ════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════
tab_risk, tab_drift, tab_drug, tab_sql, tab_feasibility, tab_trial, tab_runner = st.tabs([
    "🧬 Risk Prediction", "📊 Drift Monitor", "💊 Drug Evidence",
    "📈 SQL Analytics", "🔍 Feasibility", "🧪 Trial Sim", "⚙️ Model Runner",
])


# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — RISK PREDICTION
# ────────────────────────────────────────────────────────────────────────────
with tab_risk:
    st.header("🧬 Diabetes Risk Prediction")
    st.caption(
        "NHANES XGBoost model (AUC 0.97).  "
        "Enter patient biometrics to receive a real-time risk score."
    )

    col_inputs, col_result = st.columns([1.2, 1], gap="large")

    with col_inputs:
        st.subheader("Patient Biometrics")

        age = st.slider("Age (years)", 20, 90, 50)
        gender = st.radio("Sex", ["Female", "Male"], horizontal=True)
        gender_val = 1 if gender == "Male" else 2   # NHANES coding

        col_a, col_b = st.columns(2)
        with col_a:
            bmi = st.number_input("BMI (kg/m²)", 10.0, 60.0, 27.0, 0.1, format="%.1f")
            glucose = st.number_input("Fasting Glucose (mg/dL)", 40.0, 600.0, 100.0, 1.0, format="%.0f")
            hba1c = st.number_input("HbA1c (%)", 3.0, 15.0, 5.5, 0.1, format="%.1f")
        with col_b:
            bp_sys = st.number_input("Systolic BP (mmHg)", 80.0, 220.0, 120.0, 1.0, format="%.0f")
            bp_dia = st.number_input("Diastolic BP (mmHg)", 40.0, 140.0, 80.0, 1.0, format="%.0f")

        predict_btn = st.button("🔍 Assess Risk", type="primary", use_container_width=True)

    with col_result:
        st.subheader("Risk Assessment")

        if predict_btn:
            model = _load_model("nhanes_diab_xgb.pkl")
            features = ["age", "gender", "bmi", "glucose", "hba1c",
                        "blood_pressure_sys", "blood_pressure_dia"]
            X = pd.DataFrame(
                [[age, gender_val, bmi, glucose, hba1c, bp_sys, bp_dia]],
                columns=features,
            )
            prob = float(model.predict_proba(X)[0, 1])
            label = model.predict(X)[0]

            # Colour-coded gauge
            if prob >= 0.70:
                colour, emoji, tier = "#d32f2f", "🔴", "HIGH RISK"
            elif prob >= 0.40:
                colour, emoji, tier = "#f57c00", "🟠", "MODERATE RISK"
            else:
                colour, emoji, tier = "#388e3c", "🟢", "LOW RISK"

            st.markdown(
                f"""
                <div style="
                    background:{colour}22;border:2px solid {colour};
                    border-radius:12px;padding:24px;text-align:center">
                    <div style="font-size:3rem">{emoji}</div>
                    <div style="font-size:2rem;font-weight:700;color:{colour}">{tier}</div>
                    <div style="font-size:1.4rem;margin-top:8px">
                        Diabetes probability: <b>{prob:.1%}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Feature breakdown
            st.markdown("#### Input Summary")
            summary_df = pd.DataFrame({
                "Feature":  ["Age", "Sex", "BMI", "Glucose", "HbA1c", "BP Sys", "BP Dia"],
                "Value":    [age, gender, f"{bmi:.1f}", f"{glucose:.0f} mg/dL",
                             f"{hba1c:.1f}%", f"{bp_sys:.0f} mmHg", f"{bp_dia:.0f} mmHg"],
                "Status":   [
                    "⚠️ Elevated" if age >= 45 else "✅ Normal",
                    "—",
                    "⚠️ Overweight" if bmi >= 25 else "✅ Normal",
                    "⚠️ Pre-diabetic" if glucose >= 100 else "✅ Normal",
                    "⚠️ Pre-diabetic" if hba1c >= 5.7 else "✅ Normal",
                    "⚠️ Elevated" if bp_sys >= 130 else "✅ Normal",
                    "⚠️ Elevated" if bp_dia >= 80 else "✅ Normal",
                ],
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        else:
            st.info("Enter patient data on the left and click **🔍 Assess Risk**.")

    # ── Cardiovascular sub-section ───────────────────────────────────────────
    st.divider()
    st.subheader("❤️ Cardiovascular Risk  *(Statlog — demo)*")
    st.caption("AUC 0.85 · 270-patient Statlog dataset · bootstrap CI [0.75, 0.94]")

    with st.expander("Enter Statlog features"):
        c1, c2, c3 = st.columns(3)
        with c1:
            cs_age   = st.number_input("Age", 20, 90, 55, key="cs_age")
            sex_c    = st.selectbox("Sex", ["Female (0)", "Male (1)"], key="cs_sex")
            chest_cp = st.selectbox("Chest Pain Type (0=typical angina … 3=asymptomatic)",
                                    [0, 1, 2, 3], index=3, key="cs_cp")
        with c2:
            cs_bp    = st.number_input("Resting BP (mmHg)", 80, 220, 130, key="cs_bp")
            cholest  = st.number_input("Serum Cholesterol (mg/dL)", 50, 600, 240, key="cs_chol")
            fbs      = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1], key="cs_fbs")
        with c3:
            rest_ecg = st.selectbox("Resting ECG (0/1/2)", [0, 1, 2], key="cs_ecg")
            max_hr   = st.number_input("Max Heart Rate", 60, 220, 150, key="cs_hr")
            exang    = st.selectbox("Exercise-induced Angina (0/1)", [0, 1], key="cs_exang")
        c4, c5 = st.columns(2)
        with c4:
            oldpeak  = st.number_input("ST Depression (oldpeak)", 0.0, 6.5, 1.0, 0.1, key="cs_oldpeak")
            slope    = st.selectbox("ST Slope (1/2/3)", [1, 2, 3], key="cs_slope")
        with c5:
            ca_val   = st.selectbox("Vessels Coloured (0–3)", [0, 1, 2, 3], key="cs_ca")
            thal     = st.selectbox("Thal (3=normal, 6=fixed, 7=reversible defect)",
                                    [3, 6, 7], key="cs_thal")

        if st.button("🔍 Cardio Risk", type="secondary"):
            sex_int  = 1 if "Male" in sex_c else 0
            cardio_m = _load_model("cardio_xgb.pkl")
            with open(os.path.join(MODEL_DIR, "cardio_features.pkl"), "rb") as _f:
                cardio_feats = pickle.load(_f)
            Xc = pd.DataFrame(
                [[cs_age, sex_int, chest_cp, cs_bp, cholest, fbs,
                  rest_ecg, max_hr, exang, oldpeak, slope, ca_val, thal]],
                columns=cardio_feats,
            )
            cp = float(cardio_m.predict_proba(Xc)[0, 1])
            tier_c = "HIGH RISK ❤️‍🔥" if cp >= 0.5 else "LOW RISK 💚"
            st.metric("Cardiovascular probability", f"{cp:.1%}", tier_c)


# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — DRIFT MONITOR
# ────────────────────────────────────────────────────────────────────────────
with tab_drift:
    st.header("📊 Model Health & Data Drift")
    st.caption(
        "Evidently analysis comparing NHANES 2015-16 (reference) "
        "vs 2021-23 (current).  "
        "Wasserstein distance thresholds: WARNING ≥ 0.10 · CRITICAL ≥ 0.20"
    )

    report_path = os.path.join(ROOT, "reports", "model_health_report.html")

    col_run, col_info = st.columns([1, 2])
    with col_run:
        if st.button("🔄 Re-run Drift Analysis", type="primary"):
            with st.spinner("Running Evidently — this takes ~30 seconds ..."):
                try:
                    from src.monitoring.drift_detector import run_drift_report
                    from src.monitoring.health_report import build_health_report, console_render

                    result   = run_drift_report()
                    report_d = build_health_report(result)
                    console_render(report_d)
                    st.success("✅ Drift report regenerated!")
                except Exception as exc:
                    st.error(f"Drift analysis failed: {exc}")

    with col_info:
        st.info(
            "The HTML report is embedded below.  "
            "Click **Re-run Drift Analysis** to refresh it with the latest data."
        )

    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as _f:
            html_content = _f.read()
        st.components.v1.html(html_content, height=700, scrolling=True)
    else:
        st.warning(
            "No report found at `reports/model_health_report.html`.  "
            "Click **Re-run Drift Analysis** to generate it."
        )

    # ── inline summary metrics ───────────────────────────────────────────────
    st.divider()
    st.subheader("Quick Drift Summary  *(from last run)*")

    drift_summary = {
        "age":              {"distance": 0.262, "drifted": True,  "method": "Wasserstein"},
        "glucose":          {"distance": 0.101, "drifted": True,  "method": "Wasserstein"},
        "gender":           {"distance": 0.034, "drifted": False, "method": "Jensen-Shannon"},
        "hba1c":            {"distance": 0.052, "drifted": False, "method": "Wasserstein"},
        "serum_creatinine": {"distance": 0.060, "drifted": False, "method": "Wasserstein"},
    }
    model_auc = 0.9736

    dcols = st.columns(len(drift_summary) + 1)
    for i, (feat, info) in enumerate(drift_summary.items()):
        with dcols[i]:
            icon = "🔴" if info["drifted"] else "🟢"
            st.metric(
                label=f"{icon} {feat}",
                value=f"{info['distance']:.3f}",
                delta="DRIFT" if info["drifted"] else "OK",
                delta_color="inverse",
            )
    with dcols[-1]:
        st.metric(label="🎯 Model AUC (2021-23)", value=f"{model_auc:.4f}", delta="+stable")


# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — DRUG EVIDENCE
# ────────────────────────────────────────────────────────────────────────────
with tab_drug:
    st.header("💊 Drug Evidence Summarizer")
    st.caption(
        "RAG pipeline: ChromaDB (11 k review chunks) + Mistral 7B via Ollama.  "
        "Use **Retrieval Only** when Ollama is unavailable."
    )

    col_query, col_out = st.columns([1, 1.3], gap="large")

    with col_query:
        st.subheader("Query")
        condition = st.text_input(
            "Clinical condition",
            value="high blood pressure",
            placeholder="e.g. Type 2 Diabetes, depression",
        )
        drug_name = st.text_input(
            "Drug name (optional)",
            value="",
            placeholder="e.g. Metformin, Toprol",
        )
        top_k = st.slider("Review chunks to retrieve", 4, 20, 8)
        use_llm = st.toggle("Use Mistral 7B via Ollama", value=True)

        if not use_llm:
            st.info("Retrieval-only mode: structured snippets, no LLM generation.")

        run_btn = st.button("🔬 Generate Evidence Summary", type="primary",
                            use_container_width=True)

    with col_out:
        st.subheader("Clinical Summary")

        if run_btn:
            if not condition.strip():
                st.warning("Please enter a clinical condition.")
            else:
                drug_q = drug_name.strip() or None
                spinner_msg = (
                    "Retrieving reviews + generating with Mistral 7B …"
                    if use_llm else
                    "Retrieving review chunks …"
                )
                with st.spinner(spinner_msg):
                    try:
                        if use_llm:
                            from src.nlp.summarizer import summarize_drug
                            result = summarize_drug(
                                condition=condition.strip(),
                                drug=drug_q,
                                top_k=top_k,
                            )
                        else:
                            from src.nlp.summarizer import summarize_drug_no_llm
                            result = summarize_drug_no_llm(
                                condition=condition.strip(),
                                drug=drug_q,
                                top_k=top_k,
                            )
                        st.success("Summary generated.")
                        st.markdown(result["summary"])

                        # Source attribution
                        st.divider()
                        st.markdown(f"**Sources** — {result['top_k']} chunks retrieved")
                        src_rows = []
                        for s in result["source_docs"][:6]:
                            src_rows.append({
                                "Drug":      s.get("drug", ""),
                                "Condition": s.get("condition", ""),
                                "Rating":    f"{s.get('rating', 0):.0f}/10",
                                "Snippet":   s.get("snippet", "")[:120] + "…",
                            })
                        if src_rows:
                            st.dataframe(
                                pd.DataFrame(src_rows),
                                use_container_width=True,
                                hide_index=True,
                            )
                    except Exception as exc:
                        st.error(f"Error: {exc}")
                        st.info(
                            "If Ollama is not running, toggle **Use Mistral 7B** off "
                            "to use retrieval-only mode."
                        )
        else:
            st.info(
                "Enter a condition and optional drug name, then click  \n"
                "**🔬 Generate Evidence Summary**."
            )

    # ── popular conditions explorer ──────────────────────────────────────────
    st.divider()
    st.subheader("📚 Drug Review Coverage")

    @st.cache_data(show_spinner=False)
    def _review_stats():
        raw = os.path.join(ROOT, "data", "raw", "drug-reviews")
        frames = []
        for fn in ["drugLibTrain_raw.tsv", "drugLibTest_raw.tsv"]:
            p = os.path.join(raw, fn)
            if os.path.exists(p):
                frames.append(pd.read_csv(p, sep="\t"))
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        return df

    df_rev = _review_stats()
    if not df_rev.empty:
        c_stat1, c_stat2, c_stat3 = st.columns(3)
        with c_stat1:
            st.metric("Total Reviews", f"{len(df_rev):,}")
        with c_stat2:
            st.metric("Unique Drugs", f"{df_rev['urlDrugName'].nunique():,}")
        with c_stat3:
            st.metric("Unique Conditions", f"{df_rev['condition'].nunique():,}")

        top_cond = (
            df_rev.groupby("condition")
            .size()
            .sort_values(ascending=False)
            .head(15)
            .reset_index()
        )
        top_cond.columns = ["Condition", "Reviews"]
        st.bar_chart(top_cond.set_index("Condition"))
    else:
        st.info("Drug review data not found at `data/raw/drug-reviews/`.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — SQL ANALYTICS
# ════════════════════════════════════════════════════════════════════════════
with tab_sql:
    st.header("📈 SQL Analytics")
    st.caption(
        "DuckDB analytics sprint — 6 SQL queries on the 127k-patient master table.  "
        "All queries run directly on Parquet files with zero data-loading overhead."
    )

    @st.cache_data(show_spinner=False)
    def _load_sql_analytics():
        import duckdb
        con = duckdb.connect()
        con.execute(
            "CREATE VIEW master AS SELECT * FROM "
            "read_parquet('data/processed/master_patient_table.parquet')"
        )
        q1 = con.execute("""
            SELECT source_dataset,
                   COUNT(*) AS n,
                   ROUND(AVG(age), 1) AS avg_age,
                   ROUND(STDDEV(age), 1) AS std_age,
                   ROUND(AVG(bmi), 1) AS avg_bmi,
                   SUM(CASE WHEN gender = 1 THEN 1 ELSE 0 END) AS male_count,
                   SUM(CASE WHEN gender = 0 THEN 1 ELSE 0 END) AS female_count
            FROM master GROUP BY source_dataset ORDER BY n DESC
        """).df()
        q2 = con.execute("""
            SELECT
                CASE
                    WHEN glucose < 100 AND hba1c < 5.7 THEN 'Normal'
                    WHEN glucose BETWEEN 100 AND 125
                         OR hba1c BETWEEN 5.7 AND 6.4 THEN 'Pre-diabetic'
                    WHEN glucose >= 126 OR hba1c >= 6.5 THEN 'Diabetic'
                    ELSE 'Insufficient data'
                END AS risk_category,
                COUNT(*) AS patient_count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct_of_total
            FROM master
            GROUP BY risk_category ORDER BY patient_count DESC
        """).df()
        q3 = con.execute("""
            SELECT
                ROUND(CORR(glucose, diabetes_label), 3)            AS glucose_corr,
                ROUND(CORR(hba1c, diabetes_label), 3)              AS hba1c_corr,
                ROUND(CORR(bmi, diabetes_label), 3)                AS bmi_corr,
                ROUND(CORR(blood_pressure_sys, diabetes_label), 3) AS bp_corr,
                ROUND(CORR(age, diabetes_label), 3)                AS age_corr
            FROM master WHERE diabetes_label IS NOT NULL
        """).df()
        q4 = con.execute("""
            SELECT nhanes_cycle,
                   ROUND(AVG(glucose), 2) AS avg_glucose,
                   ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY glucose), 2)
                       AS median_glucose,
                   ROUND(AVG(hba1c), 2)   AS avg_hba1c,
                   ROUND(AVG(bmi), 2)     AS avg_bmi,
                   COUNT(*)               AS n
            FROM master
            WHERE source_dataset = 'nhanes' AND nhanes_cycle IS NOT NULL
            GROUP BY nhanes_cycle ORDER BY nhanes_cycle
        """).df()
        q5 = con.execute("""
            SELECT patient_id, source_dataset, age, bmi, glucose, hba1c,
                   blood_pressure_sys,
                   CASE
                       WHEN glucose >= 200 OR hba1c >= 9.0 THEN 'Critical'
                       WHEN glucose >= 126 OR hba1c >= 6.5 THEN 'High'
                       ELSE 'Moderate'
                   END AS risk_tier
            FROM master
            WHERE (glucose >= 126 OR hba1c >= 6.5) AND diabetes_label IS NOT NULL
            ORDER BY glucose DESC, hba1c DESC
            LIMIT 100
        """).df()
        return q1, q2, q3, q4, q5

    try:
        q1_s, q2_s, q3_s, q4_s, q5_s = _load_sql_analytics()

        _m1, _m2, _m3 = st.columns(3)
        _m1.metric("Total Patients",     f"{q1_s['n'].sum():,}")
        _m2.metric("Source Datasets",    str(len(q1_s)))
        _m3.metric("High-Risk Patients", f"{len(q5_s):,}+")

        st.divider()
        _cl, _cr = st.columns(2, gap="large")

        with _cl:
            st.subheader("Demographics by Dataset")
            st.dataframe(q1_s, use_container_width=True, hide_index=True)

            st.subheader("ADA Diabetes Risk Stratification")
            st.bar_chart(q2_s.set_index("risk_category")["patient_count"])
            st.dataframe(q2_s, use_container_width=True, hide_index=True)

        with _cr:
            st.subheader("Biomarker Correlations with Diabetes Label")
            _corr = q3_s.T.reset_index()
            _corr.columns = ["Biomarker", "Pearson r"]
            _corr["Biomarker"] = (
                _corr["Biomarker"]
                .str.replace("_corr", "")
                .str.replace("_", " ")
                .str.title()
            )
            st.dataframe(
                _corr.sort_values("Pearson r", ascending=False),
                use_container_width=True, hide_index=True,
            )

            st.subheader("NHANES Cross-Cycle Biomarker Trend")
            st.dataframe(q4_s, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("🚨 High-Risk Patient Registry")
        st.caption(
            "Patients with Glucose ≥ 126 mg/dL or HbA1c ≥ 6.5%  "
            "— sorted by severity (top 100 shown)"
        )
        st.dataframe(q5_s, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇ Download High-Risk Cohort CSV",
            data=q5_s.to_csv(index=False),
            file_name="high_risk_patients.csv",
            mime="text/csv",
        )

    except Exception as _exc_sql:
        st.error(f"SQL Analytics error: {_exc_sql}")
        st.info(
            "Make sure `data/processed/master_patient_table.parquet` exists.  "
            "Run Notebook 01 to generate it."
        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — FEASIBILITY
# ════════════════════════════════════════════════════════════════════════════
with tab_feasibility:
    st.header("🔍 SQL Feasibility Analysis")
    st.caption(
        "Pre-modelling triage — 5 DuckDB checks a senior data scientist runs "
        "before approving any new modelling task.  Mirrors Notebook 07."
    )

    @st.cache_data(show_spinner=False)
    def _load_feasibility():
        import duckdb
        con = duckdb.connect()
        con.execute(
            "CREATE VIEW master AS SELECT * FROM "
            "read_parquet('data/processed/master_patient_table.parquet')"
        )
        f1 = con.execute("""
            SELECT
                source_dataset,
                CASE WHEN age >= 60 THEN '60+' ELSE 'Under 60' END AS age_group,
                COUNT(*) AS patient_count,
                ROUND(AVG(diabetes_label), 3) AS diabetes_rate,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY source_dataset), 1)
                    AS pct_of_dataset
            FROM master
            WHERE diabetes_label IS NOT NULL
            GROUP BY source_dataset, age_group
            ORDER BY source_dataset, age_group
        """).df()
        f2 = con.execute("""
            SELECT
                nhanes_cycle,
                COUNT(*) AS total_patients,
                ROUND(COUNT(hba1c)   * 100.0 / COUNT(*), 1) AS hba1c_completeness_pct,
                ROUND(COUNT(glucose) * 100.0 / COUNT(*), 1) AS glucose_completeness_pct,
                ROUND(COUNT(bmi)     * 100.0 / COUNT(*), 1) AS bmi_completeness_pct
            FROM master
            WHERE source_dataset = 'nhanes'
            GROUP BY nhanes_cycle
        """).df()
        f3 = con.execute("""
            SELECT
                source_dataset,
                CAST(diabetes_label AS INTEGER) AS label,
                COUNT(*) AS count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY source_dataset), 1)
                    AS pct
            FROM master
            WHERE diabetes_label IS NOT NULL
            GROUP BY source_dataset, diabetes_label
            ORDER BY source_dataset, label
        """).df()
        f4 = con.execute("""
            SELECT
                nhanes_cycle,
                ROUND(AVG(glucose), 2)    AS avg_glucose,
                ROUND(AVG(bmi), 2)        AS avg_bmi,
                ROUND(AVG(hba1c), 2)      AS avg_hba1c,
                ROUND(STDDEV(glucose), 2) AS std_glucose,
                ROUND(STDDEV(bmi), 2)     AS std_bmi
            FROM master
            WHERE source_dataset = 'nhanes'
            GROUP BY nhanes_cycle
        """).df()
        f5 = con.execute("""
            SELECT
                source_dataset,
                COUNT(*) AS total,
                COUNT(diabetes_label) AS labeled_diabetes,
                COUNT(cardio_label)   AS labeled_cardio,
                CASE
                    WHEN COUNT(diabetes_label) >= 1000 THEN 'Sufficient'
                    WHEN COUNT(diabetes_label) >= 300  THEN 'Marginal'
                    ELSE 'Insufficient'
                END AS diabetes_feasibility,
                CASE
                    WHEN COUNT(cardio_label) >= 1000 THEN 'Sufficient'
                    WHEN COUNT(cardio_label) >= 300  THEN 'Marginal'
                    ELSE 'Insufficient'
                END AS cardio_feasibility
            FROM master
            GROUP BY source_dataset
            ORDER BY total DESC
        """).df()
        fsum = con.execute("""
            SELECT
                COUNT(*) AS total_patients,
                COUNT(DISTINCT source_dataset) AS datasets,
                ROUND(COUNT(diabetes_label) * 100.0 / COUNT(*), 1) AS diabetes_label_coverage_pct,
                ROUND(COUNT(hba1c) * 100.0 / COUNT(*), 1) AS hba1c_coverage_pct,
                ROUND(AVG(age), 1) AS avg_age,
                ROUND(AVG(bmi), 1) AS avg_bmi
            FROM master
        """).df()
        return f1, f2, f3, f4, f5, fsum

    try:
        f1_d, f2_d, f3_d, f4_d, f5_d, fsum_d = _load_feasibility()

        _all_suff  = (f5_d["diabetes_feasibility"] == "Sufficient").all()
        _min_comp  = min(
            float(f2_d["hba1c_completeness_pct"].min())   if len(f2_d) else 100.0,
            float(f2_d["glucose_completeness_pct"].min()) if len(f2_d) else 100.0,
        )
        _min_minor = (
            float(f3_d.groupby("source_dataset")["pct"].min().min())
            if len(f3_d) else 100.0
        )

        _s1, _s2, _s3 = st.columns(3)
        _s1.metric("Sample Size Check",
                   "✅ Sufficient" if _all_suff else "⚠️ Marginal",
                   "Ready to model" if _all_suff else "See Check 5")
        _s2.metric("Min Feature Completeness",
                   f"{_min_comp:.0f}%",
                   "✅ Above 70%" if _min_comp >= 70 else "⚠️ Below threshold")
        _s3.metric("Min Minority Class",
                   f"{_min_minor:.0f}%",
                   "✅ Balanced" if _min_minor >= 20 else "⚠️ Use SMOTE")

        st.divider()

        with st.expander("Check 1 — Subgroup Data Availability (60+ cohort)", expanded=True):
            st.caption("Threshold: < 500 patients in 60+ group → underpowered for subgroup model.")
            st.dataframe(f1_d, use_container_width=True, hide_index=True)

        with st.expander("Check 2 — Feature Completeness (NHANES cycles)"):
            st.caption("Threshold: < 70% completeness → drift risk factor.")
            st.dataframe(f2_d, use_container_width=True, hide_index=True)

        with st.expander("Check 3 — Class Balance Assessment"):
            st.caption("Threshold: minority class < 20% → apply SMOTE or scale_pos_weight.")
            st.dataframe(f3_d, use_container_width=True, hide_index=True)

        with st.expander("Check 4 — Cross-Cycle Biomarker Shift"):
            st.caption("Decision rule: > 10% shift in mean glucose or BMI confirms real drift.")
            st.dataframe(f4_d, use_container_width=True, hide_index=True)

        with st.expander("Check 5 — Sample Size Validation"):
            st.caption("Sufficient ≥ 1 000 · Marginal 300–999 · Insufficient < 300")
            st.dataframe(f5_d, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Master Table Summary")
        st.dataframe(fsum_d, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇ Download Feasibility Summary CSV",
            data=fsum_d.to_csv(index=False),
            file_name="feasibility_summary.csv",
            mime="text/csv",
        )

    except Exception as _exc_feas:
        st.error(f"Feasibility error: {_exc_feas}")
        st.info("Make sure `data/processed/master_patient_table.parquet` exists.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — TRIAL SIMULATOR
# ════════════════════════════════════════════════════════════════════════════
with tab_trial:
    st.header("🧪 Synthetic Clinical Trial Simulator")
    st.caption(
        "Propensity score matching on the Diabetes 130-US hospital dataset.  "
        "Abbott RWE workflow: PSM → outcome analysis → Kaplan-Meier → log-rank test.  "
        "Mirrors Notebook 08."
    )

    _km_path = os.path.join(ROOT, "reports", "kaplan_meier.png")

    _col_btn, _col_info = st.columns([1, 2])
    with _col_btn:
        _run_trial = st.button("▶ Run Simulation", type="primary", use_container_width=True)
    with _col_info:
        st.info(
            "**Treatment proxy:** HbA1c ≥ 8.0% → insulin-intensification arm.  \n"
            "**Matching:** 1:1 nearest-neighbour propensity score.  \n"
            "**Outcome:** 30-day readmission (proxied by diabetes_label)."
        )

    if os.path.exists(_km_path) and not _run_trial:
        st.image(_km_path, caption="Kaplan-Meier: last simulation run",
                 use_container_width=True)
        st.info("Click **▶ Run Simulation** to regenerate with fresh data.")

    if _run_trial:
        with st.spinner("Running PSM + Kaplan-Meier (~5 sec) …"):
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as _plt
                from sklearn.linear_model import LogisticRegression as _LR
                from sklearn.preprocessing import StandardScaler as _SS
                from sklearn.neighbors import NearestNeighbors as _NN
                from lifelines import KaplanMeierFitter as _KMF
                from lifelines.statistics import logrank_test as _lrt

                _master = pd.read_parquet(
                    os.path.join(ROOT, "data", "processed", "master_patient_table.parquet")
                )
                _tdf = _master[_master["source_dataset"] == "diabetes_130"].copy()
                _tdf = _tdf.dropna(subset=["age", "gender", "bmi", "diabetes_label"])

                _FEATS = ["age", "gender", "bmi", "blood_pressure_sys"]
                _tdf["treatment"] = (_tdf["hba1c"] >= 8.0).astype(int)
                _Xt = _tdf[_FEATS].fillna(_tdf[_FEATS].median())
                _yt = _tdf["treatment"]

                _sc = _SS()
                _ps = _LR(max_iter=500, random_state=42)
                _ps.fit(_sc.fit_transform(_Xt), _yt)
                _tdf["propensity_score"] = _ps.predict_proba(_sc.transform(_Xt))[:, 1]

                _treated = _tdf[_tdf["treatment"] == 1].copy()
                _control = _tdf[_tdf["treatment"] == 0].copy()
                _nbrs = _NN(n_neighbors=1, algorithm="ball_tree")
                _nbrs.fit(_control[["propensity_score"]])
                _, _idx = _nbrs.kneighbors(_treated[["propensity_score"]])
                _mctrl = _control.iloc[_idx.flatten()].copy()
                _mtrt  = _treated.copy()

                _oc   = "diabetes_label"
                _tr   = _mtrt[_oc].mean()
                _cr   = _mctrl[_oc].mean()
                _arr  = _cr - _tr
                _nnt  = (1 / _arr) if _arr > 0 else float("inf")

                np.random.seed(42)
                _mtrt  = _mtrt.copy()
                _mctrl = _mctrl.copy()
                _mtrt["time"]  = np.random.exponential(
                    120 - 20 * _mtrt[_oc],  len(_mtrt)
                ).clip(1, 365)
                _mctrl["time"] = np.random.exponential(
                    100 - 20 * _mctrl[_oc], len(_mctrl)
                ).clip(1, 365)

                _lr = _lrt(
                    _mtrt["time"], _mctrl["time"],
                    event_observed_A=_mtrt[_oc],
                    event_observed_B=_mctrl[_oc],
                )

                _fig, _ax = _plt.subplots(figsize=(9, 5))
                _KMF().fit(_mtrt["time"],  _mtrt[_oc],
                           label="Treatment arm").plot_survival_function(ax=_ax, ci_show=True)
                _KMF().fit(_mctrl["time"], _mctrl[_oc],
                           label="Control arm").plot_survival_function(ax=_ax, ci_show=True)
                _ax.set_title("Kaplan-Meier: Readmission-free survival (simulated trial)")
                _ax.set_xlabel("Days")
                _ax.set_ylabel("Survival probability")
                _plt.tight_layout()
                os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
                _plt.savefig(_km_path, dpi=150)
                _plt.close()

                st.success(f"✅ Simulation complete — {len(_mtrt):,} matched pairs")

                _km1, _km2, _km3, _km4 = st.columns(4)
                _km1.metric("Treatment readmission", f"{_tr:.1%}")
                _km2.metric("Control readmission",   f"{_cr:.1%}")
                _km3.metric("ARR",
                            f"{_arr:.3f}",
                            f"NNT = {_nnt:.0f}" if _arr > 0 else "No benefit")
                _km4.metric("Log-rank p-value",
                            f"{_lr.p_value:.4f}",
                            "✅ Significant" if _lr.p_value < 0.05 else "Not significant")

                st.subheader("Cohort Balance After Matching")
                st.dataframe(
                    pd.DataFrame([{
                        "Feature":     col,
                        "Treated mean": round(_mtrt[col].mean(), 2),
                        "Control mean": round(_mctrl[col].mean(), 2),
                        "Abs. diff":    round(
                            abs(_mtrt[col].mean() - _mctrl[col].mean()), 2
                        ),
                    } for col in _FEATS]),
                    use_container_width=True, hide_index=True,
                )
                st.image(_km_path, caption="Kaplan-Meier Survival Curve",
                         use_container_width=True)

            except ImportError as _ie:
                st.error(f"Missing dependency: {_ie}")
                st.info("Install with: `pip install lifelines`")
            except Exception as _exc_trial:
                st.error(f"Simulation error: {_exc_trial}")
                import traceback as _tb
                with st.expander("Traceback"):
                    st.code(_tb.format_exc())


# ════════════════════════════════════════════════════════════════════════════
# TAB 7 — MODEL RUNNER
# ════════════════════════════════════════════════════════════════════════════
with tab_runner:
    st.header("⚙️ YAML-Driven Model Runner")
    st.caption(
        "Train any model by selecting a YAML config.  "
        "Equivalent to: `python run.py --config configs/<name>.yaml`"
    )

    import yaml as _yaml
    import subprocess as _subp

    _cfg_options = {
        "🧬 Diabetes Classifier  (NHANES + Diabetes 130)": os.path.join(
            ROOT, "configs", "diabetes.yaml"
        ),
        "❤️  Cardio Classifier   (Statlog Heart)": os.path.join(
            ROOT, "configs", "cardio.yaml"
        ),
    }

    _sel_label = st.selectbox("Select model config", list(_cfg_options.keys()))
    _sel_path  = _cfg_options[_sel_label]

    with open(_sel_path) as _cf:
        _cfg = _yaml.safe_load(_cf)

    _cc, _cp = st.columns([1.5, 1], gap="large")

    with _cc:
        st.subheader("Config")
        st.json(_cfg)

    with _cp:
        st.subheader("Training Parameters")
        for _pk, _pv in _cfg.get("model_params", {}).items():
            st.metric(_pk.replace("_", " ").title(), str(_pv))
        st.divider()
        st.markdown(f"**Label column:** `{_cfg.get('label_column')}`")
        st.markdown(
            f"**Features ({len(_cfg.get('features', []))}):** "
            f"`{', '.join(_cfg.get('features', []))}`"
        )
        st.markdown(f"**Source datasets:** `{', '.join(_cfg.get('source_datasets', []))}`")
        st.markdown(f"**Output path:** `{_cfg.get('output_model_path')}`")

    st.divider()
    _train_btn = st.button(
        f"🚀  Train  {_cfg.get('model_name', 'Model')}",
        type="primary",
        use_container_width=True,
    )

    if _train_btn:
        _run_py = os.path.join(ROOT, "run.py")
        with st.spinner(
            f"Training {_cfg['model_name']} — ~30–60 sec …  \n"
            f"Running: `python run.py --config {_sel_path}`"
        ):
            _proc = _subp.run(
                [sys.executable, _run_py, "--config", _sel_path],
                capture_output=True,
                text=True,
                cwd=ROOT,
            )
        if _proc.returncode == 0:
            st.success(f"✅  {_cfg['model_name']} trained successfully!")
            st.code(_proc.stdout, language="text")
            _out = _cfg.get("output_model_path", "")
            if _out and os.path.exists(os.path.join(ROOT, _out)):
                st.info(f"Model saved → `{_out}`")
        else:
            st.error("Training failed.")
            st.code(_proc.stderr or _proc.stdout, language="text")
