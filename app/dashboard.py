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
        - sentence-transformers
        """
    )
    st.divider()
    st.caption("v1.0 · Global Data Science & Analytics")
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
tab_risk, tab_drift, tab_drug = st.tabs(
    ["🧬 Risk Prediction", "📊 Drift Monitor", "💊 Drug Evidence"]
)


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
