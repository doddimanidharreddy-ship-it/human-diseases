"""
Disease Prediction System Using Decision Trees
Interactive Multi-Disease Clinical Screening & Explainable Machine Learning Web Application
10214CS601 - Minor Project - I | Vel Tech R&D Institute of Science & Technology
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

import explainability
import database

# Configure Streamlit Page
st.set_page_config(
    page_title="Disease Prediction System | Decision Trees",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Medical Theme Styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f172a, #1e3a8a, #0284c7);
        color: white;
        padding: 22px 28px;
        border-radius: 14px;
        margin-bottom: 25px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.12);
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    .main-header p {
        color: #e0f2fe;
        margin: 6px 0 0 0;
        font-size: 1.05rem;
    }
    .badge-sub {
        display: inline-block;
        background: rgba(255, 255, 255, 0.18);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        margin-top: 8px;
        border: 1px solid rgba(255, 255, 255, 0.25);
    }
    .metric-card {
        background: white;
        padding: 15px 20px;
        border-radius: 10px;
        border-left: 5px solid #0284c7;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 15px;
    }
    .rule-box {
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        border-left: 5px solid #2563eb;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 8px;
        font-family: Consolas, monospace;
        font-size: 0.95rem;
    }
    .risk-high {
        background-color: #fee2e2;
        color: #991b1b;
        border-left: 6px solid #dc2626;
        padding: 16px;
        border-radius: 8px;
        font-weight: 600;
        margin-bottom: 15px;
    }
    .risk-low {
        background-color: #dcfce7;
        color: #166534;
        border-left: 6px solid #16a34a;
        padding: 16px;
        border-radius: 8px;
        font-weight: 600;
        margin-bottom: 15px;
    }
    .risk-warning {
        background-color: #fef3c7;
        color: #92400e;
        border-left: 6px solid #f59e0b;
        padding: 16px;
        border-radius: 8px;
        font-weight: 600;
        margin-bottom: 15px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f5f9;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0284c7 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(disease_name):
    model_path = os.path.join(os.path.dirname(__file__), "models", f"{disease_name}_dt_model.joblib")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_data
def load_metrics():
    metrics_path = os.path.join(os.path.dirname(__file__), "models", "metrics_summary.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    return {}

def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>🩺 Multi-Disease Prediction System Using Decision Trees</h1>
        <p>Comprehensive & Explainable Clinical Decision Support System (9 Disease Modules)</p>
        <span class="badge-sub">Batch: MI1301 | Vel Tech R&D Institute of Science & Technology | 10214CS601 Minor Project - I</span>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "veltech_logo.png")
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, use_container_width=True)
    else:
        st.sidebar.image("https://img.icons8.com/color/96/caduceus.png", width=80)
    
    st.sidebar.title("Disease Navigation")
    menu = st.sidebar.radio(
        "Select Screening Module:",
        [
            "🩸 Diabetes Screening",
            "❤️ Heart Disease Screening",
            "🧪 Kidney Disease (CKD)",
            "🧬 Liver Disease Screening",
            "🎗️ Breast Cancer (Malignancy)",
            "🧠 Stroke Risk Assessment",
            "🔊 Parkinson's (Vocal Biomarkers)",
            "🩺 Hypertension & BP Staging",
            "🌡️ Symptom-Based Classifier (16 Conditions)",
            "📊 Model Evaluation & Benchmarks",
            "🏛️ Review 1 & Architecture",
            "📜 Screening History"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Team MI1301")
    st.sidebar.markdown("""
    - **Bugga Gangadhar Kali Prasad** (VTU 28564)
    - **Allala Bhavana** (VTU 27999)
    - **Godi Mouni Pranathi** (VTU 28298)
    
    **Supervisor:**
    Dr. J. Raja (Associate Professor)
    """)
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Explainable AI:** Uses Scikit-Learn Decision Trees to provide transparent, verifiable clinical decision paths.")
    return menu

# ==================== 1. DIABETES SCREENING ====================
def screen_diabetes():
    st.header("🩸 Diabetes Risk Screening (Decision Tree)")
    st.caption("Predicts likelihood of diabetes mellitus from physiological biomarkers and glycemic indicators.")

    pack = load_model("diabetes")
    if not pack:
        st.error("Diabetes model not found. Please train models first.")
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Patient Clinical Data")
        patient_name = st.text_input("Patient Identifier / Name", "Patient-D01")
        age = st.slider("Age (Years)", 21, 90, 46)
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=17, value=1)
        glucose = st.slider("Fasting Blood Glucose (mg/dL)", 60, 260, 138, help="Normal < 100 mg/dL, Pre-diabetes 100-125, Diabetes >= 126")
        blood_pressure = st.slider("Diastolic Blood Pressure (mm Hg)", 40, 140, 78)
        skin_thickness = st.slider("Triceps Skin Fold Thickness (mm)", 8, 60, 29)
        insulin = st.slider("2-Hour Serum Insulin (mu U/ml)", 15, 500, 105)
        bmi = st.slider("Body Mass Index (BMI kg/m²)", 15.0, 55.0, 31.8, step=0.1)
        pedigree = st.number_input("Diabetes Pedigree Function (Genetic History)", 0.05, 2.50, 0.48, step=0.01)

        predict_btn = st.button("Run Diabetes Prediction", type="primary", use_container_width=True)

    with col2:
        if predict_btn:
            input_df = pd.DataFrame([{
                "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": blood_pressure,
                "SkinThickness": skin_thickness, "Insulin": insulin, "BMI": bmi,
                "DiabetesPedigreeFunction": pedigree, "Age": age
            }])
            model = pack["model"]
            pred_idx = int(model.predict(input_df)[0])
            probs = model.predict_proba(input_df)[0]
            conf = probs[pred_idx] * 100
            label = pack["label_names"][pred_idx]

            guidance = explainability.calculate_risk_and_guidance("diabetes", pred_idx, probs, pack["label_names"])
            rules = explainability.extract_decision_path(model, input_df, pack["features"])

            database.save_prediction(patient_name, "Diabetes", f"Age:{age}, Glu:{glucose}, BMI:{bmi}", label, guidance["risk_level"], conf)

            st.subheader("Prediction Result & Clinical Assessment")
            if pred_idx == 1:
                st.markdown(f"""
                <div class="risk-high">
                    <h3>⚠️ {label}</h3>
                    <p>Estimated Probability: <b>{conf:.1f}%</b> | Risk Category: <b>{guidance['risk_level']}</b></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <h3>✅ {label}</h3>
                    <p>Estimated Confidence: <b>{conf:.1f}%</b> | Risk Category: <b>{guidance['risk_level']}</b></p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### 🔍 Explainable Decision Path (Tree Traversal)")
            for rule in rules:
                st.markdown(f"""<div class="rule-box"><b>Step {rule['step']}:</b> If <code>{rule['rule']}</code> &rarr; <i>{rule['description']}</i></div>""", unsafe_allow_html=True)

            st.markdown("#### 📋 Recommended Actions & Medical Guidance")
            for rec in guidance["recommendations"]:
                st.markdown(f"- {rec}")
            st.warning(guidance["disclaimer"])

# ==================== 2. HEART DISEASE SCREENING ====================
def screen_heart():
    st.header("❤️ Heart Disease Risk Screening (Decision Tree)")
    st.caption("Assesses cardiovascular health indicators to determine coronary artery disease likelihood.")

    pack = load_model("heart")
    if not pack:
        st.error("Heart Disease model not found.")
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Cardiovascular Parameters")
        patient_name = st.text_input("Patient Identifier / Name", "Patient-H01")
        age = st.slider("Age (Years)", 25, 80, 58)
        sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
        chest_pain = st.selectbox(
            "Chest Pain Type",
            options=[("0: Typical Angina", 0), ("1: Atypical Angina", 1), ("2: Non-Anginal Pain", 2), ("3: Asymptomatic", 3)],
            format_func=lambda x: x[0]
        )[1]
        resting_bp = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 142)
        cholesterol = st.slider("Serum Cholesterol (mg/dL)", 120, 520, 270)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", options=[("False (<= 120)", 0), ("True (> 120)", 1)], format_func=lambda x: x[0])[1]
        resting_ecg = st.selectbox("Resting Electrocardiogram (ECG)", options=[("0: Normal", 0), ("1: ST-T Wave Abnormality", 1), ("2: Left Ventricular Hypertrophy", 2)], format_func=lambda x: x[0])[1]
        max_hr = st.slider("Maximum Heart Rate Achieved (bpm)", 70, 210, 136)
        exercise_angina = st.selectbox("Exercise Induced Angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        oldpeak = st.slider("ST Depression Induced by Exercise (Oldpeak)", 0.0, 6.2, 2.1, step=0.1)
        st_slope = st.selectbox("Slope of the Peak Exercise ST Segment", options=[("0: Upsloping", 0), ("1: Flat", 1), ("2: Downsloping", 2)], format_func=lambda x: x[0])[1]

        predict_btn = st.button("Run Heart Disease Prediction", type="primary", use_container_width=True)

    with col2:
        if predict_btn:
            input_df = pd.DataFrame([{
                "Age": age, "Sex": sex, "ChestPainType": chest_pain, "RestingBP": resting_bp,
                "Cholesterol": cholesterol, "FastingBS": fasting_bs, "RestingECG": resting_ecg,
                "MaxHR": max_hr, "ExerciseAngina": exercise_angina, "Oldpeak": oldpeak, "ST_Slope": st_slope
            }])
            model = pack["model"]
            pred_idx = int(model.predict(input_df)[0])
            probs = model.predict_proba(input_df)[0]
            conf = probs[pred_idx] * 100
            label = pack["label_names"][pred_idx]

            guidance = explainability.calculate_risk_and_guidance("heart", pred_idx, probs, pack["label_names"])
            rules = explainability.extract_decision_path(model, input_df, pack["features"])

            database.save_prediction(patient_name, "Heart Disease", f"Age:{age}, Chol:{cholesterol}, BP:{resting_bp}", label, guidance["risk_level"], conf)

            st.subheader("Prediction Result & Clinical Assessment")
            if pred_idx == 1:
                st.markdown(f"""
                <div class="risk-high">
                    <h3>⚠️ {label}</h3>
                    <p>Estimated Probability: <b>{conf:.1f}%</b> | Risk Category: <b>{guidance['risk_level']}</b></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <h3>✅ {label}</h3>
                    <p>Estimated Confidence: <b>{conf:.1f}%</b> | Risk Category: <b>{guidance['risk_level']}</b></p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### 🔍 Explainable Decision Path")
            for rule in rules:
                st.markdown(f"""<div class="rule-box"><b>Step {rule['step']}:</b> If <code>{rule['rule']}</code> &rarr; <i>{rule['description']}</i></div>""", unsafe_allow_html=True)

            st.markdown("#### 📋 Cardiologist Guidance & Precautions")
            for rec in guidance["recommendations"]:
                st.markdown(f"- {rec}")
            st.warning(guidance["disclaimer"])

# ==================== 3. KIDNEY DISEASE SCREENING ====================
def screen_kidney():
    st.header("🧪 Chronic Kidney Disease (CKD) Screening")
    st.caption("Analyzes renal biomarkers (Creatinine, Urea, Albumin, Specific Gravity) using Decision Trees.")

    pack = load_model("kidney")
    if not pack:
        st.error("Kidney Disease model not found.")
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Renal Function & Lab Indicators")
        patient_name = st.text_input("Patient Identifier / Name", "Patient-K01")
        age = st.slider("Age (Years)", 18, 85, 54)
        blood_pressure = st.slider("Blood Pressure (mm Hg)", 50, 160, 88)
        specific_gravity = st.selectbox("Urine Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025], index=1)
        albumin = st.slider("Urine Albumin (0 to 5)", 0, 5, 3)
        sugar = st.slider("Urine Sugar (0 to 5)", 0, 5, 0)
        red_blood_cells = st.selectbox("Red Blood Cells in Urine", options=[("Normal", 0), ("Abnormal", 1)], format_func=lambda x: x[0])[1]
        blood_glucose_random = st.slider("Random Blood Glucose (mg/dL)", 60, 450, 160)
        blood_urea = st.slider("Blood Urea (mg/dL)", 10, 250, 82)
        serum_creatinine = st.slider("Serum Creatinine (mg/dL)", 0.4, 12.0, 3.1, step=0.1)
        sodium = st.slider("Serum Sodium (mEq/L)", 110.0, 155.0, 132.0, step=0.5)
        potassium = st.slider("Serum Potassium (mEq/L)", 2.5, 7.5, 5.0, step=0.1)
        hemoglobin = st.slider("Hemoglobin Level (g/dL)", 4.0, 18.0, 9.8, step=0.1)

        predict_btn = st.button("Run Kidney Disease Prediction", type="primary", use_container_width=True)

    with col2:
        if predict_btn:
            input_df = pd.DataFrame([{
                "Age": age, "BloodPressure": blood_pressure, "SpecificGravity": specific_gravity,
                "Albumin": albumin, "Sugar": sugar, "RedBloodCells": red_blood_cells,
                "BloodGlucoseRandom": blood_glucose_random, "BloodUrea": blood_urea,
                "SerumCreatinine": serum_creatinine, "Sodium": sodium, "Potassium": potassium,
                "Hemoglobin": hemoglobin
            }])
            model = pack["model"]
            pred_idx = int(model.predict(input_df)[0])
            probs = model.predict_proba(input_df)[0]
            conf = probs[pred_idx] * 100
            label = pack["label_names"][pred_idx]

            guidance = explainability.calculate_risk_and_guidance("kidney", pred_idx, probs, pack["label_names"])
            rules = explainability.extract_decision_path(model, input_df, pack["features"])

            database.save_prediction(patient_name, "Kidney Disease", f"Age:{age}, Creatinine:{serum_creatinine}, Urea:{blood_urea}", label, guidance["risk_level"], conf)

            st.subheader("Prediction Result & Clinical Assessment")
            if pred_idx == 1:
                st.markdown(f"""
                <div class="risk-high">
                    <h3>⚠️ {label}</h3>
                    <p>Estimated Probability: <b>{conf:.1f}%</b> | Risk Category: <b>{guidance['risk_level']}</b></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <h3>✅ {label}</h3>
                    <p>Estimated Confidence: <b>{conf:.1f}%</b> | Risk Category: <b>{guidance['risk_level']}</b></p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### 🔍 Explainable Decision Path")
            for rule in rules:
                st.markdown(f"""<div class="rule-box"><b>Step {rule['step']}:</b> If <code>{rule['rule']}</code> &rarr; <i>{rule['description']}</i></div>""", unsafe_allow_html=True)

            st.markdown("#### 📋 Nephrologist Guidance & Precautions")
            for rec in guidance["recommendations"]:
                st.markdown(f"- {rec}")
            st.warning(guidance["disclaimer"])

# ==================== 4. LIVER DISEASE SCREENING ====================
def screen_liver():
    st.header("🧬 Liver Disease Risk Screening")
    st.caption("Examines liver function tests (Bilirubin, Enzymes ALT/AST, Albumin) with Decision Tree classifiers.")

    pack = load_model("liver")
    if not pack:
        st.error("Liver Disease model not found.")
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Liver Function Test (LFT) Values")
        patient_name = st.text_input("Patient Identifier / Name", "Patient-L01")
        age = st.slider("Age (Years)", 18, 80, 50)
        gender = st.selectbox("Gender", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
        total_bilirubin = st.slider("Total Bilirubin (mg/dL)", 0.4, 25.0, 3.8, step=0.1)
        direct_bilirubin = st.slider("Direct Bilirubin (mg/dL)", 0.1, 15.0, 1.6, step=0.1)
        alkaline_phosphotase = st.slider("Alkaline Phosphatase - ALP (IU/L)", 60, 1500, 320)
        alt = st.slider("Alamine Aminotransferase - ALT / SGPT (IU/L)", 10, 800, 96)
        ast = st.slider("Aspartate Aminotransferase - AST / SGOT (IU/L)", 10, 800, 110)
        total_proteins = st.slider("Total Proteins (g/dL)", 2.5, 10.0, 6.0, step=0.1)
        albumin = st.slider("Albumin (g/dL)", 1.0, 5.5, 2.6, step=0.1)
        ag_ratio = st.slider("Albumin and Globulin Ratio (A/G)", 0.3, 2.8, 0.75, step=0.01)

        predict_btn = st.button("Run Liver Disease Prediction", type="primary", use_container_width=True)

    with col2:
        if predict_btn:
            input_df = pd.DataFrame([{
                "Age": age, "Gender": gender, "TotalBilirubin": total_bilirubin,
                "DirectBilirubin": direct_bilirubin, "AlkalinePhosphotase": alkaline_phosphotase,
                "AlamineAminotransferase": alt, "AspartateAminotransferase": ast,
                "TotalProteins": total_proteins, "Albumin": albumin, "AlbuminGlobulinRatio": ag_ratio
            }])
            model = pack["model"]
            pred_idx = int(model.predict(input_df)[0])
            probs = model.predict_proba(input_df)[0]
            conf = probs[pred_idx] * 100
            label = pack["label_names"][pred_idx]

            guidance = explainability.calculate_risk_and_guidance("liver", pred_idx, probs, pack["label_names"])
            rules = explainability.extract_decision_path(model, input_df, pack["features"])

            database.save_prediction(patient_name, "Liver Disease", f"Age:{age}, TBil:{total_bilirubin}, ALT:{alt}", label, guidance["risk_level"], conf)

            st.subheader("Prediction Result & Clinical Assessment")
            if pred_idx == 1:
                st.markdown(f"""
                <div class="risk-high">
                    <h3>⚠️ {label}</h3>
                    <p>Estimated Probability: <b>{conf:.1f}%</b> | Risk Category: <b>{guidance['risk_level']}</b></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <h3>✅ {label}</h3>
                    <p>Estimated Confidence: <b>{conf:.1f}%</b> | Risk Category: <b>{guidance['risk_level']}</b></p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### 🔍 Explainable Decision Path")
            for rule in rules:
                st.markdown(f"""<div class="rule-box"><b>Step {rule['step']}:</b> If <code>{rule['rule']}</code> &rarr; <i>{rule['description']}</i></div>""", unsafe_allow_html=True)

            st.markdown("#### 📋 Hepatology Guidance & Dietary Precautions")
            for rec in guidance["recommendations"]:
                st.markdown(f"- {rec}")
            st.warning(guidance["disclaimer"])

# ==================== 5. BREAST CANCER SCREENING ====================
def screen_breast_cancer():
    st.header("🎗️ Breast Cancer Malignancy Risk Screening")
    st.caption("Evaluates cellular and tissue morphological attributes (Wisconsin Diagnostic Benchmark) using Decision Trees.")

    pack = load_model("breast_cancer")
    if not pack:
        st.error("Breast Cancer model not found.")
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Cell Nuclei Morphological Parameters")
        patient_name = st.text_input("Patient Identifier / Name", "Patient-BC01")
        radius_mean = st.slider("Radius Mean (Distance from Center to Perimeter)", 6.5, 30.0, 18.2, step=0.1)
        texture_mean = st.slider("Texture Mean (Standard Deviation of Gray-Scale)", 9.0, 40.0, 22.4, step=0.1)
        perimeter_mean = st.slider("Perimeter Mean", 40.0, 190.0, 118.0, step=0.5)
        area_mean = st.slider("Area Mean", 140.0, 2500.0, 1020.0, step=5.0)
        smoothness_mean = st.slider("Smoothness Mean (Local Variation in Radius)", 0.05, 0.17, 0.108, step=0.001)
        compactness_mean = st.slider("Compactness Mean (Perimeter² / Area - 1.0)", 0.02, 0.35, 0.165, step=0.001)
        concavity_mean = st.slider("Concavity Mean (Severity of Concave Portions)", 0.0, 0.45, 0.185, step=0.005)
        concave_points = st.slider("Concave Points Mean (Number of Concave Portions)", 0.0, 0.21, 0.095, step=0.005)
        symmetry_mean = st.slider("Symmetry Mean", 0.10, 0.31, 0.210, step=0.005)
        fractal_dim = st.slider("Fractal Dimension Mean ('Coastline Approximation')", 0.04, 0.10, 0.068, step=0.001)

        predict_btn = st.button("Run Malignancy Assessment", type="primary", use_container_width=True)

    with col2:
        if predict_btn:
            input_df = pd.DataFrame([{
                "RadiusMean": radius_mean, "TextureMean": texture_mean, "PerimeterMean": perimeter_mean,
                "AreaMean": area_mean, "SmoothnessMean": smoothness_mean, "CompactnessMean": compactness_mean,
                "ConcavityMean": concavity_mean, "ConcavePointsMean": concave_points,
                "SymmetryMean": symmetry_mean, "FractalDimension": fractal_dim
            }])
            model = pack["model"]
            pred_idx = int(model.predict(input_df)[0])
            probs = model.predict_proba(input_df)[0]
            conf = probs[pred_idx] * 100
            label = pack["label_names"][pred_idx]

            guidance = explainability.calculate_risk_and_guidance("breast_cancer", pred_idx, probs, pack["label_names"])
            rules = explainability.extract_decision_path(model, input_df, pack["features"])

            database.save_prediction(patient_name, "Breast Cancer", f"Radius:{radius_mean}, Concavity:{concavity_mean}", label, guidance["risk_level"], conf)

            st.subheader("Prediction Result & Clinical Assessment")
            if pred_idx == 1:
                st.markdown(f"""
                <div class="risk-high">
                    <h3>⚠️ {label}</h3>
                    <p>Estimated Probability: <b>{conf:.1f}%</b> | Risk Category: <b>{guidance['risk_level']}</b></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <h3>✅ {label}</h3>
                    <p>Estimated Confidence: <b>{conf:.1f}%</b> | Risk Category: <b>{guidance['risk_level']}</b></p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### 🔍 Explainable Decision Path")
            for rule in rules:
                st.markdown(f"""<div class="rule-box"><b>Step {rule['step']}:</b> If <code>{rule['rule']}</code> &rarr; <i>{rule['description']}</i></div>""", unsafe_allow_html=True)

            st.markdown("#### 📋 Oncology Guidance & Diagnostic Pathway")
            for rec in guidance["recommendations"]:
                st.markdown(f"- {rec}")
            st.warning(guidance["disclaimer"])

# ==================== 6. STROKE RISK SCREENING ====================
def screen_stroke():
    st.header("🧠 Stroke Risk Prediction & Cerebrovascular Screening")
    st.caption("Assesses cardiovascular, metabolic, and lifestyle determinants to quantify acute ischemic stroke risk.")

    pack = load_model("stroke")
    if not pack:
        st.error("Stroke model not found.")
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Patient Clinical & Demographic Profile")
        patient_name = st.text_input("Patient Identifier / Name", "Patient-ST01")
        age = st.slider("Age (Years)", 25, 90, 67)
        hypertension = st.selectbox("Hypertension Diagnosis", options=[("No (Normal Blood Pressure)", 0), ("Yes (Hypertensive)", 1)], format_func=lambda x: x[0])[1]
        heart_disease = st.selectbox("Pre-existing Heart Disease", options=[("No History", 0), ("Yes (Cardiovascular Condition)", 1)], format_func=lambda x: x[0])[1]
        avg_glucose = st.slider("Average Blood Glucose Level (mg/dL)", 55.0, 290.0, 165.0, step=1.0)
        bmi = st.slider("Body Mass Index (BMI kg/m²)", 14.0, 58.0, 32.5, step=0.1)
        smoking = st.selectbox("Smoking Status", options=[("Never Smoked", 0), ("Formerly Smoked", 1), ("Currently Smokes", 2)], format_func=lambda x: x[0])[1]
        work_type = st.selectbox("Work Environment", options=[("Government Job", 0), ("Private Sector", 1), ("Self-Employed", 2)], format_func=lambda x: x[0])[1]

        predict_btn = st.button("Run Stroke Risk Evaluation", type="primary", use_container_width=True)

    with col2:
        if predict_btn:
            input_df = pd.DataFrame([{
                "Age": age, "Hypertension": hypertension, "HeartDisease": heart_disease,
                "AvgGlucoseLevel": avg_glucose, "BMI": bmi, "SmokingStatus": smoking, "WorkType": work_type
            }])
            model = pack["model"]
            pred_idx = int(model.predict(input_df)[0])
            probs = model.predict_proba(input_df)[0]
            conf = probs[pred_idx] * 100
            label = pack["label_names"][pred_idx]

            guidance = explainability.calculate_risk_and_guidance("stroke", pred_idx, probs, pack["label_names"])
            rules = explainability.extract_decision_path(model, input_df, pack["features"])

            database.save_prediction(patient_name, "Stroke Risk", f"Age:{age}, HTN:{hypertension}, Glucose:{avg_glucose}", label, guidance["risk_level"], conf)

            st.subheader("Prediction Result & Clinical Assessment")
            if pred_idx == 1:
                st.markdown(f"""
                <div class="risk-high">
                    <h3>⚠️ {label}</h3>
                    <p>Estimated Probability: <b>{conf:.1f}%</b> | Risk Category: <b>{guidance['risk_level']}</b></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <h3>✅ {label}</h3>
                    <p>Estimated Confidence: <b>{conf:.1f}%</b> | Risk Category: <b>{guidance['risk_level']}</b></p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### 🔍 Explainable Decision Path")
            for rule in rules:
                st.markdown(f"""<div class="rule-box"><b>Step {rule['step']}:</b> If <code>{rule['rule']}</code> &rarr; <i>{rule['description']}</i></div>""", unsafe_allow_html=True)

            st.markdown("#### 📋 Neurological Precautions & FAST Warning")
            for rec in guidance["recommendations"]:
                st.markdown(f"- {rec}")
            st.warning(guidance["disclaimer"])

# ==================== 7. PARKINSON'S DISEASE SCREENING ====================
def screen_parkinsons():
    st.header("🔊 Parkinson's Disease (Voice & Motor Biomarkers)")
    st.caption("Analyzes phonation micro-perturbations and nonlinear vocal dynamics (Oxford/UCI Benchmark).")

    pack = load_model("parkinsons")
    if not pack:
        st.error("Parkinson's model not found.")
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Vocal Acoustic & Dynamical Biomarkers")
        patient_name = st.text_input("Patient Identifier / Name", "Patient-PD01")
        fo = st.slider("MDVP:Fo(Hz) - Average Vocal Fundamental Frequency", 80.0, 270.0, 145.0, step=0.5)
        fhi = st.slider("MDVP:Fhi(Hz) - Maximum Vocal Fundamental Frequency", 100.0, 450.0, 185.0, step=1.0)
        flo = st.slider("MDVP:Flo(Hz) - Minimum Vocal Fundamental Frequency", 60.0, 240.0, 110.0, step=0.5)
        jitter = st.slider("MDVP:Jitter(%) - Fundamental Frequency Perturbation", 0.001, 0.035, 0.0125, step=0.0005, format="%.4f")
        shimmer = st.slider("MDVP:Shimmer - Amplitude Perturbation", 0.008, 0.120, 0.0480, step=0.001, format="%.3f")
        hnr = st.slider("HNR - Harmonics-to-Noise Ratio (dB)", 8.0, 35.0, 18.2, step=0.2)
        rpde = st.slider("RPDE - Recurrence Period Density Entropy", 0.20, 0.75, 0.540, step=0.005)
        dfa = st.slider("DFA - Detrended Fluctuation Analysis", 0.55, 0.85, 0.745, step=0.005)
        spread1 = st.slider("Spread1 - Nonlinear Measure of Frequency Variation", -8.0, -2.0, -4.8, step=0.1)
        ppe = st.slider("PPE - Pitch Period Entropy", 0.04, 0.55, 0.320, step=0.005)

        predict_btn = st.button("Run Parkinson's Speech Analysis", type="primary", use_container_width=True)

    with col2:
        if predict_btn:
            input_df = pd.DataFrame([{
                "MDVP_Fo_Hz": fo, "MDVP_Fhi_Hz": fhi, "MDVP_Flo_Hz": flo,
                "MDVP_Jitter_Pct": jitter, "MDVP_Shimmer": shimmer, "HNR": hnr,
                "RPDE": rpde, "DFA": dfa, "Spread1": spread1, "PPE": ppe
            }])
            model = pack["model"]
            pred_idx = int(model.predict(input_df)[0])
            probs = model.predict_proba(input_df)[0]
            conf = probs[pred_idx] * 100
            label = pack["label_names"][pred_idx]

            guidance = explainability.calculate_risk_and_guidance("parkinsons", pred_idx, probs, pack["label_names"])
            rules = explainability.extract_decision_path(model, input_df, pack["features"])

            database.save_prediction(patient_name, "Parkinson's", f"Jitter:{jitter:.4f}, HNR:{hnr:.1f}, PPE:{ppe:.3f}", label, guidance["risk_level"], conf)

            st.subheader("Prediction Result & Clinical Assessment")
            if pred_idx == 1:
                st.markdown(f"""
                <div class="risk-high">
                    <h3>⚠️ {label}</h3>
                    <p>Estimated Probability: <b>{conf:.1f}%</b> | Risk Category: <b>{guidance['risk_level']}</b></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <h3>✅ {label}</h3>
                    <p>Estimated Confidence: <b>{conf:.1f}%</b> | Risk Category: <b>{guidance['risk_level']}</b></p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### 🔍 Explainable Decision Path")
            for rule in rules:
                st.markdown(f"""<div class="rule-box"><b>Step {rule['step']}:</b> If <code>{rule['rule']}</code> &rarr; <i>{rule['description']}</i></div>""", unsafe_allow_html=True)

            st.markdown("#### 📋 Neurological Guidance & Movement Therapies")
            for rec in guidance["recommendations"]:
                st.markdown(f"- {rec}")
            st.warning(guidance["disclaimer"])

# ==================== 8. HYPERTENSION SCREENING ====================
def screen_hypertension():
    st.header("🩺 Hypertension & Blood Pressure Staging")
    st.caption("Classifies blood pressure stages (Normal, Elevated/Stage 1, Stage 2/Crisis) and lifestyle risk drivers.")

    pack = load_model("hypertension")
    if not pack:
        st.error("Hypertension model not found.")
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Hemodynamic & Lifestyle Measurements")
        patient_name = st.text_input("Patient Identifier / Name", "Patient-HT01")
        age = st.slider("Age (Years)", 20, 85, 52)
        systolic = st.slider("Systolic Blood Pressure (mm Hg)", 90, 210, 146)
        diastolic = st.slider("Diastolic Blood Pressure (mm Hg)", 55, 130, 94)
        hr = st.slider("Resting Heart Rate (bpm)", 50, 120, 82)
        bmi = st.slider("Body Mass Index (BMI kg/m²)", 16.0, 50.0, 29.5, step=0.1)
        sodium = st.slider("Estimated Daily Sodium Intake (mg/day)", 1000, 6000, 3800, step=100)
        activity = st.slider("Weekly Physical Activity (Hours)", 0.0, 12.0, 1.5, step=0.5)
        stress = st.selectbox("Reported Stress Level", options=[("Low Stress", 0), ("Moderate Stress", 1), ("High / Chronic Stress", 2)], format_func=lambda x: x[0])[1]
        family_history = st.selectbox("Family History of Hypertension", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]

        predict_btn = st.button("Evaluate Hypertension Stage", type="primary", use_container_width=True)

    with col2:
        if predict_btn:
            input_df = pd.DataFrame([{
                "Age": age, "SystolicBP": systolic, "DiastolicBP": diastolic,
                "HeartRate": hr, "BMI": bmi, "SodiumIntake": sodium,
                "PhysicalActivityHours": activity, "StressLevel": stress, "FamilyHistory": family_history
            }])
            model = pack["model"]
            pred_idx = int(model.predict(input_df)[0])
            probs = model.predict_proba(input_df)[0]
            conf = probs[pred_idx] * 100
            label = pack["label_names"][pred_idx]

            guidance = explainability.calculate_risk_and_guidance("hypertension", pred_idx, probs, pack["label_names"])
            rules = explainability.extract_decision_path(model, input_df, pack["features"])

            database.save_prediction(patient_name, "Hypertension", f"BP:{systolic}/{diastolic}, Sodium:{sodium}mg", label, guidance["risk_level"], conf)

            st.subheader("Prediction Result & Clinical Assessment")
            if pred_idx == 2:
                st.markdown(f"""
                <div class="risk-high">
                    <h3>⚠️ {label}</h3>
                    <p>Estimated Probability: <b>{conf:.1f}%</b> | Classification: <b>Stage 2 Hypertension</b></p>
                </div>
                """, unsafe_allow_html=True)
            elif pred_idx == 1:
                st.markdown(f"""
                <div class="risk-warning">
                    <h3>⚠️ {label}</h3>
                    <p>Estimated Probability: <b>{conf:.1f}%</b> | Classification: <b>Elevated / Pre-hypertension</b></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <h3>✅ {label}</h3>
                    <p>Estimated Confidence: <b>{conf:.1f}%</b> | Classification: <b>Normal Blood Pressure</b></p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### 🔍 Explainable Decision Path")
            for rule in rules:
                st.markdown(f"""<div class="rule-box"><b>Step {rule['step']}:</b> If <code>{rule['rule']}</code> &rarr; <i>{rule['description']}</i></div>""", unsafe_allow_html=True)

            st.markdown("#### 📋 Cardiovascular & Blood Pressure Guidance")
            for rec in guidance["recommendations"]:
                st.markdown(f"- {rec}")
            st.warning(guidance["disclaimer"])

# ==================== 9. EXPANDED SYMPTOM-BASED CLASSIFIER ====================
def screen_symptoms():
    st.header("🌡️ Expanded Symptom-Based Multi-Disease Diagnosis")
    st.caption("Matches reported physical symptoms across 16 acute and chronic conditions using Decision Trees.")

    pack = load_model("symptom")
    if not pack:
        st.error("Symptom model not found.")
        return

    features = pack["features"]

    st.subheader("Select Present Symptoms:")
    patient_name = st.text_input("Patient Identifier / Name", "Patient-S01")

    cols = st.columns(3)
    symptom_inputs = {}

    symptom_labels = {
        "fever": "🌡️ High Body Temperature (Fever)",
        "cough": "🗣️ Persistent Cough",
        "fatigue": "🥱 Fatigue / Generalized Weakness",
        "shortness_of_breath": "🫁 Shortness of Breath / Dyspnea",
        "headache": "🤕 Severe Headache",
        "chest_pain": "💔 Chest Pain or Tightness",
        "sore_throat": "🧣 Sore Throat / Pharyngitis",
        "nausea": "🤢 Nausea",
        "vomiting": "🤮 Vomiting",
        "diarrhea": "💧 Diarrhea / Loose Stools",
        "joint_pain": "🦵 Severe Joint & Bone Pain",
        "loss_of_taste_smell": "👃 Loss of Taste / Smell (Anosmia)",
        "rash": "🩹 Skin Rash / Petechiae",
        "high_fever_with_chills": "🥶 Shivering & Severe Chills",
        "abdominal_pain": "🤰 Abdominal Cramps / Pain",
        "wheezing": "🌬️ Wheezing / Whistling Breath",
        "jaundice_yellow_eyes": "🟡 Jaundice (Yellow Sclera / Eyes)",
        "chronic_cough_with_blood": "🩸 Chronic Cough Producing Blood"
    }

    for i, feat in enumerate(features):
        col_target = cols[i % 3]
        label = symptom_labels.get(feat, feat.replace("_", " ").title())
        symptom_inputs[feat] = col_target.checkbox(label, value=False)

    predict_btn = st.button("Diagnose from Symptoms", type="primary", use_container_width=True)

    if predict_btn:
        active_count = sum(symptom_inputs.values())
        if active_count == 0:
            st.warning("Please select at least 1 or 2 symptoms to evaluate.")
            return

        input_df = pd.DataFrame([{feat: (1 if symptom_inputs[feat] else 0) for feat in features}])
        model = pack["model"]
        predicted_disease = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        classes = model.classes_

        class_prob_dict = {cls: float(prob) * 100 for cls, prob in zip(classes, probabilities)}
        sorted_probs = sorted(class_prob_dict.items(), key=lambda x: x[1], reverse=True)

        top_disease, top_conf = sorted_probs[0]
        guidance = explainability.calculate_risk_and_guidance("symptom", top_disease, probabilities, classes)
        rules = explainability.extract_decision_path(model, input_df, features)

        active_symptoms_str = ", ".join([k.replace('_', ' ') for k, v in symptom_inputs.items() if v])
        database.save_prediction(patient_name, "Symptom Diagnosis", active_symptoms_str[:80], top_disease, guidance["risk_level"], top_conf)

        st.markdown(f"""
        <div class="risk-high" style="background-color: #eff6ff; color: #1e40af; border-left: 6px solid #2563eb;">
            <h3>🔬 Most Likely Condition: <b>{top_disease}</b></h3>
            <p>Confidence: <b>{top_conf:.1f}%</b> | Selected Symptoms: <b>{active_symptoms_str}</b></p>
        </div>
        """, unsafe_allow_html=True)

        colA, colB = st.columns([1, 1])
        with colA:
            st.markdown("#### 📊 Differential Diagnoses (Top Candidates)")
            for dis, prob in sorted_probs[:5]:
                st.progress(prob / 100.0, text=f"{dis}: {prob:.1f}%")

        with colB:
            st.markdown("#### 🔍 Decision Tree Logic Sequence")
            for rule in rules:
                val_repr = "Present (1)" if rule['patient_value'] > 0.5 else "Absent (0)"
                st.markdown(f"""<div class="rule-box"><b>Step {rule['step']}:</b> {rule['feature'].replace('_',' ').title()} is <b>{val_repr}</b> &rarr; <i>{rule['description']}</i></div>""", unsafe_allow_html=True)

        st.markdown("#### 📋 Medical Care Recommendations")
        for rec in guidance["recommendations"]:
            st.markdown(f"- {rec}")
        st.warning(guidance["disclaimer"])

# ==================== 10. MODEL EVALUATION & BENCHMARKS ====================
def view_model_analytics():
    st.header("📊 Multi-Disease Model Evaluation & Benchmarks")
    st.caption("Quantitative performance metrics assessed across 9 specialized Decision Tree models.")

    metrics_summary = load_metrics()
    if not metrics_summary:
        st.info("Metrics not loaded. Please ensure train_models.py has completed.")
        return

    rows = []
    for k, v in metrics_summary.items():
        rows.append({
            "Disease Model": k.replace("_", " ").upper(),
            "Accuracy (%)": f"{v['accuracy'] * 100:.2f}%",
            "Precision (%)": f"{v['precision'] * 100:.2f}%",
            "Recall (%)": f"{v['recall'] * 100:.2f}%",
            "F1-Score (%)": f"{v['f1_score'] * 100:.2f}%",
            "Test Samples": v["test_samples"]
        })
    df_metrics = pd.DataFrame(rows)
    st.dataframe(df_metrics, use_container_width=True)

    st.markdown("---")
    st.subheader("Confusion Matrices & Key Feature Importances")
    
    model_choices = list(metrics_summary.keys())
    selected_model_view = st.selectbox(
        "Select Model for Detailed Visualizations:",
        model_choices,
        format_func=lambda x: f"{x.replace('_', ' ').upper()} Model"
    )

    col1, col2 = st.columns(2)
    plots_dir = os.path.join(os.path.dirname(__file__), "models", "plots")
    
    cm_img_path = os.path.join(plots_dir, f"{selected_model_view}_confusion_matrix.png")
    fi_img_path = os.path.join(plots_dir, f"{selected_model_view}_feature_importance.png")

    with col1:
        st.markdown("#### Confusion Matrix")
        if os.path.exists(cm_img_path):
            st.image(cm_img_path, use_container_width=True)
        else:
            st.info("Confusion matrix plot not found.")

    with col2:
        st.markdown("#### Feature Importance Analysis")
        if os.path.exists(fi_img_path):
            st.image(fi_img_path, use_container_width=True)
        else:
            st.info("Feature importance plot not found.")

# ==================== 11. REVIEW 1 & ARCHITECTURE ====================
def view_review1_agenda():
    st.header("🏛️ Minor Project - Review 1 Specification & Architecture")
    st.caption("Structured in alignment with the official Review 1 Agenda and System Blueprint.")

    tabs = st.tabs([
        "1. Introduction",
        "2. Literature Survey",
        "3. Problem Statement",
        "4. Proposed Methodology",
        "5. System Architecture",
        "6. Implementation Timeline"
    ])

    with tabs[0]:
        st.markdown("""
        ### Introduction
        - **Project Title:** Disease Prediction System Using Decision Trees
        - **Expanded Scope:** To develop an intelligent, application-based system that estimates the risk of 9 major clinical domains:
          1. **Diabetes Mellitus** (Pima Indians Clinical Model)
          2. **Coronary Heart Disease** (Cleveland Cardiovascular Benchmark)
          3. **Chronic Kidney Disease** (Renal Glomerular Function)
          4. **Liver Disease** (Hepatic Enzyme Panel)
          5. **Breast Cancer** (Cellular Morphological Malignancy)
          6. **Cerebrovascular Stroke** (FAST Stroke Risk Quantification)
          7. **Parkinson's Disease** (Vocal Phonation Biomarkers)
          8. **Hypertension Staging** (Hemodynamic Classification)
          9. **Symptom-Based Differential Diagnosis** (16 acute conditions)
        - **Core Distinction:** Interpretable Explainable AI (XAI) using Decision Trees with full root-to-leaf decision paths.
        """)

    with tabs[1]:
        st.markdown("""
        ### Literature Survey
        A systematic comparison between conventional diagnostic approaches and the proposed multi-disease system:
        """)
        lit_data = [
            {"Approach": "Traditional Clinical Diagnosis", "Mechanism": "Physician evaluation, manual lab panels", "Pros": "High diagnostic authority", "Cons": "High delays, high cost, limited accessibility"},
            {"Approach": "Rule-Based Expert Systems", "Mechanism": "Static IF-THEN medical rules", "Pros": "Deterministic explanations", "Cons": "Rigid, fails on non-linear or noisy patient data"},
            {"Approach": "Deep Learning / Black-Box ML", "Mechanism": "Multi-layer Neural Networks / Ensembles", "Pros": "High mathematical curve fitting", "Cons": "Zero interpretability; doctors cannot verify reasoning"},
            {"Approach": "Single-Disease Web Predictors", "Mechanism": "Standalone ML models for 1 illness", "Pros": "Targeted accuracy", "Cons": "Inconvenient; lacks holistic multi-organ evaluation"},
            {"Approach": "Proposed System (Decision Trees)", "Mechanism": "Optimized Multi-Disease Decision Trees + XAI", "Pros": "9 disease modules, explainable paths, low latency, clinically aligned", "Cons": "Requires calibrated clinical benchmark datasets"}
        ]
        st.table(pd.DataFrame(lit_data))

    with tabs[2]:
        st.markdown("""
        ### Problem Statement & Challenges
        - **Limited Disease Coverage:** Existing tools typically address only one specific pathology, requiring users to navigate multiple fragmented platforms.
        - **The Black-Box Dilemma:** Deep neural models generate probability scores without explainability, which hinders adoption by medical professionals who require verifiable reasoning.
        - **Delayed Early Detection:** Patients frequently delay seeking medical intervention because laboratory reports contain complex clinical jargon that is difficult to interpret without physician assistance.
        - **Data Quality & Inconsistency:** Clinical data commonly exhibits missing values, duplicate entries, and multi-scale units requiring systematic automated preprocessing.
        """)

    with tabs[3]:
        st.markdown("""
        ### Proposed Methodology
        The system follows a multi-stage machine learning pipeline:
        1. **Data Preprocessing & Cleaning:** Handling missing observations, encoding categorical features, and min-max normalization.
        2. **Decision Tree Classification:**
           - Splitting criteria: Gini Impurity \\(I_G(p) = 1 - \\sum p_i^2\\) and Shannon Entropy \\(H(X) = -\\sum p_i \\log_2 p_i\\).
           - Pruning & regularization: Controlling `max_depth` (5-9) and `min_samples_split` to eliminate tree overfitting and maintain readability.
        3. **Explainable Rule Extraction:** Generating traversable decision paths with feature thresholds for every single prediction.
        4. **Evaluation Protocol:** Standardized validation using independent 80/20 train-test splits and confusion matrix calculation.
        """)

    with tabs[4]:
        st.markdown("""
        ### Proposed System Architecture
        """)
        arch_img_path = os.path.join(os.path.dirname(__file__), "assets", "system_architecture_diagram.png")
        if os.path.exists(arch_img_path):
            st.image(arch_img_path, caption="Figure 1: Proposed System Architecture (Disease Prediction System Using Decision Trees)", use_container_width=True)

    with tabs[5]:
        st.markdown("""
        ### Implementation Plan & Project Milestones
        """)
        timeline = [
            {"Milestone / Period": "Week 1 - 2", "Tasks": "Problem formulation, literature review, clinical dataset gathering, and requirements specification.", "Status": "Completed (Review 1)"},
            {"Milestone / Period": "Week 3 - 4", "Tasks": "Data preprocessing pipelines, Decision Tree model development for Diabetes and Heart Disease modules.", "Status": "Completed"},
            {"Milestone / Period": "Week 5 - 6", "Tasks": "Development of Kidney, Liver, Cancer, Stroke, Parkinson's, and Hypertension modules.", "Status": "Completed"},
            {"Milestone / Period": "Week 7 - 8", "Tasks": "Unified Streamlit web application integration, XAI decision path visualization, database history logging, user testing, and final documentation.", "Status": "Completed"}
        ]
        st.table(pd.DataFrame(timeline))

# ==================== 12. SCREENING HISTORY ====================
def view_history():
    st.header("📜 Patient Screening History")
    st.caption("Persistent log of all completed screening assessments stored in SQLite database.")

    history_records = database.get_history(limit=50)
    if not history_records:
        st.info("No screening assessments recorded yet. Run a prediction from any module to log patient data.")
        return

    df_hist = pd.DataFrame(history_records, columns=[
        "ID", "Timestamp", "Patient Name", "Disease Module", "Clinical Summary", "Prediction", "Risk Level", "Confidence (%)"
    ])
    df_hist["Confidence (%)"] = df_hist["Confidence (%)"].map(lambda x: f"{x:.1f}%")

    st.dataframe(df_hist, use_container_width=True)

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Clear History Log", type="secondary"):
            database.clear_history()
            st.rerun()

# ==================== MAIN ROUTER ====================
def main():
    render_header()
    selected_menu = render_sidebar()

    if selected_menu == "🩸 Diabetes Screening":
        screen_diabetes()
    elif selected_menu == "❤️ Heart Disease Screening":
        screen_heart()
    elif selected_menu == "🧪 Kidney Disease (CKD)":
        screen_kidney()
    elif selected_menu == "🧬 Liver Disease Screening":
        screen_liver()
    elif selected_menu == "🎗️ Breast Cancer (Malignancy)":
        screen_breast_cancer()
    elif selected_menu == "🧠 Stroke Risk Assessment":
        screen_stroke()
    elif selected_menu == "🔊 Parkinson's (Vocal Biomarkers)":
        screen_parkinsons()
    elif selected_menu == "🩺 Hypertension & BP Staging":
        screen_hypertension()
    elif selected_menu == "🌡️ Symptom-Based Classifier (16 Conditions)":
        screen_symptoms()
    elif selected_menu == "📊 Model Evaluation & Benchmarks":
        view_model_analytics()
    elif selected_menu == "🏛️ Review 1 & Architecture":
        view_review1_agenda()
    elif selected_menu == "📜 Screening History":
        view_history()

if __name__ == "__main__":
    main()
