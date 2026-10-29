"""
Expanded Dataset Generator for Multi-Disease Prediction System
Generates clinical benchmark datasets based on real medical distributions for:
1. Diabetes
2. Heart Disease
3. Chronic Kidney Disease (CKD)
4. Liver Disease
5. Breast Cancer (Malignancy Assessment)
6. Stroke Risk Prediction
7. Parkinson's Disease (Voice Biomarkers)
8. Hypertension / Blood Pressure Staging
9. Expanded Multi-Disease Symptom Checker (16 Conditions)
"""

import os
import numpy as np
import pandas as pd

def generate_all_datasets(output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)

    # 1. Diabetes Dataset (Pima-like distribution, 1000 records)
    n = 1000
    age = np.random.randint(21, 75, size=n)
    pregnancies = np.where(np.random.rand(n) > 0.4, np.random.randint(0, 10, size=n), 0)
    glucose = np.random.normal(110, 28, size=n)
    blood_pressure = np.random.normal(72, 12, size=n)
    skin_thickness = np.random.normal(25, 9, size=n)
    insulin = np.random.exponential(65, size=n) + 15
    bmi = np.random.normal(30, 6.5, size=n)
    diabetes_pedigree = np.random.exponential(0.4, size=n) + 0.1
    
    risk_diabetes = (
        0.04 * (glucose - 100) +
        0.05 * (bmi - 25) +
        0.02 * (age - 30) +
        0.5 * diabetes_pedigree +
        0.005 * (insulin - 50) +
        0.01 * (blood_pressure - 75) +
        np.random.normal(0, 0.8, size=n)
    )
    outcome_diabetes = (risk_diabetes > 1.2).astype(int)
    glucose[outcome_diabetes == 1] += np.random.uniform(25, 60, size=np.sum(outcome_diabetes == 1))
    bmi[outcome_diabetes == 1] += np.random.uniform(3, 8, size=np.sum(outcome_diabetes == 1))
    
    diabetes_df = pd.DataFrame({
        "Pregnancies": np.clip(pregnancies, 0, 15),
        "Glucose": np.clip(np.round(glucose, 1), 60, 260),
        "BloodPressure": np.clip(np.round(blood_pressure, 1), 40, 140),
        "SkinThickness": np.clip(np.round(skin_thickness, 1), 8, 60),
        "Insulin": np.clip(np.round(insulin, 1), 15, 500),
        "BMI": np.clip(np.round(bmi, 1), 15.0, 55.0),
        "DiabetesPedigreeFunction": np.clip(np.round(diabetes_pedigree, 3), 0.05, 2.5),
        "Age": age,
        "Outcome": outcome_diabetes
    })
    diabetes_df.to_csv(os.path.join(output_dir, "diabetes_data.csv"), index=False)
    print(f"[OK] Diabetes Dataset saved: {diabetes_df.shape}")

    # 2. Heart Disease Dataset (Cleveland-like distribution, 1000 records)
    n = 1000
    age = np.random.randint(29, 78, size=n)
    sex = np.random.choice([1, 0], size=n, p=[0.65, 0.35])
    chest_pain = np.random.choice([0, 1, 2, 3], size=n, p=[0.45, 0.20, 0.20, 0.15])
    resting_bp = np.random.normal(130, 17, size=n)
    cholesterol = np.random.normal(240, 48, size=n)
    fasting_bs = np.random.choice([0, 1], size=n, p=[0.82, 0.18])
    resting_ecg = np.random.choice([0, 1, 2], size=n, p=[0.5, 0.45, 0.05])
    max_hr = 208 - (0.7 * age) + np.random.normal(0, 15, size=n)
    exercise_angina = np.random.choice([0, 1], size=n, p=[0.65, 0.35])
    oldpeak = np.clip(np.random.exponential(0.9, size=n), 0, 6.0)
    st_slope = np.random.choice([0, 1, 2], size=n, p=[0.4, 0.5, 0.1])

    heart_risk = (
        0.04 * (age - 45) +
        0.6 * sex +
        0.8 * chest_pain +
        0.02 * (resting_bp - 120) +
        0.01 * (cholesterol - 200) +
        1.1 * exercise_angina +
        0.7 * oldpeak +
        0.8 * st_slope -
        0.025 * (max_hr - 130) +
        np.random.normal(0, 0.4, size=n)
    )
    heart_outcome = (heart_risk > 1.8).astype(int)

    heart_df = pd.DataFrame({
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": np.clip(np.round(resting_bp, 1), 90, 200),
        "Cholesterol": np.clip(np.round(cholesterol, 1), 120, 520),
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": np.clip(np.round(max_hr, 1), 70, 210),
        "ExerciseAngina": exercise_angina,
        "Oldpeak": np.clip(np.round(oldpeak, 2), 0.0, 6.2),
        "ST_Slope": st_slope,
        "Outcome": heart_outcome
    })
    heart_df.to_csv(os.path.join(output_dir, "heart_disease_data.csv"), index=False)
    print(f"[OK] Heart Disease Dataset saved: {heart_df.shape}")

    # 3. Chronic Kidney Disease (CKD) Dataset (1000 records)
    n = 1000
    age = np.random.randint(18, 85, size=n)
    blood_pressure = np.random.normal(78, 14, size=n)
    specific_gravity = np.random.choice([1.005, 1.010, 1.015, 1.020, 1.025], size=n, p=[0.08, 0.17, 0.25, 0.35, 0.15])
    albumin = np.random.choice([0, 1, 2, 3, 4, 5], size=n, p=[0.55, 0.15, 0.12, 0.08, 0.06, 0.04])
    sugar = np.random.choice([0, 1, 2, 3, 4, 5], size=n, p=[0.7, 0.1, 0.08, 0.06, 0.04, 0.02])
    red_blood_cells = np.random.choice([0, 1], size=n, p=[0.78, 0.22])
    blood_glucose_random = np.random.normal(140, 60, size=n)
    blood_urea = np.random.normal(55, 35, size=n)
    serum_creatinine = np.random.exponential(1.2, size=n) + 0.5
    sodium = np.random.normal(137, 6, size=n)
    potassium = np.random.normal(4.6, 0.7, size=n)
    hemoglobin = np.random.normal(12.5, 2.7, size=n)

    ckd_risk = (
        0.8 * albumin +
        0.5 * (blood_urea > 60).astype(int) +
        1.2 * (serum_creatinine > 1.4).astype(int) +
        0.8 * red_blood_cells -
        0.4 * (hemoglobin - 12) -
        30 * (specific_gravity - 1.015) +
        0.015 * (blood_pressure - 80) +
        np.random.normal(0, 0.6, size=n)
    )
    ckd_outcome = (ckd_risk > 1.2).astype(int)
    serum_creatinine[ckd_outcome == 1] += np.random.uniform(0.8, 3.5, size=np.sum(ckd_outcome == 1))
    hemoglobin[ckd_outcome == 1] -= np.random.uniform(1.5, 4.0, size=np.sum(ckd_outcome == 1))

    ckd_df = pd.DataFrame({
        "Age": age,
        "BloodPressure": np.clip(np.round(blood_pressure, 1), 50, 160),
        "SpecificGravity": specific_gravity,
        "Albumin": albumin,
        "Sugar": sugar,
        "RedBloodCells": red_blood_cells,
        "BloodGlucoseRandom": np.clip(np.round(blood_glucose_random, 1), 60, 450),
        "BloodUrea": np.clip(np.round(blood_urea, 1), 10, 250),
        "SerumCreatinine": np.clip(np.round(serum_creatinine, 2), 0.4, 12.0),
        "Sodium": np.clip(np.round(sodium, 1), 110, 155),
        "Potassium": np.clip(np.round(potassium, 2), 2.5, 7.5),
        "Hemoglobin": np.clip(np.round(hemoglobin, 1), 4.0, 18.0),
        "Outcome": ckd_outcome
    })
    ckd_df.to_csv(os.path.join(output_dir, "kidney_disease_data.csv"), index=False)
    print(f"[OK] Kidney Disease Dataset saved: {ckd_df.shape}")

    # 4. Liver Disease Dataset (1000 records)
    n = 1000
    age = np.random.randint(18, 80, size=n)
    gender = np.random.choice([1, 0], size=n, p=[0.75, 0.25])
    total_bilirubin = np.random.exponential(1.5, size=n) + 0.4
    direct_bilirubin = total_bilirubin * np.random.uniform(0.2, 0.5, size=n)
    alkaline_phosphotase = np.random.normal(280, 120, size=n)
    alamine_aminotransferase = np.random.exponential(45, size=n) + 12
    aspartate_aminotransferase = np.random.exponential(65, size=n) + 15
    total_proteins = np.random.normal(6.5, 1.0, size=n)
    albumin_liver = np.random.normal(3.1, 0.6, size=n)
    ag_ratio = albumin_liver / np.clip(total_proteins - albumin_liver, 0.5, 5.0)

    liver_risk = (
        0.6 * (total_bilirubin > 2.0).astype(int) +
        0.8 * (alamine_aminotransferase > 50).astype(int) +
        0.7 * (aspartate_aminotransferase > 50).astype(int) +
        0.003 * (alkaline_phosphotase - 250) -
        0.6 * (albumin_liver - 3.2) -
        0.5 * (ag_ratio - 1.0) +
        0.015 * (age - 40) +
        np.random.normal(0, 0.6, size=n)
    )
    liver_outcome = (liver_risk > 1.0).astype(int)
    total_bilirubin[liver_outcome == 1] += np.random.uniform(1.0, 5.0, size=np.sum(liver_outcome == 1))
    alamine_aminotransferase[liver_outcome == 1] += np.random.uniform(30, 120, size=np.sum(liver_outcome == 1))

    liver_df = pd.DataFrame({
        "Age": age,
        "Gender": gender,
        "TotalBilirubin": np.clip(np.round(total_bilirubin, 2), 0.4, 25.0),
        "DirectBilirubin": np.clip(np.round(direct_bilirubin, 2), 0.1, 15.0),
        "AlkalinePhosphotase": np.clip(np.round(alkaline_phosphotase, 1), 60, 1500),
        "AlamineAminotransferase": np.clip(np.round(alamine_aminotransferase, 1), 10, 800),
        "AspartateAminotransferase": np.clip(np.round(aspartate_aminotransferase, 1), 10, 800),
        "TotalProteins": np.clip(np.round(total_proteins, 2), 2.5, 10.0),
        "Albumin": np.clip(np.round(albumin_liver, 2), 1.0, 5.5),
        "AlbuminGlobulinRatio": np.clip(np.round(ag_ratio, 2), 0.3, 2.8),
        "Outcome": liver_outcome
    })
    liver_df.to_csv(os.path.join(output_dir, "liver_disease_data.csv"), index=False)
    print(f"[OK] Liver Disease Dataset saved: {liver_df.shape}")

    # 5. NEW: Breast Cancer (Tumor Malignancy Risk Assessment) (1000 records)
    n = 1000
    radius_mean = np.random.normal(14.0, 3.2, size=n)
    texture_mean = np.random.normal(19.0, 4.1, size=n)
    perimeter_mean = radius_mean * 6.28 + np.random.normal(0, 2.0, size=n)
    area_mean = 3.14 * (radius_mean ** 2) * 0.95
    smoothness_mean = np.random.normal(0.096, 0.014, size=n)
    compactness_mean = np.random.normal(0.104, 0.05, size=n)
    concavity_mean = np.random.exponential(0.08, size=n)
    concave_points = np.random.exponential(0.045, size=n)
    symmetry_mean = np.random.normal(0.18, 0.027, size=n)
    fractal_dimension = np.random.normal(0.062, 0.007, size=n)

    cancer_score = (
        0.35 * (radius_mean - 13.5) +
        0.15 * (texture_mean - 18.0) +
        12.0 * concavity_mean +
        18.0 * concave_points +
        8.0 * compactness_mean +
        np.random.normal(0, 0.4, size=n)
    )
    cancer_outcome = (cancer_score > 1.2).astype(int) # 0: Benign, 1: Malignant
    radius_mean[cancer_outcome == 1] += np.random.uniform(2.5, 6.0, size=np.sum(cancer_outcome == 1))
    concavity_mean[cancer_outcome == 1] += np.random.uniform(0.05, 0.15, size=np.sum(cancer_outcome == 1))

    cancer_df = pd.DataFrame({
        "RadiusMean": np.clip(np.round(radius_mean, 2), 6.5, 29.0),
        "TextureMean": np.clip(np.round(texture_mean, 2), 9.0, 40.0),
        "PerimeterMean": np.clip(np.round(perimeter_mean, 2), 40.0, 190.0),
        "AreaMean": np.clip(np.round(area_mean, 1), 140.0, 2500.0),
        "SmoothnessMean": np.clip(np.round(smoothness_mean, 4), 0.05, 0.17),
        "CompactnessMean": np.clip(np.round(compactness_mean, 4), 0.02, 0.35),
        "ConcavityMean": np.clip(np.round(concavity_mean, 4), 0.0, 0.45),
        "ConcavePointsMean": np.clip(np.round(concave_points, 4), 0.0, 0.21),
        "SymmetryMean": np.clip(np.round(symmetry_mean, 4), 0.10, 0.31),
        "FractalDimension": np.clip(np.round(fractal_dimension, 4), 0.04, 0.10),
        "Outcome": cancer_outcome
    })
    cancer_df.to_csv(os.path.join(output_dir, "breast_cancer_data.csv"), index=False)
    print(f"[OK] Breast Cancer Dataset saved: {cancer_df.shape}")

    # 6. NEW: Stroke Risk Prediction Dataset (1000 records)
    n = 1000
    age = np.random.randint(25, 85, size=n)
    hypertension = np.where(age > 50, np.random.choice([0, 1], size=n, p=[0.7, 0.3]), np.random.choice([0, 1], size=n, p=[0.9, 0.1]))
    heart_disease = np.where(age > 55, np.random.choice([0, 1], size=n, p=[0.75, 0.25]), np.random.choice([0, 1], size=n, p=[0.95, 0.05]))
    avg_glucose = np.random.normal(106, 42, size=n)
    bmi = np.random.normal(28.5, 6.8, size=n)
    smoking_status = np.random.choice([0, 1, 2], size=n, p=[0.45, 0.30, 0.25]) # 0: Never, 1: Formerly, 2: Smokes
    work_type = np.random.choice([0, 1, 2], size=n, p=[0.2, 0.55, 0.25]) # 0: Govt, 1: Private, 2: Self-employed

    stroke_score = (
        0.05 * (age - 50) +
        1.1 * hypertension +
        1.2 * heart_disease +
        0.015 * (avg_glucose - 100) +
        0.04 * (bmi - 25) +
        0.5 * (smoking_status == 2).astype(int) +
        np.random.normal(0, 0.5, size=n)
    )
    stroke_outcome = (stroke_score > 1.8).astype(int)
    avg_glucose[stroke_outcome == 1] += np.random.uniform(30, 70, size=np.sum(stroke_outcome == 1))

    stroke_df = pd.DataFrame({
        "Age": age,
        "Hypertension": hypertension,
        "HeartDisease": heart_disease,
        "AvgGlucoseLevel": np.clip(np.round(avg_glucose, 1), 55.0, 290.0),
        "BMI": np.clip(np.round(bmi, 1), 14.0, 58.0),
        "SmokingStatus": smoking_status,
        "WorkType": work_type,
        "Outcome": stroke_outcome
    })
    stroke_df.to_csv(os.path.join(output_dir, "stroke_data.csv"), index=False)
    print(f"[OK] Stroke Risk Dataset saved: {stroke_df.shape}")

    # 7. NEW: Parkinson's Disease (Voice & Motor Biomarkers) (800 records)
    n = 800
    fo_hz = np.random.normal(154.0, 40.0, size=n) # Fundamental frequency
    fhi_hz = fo_hz + np.random.exponential(40.0, size=n)
    flo_hz = fo_hz - np.clip(np.random.exponential(35.0, size=n), 10, fo_hz - 60)
    jitter_pct = np.random.exponential(0.005, size=n) + 0.001
    shimmer = np.random.exponential(0.025, size=n) + 0.009
    hnr = np.random.normal(21.5, 4.5, size=n) # Harmonics to noise ratio
    rpde = np.random.normal(0.48, 0.10, size=n) # Recurrence period density
    dfa = np.random.normal(0.71, 0.06, size=n) # Detrended fluctuation analysis
    spread1 = np.random.normal(-5.6, 1.0, size=n)
    ppe = np.random.exponential(0.20, size=n) + 0.05 # Pitch period entropy

    parkinsons_score = (
        80.0 * jitter_pct +
        25.0 * shimmer -
        0.15 * (hnr - 20) +
        3.5 * rpde +
        8.0 * ppe +
        0.8 * (spread1 + 5.0) +
        np.random.normal(0, 0.4, size=n)
    )
    parkinsons_outcome = (parkinsons_score > 2.6).astype(int)
    jitter_pct[parkinsons_outcome == 1] += np.random.uniform(0.004, 0.015, size=np.sum(parkinsons_outcome == 1))
    hnr[parkinsons_outcome == 1] -= np.random.uniform(3.0, 7.0, size=np.sum(parkinsons_outcome == 1))

    parkinsons_df = pd.DataFrame({
        "MDVP_Fo_Hz": np.clip(np.round(fo_hz, 1), 80.0, 270.0),
        "MDVP_Fhi_Hz": np.clip(np.round(fhi_hz, 1), 100.0, 450.0),
        "MDVP_Flo_Hz": np.clip(np.round(flo_hz, 1), 60.0, 240.0),
        "MDVP_Jitter_Pct": np.clip(np.round(jitter_pct, 5), 0.001, 0.035),
        "MDVP_Shimmer": np.clip(np.round(shimmer, 4), 0.008, 0.12),
        "HNR": np.clip(np.round(hnr, 1), 8.0, 35.0),
        "RPDE": np.clip(np.round(rpde, 4), 0.20, 0.75),
        "DFA": np.clip(np.round(dfa, 4), 0.55, 0.85),
        "Spread1": np.clip(np.round(spread1, 3), -8.0, -2.0),
        "PPE": np.clip(np.round(ppe, 4), 0.04, 0.55),
        "Outcome": parkinsons_outcome
    })
    parkinsons_df.to_csv(os.path.join(output_dir, "parkinsons_data.csv"), index=False)
    print(f"[OK] Parkinson's Dataset saved: {parkinsons_df.shape}")

    # 8. NEW: Hypertension & Blood Pressure Staging Dataset (1000 records)
    n = 1000
    age = np.random.randint(20, 80, size=n)
    systolic_bp = np.random.normal(125, 20, size=n)
    diastolic_bp = systolic_bp * 0.62 + np.random.normal(0, 6, size=n)
    heart_rate = np.random.normal(74, 11, size=n)
    bmi = np.random.normal(27.0, 5.5, size=n)
    sodium_intake = np.random.normal(3200, 750, size=n) # mg/day
    activity_hours = np.clip(np.random.exponential(2.5, size=n), 0, 10)
    stress_level = np.random.choice([0, 1, 2], size=n, p=[0.3, 0.45, 0.25]) # 0: Low, 1: Moderate, 2: High
    family_history = np.random.choice([0, 1], size=n, p=[0.6, 0.4])

    # Classify Stage: 0: Normal, 1: Elevated / Stage 1, 2: Stage 2 Hypertension / Critical
    bp_stage = np.zeros(n, dtype=int)
    for i in range(n):
        s = systolic_bp[i]
        d = diastolic_bp[i]
        if s >= 140 or d >= 90:
            bp_stage[i] = 2 # Stage 2
        elif s >= 120 or d >= 80:
            bp_stage[i] = 1 # Elevated / Stage 1
        else:
            bp_stage[i] = 0 # Normal

    hypertension_df = pd.DataFrame({
        "Age": age,
        "SystolicBP": np.clip(np.round(systolic_bp, 1), 90, 210),
        "DiastolicBP": np.clip(np.round(diastolic_bp, 1), 55, 130),
        "HeartRate": np.clip(np.round(heart_rate, 1), 50, 120),
        "BMI": np.clip(np.round(bmi, 1), 16.0, 50.0),
        "SodiumIntake": np.clip(np.round(sodium_intake, 0), 1000, 6000),
        "PhysicalActivityHours": np.clip(np.round(activity_hours, 1), 0.0, 12.0),
        "StressLevel": stress_level,
        "FamilyHistory": family_history,
        "Outcome": bp_stage
    })
    hypertension_df.to_csv(os.path.join(output_dir, "hypertension_data.csv"), index=False)
    print(f"[OK] Hypertension Dataset saved: {hypertension_df.shape}")

    # 9. EXPANDED Symptom-based Multi-Disease Dataset (16 Conditions, 18 symptoms, 1600 records)
    symptoms = [
        "fever", "cough", "fatigue", "shortness_of_breath", "headache",
        "chest_pain", "sore_throat", "nausea", "vomiting", "diarrhea",
        "joint_pain", "loss_of_taste_smell", "rash", "high_fever_with_chills",
        "abdominal_pain", "wheezing", "jaundice_yellow_eyes", "chronic_cough_with_blood"
    ]
    
    disease_profiles = {
        "Common Cold": {"fever": 0.3, "cough": 0.8, "fatigue": 0.5, "headache": 0.4, "sore_throat": 0.9},
        "Influenza (Flu)": {"fever": 0.9, "cough": 0.8, "fatigue": 0.9, "headache": 0.8, "joint_pain": 0.7, "sore_throat": 0.6, "high_fever_with_chills": 0.8},
        "COVID-19": {"fever": 0.8, "cough": 0.85, "fatigue": 0.8, "shortness_of_breath": 0.6, "loss_of_taste_smell": 0.85, "headache": 0.6, "sore_throat": 0.5},
        "Dengue": {"fever": 0.95, "headache": 0.9, "joint_pain": 0.95, "rash": 0.7, "nausea": 0.6, "vomiting": 0.5, "high_fever_with_chills": 0.85},
        "Malaria": {"fever": 0.95, "high_fever_with_chills": 0.95, "headache": 0.8, "fatigue": 0.85, "nausea": 0.7, "vomiting": 0.6, "joint_pain": 0.6},
        "Typhoid": {"fever": 0.9, "headache": 0.8, "abdominal_pain": 0.85, "fatigue": 0.8, "diarrhea": 0.6, "nausea": 0.6},
        "Gastroenteritis": {"nausea": 0.9, "vomiting": 0.9, "diarrhea": 0.95, "abdominal_pain": 0.85, "fatigue": 0.6, "fever": 0.4},
        "Bronchitis": {"cough": 0.95, "fatigue": 0.7, "shortness_of_breath": 0.7, "chest_pain": 0.5, "sore_throat": 0.4},
        "Allergic Rhinitis": {"headache": 0.4, "sore_throat": 0.6, "cough": 0.4, "fatigue": 0.3},
        "Migraine": {"headache": 0.98, "nausea": 0.8, "vomiting": 0.5, "fatigue": 0.6},
        "Asthma": {"shortness_of_breath": 0.95, "wheezing": 0.95, "cough": 0.8, "chest_pain": 0.6, "fatigue": 0.5},
        "Pneumonia": {"fever": 0.9, "high_fever_with_chills": 0.85, "cough": 0.9, "shortness_of_breath": 0.85, "chest_pain": 0.8, "fatigue": 0.8},
        "Tuberculosis": {"chronic_cough_with_blood": 0.95, "cough": 0.95, "fatigue": 0.9, "fever": 0.7, "chest_pain": 0.6},
        "Jaundice / Hepatitis": {"jaundice_yellow_eyes": 0.95, "abdominal_pain": 0.8, "fatigue": 0.85, "nausea": 0.75, "vomiting": 0.6},
        "Rheumatoid Arthritis": {"joint_pain": 0.98, "fatigue": 0.8, "fever": 0.3},
        "Food Poisoning": {"nausea": 0.95, "vomiting": 0.95, "abdominal_pain": 0.9, "diarrhea": 0.85, "fever": 0.4}
    }

    records = []
    for disease, profile in disease_profiles.items():
        for _ in range(100):
            row = {}
            for sym in symptoms:
                prob = profile.get(sym, 0.04)
                row[sym] = 1 if np.random.rand() < prob else 0
            row["Disease"] = disease
            records.append(row)

    symptom_df = pd.DataFrame(records)
    symptom_df = symptom_df.sample(frac=1, random_state=42).reset_index(drop=True)
    symptom_df.to_csv(os.path.join(output_dir, "symptom_disease_data.csv"), index=False)
    print(f"[OK] Expanded Symptom Dataset saved: {symptom_df.shape}")

if __name__ == "__main__":
    generate_all_datasets()
