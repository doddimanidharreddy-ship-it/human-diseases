"""
Comprehensive Test Suite for 9 Multi-Disease Prediction Models
Verifies model inference, explainability rules, and SQLite logging.
"""

import os
import joblib
import pandas as pd
import explainability
import database

def test_all():
    print("[*] Running Complete Multi-Disease System Test Suite...")
    database.init_db()

    tests = [
        ("diabetes", "Diabetes", pd.DataFrame([{
            "Pregnancies": 2, "Glucose": 155.0, "BloodPressure": 80.0,
            "SkinThickness": 28.0, "Insulin": 150.0, "BMI": 32.5,
            "DiabetesPedigreeFunction": 0.55, "Age": 45
        }])),
        ("heart", "Heart Disease", pd.DataFrame([{
            "Age": 60, "Sex": 1, "ChestPainType": 2, "RestingBP": 140.0,
            "Cholesterol": 265.0, "FastingBS": 1, "RestingECG": 1,
            "MaxHR": 135.0, "ExerciseAngina": 1, "Oldpeak": 2.0, "ST_Slope": 1
        }])),
        ("kidney", "Kidney Disease", pd.DataFrame([{
            "Age": 52, "BloodPressure": 85.0, "SpecificGravity": 1.010, "Albumin": 2,
            "Sugar": 0, "RedBloodCells": 1, "BloodGlucoseRandom": 150.0, "BloodUrea": 75.0,
            "SerumCreatinine": 2.8, "Sodium": 133.0, "Potassium": 4.9, "Hemoglobin": 10.0
        }])),
        ("liver", "Liver Disease", pd.DataFrame([{
            "Age": 48, "Gender": 1, "TotalBilirubin": 3.6, "DirectBilirubin": 1.5,
            "AlkalinePhosphotase": 310.0, "AlamineAminotransferase": 95.0,
            "AspartateAminotransferase": 105.0, "TotalProteins": 6.1, "Albumin": 2.7,
            "AlbuminGlobulinRatio": 0.76
        }])),
        ("breast_cancer", "Breast Cancer", pd.DataFrame([{
            "RadiusMean": 18.5, "TextureMean": 22.0, "PerimeterMean": 120.0,
            "AreaMean": 1050.0, "SmoothnessMean": 0.110, "CompactnessMean": 0.170,
            "ConcavityMean": 0.190, "ConcavePointsMean": 0.100,
            "SymmetryMean": 0.215, "FractalDimension": 0.069
        }])),
        ("stroke", "Stroke Risk", pd.DataFrame([{
            "Age": 68, "Hypertension": 1, "HeartDisease": 1,
            "AvgGlucoseLevel": 170.0, "BMI": 33.0, "SmokingStatus": 2, "WorkType": 1
        }])),
        ("parkinsons", "Parkinson's", pd.DataFrame([{
            "MDVP_Fo_Hz": 140.0, "MDVP_Fhi_Hz": 180.0, "MDVP_Flo_Hz": 105.0,
            "MDVP_Jitter_Pct": 0.0135, "MDVP_Shimmer": 0.051, "HNR": 17.5,
            "RPDE": 0.56, "DFA": 0.75, "Spread1": -4.6, "PPE": 0.34
        }])),
        ("hypertension", "Hypertension", pd.DataFrame([{
            "Age": 54, "SystolicBP": 150.0, "DiastolicBP": 96.0,
            "HeartRate": 84.0, "BMI": 30.2, "SodiumIntake": 3900.0,
            "PhysicalActivityHours": 1.0, "StressLevel": 2, "FamilyHistory": 1
        }])),
        ("symptom", "Symptom Diagnosis", pd.DataFrame([{
            "fever": 1, "cough": 1, "fatigue": 1, "shortness_of_breath": 1, "headache": 1,
            "chest_pain": 0, "sore_throat": 1, "nausea": 0, "vomiting": 0, "diarrhea": 0,
            "joint_pain": 0, "loss_of_taste_smell": 1, "rash": 0, "high_fever_with_chills": 0,
            "abdominal_pain": 0, "wheezing": 0, "jaundice_yellow_eyes": 0, "chronic_cough_with_blood": 0
        }]))
    ]

    for model_name, display_title, sample_df in tests:
        pack = joblib.load(f"models/{model_name}_dt_model.joblib")
        model = pack["model"]
        pred = model.predict(sample_df)[0]
        probs = model.predict_proba(sample_df)[0]
        rules = explainability.extract_decision_path(model, sample_df, pack["features"])
        label_names = pack["label_names"]
        
        pred_label = label_names[int(pred)] if model_name != "symptom" and not isinstance(pred, str) else str(pred)
        guidance = explainability.calculate_risk_and_guidance(model_name, pred if isinstance(pred, int) else 0, probs, label_names)
        database.save_prediction(f"Test-{model_name}", display_title, "Verified Sample", str(pred_label), guidance["risk_level"], float(max(probs))*100)
        print(f"[OK] {display_title.upper()}: Pred={pred_label}, Rules={len(rules)}, Risk={guidance['risk_level']}")

    history = database.get_history(limit=10)
    print(f"\n[SUCCESS] All 9 Disease models tested successfully! {len(history)} records in SQLite database.")

if __name__ == "__main__":
    test_all()
