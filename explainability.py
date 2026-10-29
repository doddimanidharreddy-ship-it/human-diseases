"""
Explainability & Decision Path Analysis Engine
Extracts transparent decision rules from Scikit-Learn Decision Trees
to satisfy medical interpretability and user guidance requirements.
"""

import numpy as np
import pandas as pd

def extract_decision_path(model, sample_df, feature_names):
    """
    Extracts the sequence of decision rules evaluated by the Decision Tree
    for a given patient sample.
    """
    tree = model.tree_
    feature = tree.feature
    threshold = tree.threshold

    # Get node path for the first sample
    node_indicator = model.decision_path(sample_df)
    leaf_id = model.apply(sample_df)

    node_index = node_indicator.indices[
        node_indicator.indptr[0] : node_indicator.indptr[1]
    ]

    rules = []
    sample_values = sample_df.iloc[0].to_dict()

    for node_id in node_index:
        if leaf_id[0] == node_id:
            continue # Reached leaf node

        feat_idx = feature[node_id]
        feat_name = feature_names[feat_idx]
        thresh_val = threshold[node_id]
        patient_val = sample_values[feat_name]

        if patient_val <= thresh_val:
            decision_text = f"<= {thresh_val:.2f}"
            branch_desc = f"Below or equal to clinical threshold ({thresh_val:.2f})"
            condition_met = True
        else:
            decision_text = f"> {thresh_val:.2f}"
            branch_desc = f"Above clinical threshold ({thresh_val:.2f})"
            condition_met = True

        rules.append({
            "step": len(rules) + 1,
            "feature": feat_name,
            "patient_value": round(float(patient_val), 2),
            "threshold": round(float(thresh_val), 2),
            "rule": f"{feat_name} ({patient_val:.2f}) {decision_text}",
            "description": branch_desc
        })

    return rules

def calculate_risk_and_guidance(disease_name, prediction, probabilities, class_names):
    """
    Determines clinical risk level and returns personalized doctor advice
    and lifestyle recommendations.
    """
    if len(probabilities) == 2:
        # Binary classification
        pos_prob = float(probabilities[1])
        if pos_prob < 0.25:
            risk_level = "Low Risk"
            badge_color = "success"
        elif pos_prob < 0.55:
            risk_level = "Moderate Risk (Borderline)"
            badge_color = "warning"
        elif pos_prob < 0.80:
            risk_level = "Elevated Risk"
            badge_color = "danger"
        else:
            risk_level = "High / Critical Risk"
            badge_color = "danger"
    else:
        # Multi-class
        top_prob = float(max(probabilities))
        if top_prob > 0.65:
            risk_level = "High Correlation"
            badge_color = "warning"
        else:
            risk_level = "Moderate Correlation"
            badge_color = "info"

    # Disease-specific guidance
    guidelines = {
        "diabetes": {
            "High / Critical Risk": [
                "Schedule a fasting blood glucose and HbA1c test with a certified physician.",
                "Limit high glycemic carbohydrates, sugary beverages, and processed foods.",
                "Incorporate at least 30 minutes of moderate aerobic exercise daily.",
                "Maintain a daily log of blood glucose levels."
            ],
            "Elevated Risk": [
                "Consult a healthcare professional for pre-diabetes screening.",
                "Adopt a balanced Mediterranean or low-glycemic dietary plan.",
                "Monitor body weight and aim for gradual BMI reduction."
            ],
            "Low Risk": [
                "Maintain healthy eating patterns rich in whole grains and vegetables.",
                "Undergo routine annual wellness checkups.",
                "Maintain active physical lifestyle."
            ]
        },
        "heart": {
            "High / Critical Risk": [
                "IMMEDIATE ACTION: Consult a cardiologist for comprehensive ECG and stress test evaluation.",
                "If experiencing chest pressure, radiation to left arm, or shortness of breath, seek emergency medical services immediately.",
                "Strictly reduce sodium intake (< 2,000 mg/day) and eliminate trans fats.",
                "Refrain from vigorous exertion until medical clearance is granted."
            ],
            "Elevated Risk": [
                "Consult a cardiologist for lipid profile and blood pressure optimization.",
                "Engage in medically supervised cardiovascular exercises.",
                "Avoid smoking and strictly moderate alcohol intake."
            ],
            "Low Risk": [
                "Maintain cardiovascular health with regular aerobic exercise (150 mins/week).",
                "Keep systolic BP below 120 mmHg through mindful diet and stress management."
            ]
        },
        "kidney": {
            "High / Critical Risk": [
                "Consult a nephrologist immediately for serum creatinine, eGFR, and urine protein testing.",
                "Carefully monitor blood pressure; hypertension accelerates kidney decline.",
                "Avoid over-the-counter NSAID pain relievers (such as ibuprofen), which stress renal function.",
                "Maintain optimal hydration unless under fluid restriction."
            ],
            "Elevated Risk": [
                "Schedule renal function tests and repeat urinalysis.",
                "Reduce dietary protein load and salt intake as advised by a dietitian.",
                "Ensure strict control of blood sugar and blood pressure."
            ],
            "Low Risk": [
                "Drink 2-3 liters of clean water daily to assist renal clearance.",
                "Avoid excessive intake of sodium and unprescribed supplements."
            ]
        },
        "liver": {
            "High / Critical Risk": [
                "Consult a gastroenterologist/hepatologist for ultrasound and comprehensive LFT panel.",
                "Completely abstain from alcohol consumption and hepatotoxic medications.",
                "Adopt a lean, non-fatty liver protective diet rich in antioxidants.",
                "Watch for jaundice symptoms (yellowing eyes/skin, dark urine)."
            ],
            "Elevated Risk": [
                "Undergo repeat liver enzyme panel (ALT, AST, ALP).",
                "Eliminate deep-fried foods and refined sugars to reverse early hepatic steatosis.",
                "Ensure Hepatitis B and C screening if not previously performed."
            ],
            "Low Risk": [
                "Maintain healthy liver function with balanced diet and active lifestyle.",
                "Limit alcohol consumption and avoid unnecessary medications."
            ]
        },
        "breast_cancer": {
            "High / Critical Risk": [
                "URGENT: Consult an oncologist or breast surgeon for diagnostic mammography and core needle biopsy.",
                "Undergo high-resolution ultrasound imaging to assess mass vascularity and margins.",
                "Review family oncology history with a genetic counselor (BRCA1/BRCA2 mutation screening).",
                "Do not delay clinical evaluation; early intervention correlates with >95% 5-year survival."
            ],
            "Elevated Risk": [
                "Schedule a follow-up clinical breast examination and ultrasound.",
                "Monitor for localized tissue alterations, skin dimpling, or nipple changes.",
                "Maintain regular monthly self-examinations."
            ],
            "Low Risk": [
                "Cellular dimensions and morphological characteristics are within benign physiological limits.",
                "Follow age-appropriate routine annual screening mammograms.",
                "Maintain healthy weight and regular physical activity."
            ]
        },
        "stroke": {
            "High / Critical Risk": [
                "CRITICAL WARNING: If experiencing sudden facial drooping, arm weakness, or slurred speech, call 911 / emergency services immediately (FAST Protocol).",
                "Consult a neurologist for carotid duplex ultrasound, echocardiogram, and brain MRI/CT angiogram.",
                "Strict control of blood pressure, blood glucose, and dyslipidemia is mandatory.",
                "Smoking cessation is required immediately to prevent cerebral thrombosis."
            ],
            "Elevated Risk": [
                "Consult a physician to establish an aggressive cardiovascular risk-reduction plan.",
                "Monitor daily blood pressure and maintain target below 120/80 mmHg.",
                "Adopt the DASH (Dietary Approaches to Stop Hypertension) diet."
            ],
            "Low Risk": [
                "Continue maintaining healthy vascular habits with routine exercise and balanced nutrition.",
                "Avoid tobacco smoke and maintain optimal hydration."
            ]
        },
        "parkinsons": {
            "High / Critical Risk": [
                "Consult a movement disorder specialist / neurologist for clinical neurological examination.",
                "Consider DaTscan (dopamine transporter imaging) and electromyography (EMG).",
                "Engage in targeted physical and vocal speech therapy exercises.",
                "Discuss personalized neuroprotective and dopaminergic management options."
            ],
            "Elevated Risk": [
                "Monitor for subtle resting tremors, bradykinesia (slowed movements), or voice pitch micro-variations.",
                "Participate in regular balance and aerobic exercises (e.g. Tai Chi, brisk walking).",
                "Schedule an evaluation with a neurologist for baseline motor scoring."
            ],
            "Low Risk": [
                "Voice acoustic and motor biomarkers indicate normal neurological function.",
                "Maintain regular cognitive, aerobic, and social activities."
            ]
        },
        "hypertension": {
            "High / Critical Risk": [
                "CRITICAL: Blood pressure indicates Stage 2 Hypertension or Hypertensive Urgency.",
                "Consult a physician or visit an urgent care clinic immediately for antihypertensive therapy.",
                "Reduce daily sodium intake to strictly below 1,500 mg.",
                "Avoid sudden physical exertion, energy drinks, and excessive stimulants."
            ],
            "Elevated Risk": [
                "Blood pressure is in the elevated / Stage 1 hypertension category.",
                "Initiate lifestyle modifications: low-sodium DASH diet, weight management, and stress reduction.",
                "Log blood pressure twice daily (morning and evening) for 2 weeks."
            ],
            "Low Risk": [
                "Blood pressure is optimal. Continue heart-healthy nutrition and active lifestyle.",
                "Check blood pressure at least once every 6 months."
            ]
        },
        "symptom": {
            "General": [
                "This assessment evaluates symptomatic pattern matching and does not substitute laboratory diagnostic tests.",
                "If symptoms persist for over 48-72 hours or worsen, visit a clinic promptly.",
                "Seek immediate emergency care if experiencing breathing difficulties, blue lips, severe chest pain, or confusion.",
                "Stay well hydrated and isolate if contagious viral illness is suspected."
            ]
        }
    }

    category_key = guidelines.get(disease_name, {})
    if disease_name == "symptom":
        advice_list = category_key.get("General", [])
    else:
        advice_list = category_key.get(risk_level, category_key.get("Low Risk", []))

    return {
        "risk_level": risk_level,
        "badge_color": badge_color,
        "recommendations": advice_list,
        "disclaimer": "DISCLAIMER: This system is an educational and screening support tool powered by Machine Learning Decision Trees. It is NOT a certified clinical diagnosis. Always consult a licensed medical physician."
    }
