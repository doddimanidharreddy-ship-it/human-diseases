# Disease Prediction System Using Decision Trees

**Vel Tech Rangarajan Dr. Sagunthala R&D Institute of Science and Technology**  
*School of Computing | Department of Computer Science & Engineering*  
**10214CS601 - Minor Project - I | Academic Year: 2026-2027 (Summer Semester)**  
**Batch:** MI1301  

---

## 👥 Project Team & Supervision
- **Bugga Gangadhar Kali Prasad** (VTU 28564) (24UECS0049)
- **Allala Bhavana** (VTU 27999) (24UECS0008)
- **Godi Mouni Pranathi** (VTU 28298) (24UECS0107)
- **Supervisor:** Dr. J. Raja (Associate Professor, Dept. of CSE)

---

## 📌 Project Overview
The **Disease Prediction System Using Decision Trees** is an Explainable Machine Learning Clinical Decision Support System. Unlike opaque black-box neural networks, this system leverages interpretable **Decision Tree Classifiers** to evaluate multi-organ clinical parameters and physical symptoms, generating transparent root-to-leaf decision paths alongside estimated risk levels and actionable medical recommendations.

### 🩺 Supported Modules:
1. **Diabetes Risk Screening:** Fasting glucose, blood pressure, BMI, age, insulin, and pedigree function.
2. **Heart Disease Risk Screening:** Age, resting blood pressure, cholesterol, chest pain category, ST depression, ECG, and exercise angina.
3. **Chronic Kidney Disease (CKD) Screening:** Serum creatinine, blood urea, specific gravity, urine albumin, and hemoglobin.
4. **Liver Disease Screening:** Total/direct bilirubin, alkaline phosphatase, ALT (SGPT), AST (SGOT), albumin, and A/G ratio.
5. **Symptom-Based Multi-Disease Classifier:** Interactive selection of 15 physical symptoms differential diagnosis across 10 common conditions (COVID-19, Influenza, Dengue, Malaria, Typhoid, etc.).
6. **Model Analytics & Evaluation:** Live confusion matrices, feature importance graphs, and precision/recall/F1 metrics.
7. **Screening History Logger:** Persistent storage of patient screening evaluations using SQLite.

---

## 🚀 How to Run the Application

### 1. Navigate to Project Directory
```powershell
cd "C:\Users\manid\.gemini\antigravity\scratch\disease_prediction_system"
```

### 2. Run the Streamlit Web Application
```powershell
streamlit run app.py
```
The application will open automatically in your browser at `http://localhost:8501`.

### 3. Run Automated System Tests
```powershell
python test_system.py
```

### 4. Regenerate / Retrain Models (Optional)
```powershell
python generate_datasets.py
python train_models.py
```

### 5. Generate Review 1 PowerPoint Presentation (.pptx)
```powershell
python generate_review1_pptx.py
```
Output files created:
- `C:\Users\manid\.gemini\antigravity\scratch\disease_prediction_system\DISEASE_PREDICTION_SYSTEM_REVIEW_1.pptx`
- `C:\Users\manid\Downloads\CSE-MINOR_PROJECT-1_DISEASE_PREDICTION.pptx`

---

## 📊 Model Benchmark Results
| Disease Model | Test Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Diabetes** | 95.00% | 96.36% | 94.64% | 95.50% |
| **Heart Disease** | 80.50% | 87.69% | 83.21% | 85.39% |
| **Kidney Disease (CKD)** | 94.50% | 93.52% | 96.19% | 94.84% |
| **Liver Disease** | 98.50% | 99.13% | 98.28% | 98.70% |
| **Symptom Diagnosis (10 classes)** | 74.58% | 76.84% | 74.58% | 75.06% |

---

## ⚖️ Clinical Disclaimer
This system is an educational screening support tool powered by machine learning algorithms and is **NOT** a substitute for certified clinical diagnosis by licensed medical professionals.
