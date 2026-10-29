"""
Expanded Model Training & Evaluation Pipeline for Multi-Disease Prediction System
Trains Scikit-learn Decision Tree Classifiers for:
1. Diabetes
2. Heart Disease
3. Chronic Kidney Disease (CKD)
4. Liver Disease
5. Breast Cancer
6. Stroke Risk
7. Parkinson's Disease
8. Hypertension / BP Staging
9. Multi-Condition Symptom Checker (16 Diseases)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_and_evaluate(data_dir="data", models_dir="models", plots_dir="models/plots"):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    configs = [
        {
            "name": "diabetes",
            "file": "diabetes_data.csv",
            "target": "Outcome",
            "label_names": ["No Diabetes (Low Risk)", "Diabetes Detected (High Risk)"],
            "max_depth": 5,
            "criterion": "gini"
        },
        {
            "name": "heart",
            "file": "heart_disease_data.csv",
            "target": "Outcome",
            "label_names": ["Normal / Low Risk", "Heart Disease Detected (High Risk)"],
            "max_depth": 5,
            "criterion": "gini"
        },
        {
            "name": "kidney",
            "file": "kidney_disease_data.csv",
            "target": "Outcome",
            "label_names": ["No CKD (Normal)", "Chronic Kidney Disease Detected"],
            "max_depth": 5,
            "criterion": "entropy"
        },
        {
            "name": "liver",
            "file": "liver_disease_data.csv",
            "target": "Outcome",
            "label_names": ["Normal Liver Health", "Liver Disease Detected"],
            "max_depth": 5,
            "criterion": "gini"
        },
        {
            "name": "breast_cancer",
            "file": "breast_cancer_data.csv",
            "target": "Outcome",
            "label_names": ["Benign (Non-Cancerous)", "Malignant (High Cancer Risk)"],
            "max_depth": 5,
            "criterion": "entropy"
        },
        {
            "name": "stroke",
            "file": "stroke_data.csv",
            "target": "Outcome",
            "label_names": ["Low Stroke Risk", "High Stroke Risk Detected"],
            "max_depth": 5,
            "criterion": "gini"
        },
        {
            "name": "parkinsons",
            "file": "parkinsons_data.csv",
            "target": "Outcome",
            "label_names": ["Healthy / Low Risk", "Parkinson's Indicators Detected"],
            "max_depth": 5,
            "criterion": "entropy"
        },
        {
            "name": "hypertension",
            "file": "hypertension_data.csv",
            "target": "Outcome",
            "label_names": ["Normal BP", "Elevated / Stage 1 HTN", "Stage 2 HTN / High Risk"],
            "max_depth": 6,
            "criterion": "gini"
        },
        {
            "name": "symptom",
            "file": "symptom_disease_data.csv",
            "target": "Disease",
            "label_names": None, # Multi-class
            "max_depth": 9,
            "criterion": "entropy"
        }
    ]

    summary_metrics = {}

    for cfg in configs:
        name = cfg["name"]
        csv_path = os.path.join(data_dir, cfg["file"])
        df = pd.read_csv(csv_path)

        X = df.drop(columns=[cfg["target"]])
        y = df[cfg["target"]]
        feature_names = list(X.columns)

        is_multiclass = (name in ["symptom", "hypertension"])
        if name == "symptom":
            unique_classes = sorted(y.unique().tolist())
            cfg["label_names"] = unique_classes

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        clf = DecisionTreeClassifier(
            criterion=cfg["criterion"],
            max_depth=cfg["max_depth"],
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        average_mode = "weighted" if is_multiclass else "binary"
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average=average_mode, zero_division=0))
        rec = float(recall_score(y_test, y_pred, average=average_mode, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average=average_mode, zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()

        feature_importances = dict(zip(feature_names, [round(float(v), 4) for v in clf.feature_importances_]))

        metrics = {
            "name": name,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "features": feature_names,
            "label_names": cfg["label_names"],
            "feature_importances": feature_importances,
            "confusion_matrix": cm,
            "test_samples": len(y_test)
        }
        summary_metrics[name] = metrics

        model_pack = {
            "model": clf,
            "features": feature_names,
            "label_names": cfg["label_names"],
            "metrics": metrics,
            "is_multiclass": is_multiclass
        }
        model_save_path = os.path.join(models_dir, f"{name}_dt_model.joblib")
        joblib.dump(model_pack, model_save_path)

        # Generate Confusion Matrix Plot
        plt.figure(figsize=(6.5, 5))
        xtick_labels = [c[:8] for c in cfg["label_names"]] if is_multiclass else cfg["label_names"]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=xtick_labels,
            yticklabels=xtick_labels
        )
        plt.title(f"{name.replace('_', ' ').title()} - Confusion Matrix\nAccuracy: {acc*100:.1f}%")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{name}_confusion_matrix.png"), dpi=200)
        plt.close()

        # Generate Feature Importance Plot
        plt.figure(figsize=(7, 4.5))
        sorted_fi = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:10]
        feats = [x[0] for x in sorted_fi]
        vals = [x[1] for x in sorted_fi]
        plt.barh(feats[::-1], vals[::-1], color="#1d4ed8")
        plt.xlabel("Feature Importance Score")
        plt.title(f"Key Decision Tree Predictors ({name.replace('_', ' ').title()})")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{name}_feature_importance.png"), dpi=200)
        plt.close()

        print(f"[OK] Trained {name.upper()}: Accuracy={acc*100:.2f}%, Precision={prec*100:.2f}%, Recall={rec*100:.2f}%, F1={f1*100:.2f}%")

    with open(os.path.join(models_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary_metrics, f, indent=2)

    print("\n[OK] All 9 Multi-Disease Decision Tree models trained, evaluated, and saved successfully.")

if __name__ == "__main__":
    train_and_evaluate()
