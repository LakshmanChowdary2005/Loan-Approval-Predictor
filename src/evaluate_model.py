"""
Standalone Model Evaluation & Subgroup Fairness Module.
Computes test metrics, confusion matrix, ROC-AUC, and fairness disparity metrics
across demographic categories with documented ethical disclosures.
"""
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import json

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

from config import DATA_PATH, MODEL_DIR, CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COLUMN, TARGET_MAP
from src.data_loader import DataLoader

def run_evaluation(model_name="Random Forest", test_size=0.20, random_state=42):
    """
    Evaluates specified model on held-out test data and analyzes demographic fairness.
    """
    model_path = os.path.join(MODEL_DIR, "loan_model.pkl" if model_name == "Random Forest" else f"{model_name.lower().replace(' ', '_')}.pkl")
    if not os.path.exists(model_path):
        # Fallback to loan_model.pkl
        model_path = os.path.join(MODEL_DIR, "loan_model.pkl")

    model = joblib.load(model_path)
    
    loader = DataLoader(DATA_PATH)
    df = loader.clean_data()
    df['Target'] = df[TARGET_COLUMN].map(TARGET_MAP)
    df = df.dropna(subset=['Target'])

    X = df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    y = df['Target'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model_name": model_name,
        "test_samples": len(y_test),
        "accuracy": round(float(accuracy_score(y_test, y_pred) * 100), 2),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0) * 100), 2),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0) * 100), 2),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0) * 100), 2),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob) * 100), 2),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    # Subgroup Fairness Analysis on Test Split
    test_eval_df = X_test.copy()
    test_eval_df['Actual'] = y_test
    test_eval_df['Predicted'] = y_pred

    fairness_subgroups = {}
    for group_col in ['Gender', 'Education', 'Property_Area']:
        if group_col in test_eval_df.columns:
            group_metrics = {}
            for val, group in test_eval_df.groupby(group_col):
                if len(group) > 0:
                    group_acc = accuracy_score(group['Actual'], group['Predicted']) * 100
                    app_rate = (group['Predicted'] == 1).mean() * 100
                    group_metrics[str(val)] = {
                        "sample_count": int(len(group)),
                        "approval_rate": round(float(app_rate), 1),
                        "subgroup_accuracy": round(float(group_acc), 1)
                    }
            fairness_subgroups[group_col] = group_metrics

    metrics["fairness_analysis"] = fairness_subgroups
    metrics["ethical_disclaimer"] = (
        "Statistical disparity in historical training data does not imply algorithmic intent. "
        "Underwriting models in production must undergo statutory disparate impact assessments, "
        "and lending decisions should comply with Equal Credit Opportunity Act (ECOA) regulations."
    )
    return metrics

if __name__ == "__main__":
    res = run_evaluation()
    print(json.dumps(res, indent=2))
