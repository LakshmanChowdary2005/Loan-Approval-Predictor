"""
Training Pipeline for AI-Powered Loan Approval Prediction System.
Trains 5 classification algorithms, evaluates performance across cross-validation
and held-out test splits, extracts feature importances, and persists models & metadata.
"""
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import json
import joblib
from datetime import date

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, balanced_accuracy_score, confusion_matrix, classification_report
)

from config import (
    DATA_PATH, MODEL_DIR, EDA_IMAGE_DIR, METADATA_FILE,
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COLUMN, TARGET_MAP
)
from src.data_loader import DataLoader
from src.preprocessing import build_preprocessor, get_transformed_feature_names

def train_and_evaluate(random_state=42):
    """Executes end-to-end model training, validation, visualization, and export."""
    print("=" * 70)
    print("AI-POWERED LOAN APPROVAL PREDICTION - MODEL TRAINING ENGINE")
    print("=" * 70)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(EDA_IMAGE_DIR, exist_ok=True)

    # 1. Load Data
    loader = DataLoader(DATA_PATH)
    df_clean = loader.clean_data()
    print(f"[1/6] Cleaned dataset shape: {df_clean.shape}")

    # Target encoding
    df_clean['Target'] = df_clean[TARGET_COLUMN].map(TARGET_MAP)
    valid_df = df_clean.dropna(subset=['Target'])

    X = valid_df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    y = valid_df['Target'].astype(int)

    # 2. Stratified Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    print(f"[2/6] Train set: {X_train.shape[0]} | Test set: {X_test.shape[0]}")
    print(f"Class ratio in train: Approved={y_train.sum()} ({y_train.mean()*100:.1f}%), Rejected={len(y_train)-y_train.sum()} ({(1-y_train.mean())*100:.1f}%)")

    # 3. Fit Preprocessing Pipeline on X_train only (Strict Anti-Leakage)
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessing.pkl"))
    print("[3/6] Preprocessing pipeline fitted and saved.")

    feature_names = get_transformed_feature_names(preprocessor)

    # 4. Define Candidate Classifiers
    candidate_models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=250, max_depth=6, min_samples_split=4, min_samples_leaf=2, random_state=random_state
        ),
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, random_state=random_state
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=5, random_state=random_state
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=120, max_depth=3, learning_rate=0.05, random_state=random_state
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200, max_depth=5, min_samples_split=4, random_state=random_state
        )
    }

    # Optional: Try XGBoost if installed
    try:
        from xgboost import XGBClassifier
        candidate_models["XGBoost"] = XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.08, random_state=random_state, eval_metric='logloss'
        )
    except Exception:
        pass

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    benchmarks = {}
    trained_pipelines = {}

    print("\n[4/6] Training and Cross-Validating Models...")
    for name, clf in candidate_models.items():
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])

        cv_results = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "balanced_accuracy": "balanced_accuracy",
                "macro_f1": "f1_macro"
            }
        )
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()

        benchmarks[name] = {
            "cv_accuracy_mean": round(float(cv_results["test_accuracy"].mean() * 100), 2),
            "cv_accuracy_std": round(float(cv_results["test_accuracy"].std() * 100), 2),
            "cv_balanced_accuracy_mean": round(float(cv_results["test_balanced_accuracy"].mean() * 100), 2),
            "cv_macro_f1_mean": round(float(cv_results["test_macro_f1"].mean() * 100), 2),
            "test_accuracy": round(float(acc * 100), 2),
            "test_balanced_accuracy": round(float(balanced_acc * 100), 2),
            "precision": round(float(prec * 100), 2),
            "recall": round(float(rec * 100), 2),
            "f1_score": round(float(f1 * 100), 2),
            "macro_f1": round(float(f1_score(y_test, y_pred, average='macro', zero_division=0) * 100), 2),
            "roc_auc": round(float(roc * 100), 2),
            "confusion_matrix": cm,
            "classification_report": classification_report(y_test, y_pred, target_names=['Rejected', 'Approved'], output_dict=True)
        }
        print(
            f" -> {name:20s} | CV Acc: {cv_results['test_accuracy'].mean()*100:5.2f}%"
            f" | CV Bal Acc: {cv_results['test_balanced_accuracy'].mean()*100:5.2f}%"
            f" | Test Acc: {acc*100:5.2f}% | F1: {f1*100:5.2f}% | ROC-AUC: {roc*100:5.2f}%"
        )

    # 5. Export Pipelines
    print("\n[5/6] Exporting Model Artifacts...")
    model_export_map = {
        "Random Forest": "loan_model.pkl",
        "Logistic Regression": "lr_model.pkl",
        "Decision Tree": "dt_model.pkl",
        "Gradient Boosting": "gb_model.pkl",
        "Extra Trees": "et_model.pkl"
    }
    if "XGBoost" in trained_pipelines:
        model_export_map["XGBoost"] = "xgb_model.pkl"

    for name, filename in model_export_map.items():
        if name in trained_pipelines:
            joblib.dump(trained_pipelines[name], os.path.join(MODEL_DIR, filename))

    # Feature importances from Random Forest
    rf_clf = trained_pipelines["Random Forest"].named_steps['classifier']
    importances = rf_clf.feature_importances_
    feat_imp = []
    for raw_f, imp in zip(feature_names, importances):
        friendly = raw_f.replace('cat__', '').replace('num__', '').replace('_', ' ').title()
        feat_imp.append({"raw_feature": raw_f, "feature": friendly, "importance": round(float(imp * 100), 2)})
    feat_imp.sort(key=lambda x: x['importance'], reverse=True)

    # 6. Generate Static EDA and Model Evaluation Plots
    print("\n[6/6] Generating EDA and Model Diagnostic Charts...")
    sns.set_theme(style="whitegrid", palette="muted")

    # Chart 1: Loan Status Distribution
    plt.figure(figsize=(6, 4))
    counts = df_clean[TARGET_COLUMN].value_counts()
    colors = ['#10B981', '#EF4444']
    ax = sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=colors, legend=False)
    plt.title("Loan Approval Distribution (Dataset)", fontsize=12, fontweight='bold')
    plt.xlabel("Status (Y: Approved, N: Rejected)")
    plt.ylabel("Number of Applications")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())} ({p.get_height()/len(df_clean)*100:.1f}%)",
                    (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                    ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_IMAGE_DIR, "loan_status_distribution.png"), dpi=180)
    plt.close()

    # Chart 2: Credit History vs Approval
    plt.figure(figsize=(6, 4))
    ch_df = df_clean.dropna(subset=['Credit_History'])
    ct = pd.crosstab(ch_df['Credit_History'], ch_df[TARGET_COLUMN], normalize='index') * 100
    ax = ct.plot(kind='bar', stacked=True, color=['#EF4444', '#10B981'], figsize=(6, 4))
    plt.title("Credit History vs Approval Rate (%)", fontsize=12, fontweight='bold')
    plt.xlabel("Credit History (0: Defaulter/Bad, 1: Good/Meets Criteria)")
    plt.ylabel("Percentage (%)")
    plt.xticks(ticks=[0, 1], labels=['0.0 (Unfavorable)', '1.0 (Prime/Good)'], rotation=0)
    plt.legend(title="Loan Status", labels=['Rejected (N)', 'Approved (Y)'])
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_IMAGE_DIR, "credit_history_vs_approval.png"), dpi=180)
    plt.close()

    # Chart 3: Education vs Approval
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df_clean, x='Education', hue=TARGET_COLUMN, palette=['#10B981', '#EF4444'])
    plt.title("Education Level vs Loan Approval", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_IMAGE_DIR, "education_vs_approval.png"), dpi=180)
    plt.close()

    # Chart 4: Property Area vs Approval
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df_clean, x='Property_Area', hue=TARGET_COLUMN, palette=['#10B981', '#EF4444'])
    plt.title("Property Area vs Loan Approval", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_IMAGE_DIR, "property_area_vs_approval.png"), dpi=180)
    plt.close()

    # Chart 5: Model Comparison
    plt.figure(figsize=(8, 4.5))
    model_names = list(benchmarks.keys())
    test_accs = [benchmarks[m]['test_accuracy'] for m in model_names]
    cv_accs = [benchmarks[m]['cv_accuracy_mean'] for m in model_names]
    x_idx = np.arange(len(model_names))
    width = 0.35
    plt.bar(x_idx - width/2, cv_accs, width, label='5-Fold CV Accuracy (%)', color='#0284C7')
    plt.bar(x_idx + width/2, test_accs, width, label='Test Accuracy (%)', color='#10B981')
    plt.xticks(x_idx, model_names, rotation=15, fontsize=10)
    plt.ylim(65, 95)
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy Benchmarks (CV vs Test)", fontsize=13, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_IMAGE_DIR, "model_comparison_bar.png"), dpi=180)
    plt.close()

    # Chart 6: Feature Importance Bar
    plt.figure(figsize=(8, 5))
    top_f = feat_imp[:8]
    sns.barplot(x=[f['importance'] for f in top_f], y=[f['feature'] for f in top_f], palette="mako")
    plt.title("Random Forest: Top Feature Importance Factors (%)", fontsize=12, fontweight='bold')
    plt.xlabel("Importance Score (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_IMAGE_DIR, "feature_importance_bar.png"), dpi=180)
    plt.close()

    # 7. Metadata JSON Export
    meta = {
        "dataset_summary": {
            "total_records": int(df_clean.shape[0]),
            "total_features": int(df_clean.shape[1]),
            "approved_count": int((df_clean[TARGET_COLUMN] == 'Y').sum()),
            "rejected_count": int((df_clean[TARGET_COLUMN] == 'N').sum()),
            "approval_rate": round(float((df_clean[TARGET_COLUMN] == 'Y').mean() * 100), 2),
            "applicant_income_median": float(df_clean['ApplicantIncome'].median()),
            "loan_amount_median": float(df_clean['LoanAmount'].median())
        },
        "benchmark_metrics": benchmarks,
        "feature_importance": feat_imp,
        "default_model": "Random Forest",
        "supported_models": list(trained_pipelines.keys()),
        "trained_date": date.today().isoformat()
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)
    print(f"Metadata exported to {METADATA_FILE}")
    print("=" * 70)
    return meta

if __name__ == "__main__":
    train_and_evaluate()
