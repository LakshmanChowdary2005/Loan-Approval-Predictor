"""
Loan Approval Prediction - Machine Learning Model Training Pipeline
====================================================================
This script trains, evaluates, and exports multiple classification models
for predicting loan approval with high accuracy, explainability, and robust preprocessing.
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

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

def run_training():
    print("=" * 70)
    print("LOAN APPROVAL PREDICTION - MODEL TRAINING & EVALUATION PIPELINE")
    print("=" * 70)
    
    # 1. Directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "dataset", "loan_data.csv")
    model_dir = os.path.join(base_dir, "model")
    img_dir = os.path.join(base_dir, "static", "images", "eda")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    
    # 2. Load Dataset
    print(f"\n[1/7] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 3. Clean & Prepare Dataset
    print("\n[2/7] Preprocessing & Cleaning raw records...")
    df_clean = df.copy()
    
    # Clean Dependents: remove '+' and handle missing
    df_clean['Dependents'] = df_clean['Dependents'].astype(str).str.replace('+', '', regex=False)
    df_clean['Dependents'] = df_clean['Dependents'].replace('nan', np.nan)
    
    # Target encoding: Y -> 1, N -> 0
    df_clean['Loan_Status_Code'] = df_clean['Loan_Status'].map({'Y': 1, 'N': 0})
    
    # Define categorical and numerical features
    cat_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    num_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    
    X = df_clean[cat_features + num_features]
    y = df_clean['Loan_Status_Code']
    
    # Stratified Train/Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
    print(f"Class distribution: Approved={y.sum()} ({y.mean()*100:.1f}%), Rejected={len(y)-y.sum()} ({(1-y.mean())*100:.1f}%)")
    
    # 4. Generate & Save EDA Visualizations
    print("\n[3/7] Generating Exploratory Data Analysis (EDA) visualizations...")
    sns.set_theme(style="whitegrid", palette="muted")
    
    # Visual 1: Loan Status Distribution
    plt.figure(figsize=(7, 5))
    counts = df_clean['Loan_Status'].value_counts()
    colors = ['#22c55e', '#ef4444']
    ax = sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=colors, legend=False)
    plt.title("Loan Approval Status Distribution", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Loan Status (Y: Approved, N: Rejected)", fontsize=11)
    plt.ylabel("Number of Applications", fontsize=11)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())} ({p.get_height()/len(df_clean)*100:.1f}%)", 
                    (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                    ha='center', va='center', fontsize=12, color='white', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "loan_status_distribution.png"), dpi=200)
    plt.close()
    
    # Visual 2: Credit History vs Loan Status
    plt.figure(figsize=(7, 5))
    ch_df = df_clean.dropna(subset=['Credit_History'])
    ch_crosstab = pd.crosstab(ch_df['Credit_History'], ch_df['Loan_Status'], normalize='index') * 100
    ax = ch_crosstab.plot(kind='bar', stacked=True, color=['#ef4444', '#22c55e'], figsize=(7, 5))
    plt.title("Credit History vs Loan Approval Rate (%)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Credit History (0.0: Bad Credit, 1.0: Good Credit)", fontsize=11)
    plt.ylabel("Percentage (%)", fontsize=11)
    plt.xticks(ticks=[0, 1], labels=['0.0 (Poor/Defaulted)', '1.0 (Meets Guidelines)'], rotation=0)
    plt.legend(title="Loan Status", labels=['Rejected', 'Approved'])
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "credit_history_vs_approval.png"), dpi=200)
    plt.close()
    
    # Visual 3: Education vs Loan Status
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df_clean, x='Education', hue='Loan_Status', palette=['#22c55e', '#ef4444'])
    plt.title("Applicant Education vs Loan Approval", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Education Level", fontsize=11)
    plt.ylabel("Number of Applications", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "education_vs_approval.png"), dpi=200)
    plt.close()
    
    # Visual 4: Property Area vs Loan Status
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df_clean, x='Property_Area', hue='Loan_Status', palette=['#22c55e', '#ef4444'])
    plt.title("Property Area Location vs Loan Approval", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Property Area", fontsize=11)
    plt.ylabel("Number of Applications", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "property_area_vs_approval.png"), dpi=200)
    plt.close()
    
    # Visual 5: Income vs Loan Amount distribution
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df_clean, x='ApplicantIncome', y='LoanAmount',
        hue='Loan_Status', palette={'Y': '#22c55e', 'N': '#ef4444'},
        alpha=0.8, s=60
    )
    plt.title("Applicant Income vs Requested Loan Amount ($000s)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Applicant Income ($)", fontsize=11)
    plt.ylabel("Loan Amount ($k)", fontsize=11)
    plt.xlim(0, 30000)
    plt.ylim(0, 500)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "income_vs_loan_scatter.png"), dpi=200)
    plt.close()
    
    # Visual 6: Correlation Heatmap
    plt.figure(figsize=(8, 6))
    num_df = df_clean[num_features + ['Loan_Status_Code']].copy()
    num_df = num_df.fillna(num_df.median())
    corr = num_df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
    plt.title("Feature Correlation Matrix with Loan Approval", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "correlation_heatmap.png"), dpi=200)
    plt.close()
    
    print("EDA Visualizations successfully saved to static/images/eda/")
    
    # 5. Build Preprocessing Pipelines
    print("\n[4/7] Constructing robust Scikit-Learn Preprocessing Pipeline...")
    
    # Numeric pipeline: median imputation + standard scaling
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline: most frequent imputation + OneHotEncoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ])
    
    # Fit preprocessor on training data
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, os.path.join(model_dir, "preprocessing.pkl"))
    print("Preprocessing pipeline fitted and saved to model/preprocessing.pkl")
    
    # Extract encoded feature names for explainability
    encoded_cat_names = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(cat_features)
    all_feature_names = list(num_features) + list(encoded_cat_names)
    
    # 6. Define & Train Multiple Classification Models
    print("\n[5/7] Training and evaluating candidate classification algorithms...")
    candidate_models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=250, max_depth=6, min_samples_split=4, min_samples_leaf=2, random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=5, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=120, max_depth=3, learning_rate=0.05, random_state=42
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200, max_depth=5, min_samples_split=4, random_state=42
        )
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    benchmark_metrics = {}
    trained_pipelines = {}
    
    for name, clf in candidate_models.items():
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        # 5-Fold Stratified Cross Validation
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='accuracy')
        
        # Train on full training split
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe
        
        # Predictions & Probabilities on Test set
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        benchmark_metrics[name] = {
            "cv_accuracy_mean": round(float(cv_scores.mean() * 100), 2),
            "cv_accuracy_std": round(float(cv_scores.std() * 100), 2),
            "test_accuracy": round(float(acc * 100), 2),
            "precision": round(float(prec * 100), 2),
            "recall": round(float(rec * 100), 2),
            "f1_score": round(float(f1 * 100), 2),
            "roc_auc": round(float(roc * 100), 2),
            "confusion_matrix": cm,
            "classification_report": classification_report(y_test, y_pred, target_names=['Rejected (0)', 'Approved (1)'], output_dict=True)
        }
        
        print(f" -> {name:20s} | CV Acc: {cv_scores.mean()*100:5.2f}% (+/- {cv_scores.std()*100:4.2f}%) | Test Acc: {acc*100:5.2f}% | F1: {f1*100:5.2f}% | ROC-AUC: {roc*100:5.2f}%")

    # 7. Select & Save Models
    print("\n[6/7] Exporting trained models and feature importance...")
    
    # Save individual pipelines
    joblib.dump(trained_pipelines["Random Forest"], os.path.join(model_dir, "loan_model.pkl")) # Default production model
    joblib.dump(trained_pipelines["Logistic Regression"], os.path.join(model_dir, "lr_model.pkl"))
    joblib.dump(trained_pipelines["Decision Tree"], os.path.join(model_dir, "dt_model.pkl"))
    joblib.dump(trained_pipelines["Gradient Boosting"], os.path.join(model_dir, "gb_model.pkl"))
    joblib.dump(trained_pipelines["Extra Trees"], os.path.join(model_dir, "et_model.pkl"))
    
    # Extract Feature Importances from Random Forest
    rf_clf = trained_pipelines["Random Forest"].named_steps['classifier']
    importances = rf_clf.feature_importances_
    
    feature_imp_list = []
    for feat_name, imp_val in zip(all_feature_names, importances):
        # Friendly readable names
        friendly_name = feat_name.replace('num__', '').replace('cat__', '').replace('_', ' ')
        feature_imp_list.append({
            "raw_feature": feat_name,
            "feature": friendly_name.title(),
            "importance": round(float(imp_val * 100), 2)
        })
    
    feature_imp_list = sorted(feature_imp_list, key=lambda x: x['importance'], reverse=True)
    
    # Visual 7: Feature Importance Bar Plot
    plt.figure(figsize=(9, 6))
    top_feats = feature_imp_list[:8]
    sns.barplot(
        x=[f['importance'] for f in top_feats],
        y=[f['feature'] for f in top_feats],
        palette="viridis"
    )
    plt.title("Random Forest: Top Feature Importance Factors (%)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Relative Importance Score (%)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "feature_importance_bar.png"), dpi=200)
    plt.close()
    
    # Visual 8: Model Comparison Bar Chart
    plt.figure(figsize=(9, 5))
    model_names = list(benchmark_metrics.keys())
    test_accs = [benchmark_metrics[m]['test_accuracy'] for m in model_names]
    cv_accs = [benchmark_metrics[m]['cv_accuracy_mean'] for m in model_names]
    
    x_idx = np.arange(len(model_names))
    width = 0.35
    plt.bar(x_idx - width/2, cv_accs, width, label='5-Fold CV Accuracy (%)', color='#6366f1')
    plt.bar(x_idx + width/2, test_accs, width, label='Test Accuracy (%)', color='#10b981')
    plt.xticks(x_idx, model_names, rotation=15, fontsize=10)
    plt.ylim(65, 95)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.title("Candidate Model Performance Comparison", fontsize=14, fontweight='bold', pad=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "model_comparison_bar.png"), dpi=200)
    plt.close()
    
    # Visual 9: Confusion Matrix for Random Forest
    rf_cm = np.array(benchmark_metrics["Random Forest"]["confusion_matrix"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['Predicted Rejected (N)', 'Predicted Approved (Y)'],
                yticklabels=['Actual Rejected (N)', 'Actual Approved (Y)'])
    plt.title("Random Forest Confusion Matrix", fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, "confusion_matrix_rf.png"), dpi=200)
    plt.close()
    
    # 8. Save Comprehensive Metadata JSON
    print("\n[7/7] Generating and saving metadata package (model_metadata.json)...")
    dataset_summary = {
        "total_records": int(df.shape[0]),
        "total_features": int(df.shape[1]),
        "approved_count": int((df['Loan_Status'] == 'Y').sum()),
        "rejected_count": int((df['Loan_Status'] == 'N').sum()),
        "approval_rate": round(float((df['Loan_Status'] == 'Y').mean() * 100), 2),
        "rejection_rate": round(float((df['Loan_Status'] == 'N').mean() * 100), 2),
        "missing_values": {col: int(df[col].isnull().sum()) for col in df.columns},
        "applicant_income_median": float(df['ApplicantIncome'].median()),
        "loan_amount_median": float(df['LoanAmount'].median()),
        "credit_history_mode": 1.0,
        "features": {
            "categorical": cat_features,
            "numerical": num_features,
            "target": "Loan_Status"
        }
    }
    
    metadata = {
        "dataset_summary": dataset_summary,
        "benchmark_metrics": benchmark_metrics,
        "feature_importance": feature_imp_list,
        "default_model": "Random Forest",
        "supported_models": list(candidate_models.keys()),
        "trained_date": "2026-10-04"
    }
    
    meta_path = os.path.join(model_dir, "model_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Model metadata saved to: {meta_path}")
    print("=" * 70)
    print("TRAINING PIPELINE COMPLETE! All models, artifacts, and figures ready.")
    print("=" * 70)

if __name__ == "__main__":
    run_training()
