"""
Unified Prediction Service.
Coordinates model loading, single application inference, ensemble consensus,
batch CSV scoring, sensitivity curve generation, and SHAP explainability.
"""
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import joblib

import pandas as pd
import numpy as np
from config import MODEL_DIR, MODEL_FILES, DEFAULT_MODEL, CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from src.risk_analysis import calculate_financial_risk_profile
from src.explainability import ModelExplainer

class PredictionService:
    def __init__(self):
        self.models = {}
        self.explainers = {}
        self.preprocessor = None
        self._load_models()

    def _load_models(self):
        """Loads available persisted models from disk."""
        for name, filename in MODEL_FILES.items():
            path = os.path.join(MODEL_DIR, filename)
            if os.path.exists(path):
                try:
                    loaded = joblib.load(path)
                    self.models[name] = loaded
                    try:
                        self.explainers[name] = ModelExplainer(loaded)
                    except Exception:
                        pass
                except Exception as e:
                    print(f"Notice: Could not load model {name} from {filename}: {e}")

        prep_path = os.path.join(MODEL_DIR, "preprocessing.pkl")
        if os.path.exists(prep_path):
            try:
                self.preprocessor = joblib.load(prep_path)
            except Exception as e:
                print(f"Notice: Preprocessor load notice: {e}")

    def get_supported_models(self):
        """Returns list of currently loaded model algorithm names."""
        return list(self.models.keys())

    def _build_input_df(self, data):
        """Constructs standardized 1-row DataFrame for pipeline input."""
        return pd.DataFrame([{
            'Gender': str(data.get('Gender', 'Male')).strip(),
            'Married': str(data.get('Married', 'Yes')).strip(),
            'Dependents': str(data.get('Dependents', '0')).replace('+', '').strip(),
            'Education': str(data.get('Education', 'Graduate')).strip(),
            'Self_Employed': str(data.get('Self_Employed', 'No')).strip(),
            'ApplicantIncome': float(data.get('ApplicantIncome', 4000)),
            'CoapplicantIncome': float(data.get('CoapplicantIncome', 0)),
            'LoanAmount': float(data.get('LoanAmount', 120)),
            'Loan_Amount_Term': float(data.get('Loan_Amount_Term', 360)),
            'Credit_History': float(data.get('Credit_History', 1.0)),
            'Property_Area': str(data.get('Property_Area', 'Semiurban')).strip()
        }])

    def predict_single(self, applicant_data, model_name=None):
        """
        Runs prediction, probabilities, financial risk metrics, and SHAP attribution.
        """
        chosen_model = model_name if model_name in self.models else (
            DEFAULT_MODEL if DEFAULT_MODEL in self.models else list(self.models.keys())[0]
        )
        pipeline = self.models[chosen_model]
        input_df = self._build_input_df(applicant_data)

        pred = int(pipeline.predict(input_df)[0])
        probabilities = pipeline.predict_proba(input_df)[0]

        prob_approved = round(float(probabilities[1]) * 100, 2)
        prob_rejected = round(float(probabilities[0]) * 100, 2)
        confidence = prob_approved if pred == 1 else prob_rejected

        # Risk Analysis
        risk_profile = calculate_financial_risk_profile(applicant_data)

        # Explainability
        explainer = self.explainers.get(chosen_model)
        shap_summary = {}
        if explainer:
            try:
                shap_summary = explainer.explain_instance(input_df)
            except Exception as e:
                shap_summary = {"natural_language_explanation": f"Explanation summary: {e}"}

        decision = "APPROVED" if pred == 1 else "REJECTED"
        theme = "success" if pred == 1 else "danger"

        return {
            "prediction": pred,
            "decision": decision,
            "theme": theme,
            "confidence": confidence,
            "prob_approved": prob_approved,
            "prob_rejected": prob_rejected,
            "model_name": chosen_model,
            "risk_profile": risk_profile,
            "shap_summary": shap_summary,
            "input_df": input_df
        }

    def predict_consensus(self, applicant_data):
        """
        Ensemble consensus: queries all available candidate models.
        """
        input_df = self._build_input_df(applicant_data)
        votes = []
        approved_count = 0
        total_models = len(self.models)

        for name, clf in self.models.items():
            pred = int(clf.predict(input_df)[0])
            probs = clf.predict_proba(input_df)[0]
            if pred == 1:
                approved_count += 1
            votes.append({
                "model": name,
                "prediction": pred,
                "decision": "APPROVED" if pred == 1 else "REJECTED",
                "confidence": round(float(probs[pred]) * 100, 2),
                "approval_prob": round(float(probs[1]) * 100, 2),
                "rejection_prob": round(float(probs[0]) * 100, 2)
            })

        consensus_decision = "APPROVED" if approved_count >= (total_models / 2.0) else "REJECTED"
        agreement_pct = round((max(approved_count, total_models - approved_count) / max(total_models, 1)) * 100, 1)

        return {
            "consensus_decision": consensus_decision,
            "approved_votes": approved_count,
            "rejected_votes": total_models - approved_count,
            "total_models": total_models,
            "agreement_percentage": agreement_pct,
            "votes": votes
        }

    def generate_sensitivity_curve(self, applicant_data, model_name=None):
        """
        Computes dynamic probability response curves across Loan Amounts and Applicant Incomes.
        """
        chosen_model = model_name if model_name in self.models else DEFAULT_MODEL
        pipeline = self.models.get(chosen_model, list(self.models.values())[0])
        base_df = self._build_input_df(applicant_data)

        # Loan Amount Sweep ($20k to $450k)
        loan_steps = np.linspace(20, 450, 15)
        loan_curve = []
        for amt in loan_steps:
            tmp = base_df.copy()
            tmp['LoanAmount'] = float(amt)
            prob = float(pipeline.predict_proba(tmp)[0][1]) * 100.0
            loan_curve.append({
                "loan_amount": round(float(amt), 0),
                "approval_prob": round(prob, 2)
            })

        # Income Sweep ($1,500 to $20,000)
        inc_steps = np.linspace(1500, 20000, 12)
        income_curve = []
        for inc in inc_steps:
            tmp = base_df.copy()
            tmp['ApplicantIncome'] = float(inc)
            prob = float(pipeline.predict_proba(tmp)[0][1]) * 100.0
            income_curve.append({
                "income": round(float(inc), 0),
                "approval_prob": round(prob, 2)
            })

        return {
            "model_name": chosen_model,
            "loan_curve": loan_curve,
            "income_curve": income_curve
        }

    def batch_predict(self, valid_rows, model_name=None):
        """
        Scores an array of pre-validated applicant dictionaries.
        """
        chosen_model = model_name if model_name in self.models else DEFAULT_MODEL
        pipeline = self.models.get(chosen_model, list(self.models.values())[0])

        scored_records = []
        approved_count = 0

        for row in valid_rows:
            input_df = self._build_input_df(row)
            pred = int(pipeline.predict(input_df)[0])
            probs = pipeline.predict_proba(input_df)[0]
            if pred == 1:
                approved_count += 1

            app_inc = float(row.get('ApplicantIncome', 0))
            coapp_inc = float(row.get('CoapplicantIncome', 0))
            loan_k = float(row.get('LoanAmount', 0))
            term = float(row.get('Loan_Amount_Term', 360))
            ch = float(row.get('Credit_History', 1.0))

            # Quick risk tier
            tot_inc = app_inc + coapp_inc
            tier = "LOW" if ch == 1.0 and tot_inc > 3500 else ("MODERATE" if ch == 1.0 else "HIGH")

            scored_records.append({
                "Loan_ID": row.get('Loan_ID', 'N/A'),
                "Gender": row.get('Gender', 'Male'),
                "Married": row.get('Married', 'Yes'),
                "Dependents": row.get('Dependents', '0'),
                "Education": row.get('Education', 'Graduate'),
                "Self_Employed": row.get('Self_Employed', 'No'),
                "ApplicantIncome": app_inc,
                "CoapplicantIncome": coapp_inc,
                "LoanAmount": loan_k,
                "Loan_Amount_Term": term,
                "Credit_History": ch,
                "Property_Area": row.get('Property_Area', 'Urban'),
                "Prediction": pred,
                "Decision": "APPROVED" if pred == 1 else "REJECTED",
                "Confidence": round(float(probs[pred]) * 100, 2),
                "Approved_Prob": round(float(probs[1]) * 100, 2),
                "Risk_Tier": tier
            })

        total = len(scored_records)
        return {
            "total_scored": total,
            "approved_count": approved_count,
            "rejected_count": total - approved_count,
            "approval_rate": round((approved_count / max(total, 1)) * 100, 1),
            "records": scored_records
        }
