"""
Flask Web Application for Loan Approval Prediction System
=========================================================
Enterprise Fintech Intelligence Platform:
- Real-Time Multi-Model Underwriting (Random Forest, Logistic Regression, Decision Tree, Gradient Boosting, Extra Trees)
- All-Models Consensus Engine (Ensemble agreement & confidence distribution)
- What-If Sensitivity Simulator API (Dynamic probability vs loan amount & income curve)
- Batch Processing API (CSV bulk scoring and export)
- Loan Affordability & EMI Amortization Calculator
- Interactive Dashboard with EDA, Live Charts, and Historical Dataset Explorer
"""

import os
import io
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__)

# Paths configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_PATH = os.path.join(BASE_DIR, "dataset", "loan_data.csv")
META_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

# Load trained models & preprocessor
models = {}
preprocessor = None
metadata = {}

model_files = {
    "Random Forest": "loan_model.pkl",
    "Logistic Regression": "lr_model.pkl",
    "Decision Tree": "dt_model.pkl",
    "Gradient Boosting": "gb_model.pkl",
    "Extra Trees": "et_model.pkl"
}

for name, filename in model_files.items():
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        try:
            models[name] = joblib.load(path)
        except Exception as e:
            print(f"Warning: Could not load {name} ({filename}): {e}")

if os.path.exists(os.path.join(MODEL_DIR, "preprocessing.pkl")):
    try:
        preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessing.pkl"))
    except Exception as e:
        print(f"Warning: Could not load preprocessing.pkl: {e}")

if os.path.exists(META_PATH):
    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load metadata: {e}")

print(f"Loaded {len(models)} models and preprocessor successfully.")

# Load cached dataset for table explorer
loan_df = None
if os.path.exists(DATA_PATH):
    loan_df = pd.read_csv(DATA_PATH)


def parse_applicant_df(data):
    """Sanitize and build DataFrame for pipeline inference"""
    gender = str(data.get('Gender', 'Male')).strip()
    married = str(data.get('Married', 'Yes')).strip()
    dependents = str(data.get('Dependents', '0')).strip().replace('+', '')
    education = str(data.get('Education', 'Graduate')).strip()
    self_employed = str(data.get('Self_Employed', 'No')).strip()
    property_area = str(data.get('Property_Area', 'Semiurban')).strip()
    
    applicant_income = float(data.get('ApplicantIncome', 5000))
    coapplicant_income = float(data.get('CoapplicantIncome', 0))
    loan_amount = float(data.get('LoanAmount', 150))
    loan_term = float(data.get('Loan_Amount_Term', 360))
    credit_history = float(data.get('Credit_History', 1.0))
    
    return pd.DataFrame([{
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Property_Area': property_area
    }])


@app.route('/')
def home():
    """Landing and executive overview page"""
    return render_template('index.html', metadata=metadata)


@app.route('/predict')
def predict_page():
    """Interactive Loan Application Form & Prediction Engine"""
    return render_template('prediction.html', metadata=metadata, models=list(models.keys()))


@app.route('/dashboard')
def dashboard_page():
    """Analytics, Exploratory Data Analysis, and Model Comparison Hub"""
    return render_template('dashboard.html', metadata=metadata)


@app.route('/simulator')
def simulator_page():
    """Interactive What-If Sensitivity & Policy Simulator"""
    return render_template('simulator.html', metadata=metadata, models=list(models.keys()))


@app.route('/batch')
def batch_page():
    """Bulk Batch CSV Application Scoring & Export"""
    return render_template('batch.html', metadata=metadata)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Real-time prediction API
    Expects JSON with applicant details & optional model choice.
    Returns approval status, confidence, probabilities, monthly EMI, and factor breakdown.
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'success': False, 'error': 'No input data provided'}), 400
        
        selected_model_name = data.get('model', 'Random Forest')
        if selected_model_name not in models:
            selected_model_name = "Random Forest" if "Random Forest" in models else list(models.keys())[0]
            
        model = models[selected_model_name]
        input_df = parse_applicant_df(data)
        
        pred = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
        
        prob_approved = round(float(probabilities[1]) * 100, 2)
        prob_rejected = round(float(probabilities[0]) * 100, 2)
        confidence = prob_approved if pred == 1 else prob_rejected
        
        # Financial analytics
        app_inc = float(input_df['ApplicantIncome'].iloc[0])
        coapp_inc = float(input_df['CoapplicantIncome'].iloc[0])
        total_income = app_inc + coapp_inc
        loan_k = float(input_df['LoanAmount'].iloc[0])
        loan_term_mos = float(input_df['Loan_Amount_Term'].iloc[0])
        loan_amount_actual = loan_k * 1000
        
        # Monthly EMI calculation (8.5% annual interest)
        annual_rate = 0.085
        monthly_rate = annual_rate / 12
        if loan_term_mos > 0:
            emi = (loan_amount_actual * monthly_rate * ((1 + monthly_rate) ** loan_term_mos)) / (((1 + monthly_rate) ** loan_term_mos) - 1)
        else:
            emi = loan_amount_actual / 360
            
        total_repayment = emi * loan_term_mos
        total_interest = total_repayment - loan_amount_actual
        dti_ratio = round((emi / max(total_income, 1)) * 100, 2)
        
        # Maximum safe loan amount (36% max DTI rule)
        max_safe_emi = total_income * 0.36
        if monthly_rate > 0:
            max_safe_loan = (max_safe_emi * (((1 + monthly_rate) ** loan_term_mos) - 1)) / (monthly_rate * ((1 + monthly_rate) ** loan_term_mos))
        else:
            max_safe_loan = max_safe_emi * loan_term_mos
            
        # Actionable recommendations & mitigation strategies
        recommendations = []
        factors = []
        credit_history = float(input_df['Credit_History'].iloc[0])
        property_area = str(input_df['Property_Area'].iloc[0])
        education = str(input_df['Education'].iloc[0])
        
        if credit_history == 1.0:
            factors.append({
                "factor": "Credit History",
                "impact": "Prime (+45%)",
                "type": "positive",
                "text": "Prime institutional credit history with zero recorded defaults."
            })
        else:
            factors.append({
                "factor": "Credit History",
                "impact": "High Risk (-65%)",
                "type": "negative",
                "text": "Negative credit trail or historical payment defaults detected."
            })
            recommendations.append("Clear overdue obligations and build a 12-month consecutive payment history.")
            
        if total_income >= 6000:
            factors.append({
                "factor": "Combined Income",
                "impact": "Strong (+18%)",
                "type": "positive",
                "text": f"Substantial household earnings cushion (${total_income:,.0f}/mo)."
            })
        elif total_income < 3000:
            factors.append({
                "factor": "Combined Income",
                "impact": "Restricted (-15%)",
                "type": "negative",
                "text": f"Modest household income reserve (${total_income:,.0f}/mo)."
            })
            recommendations.append("Consider onboarding an earning co-applicant to strengthen solvency.")
            
        if dti_ratio <= 35:
            factors.append({
                "factor": "Debt-to-Income (DTI)",
                "impact": "Safe (+14%)",
                "type": "positive",
                "text": f"Comfortable repayment threshold ({dti_ratio}% of monthly income)."
            })
        else:
            factors.append({
                "factor": "Debt-to-Income (DTI)",
                "impact": "Elevated (-22%)",
                "type": "negative",
                "text": f"Debt burden at {dti_ratio}% exceeds institutional 35% guideline."
            })
            suggested_down = max(0, loan_amount_actual - max_safe_loan)
            if suggested_down > 5000:
                recommendations.append(f"Reduce requested loan by ${suggested_down:,.0f} or provide additional down payment.")
            recommendations.append("Extend loan term to 360 months to reduce monthly installment pressure.")
            
        if property_area == 'Semiurban':
            factors.append({
                "factor": "Property Demographics",
                "impact": "Favorable (+10%)",
                "type": "positive",
                "text": "Semiurban collateral exhibits highest institutional liquidity (76.8% approval)."
            })
        elif property_area == 'Rural':
            factors.append({
                "factor": "Property Demographics",
                "impact": "Conservative (-5%)",
                "type": "neutral",
                "text": "Rural asset valuation mandates conservative loan-to-value limits."
            })
            
        if education == 'Graduate':
            factors.append({
                "factor": "Educational Attainment",
                "impact": "Positive (+8%)",
                "type": "positive",
                "text": "Graduate qualification indicates resilient lifetime earning trajectory."
            })

        # Institutional verdict summary & APR recommendation
        if pred == 1:
            decision = "APPROVED"
            theme = "success"
            summary = "Application satisfies solvency and credit benchmarks. Recommended for fast-track underwriting approval."
            apr_tier = "Prime: 6.75% Fixed APR" if credit_history == 1.0 and dti_ratio <= 30 else "Standard: 7.95% APR"
        else:
            decision = "REJECTED"
            theme = "danger"
            summary = "Application fails automated threshold. Structured collateral, co-signer, or credit rehabilitation required."
            apr_tier = "Not Applicable (High Default Probability)"

        return jsonify({
            'success': True,
            'prediction': pred,
            'decision': decision,
            'theme': theme,
            'confidence': confidence,
            'prob_approved': prob_approved,
            'prob_rejected': prob_rejected,
            'active_model': selected_model_name,
            'emi_monthly': round(emi, 2),
            'emi_ratio': dti_ratio,
            'total_income': total_income,
            'principal': loan_amount_actual,
            'total_interest': round(total_interest, 2),
            'total_repayment': round(total_repayment, 2),
            'max_safe_loan': round(max_safe_loan, 2),
            'apr_tier': apr_tier,
            'factors': factors,
            'recommendations': recommendations,
            'summary': summary
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/predict/all-models', methods=['POST'])
def api_predict_all_models():
    """
    Ensemble Consensus Engine:
    Evaluates applicant across all 5 candidate models simultaneously.
    Returns consensus vote, individual predictions, and agreement metric.
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'success': False, 'error': 'No input data'}), 400
            
        input_df = parse_applicant_df(data)
        
        comparison = []
        approved_votes = 0
        total_models = len(models)
        
        for name, clf in models.items():
            pred = int(clf.predict(input_df)[0])
            prob = clf.predict_proba(input_df)[0]
            if pred == 1:
                approved_votes += 1
                
            comparison.append({
                "model": name,
                "prediction": pred,
                "decision": "APPROVED" if pred == 1 else "REJECTED",
                "confidence": round(float(prob[pred]) * 100, 2),
                "approval_prob": round(float(prob[1]) * 100, 2),
                "rejection_prob": round(float(prob[0]) * 100, 2)
            })
            
        consensus_decision = "APPROVED" if approved_votes >= (total_models / 2.0) else "REJECTED"
        agreement_pct = round((max(approved_votes, total_models - approved_votes) / total_models) * 100, 1)
        
        return jsonify({
            'success': True,
            'consensus_decision': consensus_decision,
            'approved_votes': approved_votes,
            'rejected_votes': total_models - approved_votes,
            'total_models': total_models,
            'agreement_percentage': agreement_pct,
            'models_comparison': comparison
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/sensitivity', methods=['POST'])
def api_sensitivity():
    """
    What-If Sensitivity Curve Generator:
    Sweeps requested loan amount ($20k to $400k) and applicant income ($1.5k to $20k)
    returning live probability curve coordinates for interactive plotting.
    """
    try:
        data = request.get_json(force=True)
        model_name = data.get('model', 'Random Forest')
        model = models.get(model_name, list(models.values())[0])
        
        base_df = parse_applicant_df(data)
        
        # 1. Sweep Loan Amounts
        loan_steps = np.linspace(20, 450, 15)
        loan_curve = []
        for amt in loan_steps:
            temp_df = base_df.copy()
            temp_df['LoanAmount'] = float(amt)
            prob_app = float(model.predict_proba(temp_df)[0][1]) * 100
            loan_curve.append({
                'loan_amount': round(float(amt), 0),
                'approval_prob': round(prob_app, 2)
            })
            
        # 2. Sweep Incomes
        income_steps = np.linspace(1500, 20000, 12)
        income_curve = []
        for inc in income_steps:
            temp_df = base_df.copy()
            temp_df['ApplicantIncome'] = float(inc)
            prob_app = float(model.predict_proba(temp_df)[0][1]) * 100
            income_curve.append({
                'income': round(float(inc), 0),
                'approval_prob': round(prob_app, 2)
            })
            
        return jsonify({
            'success': True,
            'model': model_name,
            'loan_curve': loan_curve,
            'income_curve': income_curve
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/batch-predict', methods=['POST'])
def api_batch_predict():
    """
    Batch Scoring Engine:
    Processes uploaded CSV file or array of applications.
    Scores each row, calculates aggregate metrics, and returns scored records.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        if not file.filename.endswith('.csv'):
            return jsonify({'success': False, 'error': 'Only CSV files supported'}), 400
            
        df_uploaded = pd.read_csv(file)
        
        # Clean columns
        clean_df = df_uploaded.copy()
        if 'Dependents' in clean_df.columns:
            clean_df['Dependents'] = clean_df['Dependents'].astype(str).str.replace('+', '', regex=False)
            
        model = models.get('Random Forest', list(models.values())[0])
        
        scored_records = []
        approved_count = 0
        
        for idx, row in clean_df.iterrows():
            app_id = row.get('Loan_ID', f"APP-{idx+1:04d}")
            input_row = pd.DataFrame([{
                'Gender': row.get('Gender', 'Male'),
                'Married': row.get('Married', 'Yes'),
                'Dependents': str(row.get('Dependents', '0')).replace('+', ''),
                'Education': row.get('Education', 'Graduate'),
                'Self_Employed': row.get('Self_Employed', 'No'),
                'ApplicantIncome': float(row.get('ApplicantIncome', 4000)),
                'CoapplicantIncome': float(row.get('CoapplicantIncome', 0)),
                'LoanAmount': float(row.get('LoanAmount', 130)),
                'Loan_Amount_Term': float(row.get('Loan_Amount_Term', 360)),
                'Credit_History': float(row.get('Credit_History', 1.0)),
                'Property_Area': row.get('Property_Area', 'Urban')
            }])
            
            pred = int(model.predict(input_row)[0])
            probs = model.predict_proba(input_row)[0]
            if pred == 1:
                approved_count += 1
                
            scored_records.append({
                'Loan_ID': app_id,
                'ApplicantIncome': float(input_row['ApplicantIncome'].iloc[0]),
                'LoanAmount': float(input_row['LoanAmount'].iloc[0]),
                'Credit_History': float(input_row['Credit_History'].iloc[0]),
                'Property_Area': str(input_row['Property_Area'].iloc[0]),
                'Prediction': pred,
                'Decision': 'APPROVED' if pred == 1 else 'REJECTED',
                'Confidence': round(float(probs[pred]) * 100, 2),
                'Approved_Prob': round(float(probs[1]) * 100, 2)
            })
            
        total = len(scored_records)
        return jsonify({
            'success': True,
            'total_scored': total,
            'approved_count': approved_count,
            'rejected_count': total - approved_count,
            'batch_approval_rate': round((approved_count / max(total, 1)) * 100, 1),
            'records': scored_records
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/sample-csv', methods=['GET'])
def api_sample_csv():
    """Download a template CSV for batch testing"""
    sample_data = """Loan_ID,Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area
BATCH001,Male,Yes,1,Graduate,No,6500,2000,150,360,1.0,Semiurban
BATCH002,Female,No,0,Graduate,No,4200,0,110,360,1.0,Urban
BATCH003,Male,Yes,2,Not Graduate,Yes,2200,0,180,360,0.0,Rural
BATCH004,Male,No,0,Graduate,No,8500,0,220,180,1.0,Urban
BATCH005,Female,Yes,1,Graduate,Yes,5000,3500,200,360,1.0,Semiurban
BATCH006,Male,Yes,0,Graduate,No,3100,1800,135,360,0.0,Rural
"""
    mem = io.BytesIO()
    mem.write(sample_data.encode('utf-8'))
    mem.seek(0)
    return send_file(
        mem,
        mimetype='text/csv',
        as_attachment=True,
        download_name='batch_loan_template.csv'
    )


@app.route('/api/metadata', methods=['GET'])
def api_metadata():
    """Return model benchmarks, feature importance, and dataset statistics"""
    return jsonify(metadata)


@app.route('/api/data', methods=['GET'])
def api_data():
    """Return paginated/filtered dataset rows for table explorer"""
    if loan_df is None:
        return jsonify({'records': [], 'total': 0})
    
    page = int(request.args.get('page', 1))
    limit = int(request.args.get('limit', 15))
    search = request.args.get('search', '').lower().strip()
    status_filter = request.args.get('status', 'all')
    
    filtered_df = loan_df.copy()
    if status_filter in ['Y', 'N']:
        filtered_df = filtered_df[filtered_df['Loan_Status'] == status_filter]
        
    if search:
        filtered_df = filtered_df[
            filtered_df['Loan_ID'].astype(str).str.lower().str.contains(search) |
            filtered_df['Property_Area'].astype(str).str.lower().str.contains(search) |
            filtered_df['Education'].astype(str).str.lower().str.contains(search)
        ]
        
    total_count = len(filtered_df)
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    
    records = filtered_df.iloc[start_idx:end_idx].fillna("N/A").to_dict(orient='records')
    
    return jsonify({
        'page': page,
        'limit': limit,
        'total': total_count,
        'records': records
    })


if __name__ == '__main__':
    print("=" * 60)
    print("CreditPulse AI Platform running on http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host='127.0.0.1', port=5000, debug=True)
