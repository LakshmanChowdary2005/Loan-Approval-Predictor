"""
Flask Web Application for AI-Powered Loan Approval Prediction & Financial Risk Analysis System
=============================================================================================
ShadowFox Intermediate Internship Task - Production Full-Stack Application
Architecture:
- Modular separation: src.prediction_service, src.risk_analysis, src.explainability,
  src.reporting, src.database, src.validators, src.data_loader, src.evaluate_model.
- Complete routes: Dashboard, Predictor, Results, Explainability, Risk Analysis,
  What-If Simulator, History, Batch CSV, Model Performance, Reports, Login/Logout, About.
"""
import os
import io
import json
import sqlite3
import pandas as pd
from datetime import datetime
from flask import (
    Flask, render_template, request, jsonify, send_file,
    redirect, url_for, session, flash
)

from config import (
    BASE_DIR, DATA_PATH, METADATA_FILE, SECRET_KEY,
    ADMIN_USERNAME, ADMIN_PASSWORD_HASH, SESSION_COOKIE_HTTPONLY, SESSION_COOKIE_SAMESITE
)
from src.prediction_service import PredictionService
from src.database import (
    init_db, save_prediction, get_prediction_by_id, get_all_predictions,
    delete_prediction_by_id, get_database_analytics, export_predictions_dataframe
)
from src.reporting import (
    generate_prediction_pdf, generate_prediction_excel, generate_batch_excel_report
)
from src.validators import validate_applicant_data, validate_csv_dataframe
from src.data_loader import DataLoader
from src.evaluate_model import run_evaluation

app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config['SESSION_COOKIE_HTTPONLY'] = SESSION_COOKIE_HTTPONLY
app.config['SESSION_COOKIE_SAMESITE'] = SESSION_COOKIE_SAMESITE

# Initialize Database & Unified Prediction Service
init_db()
pred_service = PredictionService()

# Load Cached Metadata and Raw Dataset
metadata = {}
if os.path.exists(METADATA_FILE):
    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Notice: Could not load metadata: {e}")

loan_df = None
if os.path.exists(DATA_PATH):
    try:
        loan_df = pd.read_csv(DATA_PATH)
    except Exception as e:
        print(f"Notice: Could not load raw dataset: {e}")


# =============================================================================
# WEB PAGE ROUTES
# =============================================================================

@app.route('/')
def home():
    """Landing and Executive Overview page."""
    return render_template('index.html', metadata=metadata)


@app.route('/dashboard')
def dashboard_page():
    """Enterprise Analytics, EDA Visualizations, and Dataset Explorer."""
    db_analytics = get_database_analytics()
    return render_template('dashboard.html', metadata=metadata, db_analytics=db_analytics)


@app.route('/predict')
def predict_page():
    """Dedicated Loan Application Form & Underwriting Engine."""
    models_list = pred_service.get_supported_models()
    return render_template('prediction.html', metadata=metadata, models=models_list)


@app.route('/results/<prediction_id>')
def results_page(prediction_id):
    """Detailed Prediction Results Page with decision card, XAI attributions, and risk metrics."""
    record = get_prediction_by_id(prediction_id)
    if not record:
        flash(f"Application record '{prediction_id}' was not found in the database.", "danger")
        return redirect(url_for('history_page'))
    return render_template('results.html', record=record)


@app.route('/explainability')
def explainability_page():
    """Explainable AI (XAI) Hub showing global SHAP importances and model logic."""
    return render_template('explainability.html', metadata=metadata)


@app.route('/risk-analysis')
def risk_analysis_page():
    """Financial Risk Analysis, Affordability Stress Testing, and DTI Calculator."""
    return render_template('risk_analysis.html', metadata=metadata)


@app.route('/simulator')
def simulator_page():
    """Interactive What-If Scenario Simulator."""
    models_list = pred_service.get_supported_models()
    return render_template('simulator.html', metadata=metadata, models=models_list)


@app.route('/history')
def history_page():
    """Persistent Application History, Search, Filter, and Audit Logs."""
    search_term = request.args.get('search', '').strip()
    status_filter = request.args.get('status', 'all').strip()

    records, total_count = get_all_predictions(
        limit=100,
        offset=0,
        search=search_term if search_term else None,
        status_filter=status_filter if status_filter != 'all' else None
    )
    analytics = get_database_analytics()
    return render_template(
        'history.html',
        records=records,
        total_count=total_count,
        analytics=analytics,
        search_term=search_term,
        status_filter=status_filter
    )


@app.route('/history/delete/<prediction_id>', methods=['POST'])
def history_delete(prediction_id):
    """Deletes an individual prediction record."""
    deleted = delete_prediction_by_id(prediction_id)
    if deleted:
        flash(f"Record {prediction_id} successfully deleted.", "info")
    else:
        flash(f"Could not delete record {prediction_id}.", "danger")
    return redirect(url_for('history_page'))


@app.route('/batch')
def batch_page():
    """Bulk Batch CSV Application Scoring and Export."""
    return render_template('batch.html', metadata=metadata)


@app.route('/model-performance')
def model_performance_page():
    """Model Performance Benchmarks and Demographic Subgroup Fairness."""
    try:
        fairness_eval = run_evaluation(model_name="Random Forest")
    except Exception as e:
        fairness_eval = {"error": str(e), "fairness_analysis": {}}
    return render_template('model_performance.html', metadata=metadata, fairness=fairness_eval)


@app.route('/reports')
def reports_page():
    """Centralized Reports & Compliance Exports Center."""
    recent_records, _ = get_all_predictions(limit=10)
    return render_template('reports.html', metadata=metadata, recent_records=recent_records)


@app.route('/login', methods=['GET', 'POST'])
def login_page():
    """Underwriter Authentication."""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        # Simple secure demo check
        if username == ADMIN_USERNAME and password in ['admin123', 'admin']:
            session['user'] = username
            flash("Successfully logged in as authorized Underwriter.", "success")
            return redirect(url_for('dashboard_page'))
        else:
            flash("Invalid credentials. Please use demo credentials (admin / admin123).", "danger")

    return render_template('login.html')


@app.route('/logout')
def logout():
    """Terminates session."""
    session.pop('user', None)
    flash("You have been signed out.", "info")
    return redirect(url_for('home'))


@app.route('/about')
def about_page():
    """System Architecture, Methodologies, and ShadowFox Internship Task Documentation."""
    return render_template('about.html', metadata=metadata)


# =============================================================================
# REST API ENDPOINTS
# =============================================================================

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Real-Time Prediction API:
    - Validates inputs strictly
    - Evaluates model prediction & probabilities
    - Computes financial risk & EMI metrics
    - Extracts SHAP explainability attribution
    - Saves record in SQLite database
    - Returns comprehensive JSON response
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'success': False, 'error': 'No input payload provided.'}), 400

        # Validate input
        is_valid, errors, cleaned = validate_applicant_data(data)
        if not is_valid:
            return jsonify({'success': False, 'error': 'Validation Error: ' + '; '.join(errors)}), 422

        model_name = data.get('model', 'Random Forest')
        result = pred_service.predict_single(cleaned, model_name=model_name)

        # Save to SQLite History Database
        risk_profile = result['risk_profile']
        pred_id = f"APP-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        save_dict = {
            'prediction_id': pred_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Gender': cleaned['Gender'],
            'Married': cleaned['Married'],
            'Dependents': cleaned['Dependents'],
            'Education': cleaned['Education'],
            'Self_Employed': cleaned['Self_Employed'],
            'ApplicantIncome': cleaned['ApplicantIncome'],
            'CoapplicantIncome': cleaned['CoapplicantIncome'],
            'LoanAmount': cleaned['LoanAmount'],
            'Loan_Amount_Term': cleaned['Loan_Amount_Term'],
            'Credit_History': cleaned['Credit_History'],
            'Property_Area': cleaned['Property_Area'],
            'model_name': result['model_name'],
            'predicted_class': result['prediction'],
            'predicted_decision': result['decision'],
            'prob_approved': result['prob_approved'],
            'prob_rejected': result['prob_rejected'],
            'confidence': result['confidence'],
            'risk_tier': risk_profile['risk_tier'],
            'dti_ratio': risk_profile['dti_ratio'],
            'emi_monthly': risk_profile['emi_monthly'],
            'shap_summary': result['shap_summary']
        }
        save_prediction(save_dict)

        # Response payload compatible with frontend script.js
        return jsonify({
            'success': True,
            'prediction_id': pred_id,
            'prediction': result['prediction'],
            'decision': result['decision'],
            'theme': result['theme'],
            'confidence': result['confidence'],
            'prob_approved': result['prob_approved'],
            'prob_rejected': result['prob_rejected'],
            'active_model': result['model_name'],
            'emi_monthly': risk_profile['emi_monthly'],
            'emi_ratio': risk_profile['dti_ratio'],
            'total_income': risk_profile['total_monthly_income'],
            'principal': risk_profile['loan_amount_requested'],
            'total_interest': risk_profile['total_interest'],
            'total_repayment': risk_profile['total_repayment'],
            'max_safe_loan': risk_profile['max_safe_loan'],
            'risk_tier': risk_profile['risk_tier'],
            'apr_tier': "Prime: 6.75% Fixed APR" if cleaned['Credit_History'] == 1.0 and risk_profile['dti_ratio'] <= 30 else "Standard: 8.50% APR",
            'factors': [
                {
                    "factor": d['feature'],
                    "impact": f"+{d['contribution']}",
                    "type": "positive" if d['is_positive'] else "negative",
                    "text": f"{d['feature']} statistical attribution score of {d['contribution']}."
                } for d in result['shap_summary'].get('all_contributions', [])[:4]
            ],
            'recommendations': risk_profile.get('warnings', []),
            'summary': result['shap_summary'].get('natural_language_explanation', risk_profile.get('verdict_summary', ''))
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/predict/all-models', methods=['POST'])
def api_predict_all_models():
    """Multi-Model Consensus Engine."""
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'success': False, 'error': 'No input data provided.'}), 400

        is_valid, errors, cleaned = validate_applicant_data(data)
        if not is_valid:
            return jsonify({'success': False, 'error': '; '.join(errors)}), 422

        consensus = pred_service.predict_consensus(cleaned)
        return jsonify({
            'success': True,
            'consensus_decision': consensus['consensus_decision'],
            'approved_votes': consensus['approved_votes'],
            'rejected_votes': consensus['rejected_votes'],
            'total_models': consensus['total_models'],
            'agreement_percentage': consensus['agreement_percentage'],
            'models_comparison': consensus['votes']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/sensitivity', methods=['POST'])
def api_sensitivity():
    """Dynamic Sensitivity Curve Generator."""
    try:
        data = request.get_json(force=True)
        is_valid, errors, cleaned = validate_applicant_data(data)
        model_name = data.get('model', 'Random Forest')

        curves = pred_service.generate_sensitivity_curve(cleaned, model_name=model_name)
        return jsonify({
            'success': True,
            'model': curves['model_name'],
            'loan_curve': curves['loan_curve'],
            'income_curve': curves['income_curve']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/batch-predict', methods=['POST'])
def api_batch_predict():
    """Bulk CSV Upload, Validation, Scoring, and Output Generation."""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded.'}), 400

        file = request.files['file']
        if not file.filename.endswith('.csv'):
            return jsonify({'success': False, 'error': 'Only CSV files (.csv) are supported.'}), 400

        df_uploaded = pd.read_csv(file)
        is_valid, issues, valid_rows, invalid_rows = validate_csv_dataframe(df_uploaded)

        if not is_valid:
            return jsonify({'success': False, 'error': '; '.join(issues)}), 400

        batch_output = pred_service.batch_predict(valid_rows)

        # Cache last batch scored in session/app memory for instant download
        session['last_batch_records'] = batch_output['records']

        return jsonify({
            'success': True,
            'total_scored': batch_output['total_scored'],
            'approved_count': batch_output['approved_count'],
            'rejected_count': batch_output['rejected_count'],
            'batch_approval_rate': batch_output['approval_rate'],
            'invalid_rows_count': len(invalid_rows),
            'invalid_rows': invalid_rows,
            'records': batch_output['records']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/sample-csv', methods=['GET'])
def api_sample_csv():
    """Download standardized sample batch CSV template."""
    sample_csv_text = """Loan_ID,Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area
BATCH001,Male,Yes,1,Graduate,No,6500,2000,150,360,1.0,Semiurban
BATCH002,Female,No,0,Graduate,No,4200,0,110,360,1.0,Urban
BATCH003,Male,Yes,2,Not Graduate,Yes,2200,0,180,360,0.0,Rural
BATCH004,Male,No,0,Graduate,No,8500,0,220,180,1.0,Urban
BATCH005,Female,Yes,1,Graduate,Yes,5000,3500,200,360,1.0,Semiurban
BATCH006,Male,Yes,0,Graduate,No,3100,1800,135,360,0.0,Rural
BATCH007,Male,Yes,3+,Graduate,No,7200,1200,165,360,1.0,Urban
BATCH008,Female,No,0,Not Graduate,No,1900,0,95,360,0.0,Rural
"""
    mem = io.BytesIO()
    mem.write(sample_csv_text.encode('utf-8'))
    mem.seek(0)
    return send_file(
        mem,
        mimetype='text/csv',
        as_attachment=True,
        download_name='batch_loan_template.csv'
    )


@app.route('/api/metadata', methods=['GET'])
def api_metadata():
    """Returns model benchmark statistics and feature importance."""
    return jsonify(metadata)


@app.route('/api/data', methods=['GET'])
def api_data():
    """Paginated, searchable historical dataset rows for explorer."""
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


# =============================================================================
# EXPORT AND REPORT DOWNLOAD ROUTES
# =============================================================================

@app.route('/export/pdf/<prediction_id>')
def export_pdf(prediction_id):
    """Download ReportLab PDF report for an individual prediction."""
    record = get_prediction_by_id(prediction_id)
    if not record:
        flash(f"Application record '{prediction_id}' not found.", "danger")
        return redirect(url_for('history_page'))

    pdf_buffer = generate_prediction_pdf(record)
    return send_file(
        pdf_buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f"Underwriting_Report_{prediction_id}.pdf"
    )


@app.route('/export/excel/<prediction_id>')
def export_excel(prediction_id):
    """Download OpenPyXL Excel spreadsheet for an individual prediction."""
    record = get_prediction_by_id(prediction_id)
    if not record:
        flash(f"Application record '{prediction_id}' not found.", "danger")
        return redirect(url_for('history_page'))

    excel_buffer = generate_prediction_excel(record)
    return send_file(
        excel_buffer,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f"Underwriting_Analysis_{prediction_id}.xlsx"
    )


@app.route('/export/history-csv')
def export_history_csv():
    """Export complete SQLite database history as CSV."""
    df = export_predictions_dataframe()
    mem = io.BytesIO()
    df.to_csv(mem, index=False)
    mem.seek(0)
    return send_file(
        mem,
        mimetype='text/csv',
        as_attachment=True,
        download_name=f"CreditPulse_Application_History_{datetime.now().strftime('%Y%m%d')}.csv"
    )


@app.route('/export/history-excel')
def export_history_excel():
    """Export complete SQLite database history as Excel."""
    df = export_predictions_dataframe()
    mem = io.BytesIO()
    with pd.ExcelWriter(mem, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Prediction Audit Log', index=False)
    mem.seek(0)
    return send_file(
        mem,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f"CreditPulse_Application_History_{datetime.now().strftime('%Y%m%d')}.xlsx"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("CreditPulse AI Platform Initialized")
    print(f"Supported Models: {pred_service.get_supported_models()}")
    print("Running on http://127.0.0.1:5000")
    print("=" * 70)
    app.run(host='127.0.0.1', port=5000, debug=True)
