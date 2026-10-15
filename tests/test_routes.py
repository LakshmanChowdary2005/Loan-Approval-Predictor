import json
import pytest
from app import app
from src.database import save_prediction

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_page_routes(client):
    pages = [
        '/',
        '/dashboard',
        '/predict',
        '/explainability',
        '/risk-analysis',
        '/simulator',
        '/history',
        '/batch',
        '/model-performance',
        '/reports',
        '/about',
        '/login'
    ]
    for page in pages:
        res = client.get(page)
        assert res.status_code == 200, f"Page {page} returned status {res.status_code}"

def test_api_predict_valid(client):
    payload = {
        'Gender': 'Male',
        'Married': 'Yes',
        'Dependents': '1',
        'Education': 'Graduate',
        'Self_Employed': 'No',
        'ApplicantIncome': 6000,
        'CoapplicantIncome': 2000,
        'LoanAmount': 140,
        'Loan_Amount_Term': 360,
        'Credit_History': 1.0,
        'Property_Area': 'Semiurban',
        'model': 'Random Forest'
    }
    res = client.post('/api/predict', data=json.dumps(payload), content_type='application/json')
    assert res.status_code == 200
    data = res.get_json()
    assert data['success'] is True
    assert data['decision'] in ['APPROVED', 'REJECTED']
    assert 'prediction_id' in data
    assert 'emi_monthly' in data

def test_api_predict_invalid_negative(client):
    payload = {
        'ApplicantIncome': -500,
        'LoanAmount': 100
    }
    res = client.post('/api/predict', data=json.dumps(payload), content_type='application/json')
    assert res.status_code == 422
    data = res.get_json()
    assert data['success'] is False

def test_api_consensus(client):
    payload = {
        'ApplicantIncome': 5000,
        'CoapplicantIncome': 1000,
        'LoanAmount': 120,
        'Credit_History': 1.0
    }
    res = client.post('/api/predict/all-models', data=json.dumps(payload), content_type='application/json')
    assert res.status_code == 200
    data = res.get_json()
    assert data['success'] is True
    assert 'consensus_decision' in data

def test_api_sensitivity(client):
    payload = {
        'ApplicantIncome': 5000,
        'LoanAmount': 120,
        'Credit_History': 1.0
    }
    res = client.post('/api/sensitivity', data=json.dumps(payload), content_type='application/json')
    assert res.status_code == 200
    data = res.get_json()
    assert data['success'] is True
    assert len(data['loan_curve']) > 0

def test_export_pdf_and_excel(client):
    # Save a test record first
    pred_data = {
        'prediction_id': 'TEST-ROUTE-001',
        'ApplicantIncome': 5000,
        'LoanAmount': 120,
        'Credit_History': 1.0,
        'predicted_decision': 'APPROVED',
        'confidence': 85.0
    }
    save_prediction(pred_data)

    res_pdf = client.get('/export/pdf/TEST-ROUTE-001')
    assert res_pdf.status_code == 200
    assert res_pdf.mimetype == 'application/pdf'

    res_xls = client.get('/export/excel/TEST-ROUTE-001')
    assert res_xls.status_code == 200
    assert 'spreadsheet' in res_xls.mimetype

def test_export_history(client):
    res_csv = client.get('/export/history-csv')
    assert res_csv.status_code == 200
    assert 'text/csv' in res_csv.mimetype

    res_xls = client.get('/export/history-excel')
    assert res_xls.status_code == 200
