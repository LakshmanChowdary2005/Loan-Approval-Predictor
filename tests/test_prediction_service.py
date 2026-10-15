import pytest
from src.prediction_service import PredictionService

def test_prediction_service_single():
    service = PredictionService()
    assert len(service.get_supported_models()) > 0
    
    app_data = {
        'ApplicantIncome': 5000,
        'CoapplicantIncome': 2000,
        'LoanAmount': 130,
        'Loan_Amount_Term': 360,
        'Credit_History': 1.0,
        'Property_Area': 'Semiurban',
        'Education': 'Graduate'
    }
    result = service.predict_single(app_data)
    assert 'prediction' in result
    assert result['prediction'] in [0, 1]
    assert 'decision' in result
    assert result['confidence'] >= 50.0
    assert 'prob_approved' in result
    assert 'prob_rejected' in result
    assert 'risk_profile' in result

def test_prediction_service_consensus():
    service = PredictionService()
    app_data = {
        'ApplicantIncome': 6000,
        'CoapplicantIncome': 1500,
        'LoanAmount': 140,
        'Loan_Amount_Term': 360,
        'Credit_History': 1.0
    }
    consensus = service.predict_consensus(app_data)
    assert 'consensus_decision' in consensus
    assert consensus['total_models'] == len(service.get_supported_models())
    assert consensus['agreement_percentage'] >= 50.0

def test_prediction_service_sensitivity():
    service = PredictionService()
    app_data = {
        'ApplicantIncome': 5000,
        'LoanAmount': 130,
        'Credit_History': 1.0
    }
    sensitivity = service.generate_sensitivity_curve(app_data)
    assert 'loan_curve' in sensitivity
    assert 'income_curve' in sensitivity
    assert len(sensitivity['loan_curve']) > 5
