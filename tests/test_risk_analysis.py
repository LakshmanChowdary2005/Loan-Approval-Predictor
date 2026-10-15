import pytest
from src.risk_analysis import calculate_financial_risk_profile

def test_risk_analysis_low_risk():
    data = {
        'ApplicantIncome': 8000,
        'CoapplicantIncome': 2000,
        'LoanAmount': 150,
        'Loan_Amount_Term': 360,
        'Credit_History': 1.0
    }
    profile = calculate_financial_risk_profile(data)
    assert profile['risk_tier'] == 'LOW'
    assert profile['total_monthly_income'] == 10000
    assert profile['dti_ratio'] < 36.0
    assert profile['emi_monthly'] > 0
    assert len(profile['assumptions']) > 0

def test_risk_analysis_high_risk():
    data = {
        'ApplicantIncome': 1500,
        'CoapplicantIncome': 0,
        'LoanAmount': 300,
        'Loan_Amount_Term': 360,
        'Credit_History': 0.0
    }
    profile = calculate_financial_risk_profile(data)
    assert profile['risk_tier'] == 'HIGH'
    assert "Poor Credit History (0.0)" in profile['risk_factors']
