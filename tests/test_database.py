import os
import pytest
from src.database import (
    init_db, save_prediction, get_prediction_by_id,
    get_all_predictions, delete_prediction_by_id, get_database_analytics
)

def test_database_lifecycle():
    init_db()
    pred_data = {
        'prediction_id': 'TEST-APP-001',
        'ApplicantIncome': 6000,
        'CoapplicantIncome': 1000,
        'LoanAmount': 140,
        'Loan_Amount_Term': 360,
        'Credit_History': 1.0,
        'predicted_class': 1,
        'predicted_decision': 'APPROVED',
        'prob_approved': 88.5,
        'prob_rejected': 11.5,
        'confidence': 88.5,
        'risk_tier': 'LOW',
        'dti_ratio': 22.4,
        'emi_monthly': 1076.5
    }
    pred_id = save_prediction(pred_data)
    assert pred_id == 'TEST-APP-001'

    fetched = get_prediction_by_id('TEST-APP-001')
    assert fetched is not None
    assert fetched['predicted_decision'] == 'APPROVED'
    assert fetched['applicant_income'] == 6000.0

    records, count = get_all_predictions(limit=10, search='TEST-APP-001')
    assert count >= 1

    analytics = get_database_analytics()
    assert analytics['total_predictions'] >= 1

    deleted = delete_prediction_by_id('TEST-APP-001')
    assert deleted is True

    fetched_after = get_prediction_by_id('TEST-APP-001')
    assert fetched_after is None
