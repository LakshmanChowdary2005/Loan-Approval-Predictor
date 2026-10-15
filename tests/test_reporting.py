import io
import pytest
from src.reporting import generate_prediction_pdf, generate_prediction_excel, generate_batch_excel_report

def test_generate_pdf():
    record = {
        'prediction_id': 'TEST-PDF-001',
        'Gender': 'Male',
        'Married': 'Yes',
        'Dependents': '0',
        'Education': 'Graduate',
        'Self_Employed': 'No',
        'ApplicantIncome': 5500,
        'CoapplicantIncome': 1200,
        'LoanAmount': 130,
        'Loan_Amount_Term': 360,
        'Credit_History': 1.0,
        'Property_Area': 'Urban',
        'predicted_decision': 'APPROVED',
        'confidence': 85.0,
        'prob_approved': 85.0,
        'prob_rejected': 15.0,
        'risk_tier': 'LOW',
        'dti_ratio': 23.5,
        'emi_monthly': 999.5
    }
    pdf_buffer = generate_prediction_pdf(record)
    assert isinstance(pdf_buffer, io.BytesIO)
    content = pdf_buffer.getvalue()
    assert len(content) > 1000
    assert content.startswith(b'%PDF')

def test_generate_excel():
    record = {
        'prediction_id': 'TEST-XLS-001',
        'ApplicantIncome': 5000,
        'LoanAmount': 150,
        'predicted_decision': 'APPROVED',
        'confidence': 80.0
    }
    excel_buffer = generate_prediction_excel(record)
    assert isinstance(excel_buffer, io.BytesIO)
    content = excel_buffer.getvalue()
    assert len(content) > 500
