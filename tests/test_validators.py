import pytest
import pandas as pd
from src.validators import validate_applicant_data, validate_csv_dataframe

def test_validate_valid_applicant():
    data = {
        'Gender': 'Female',
        'Married': 'No',
        'Dependents': '1',
        'Education': 'Graduate',
        'Self_Employed': 'No',
        'ApplicantIncome': 4500,
        'CoapplicantIncome': 0,
        'LoanAmount': 100,
        'Loan_Amount_Term': 360,
        'Credit_History': 1.0,
        'Property_Area': 'Urban'
    }
    is_valid, errors, cleaned = validate_applicant_data(data)
    assert is_valid is True
    assert len(errors) == 0
    assert cleaned['ApplicantIncome'] == 4500.0

def test_validate_negative_income():
    data = {
        'ApplicantIncome': -500,
        'LoanAmount': 100,
        'Credit_History': 1.0
    }
    is_valid, errors, cleaned = validate_applicant_data(data)
    assert is_valid is False
    assert any("negative" in e.lower() for e in errors)

def test_validate_invalid_category():
    data = {
        'ApplicantIncome': 5000,
        'LoanAmount': 100,
        'Gender': 'UnknownAlien'
    }
    is_valid, errors, cleaned = validate_applicant_data(data)
    assert is_valid is False
    assert any("gender" in e.lower() for e in errors)

def test_validate_csv_dataframe():
    df = pd.DataFrame([
        {'Loan_ID': 'L1', 'ApplicantIncome': 4000, 'LoanAmount': 120, 'Credit_History': 1.0},
        {'Loan_ID': 'L2', 'ApplicantIncome': -100, 'LoanAmount': 120, 'Credit_History': 1.0}
    ])
    is_valid, issues, valid_rows, invalid_rows = validate_csv_dataframe(df)
    assert is_valid is True
    assert len(valid_rows) == 1
    assert len(invalid_rows) == 1
