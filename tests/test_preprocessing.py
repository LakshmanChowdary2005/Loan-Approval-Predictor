import pytest
import pandas as pd
from src.preprocessing import build_preprocessor, get_transformed_feature_names

def test_build_preprocessor():
    preprocessor = build_preprocessor()
    assert preprocessor is not None
    assert 'num' in [t[0] for t in preprocessor.transformers]
    assert 'cat' in [t[0] for t in preprocessor.transformers]

def test_preprocessor_fit_transform():
    sample_data = pd.DataFrame([{
        'Gender': 'Male',
        'Married': 'Yes',
        'Dependents': '0',
        'Education': 'Graduate',
        'Self_Employed': 'No',
        'ApplicantIncome': 5000.0,
        'CoapplicantIncome': 1500.0,
        'LoanAmount': 150.0,
        'Loan_Amount_Term': 360.0,
        'Credit_History': 1.0,
        'Property_Area': 'Semiurban'
    }])

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(sample_data)
    assert transformed.shape[0] == 1
    # Check transformed feature names extraction
    names = get_transformed_feature_names(preprocessor)
    assert len(names) == transformed.shape[1]
