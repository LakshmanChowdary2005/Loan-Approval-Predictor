"""
Input Validation and Sanitization Module for Loan Applications and CSV Batch Uploads.
Enforces data types, range bounds, allowed categories, and prevents execution exploits.
"""
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


ALLOWED_VALUES = {
    'Gender': ['Male', 'Female', 'Other'],
    'Married': ['Yes', 'No'],
    'Dependents': ['0', '1', '2', '3+'],
    'Education': ['Graduate', 'Not Graduate'],
    'Self_Employed': ['Yes', 'No'],
    'Property_Area': ['Urban', 'Semiurban', 'Rural']
}

def validate_applicant_data(data):
    """
    Validates form input fields for single prediction.
    Returns: (is_valid: bool, errors: list, cleaned_dict: dict)
    """
    errors = []
    cleaned = {}

    # Validate Categoricals
    for cat in CATEGORICAL_FEATURES:
        val = str(data.get(cat, '')).strip()
        if not val:
            # Fallback defaults
            val = 'Male' if cat == 'Gender' else ('Yes' if cat == 'Married' else ('0' if cat == 'Dependents' else ('Graduate' if cat == 'Education' else ('No' if cat == 'Self_Employed' else 'Semiurban'))))
        
        # Check against allowed list
        norm_val = val
        if cat == 'Dependents':
            norm_val = val.replace('+', '')
            if norm_val not in ['0', '1', '2', '3']:
                errors.append(f"Invalid Dependents: '{val}'. Expected 0, 1, 2, or 3+.")
        elif cat in ALLOWED_VALUES and val not in ALLOWED_VALUES[cat]:
            errors.append(f"Invalid value for {cat}: '{val}'. Allowed options: {', '.join(ALLOWED_VALUES[cat])}")
        
        cleaned[cat] = norm_val if cat == 'Dependents' else val

    # Validate Numericals
    try:
        app_inc = float(data.get('ApplicantIncome', 0))
        if app_inc < 0:
            errors.append("Applicant Income cannot be negative.")
        cleaned['ApplicantIncome'] = app_inc
    except (ValueError, TypeError):
        errors.append("Applicant Income must be a valid numeric amount.")

    try:
        coapp_inc = float(data.get('CoapplicantIncome', 0))
        if coapp_inc < 0:
            errors.append("Co-applicant Income cannot be negative.")
        cleaned['CoapplicantIncome'] = coapp_inc
    except (ValueError, TypeError):
        errors.append("Co-applicant Income must be a valid numeric amount.")

    try:
        loan_amt = float(data.get('LoanAmount', 0))
        if loan_amt <= 0:
            errors.append("Loan Amount must be strictly greater than 0 ($ thousands).")
        cleaned['LoanAmount'] = loan_amt
    except (ValueError, TypeError):
        errors.append("Loan Amount must be a valid numeric amount.")

    try:
        term = float(data.get('Loan_Amount_Term', 360))
        if term <= 0:
            errors.append("Loan Repayment Term must be positive (in months).")
        cleaned['Loan_Amount_Term'] = term
    except (ValueError, TypeError):
        errors.append("Loan Repayment Term must be a valid numeric duration.")

    try:
        ch = float(data.get('Credit_History', 1.0))
        if ch not in [0.0, 1.0]:
            errors.append("Credit History must be either 1.0 (Good) or 0.0 (Unfavorable).")
        cleaned['Credit_History'] = ch
    except (ValueError, TypeError):
        errors.append("Credit History must be 1.0 or 0.0.")

    return (len(errors) == 0, errors, cleaned)


def validate_csv_dataframe(df):
    """
    Validates uploaded batch CSV dataframe.
    Returns: (is_valid: bool, issues: list, valid_rows: list, invalid_rows: list)
    """
    issues = []
    if df is None or len(df) == 0:
        return False, ["Uploaded CSV is empty."], [], []

    required_cols = ['ApplicantIncome', 'LoanAmount']
    missing_critical = [col for col in required_cols if col not in df.columns]
    if missing_critical:
        return False, [f"Missing mandatory column(s): {', '.join(missing_critical)}"], [], []

    valid_rows = []
    invalid_rows = []

    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        is_val, row_errs, cleaned = validate_applicant_data(row_dict)
        if is_val:
            cleaned['Loan_ID'] = str(row_dict.get('Loan_ID', f"BATCH-{idx+1:04d}"))
            valid_rows.append(cleaned)
        else:
            invalid_rows.append({
                "row_index": idx + 1,
                "loan_id": str(row_dict.get('Loan_ID', f"ROW-{idx+1}")),
                "reasons": row_errs
            })

    return True, issues, valid_rows, invalid_rows
