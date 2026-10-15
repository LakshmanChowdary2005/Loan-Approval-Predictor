"""
Financial Risk Analysis & Affordability Assessment Module.
Calculates Debt-to-Income (DTI), monthly EMI amortization, Loan-to-Income ratio,
and flags high-risk borrowing scenarios with transparently documented assumptions.
"""
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import DEFAULT_ANNUAL_INTEREST_RATE, BENCHMARK_DTI_SAFE, BENCHMARK_DTI_MODERATE


def calculate_financial_risk_profile(applicant_data):
    """
    Computes financial affordability metrics, debt stress ratios, and risk indicators.
    
    Documented Assumptions:
    - Annual interest rate assumed at 8.5% fixed APR for standard retail mortgage.
    - Monthly amortization computed via standard annuity formula: [P * r * (1+r)^n] / [(1+r)^n - 1].
    - Safe Debt-To-Income (DTI) benchmark is set at 36%, with 36-43% representing moderate risk,
      and >43% representing elevated financial strain.
    - Disclaimers: Metrics provide decision support and do not represent regulatory credit bureau ratings.
    """
    app_income = float(applicant_data.get('ApplicantIncome', 0))
    coapp_income = float(applicant_data.get('CoapplicantIncome', 0))
    total_monthly_income = app_income + coapp_income

    loan_amount_k = float(applicant_data.get('LoanAmount', 0))
    loan_amount_actual = loan_amount_k * 1000.0  # LoanAmount is in thousands ($k)
    term_months = float(applicant_data.get('Loan_Amount_Term', 360))
    credit_history = float(applicant_data.get('Credit_History', 1.0))

    # Monthly installment (EMI) calculation
    monthly_interest_rate = DEFAULT_ANNUAL_INTEREST_RATE / 12.0
    if term_months > 0 and monthly_interest_rate > 0:
        factor = (1.0 + monthly_interest_rate) ** term_months
        emi = (loan_amount_actual * monthly_interest_rate * factor) / (factor - 1.0)
    elif term_months > 0:
        emi = loan_amount_actual / term_months
    else:
        emi = 0.0

    total_repayment = emi * term_months
    total_interest = total_repayment - loan_amount_actual

    # Debt-To-Income (DTI) ratio
    if total_monthly_income > 0:
        dti_ratio = round((emi / total_monthly_income) * 100.0, 2)
    else:
        dti_ratio = 100.0

    # Loan to Income (LTI) Multiplier (Annual)
    annual_income = total_monthly_income * 12.0
    lti_multiplier = round(loan_amount_actual / max(annual_income, 1.0), 2)

    # Maximum Safe Borrowing Capacity (36% max DTI rule)
    max_safe_monthly_payment = total_monthly_income * (BENCHMARK_DTI_SAFE / 100.0)
    if term_months > 0 and monthly_interest_rate > 0:
        factor = (1.0 + monthly_interest_rate) ** term_months
        max_safe_loan = (max_safe_monthly_payment * (factor - 1.0)) / (monthly_interest_rate * factor)
    else:
        max_safe_loan = max_safe_monthly_payment * term_months

    # Risk Tier Categorization & Warning Indicators
    warnings = []
    risk_factors = []
    
    if credit_history != 1.0:
        warnings.append("Deficient or absent credit repayment trail detected.")
        risk_factors.append("Poor Credit History (0.0)")

    if total_monthly_income < 2000:
        warnings.append("Household earnings below standard retail solvency buffer ($2,000/mo).")
        risk_factors.append("Low Monthly Income Cushion")

    if dti_ratio > BENCHMARK_DTI_MODERATE:
        warnings.append(f"Debt-to-Income ({dti_ratio}%) exceeds prudent lending threshold of 43%.")
        risk_factors.append("Elevated Debt-to-Income Ratio")
    elif dti_ratio > BENCHMARK_DTI_SAFE:
        warnings.append(f"Debt-to-Income ({dti_ratio}%) is within moderate review bracket (36-43%).")

    if loan_amount_actual > max_safe_loan:
        excess = loan_amount_actual - max_safe_loan
        warnings.append(f"Requested loan exceeds maximum prudent capacity by ${excess:,.0f}.")

    # Outlier / Unusual Input Detection
    unusual_inputs = []
    if total_monthly_income > 25000:
        unusual_inputs.append("Very high monthly income (> $25,000).")
    if loan_amount_k > 450:
        unusual_inputs.append("Very high loan request (> $450k).")
    if term_months < 120 or term_months > 480:
        unusual_inputs.append(f"Unusual repayment term ({term_months:.0f} months).")

    # Overall Risk Tier Assessment
    if credit_history == 1.0 and dti_ratio <= BENCHMARK_DTI_SAFE and total_monthly_income >= 3000:
        risk_tier = "LOW"
        risk_color = "success"
        verdict_summary = "Favorable financial standing with manageable debt service ratio."
    elif credit_history == 1.0 and dti_ratio <= BENCHMARK_DTI_MODERATE:
        risk_tier = "MODERATE"
        risk_color = "warning"
        verdict_summary = "Acceptable risk profile with moderate debt exposure requiring review."
    else:
        risk_tier = "HIGH"
        risk_color = "danger"
        verdict_summary = "Elevated risk exposure due to credit vulnerability or high DTI burden."

    return {
        "applicant_income": app_income,
        "coapplicant_income": coapp_income,
        "total_monthly_income": total_monthly_income,
        "loan_amount_requested": loan_amount_actual,
        "term_months": term_months,
        "emi_monthly": round(emi, 2),
        "total_interest": round(total_interest, 2),
        "total_repayment": round(total_repayment, 2),
        "dti_ratio": dti_ratio,
        "lti_multiplier": lti_multiplier,
        "max_safe_loan": round(max_safe_loan, 2),
        "risk_tier": risk_tier,
        "risk_color": risk_color,
        "verdict_summary": verdict_summary,
        "warnings": warnings,
        "risk_factors": risk_factors,
        "unusual_inputs": unusual_inputs,
        "assumptions": [
            f"Assumed fixed annual interest rate: {DEFAULT_ANNUAL_INTEREST_RATE*100:.1f}%.",
            f"Institutional safe DTI threshold: {BENCHMARK_DTI_SAFE}%.",
            "Calculations are for automated decision support and financial risk awareness."
        ]
    }
