"""
Configuration settings for the AI-Powered Loan Approval Prediction & Financial Risk Analysis System.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File Paths
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DATA_PATH = os.path.join(DATASET_DIR, "loan_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATABASE_PATH = os.path.join(BASE_DIR, "loan_system.db")
STATIC_DIR = os.path.join(BASE_DIR, "static")
IMAGE_DIR = os.path.join(STATIC_DIR, "images")
EDA_IMAGE_DIR = os.path.join(IMAGE_DIR, "eda")

# Model Files
MODEL_FILES = {
    "Random Forest": "loan_model.pkl",
    "Logistic Regression": "lr_model.pkl",
    "Decision Tree": "dt_model.pkl",
    "Gradient Boosting": "gb_model.pkl",
    "Extra Trees": "et_model.pkl"
}
DEFAULT_MODEL = "Random Forest"
METADATA_FILE = os.path.join(MODEL_DIR, "model_metadata.json")
PREPROCESSOR_FILE = os.path.join(MODEL_DIR, "preprocessing.pkl")

# Application Security
SECRET_KEY = os.environ.get("SECRET_KEY", "shadowfox-intermediate-fintech-ai-2026-key")
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD_HASH = os.environ.get("ADMIN_PASSWORD_HASH", "pbkdf2:sha256:600000$shadowfox$5f4dcc3b5aa765d61d8327deb882cf99") # default password 'admin123'
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = "Lax"

# Financial Risk Thresholds & Assumptions (Documented)
DEFAULT_ANNUAL_INTEREST_RATE = 0.085  # 8.5% annual interest for standard term calculation
BENCHMARK_DTI_SAFE = 36.0              # Max recommended Debt-To-Income percentage
BENCHMARK_DTI_MODERATE = 43.0          # Elevated DTI upper limit
MIN_INCOME_RECOMMENDED = 2500.0        # Monthly income solvency cushion guideline

# Feature Definitions
CATEGORICAL_FEATURES = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
NUMERICAL_FEATURES = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
TARGET_COLUMN = 'Loan_Status'
TARGET_MAP = {'Y': 1, 'N': 0}
TARGET_REVERSE_MAP = {1: 'Y', 0: 'N'}
