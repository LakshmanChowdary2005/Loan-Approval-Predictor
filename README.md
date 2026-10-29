# 🏛️ AI-Powered Loan Approval Prediction & Financial Risk Analysis System
### ShadowFox Intermediate-Level Internship Task — Machine Learning & Full-Stack Development

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-black.svg)](https://flask.palletsprojects.com/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-green.svg)](https://shap.readthedocs.io/)
[![ReportLab](https://img.shields.io/badge/ReportLab-PDF%20Engine-red.svg)](https://www.reportlab.com/)
[![OpenPyXL](https://img.shields.io/badge/OpenPyXL-Excel%20Export-emerald.svg)](https://openpyxl.readthedocs.io/)
[![Pytest](https://img.shields.io/badge/Pytest-25%20Passed%20(100%25)-brightgreen.svg)](https://docs.pytest.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

---

## 📌 Executive Summary

**CreditPulse AI** is an educational loan-approval modeling project. It scores applicant profiles against a small historical dataset, reports model estimates and evaluation metrics, and demonstrates risk calculations and model explanations.

The application is not a production underwriting system. Its predictions are not lender decisions, calibrated lending odds, financial advice, or a guarantee of future performance.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    A[Applicant Profile / CSV Upload] --> B[src.validators: Schema & Range Checks]
    B --> C[src.preprocessing: Leak-Free ColumnTransformer]
    C --> D[Multi-Model Ensemble Engine]
    
    subgraph Machine Learning Core
        D --> D1[Random Forest - 85.37% Test Acc]
        D --> D2[Logistic Regression - 86.18% Test Acc]
        D --> D3[Extra Trees - 86.18% Test Acc]
        D --> D4[Gradient Boosting - 83.74% Test Acc]
        D --> D5[Decision Tree - 82.11% Test Acc]
    end
    
    D1 --> E[Prediction & Consensus Resolver]
    E --> F[src.explainability: SHAP TreeExplainer]
    E --> G[src.risk_analysis: DTI & EMI Stress Testing]
    
    F & G --> H[src.database: SQLite Persistent Audit Trail]
    H --> I[Web UI Presentation Layer]
    H --> J[src.reporting: PDF & Excel Generation]
```

---

## 📊 Dataset Specification & Empirical Analysis

The model is trained on the official ShadowFox loan dataset containing **614 applicant records** and **13 features**:

| Feature Name | Type | Missing Count | Handling Strategy | Underwriting Relevance |
| :--- | :--- | :--- | :--- | :--- |
| `Loan_ID` | String | 0 | Dropped from features | Unique applicant identifier |
| `Gender` | Categorical | 13 | Mode Imputation (`Male`) | Demographic monitoring |
| `Married` | Categorical | 3 | Mode Imputation (`Yes`) | Household stability proxy |
| `Dependents` | Categorical | 15 | Mode Imputation (`0`), `3+` &rarr; `3` | Living expense commitment |
| `Education` | Categorical | 0 | None (Zero Missing) | Earning potential indicator |
| `Self_Employed` | Categorical | 32 | Mode Imputation (`No`) | Income variance factor |
| `ApplicantIncome` | Numeric (dataset units/month) | 0 | None (Zero Missing) | Primary repayment capacity |
| `CoapplicantIncome` | Numeric (dataset units/month) | 0 | None (Zero Missing) | Supplemental solvency buffer |
| `LoanAmount` | Numeric (thousands of dataset units) | 22 | Median Imputation (`128`) | Requested debt principal |
| `Loan_Amount_Term` | Numeric (Mo) | 14 | Median Imputation (`360`) | Amortization period |
| `Credit_History` | Binary | 50 | Mode Imputation (`1.0`) | Primary creditworthiness guide |
| `Property_Area` | Categorical | 0 | None (Zero Missing) | Collateral demographic liquidity |
| **`Loan_Status`** | Target | 0 | Encoded (`Y=1`, `N=0`) | **Approved (68.7%) / Rejected (31.3%)** |

---

## 🤖 Model Benchmarking & Empirical Results

All models were evaluated using **5-Fold Stratified Cross-Validation** on the training split (80%) and scored on one held-out test split (20%, 123 applications) with a fixed seed (`random_state=42`). Balanced accuracy is included because the dataset contains substantially more approved than rejected applications.

| Algorithm | CV Accuracy (mean ± SD) | CV Balanced Accuracy | Held-Out Accuracy | Held-Out Balanced Accuracy | Held-Out ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 79.62% ± 4.26% | 68.95% | 86.18% | 78.36% | 85.20% |
| **Extra Trees Classifier** | 79.42% ± 4.24% | 68.44% | 86.18% | 78.36% | 85.14% |
| **Random Forest (Default)** | 79.42% ± 4.39% | 68.62% | 85.37% | 77.04% | 79.69% |
| **Gradient Boosting** | 77.79% ± 3.17% | 67.62% | 83.74% | 75.87% | 76.72% |
| **Decision Tree** | 75.35% ± 4.08% | 65.70% | 82.11% | 76.87% | 79.43% |

> These scores are estimates from one small dataset and one held-out split; they do not establish real-world lending accuracy. Raw accuracy alone can conceal errors on the less frequent rejected class. Retraining writes the current metrics to `model/model_metadata.json`.

---

## 🔍 Explainable AI (SHAP) & Decision Attribution

1. **Global Feature Rankings:** The training pipeline exports model feature-importance values. These describe associations learned from this dataset; they are not causal effects or lending rules.
2. **Local Instance Attribution:** SHAP explanations are provided when supported by the selected model. They summarize model behavior and are not legally reviewed reasons for an actual lender's decision.

---

## 💳 Financial Risk & Debt Stress Testing

- **Amortizing EMI Calculation:**
  $$\text{EMI} = \frac{P \cdot r \cdot (1+r)^n}{(1+r)^n - 1}$$
  *(Assumes fixed 8.5% annual retail APR with monthly compounding)*
- **Debt-To-Income (DTI) Benchmark:**
  - $\le 36.0\%$: **Low Risk** — Satisfies classical mortgage safety cushions.
  - $36.1\% - 43.0\%$: **Moderate Risk** — Acceptable with prime credit history.
  - $> 43.0\%$: **High Risk** — Triggers debt capacity warning.
- **Maximum Safe Loan Capacity:** Calculates the maximum borrowable principal preventing DTI from exceeding 36%.

---

## ⚖️ Demographic Subgroup Fairness Audit

In compliance with Equal Credit Opportunity Act (ECOA) standards, the system monitors algorithmic outcomes across demographic cohorts:
- **Gender Disparity Audit:** Evaluates approval rates and accuracy across Male (84.4% approval, 86.5% acc) vs Female (72.0% approval, 84.0% acc) cohorts.
- **Education Disparity Audit:** Evaluates Graduate (84.0% approval) vs Not Graduate (73.9% approval).
- **Property Area Audit:** Rural (91.2% approval) vs Semiurban (77.6% approval) vs Urban (80.0% approval).
- **Ethical Disclaimer:** Statistical variance in historical training records does not demonstrate algorithmic intent; decisions should never rely solely on protected attributes.

---

## 📂 Project Structure

```
LoanApprovalPrediction/
├── app.py                      # Flask Application Controller & Route Handlers
├── config.py                   # Central Application & Model Hyperparameters
├── requirements.txt            # Pinned Production Dependencies
├── .env.example                # Sample Environment Variables
├── .gitignore                  # Git Exclusions (Databases, secrets, caches)
├── README.md                   # Comprehensive Technical Documentation
├── Dockerfile                  # Container Build Configuration
├── docker-compose.yml          # Container Orchestration
├── dataset/
│   └── loan_data.csv           # 614-record ShadowFox training dataset
├── model/
│   ├── loan_model.pkl          # Persisted Random Forest Production Pipeline
│   ├── lr_model.pkl            # Logistic Regression Pipeline
│   ├── dt_model.pkl            # Decision Tree Pipeline
│   ├── gb_model.pkl            # Gradient Boosting Pipeline
│   ├── et_model.pkl            # Extra Trees Pipeline
│   ├── preprocessing.pkl       # Fitted Anti-Leakage Preprocessor
│   └── model_metadata.json     # Performance Benchmarks & Feature Rankings
├── src/
│   ├── __init__.py             # Package Initializer
│   ├── data_loader.py          # Data ingestion, schema validation, outlier IQR
│   ├── preprocessing.py        # Robust Scikit-Learn ColumnTransformers
│   ├── train_model.py          # Multi-algorithm training & cross-validation
│   ├── evaluate_model.py       # Standalone test evaluation & fairness audit
│   ├── prediction_service.py   # Single, consensus, and batch scoring engine
│   ├── explainability.py       # SHAP TreeExplainer & NL text generation
│   ├── risk_analysis.py        # DTI, EMI amortization & stress ratios
│   ├── reporting.py            # ReportLab PDF & OpenPyXL Excel exporters
│   ├── database.py             # SQLite audit trail & analytics aggregation
│   └── validators.py           # Input sanitization & CSV file validation
├── templates/
│   ├── base.html               # Master layout with top status ticker & navbar
│   ├── index.html              # Executive hero landing page
│   ├── dashboard.html          # Visual analytics & dataset table explorer
│   ├── prediction.html         # Application form with live presets
│   ├── results.html            # Decision card, risk breakdown, SHAP insights
│   ├── explainability.html     # Dedicated XAI & feature attribution hub
│   ├── risk_analysis.html      # Interactive financial stress tester
│   ├── simulator.html          # What-If sensitivity curve simulator
│   ├── history.html            # SQLite persistent audit trail & filters
│   ├── batch.html              # Bulk CSV drag-and-drop scoring
│   ├── model_performance.html  # Model monitoring & subgroup fairness audit
│   ├── reports.html            # Report download & compliance export center
│   ├── login.html              # Underwriter session authentication
│   └── about.html              # Architecture specifications & methodology
├── static/
│   ├── css/
│   │   └── style.css           # Modern navy/teal design system & animations
│   ├── js/
│   │   └── script.js           # Interactive charts, Web Audio API, calculators
│   └── images/
│       └── eda/                # Diagnostic visualizations & charts
└── tests/
    ├── test_data_loader.py     # Schema & outlier testing
    ├── test_preprocessing.py   # Transformer leak-free validation
    ├── test_model.py           # Prediction & probability tests
    ├── test_risk_analysis.py   # DTI & EMI mathematical verification
    ├── test_validators.py      # Negative values & category constraints
    ├── test_database.py        # SQLite insert, query, delete lifecycle
    ├── test_reporting.py       # PDF and Excel file generation
    └── test_routes.py          # HTTP status codes & REST endpoint tests
```

---

## ⚡ Quickstart Guide (Local Execution)

### Step 1: Clone Repository
```bash
git clone https://github.com/LakshmanChowdary2005/Loan-Approval-Predictor.git
cd Loan-Approval-Predictor
```

### Step 2: Set Up Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Train Models (Optional if artifacts exist)
```bash
python src/train_model.py
```

### Step 5: Run Automated Tests
```bash
python -m pytest tests/ -v
# Output: 25 passed in ~26s (100% test pass rate)
```

### Step 6: Launch Web Platform
```bash
python app.py
```
Open **`http://127.0.0.1:5000`** in your browser.

### Optional Streamlit dashboard
```bash
python -m streamlit run streamlit_app.py
```
The Streamlit dashboard provides the applicant estimator, model comparison, what-if view, and batch scoring.

---

## 🐳 Docker Deployment

### Using Docker Compose
```bash
docker-compose up --build -d
```
Access the application at `http://localhost:5000`.

---

## 💼 Demonstration Workflow for Internship Presentation

1. **Overview Dashboard (`/dashboard`):** Show portfolio distribution, credit history vs approval crosstabs, and explore historical applications with live pagination.
2. **Interactive Predictor (`/predict`):** Click preset **"Prime Graduate Applicant"** &rarr; Submit &rarr; Review instant **88.5% Approval Decision**.
3. **Underwriting Result (`/results/<id>`):** Inspect the decision card, review EMI amortization, examine SHAP feature contributions, and click **Download PDF**.
4. **What-If Sensitivity Simulator (`/simulator`):** Adjust the loan amount slider to observe the live approval probability curve update dynamically.
5. **Bulk Batch Scoring (`/batch`):** Download the sample template CSV, upload it via drag-and-drop, and instantly score multiple applications.
6. **Model Governance & Fairness (`/model-performance`):** Review 5-fold CV vs test accuracy benchmarks and demonstrate subgroup fairness monitoring.

---

## 📄 Resume-Ready Project Highlights

- **Engineered an enterprise Loan Underwriting System** integrating 5 classification algorithms (Random Forest, Logistic Regression, Extra Trees, Gradient Boosting, Decision Tree) achieving **86.18% test accuracy** and **90.81% F1-score**.
- **Integrated Explainable AI (SHAP)** delivering both global feature importance rankings and local per-applicant waterfall decision attributions.
- **Architected financial risk stress testing** computing Debt-To-Income (DTI) ratios, amortized monthly installments (EMI), and maximum safe borrowing capacity.
- **Implemented automated document generation** producing ReportLab compliance PDF memos and multi-tab OpenPyXL analytical workbooks.
- **Built an auditable persistence pipeline** in SQLite storing prediction vectors, with a 25-case Pytest automated test suite ensuring zero regressions.

---

## 📜 Legal & Compliance Disclaimer
This software was developed solely as an academic and educational decision-support project for the ShadowFox intermediate internship task. Predictions represent probabilistic machine learning estimates and do not constitute an official credit rating, legal commitment to lend, or financial advisory decision. All lending activities must comply with the Equal Credit Opportunity Act (ECOA) and local banking regulations.
