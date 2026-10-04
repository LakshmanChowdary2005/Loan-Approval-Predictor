# 🏦 CreditPulse AI — Enterprise Loan Approval Prediction Platform

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-black.svg)](https://flask.palletsprojects.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Platform-Enterprise%20Ready-success.svg)]()

> An end-to-end Enterprise Fintech Underwriting & Machine Learning platform that automates loan eligibility evaluation (`Approved` vs `Rejected`), conducts real-time multi-model consensus voting, generates sensitivity curves, scores bulk CSV batches, and produces print-ready executive credit assessment reports.

---

## 🌟 Advanced Features & Enterprise Capabilities

### 1. 🔮 Real-Time Multi-Model Underwriting Arena
* **Instant Inference:** Score applicant profiles in &lt;15ms across 5 classification ensembles (**Random Forest**, **Logistic Regression**, **Decision Tree**, **Gradient Boosting**, **Extra Trees**).
* **Consensus Voting Engine:** View side-by-side votes from all 5 algorithms with an agreement percentage (e.g., *5 of 5 Models Recommend Approval — 100% Consensus*).
* **1-Click Demo Profiles:** Pre-configured profiles (*Prime Applicant*, *High-Risk*, *Borderline*, *Co-Applicant Cushion*) for presentation demonstrations.

### 2. 🎛️ Interactive What-If Sensitivity Simulator
* **Dynamic Parameter Sweeping:** Drag sliders for Applicant Income, Co-applicant Income, Loan Amount, and Term to observe the real-time probability gauge update dynamically without reloading.
* **Sensitivity Curve Plot:** Interactive Chart.js line plot charting requested loan amounts ($20k–$450k) against approval probabilities to identify safe borrowing ceilings.

### 3. 💰 Loan Affordability & Amortization Breakdown
* **EMI Calculation:** Accurate monthly payment formula based on loan amount, repayment term, and interest rate.
* **Debt-to-Income (DTI) Health Audit:** Clear indicator highlighting safe threshold (&le;35%), moderate risk (36–49%), or excessive burden (&ge;50%).
* **Amortization Doughnut:** Visual breakdown of Principal Borrowed vs Total Estimated Interest.

### 4. 📄 Print-Ready Official Underwriting Assessment Report
* 1-click modal to generate and print a formal credit assessment report complete with institution header, reference number, **APPROVED / REJECTED watermark and stamp**, applicant scorecard, and underwriter signature block.

### 5. 📁 Bulk Batch Application Scoring
* Drag-and-drop CSV batch upload to evaluate hundreds of loan applications simultaneously.
* Displays batch KPI summary cards (Total Scored, Approved, Rejected, Batch Approval Rate %) and an interactive scored applications data table.
* Includes 1-click **Download Template CSV** for testing.

### 6. 🎨 Premium Fintech UI / UX
* **Dual Theme Engine:** Dark mode (cyber navy & emerald glassmorphism) and Light mode (crisp executive styling) with persistent preference.
* **Audio Micro-Interactions:** Synthesized Web Audio feedback (approval chime, alert tone, click pings) with mute toggle.
* **Top Status Ticker:** Real-time system health metrics, model version, and inference latency.

---

## 📊 Dataset & Features

The dataset contains 614 historical loan applications:

| Feature Name | Description | Data Type | Imputation Strategy |
| :--- | :--- | :--- | :--- |
| `Loan_ID` | Application identifier | String | Excluded from training |
| `Gender` | Male / Female | Categorical | Mode (`Male`) |
| `Married` | Yes / No | Categorical | Mode (`Yes`) |
| `Dependents` | Number of dependents (`0`, `1`, `2`, `3+`) | Ordinal / Cat | Mode (`0`) |
| `Education` | Graduate / Not Graduate | Categorical | None (0 missing) |
| `Self_Employed` | Yes / No | Categorical | Mode (`No`) |
| `ApplicantIncome` | Primary applicant income ($/month) | Numeric | None (0 missing) |
| `CoapplicantIncome` | Co-applicant income ($/month) | Numeric | None (0 missing) |
| `LoanAmount` | Requested loan amount in thousands ($k) | Numeric | Median (`128.0`) |
| `Loan_Amount_Term` | Repayment term in months (`360`, `180`, etc.) | Numeric | Mode (`360.0`) |
| `Credit_History` | Guidelines met (`1.0` Good, `0.0` Bad) | Numeric | Mode (`1.0`) |
| `Property_Area` | Urban / Semiurban / Rural | Categorical | None (0 missing) |
| **`Loan_Status`** | **Target variable: `Y` (Approved) / `N` (Rejected)** | Binary | **Target: 1 / 0** |

---

## 🔬 Multi-Model Benchmark Comparison

Evaluated on an **80/20 Stratified Split** with **5-Fold Stratified Cross-Validation**:

| Model Algorithm | 5-Fold CV Accuracy | Test Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 79.62% (±4.26%) | **86.18%** | 84.00% | **98.82%** | **90.81%** | **85.20%** |
| **Extra Trees Classifier** | 79.42% (±4.24%) | **86.18%** | 84.00% | **98.82%** | **90.81%** | 85.14% |
| **Random Forest (Default)** | **79.83% (±4.48%)** | **85.37%** | 83.17% | **98.82%** | **90.32%** | 82.11% |
| **Gradient Boosting** | 77.79% (±3.17%) | 83.74% | 82.00% | 96.47% | 89.13% | 76.72% |
| **Decision Tree** | 75.35% (±4.08%) | 82.11% | 80.95% | 98.82% | 87.50% | 79.43% |

---

## 📁 Project Directory Structure

```
d:\Intermediate\
├── dataset/
│   └── loan_data.csv               # 614-record historical loan dataset
├── model/
│   ├── loan_model.pkl              # Production Random Forest pipeline
│   ├── lr_model.pkl                # Logistic Regression pipeline
│   ├── dt_model.pkl                # Decision Tree pipeline
│   ├── gb_model.pkl                # Gradient Boosting pipeline
│   ├── et_model.pkl                # Extra Trees pipeline
│   ├── preprocessing.pkl           # Scikit-learn ColumnTransformer
│   └── model_metadata.json         # Benchmark metrics, feature rankings, and stats
├── notebooks/
│   └── analysis.ipynb              # Comprehensive step-by-step Jupyter Notebook
├── static/
│   ├── css/style.css               # Glassmorphism design system & print styles
│   ├── js/script.js                # Client audio, consensus, simulator & batch engine
│   └── images/eda/                 # 9 Seaborn/Matplotlib visualization figures
├── templates/
│   ├── index.html                  # Executive overview & feature showcase
│   ├── prediction.html             # Multi-tab Underwriting Predictor & Arena
│   ├── simulator.html              # Dedicated What-If Sensitivity Sandbox
│   ├── batch.html                  # Bulk Batch CSV Application Scoring
│   └── dashboard.html              # Portfolio Analytics, EDA Gallery & Dataset Explorer
├── train_model.py                  # Model training and artifact generation pipeline
├── app.py                          # Production Flask application & REST API
├── streamlit_app.py                # Advanced Streamlit interactive app
├── requirements.txt                # Python dependencies
├── README.md                       # Comprehensive documentation
└── .gitignore                      # Git ignore file
```

---

## 🚀 How to Run

### Option A: Flask Production Platform (Recommended)
```powershell
py -3.12 app.py
```
Open **`http://127.0.0.1:5000`** in your browser.

* Routes available:
  * `/` — Platform Overview & Quick Stats
  * `/predict` — Multi-Tab Predictor, Consensus Arena & Amortization
  * `/simulator` — Real-Time What-If Sensitivity Policy Sandbox
  * `/batch` — Bulk CSV Upload & Batch Scoring
  * `/dashboard` — Portfolio Analytics, EDA Gallery & Dataset Explorer

### Option B: Streamlit Dashboard
```powershell
py -3.12 -m streamlit run streamlit_app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🎓 Faculty Demonstration Guide

Follow this recommended sequence for project reviews:
1. **Show the Landing Page (`/`):** Point out the real-time status ticker, KPI cards (86.2% accuracy, 98.8% recall), and pipeline architecture.
2. **Open Analytics & EDA (`/dashboard`):** Demonstrate the **Historical Applications Dataset Explorer** (search, filter, pagination) and show the EDA charts (Credit History impact, Property location, correlation heatmap).
3. **Open What-If Simulator (`/simulator`):** Move the sliders live to demonstrate how changing income or loan amount dynamically shifts approval odds in real time.
4. **Open Predictor (`/predict`):**
   - Click a preset chip (e.g. *Prime Applicant*), click **Predict Loan Approval**.
   - Show the animated **🟢 LOAN APPROVED** verdict with confidence score and audio chime.
   - Switch to the **5-Model Consensus** tab to show that all 5 models vote to approve with 100% agreement.
   - Switch to the **EMI & Amortization** tab to show the Principal vs Interest doughnut chart.
   - Click **Generate Official Underwriting Report** to display the print-ready assessment with official watermark and stamp!
5. **Open Batch Scoring (`/batch`):** Download the sample CSV template, upload it, and show bulk scoring of multiple applications simultaneously.
