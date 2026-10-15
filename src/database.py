"""
SQLite Database Storage & Query Module.
Persists application predictions, stores input parameters for auditability,
and provides search, pagination, metrics aggregation, and record deletion.
"""
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import sqlite3

import json
from datetime import datetime
import pandas as pd
from config import DATABASE_PATH

def get_db_connection():
    """Returns a connection to the SQLite database with row factory enabled."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes tables if they do not exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prediction_id TEXT UNIQUE NOT NULL,
        timestamp DATETIME NOT NULL,
        gender TEXT,
        married TEXT,
        dependents TEXT,
        education TEXT,
        self_employed TEXT,
        applicant_income REAL,
        coapplicant_income REAL,
        loan_amount REAL,
        loan_amount_term REAL,
        credit_history REAL,
        property_area TEXT,
        model_name TEXT,
        predicted_class INTEGER NOT NULL,
        predicted_decision TEXT NOT NULL,
        prob_approved REAL,
        prob_rejected REAL,
        confidence REAL,
        risk_tier TEXT,
        dti_ratio REAL,
        emi_monthly REAL,
        shap_summary TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audit_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME NOT NULL,
        action TEXT NOT NULL,
        details TEXT
    )
    """)

    conn.commit()
    conn.close()

def save_prediction(record):
    """
    Inserts a newly generated prediction record into SQLite.
    """
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    pred_id = record.get('prediction_id') or f"APP-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    timestamp = record.get('timestamp') or datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    cursor.execute("""
    INSERT OR REPLACE INTO predictions (
        prediction_id, timestamp, gender, married, dependents, education, self_employed,
        applicant_income, coapplicant_income, loan_amount, loan_amount_term, credit_history,
        property_area, model_name, predicted_class, predicted_decision,
        prob_approved, prob_rejected, confidence, risk_tier, dti_ratio, emi_monthly, shap_summary
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (

        pred_id,
        timestamp,
        record.get('Gender', 'Male'),
        record.get('Married', 'Yes'),
        str(record.get('Dependents', '0')),
        record.get('Education', 'Graduate'),
        record.get('Self_Employed', 'No'),
        float(record.get('ApplicantIncome', 0)),
        float(record.get('CoapplicantIncome', 0)),
        float(record.get('LoanAmount', 0)),
        float(record.get('Loan_Amount_Term', 360)),
        float(record.get('Credit_History', 1.0)),
        record.get('Property_Area', 'Urban'),
        record.get('model_name', 'Random Forest'),
        int(record.get('predicted_class', 1)),
        record.get('predicted_decision', 'APPROVED'),
        float(record.get('prob_approved', 0.0)),
        float(record.get('prob_rejected', 0.0)),
        float(record.get('confidence', 0.0)),
        record.get('risk_tier', 'Moderate'),
        float(record.get('dti_ratio', 0.0)),
        float(record.get('emi_monthly', 0.0)),
        json.dumps(record.get('shap_summary', {}))
    ))

    conn.commit()
    conn.close()
    return pred_id

def get_prediction_by_id(prediction_id):
    """Retrieves an individual prediction record by its unique ID."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions WHERE prediction_id = ?", (prediction_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        d = dict(row)
        if d.get('shap_summary'):
            try:
                d['shap_summary'] = json.loads(d['shap_summary'])
            except Exception:
                pass
        return d
    return None

def get_all_predictions(limit=50, offset=0, search=None, status_filter=None):
    """Retrieves paginated, filtered prediction history."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM predictions WHERE 1=1"
    params = []

    if status_filter and status_filter.upper() in ['APPROVED', 'REJECTED']:
        query += " AND predicted_decision = ?"
        params.append(status_filter.upper())

    if search:
        search_term = f"%{search.strip()}%"
        query += " AND (prediction_id LIKE ? OR property_area LIKE ? OR education LIKE ?)"
        params.extend([search_term, search_term, search_term])

    # Count total matching
    count_query = query.replace("SELECT *", "SELECT COUNT(*)")
    cursor.execute(count_query, params)
    total_count = cursor.fetchone()[0]

    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    records = [dict(r) for r in rows]
    return records, total_count

def delete_prediction_by_id(prediction_id):
    """Deletes a record from the history database."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions WHERE prediction_id = ?", (prediction_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

def get_database_analytics():
    """Aggregates metrics across user-generated predictions."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_preds = cursor.fetchone()[0]

    if total_preds == 0:
        conn.close()
        return {
            "total_predictions": 0,
            "approved_count": 0,
            "rejected_count": 0,
            "approval_rate": 0.0,
            "rejection_rate": 0.0,
            "avg_income": 0.0,
            "avg_loan": 0.0
        }

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE predicted_class = 1")
    approved = cursor.fetchone()[0]
    rejected = total_preds - approved

    cursor.execute("SELECT AVG(applicant_income), AVG(loan_amount) FROM predictions")
    avg_income, avg_loan = cursor.fetchone()

    conn.close()
    return {
        "total_predictions": total_preds,
        "approved_count": approved,
        "rejected_count": rejected,
        "approval_rate": round((approved / total_preds) * 100, 1),
        "rejection_rate": round((rejected / total_preds) * 100, 1),
        "avg_income": round(avg_income or 0.0, 2),
        "avg_loan": round(avg_loan or 0.0, 2)
    }

def export_predictions_dataframe():
    """Exports all predictions as a pandas DataFrame for reporting."""
    init_db()
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
    conn.close()
    return df
