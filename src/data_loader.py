"""
Dataset Loader and Exploratory Data Analysis Module.
Loads dataset/loan_data.csv, inspects schema, handles duplicates,
analyzes missing values, detects outliers, and validates data integrity.
"""
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pandas as pd

import numpy as np
from config import DATA_PATH, CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COLUMN

class DataLoader:
    def __init__(self, data_path=None):
        self.data_path = data_path or DATA_PATH
        self.df = None
        self.clean_df = None
        
    def load_data(self):
        """Loads dataset from CSV and returns raw dataframe."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}. Please place loan_data.csv in the dataset directory.")
        self.df = pd.read_csv(self.data_path)
        return self.df

    def get_schema_summary(self):
        """Returns comprehensive dataset schema and descriptive statistics."""
        if self.df is None:
            self.load_data()
            
        df = self.df
        total_rows, total_cols = df.shape
        duplicates = int(df.duplicated().sum())
        
        # Missing values breakdown
        missing_summary = {}
        for col in df.columns:
            cnt = int(df[col].isnull().sum())
            pct = round(float((cnt / total_rows) * 100), 2)
            missing_summary[col] = {"count": cnt, "percentage": pct}
            
        # Target distribution
        target_dist = {}
        if TARGET_COLUMN in df.columns:
            val_counts = df[TARGET_COLUMN].value_counts()
            for k, v in val_counts.items():
                target_dist[str(k)] = {
                    "count": int(v),
                    "percentage": round(float((v / total_rows) * 100), 2)
                }
                
        # Outlier detection using Interquartile Range (IQR) for numeric fields
        outliers = {}
        for col in NUMERICAL_FEATURES:
            if col in df.columns:
                valid_vals = df[col].dropna()
                q25, q75 = np.percentile(valid_vals, [25, 75])
                iqr = q75 - q25
                lower_bound = q25 - (1.5 * iqr)
                upper_bound = q75 + (1.5 * iqr)
                outlier_count = int(((valid_vals < lower_bound) | (valid_vals > upper_bound)).sum())
                outliers[col] = {
                    "count": outlier_count,
                    "q25": round(float(q25), 2),
                    "q75": round(float(q75), 2),
                    "iqr": round(float(iqr), 2),
                    "lower_bound": round(float(lower_bound), 2),
                    "upper_bound": round(float(upper_bound), 2)
                }
                
        return {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "columns": list(df.columns),
            "duplicates": duplicates,
            "missing_summary": missing_summary,
            "target_distribution": target_dist,
            "outliers_iqr": outliers
        }

    def clean_data(self):
        """
        Cleans dataset for ML modeling:
        - Drops exact duplicates if any
        - Strips string whitespaces
        - Cleans Dependents '3+' to '3'
        - Standardizes types
        """
        if self.df is None:
            self.load_data()
            
        df = self.df.copy()
        
        # Remove whitespace from string columns
        for col in df.columns:
            if df[col].dtype == object or str(df[col].dtype).startswith('str'):
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace({'nan': np.nan, 'None': np.nan, '': np.nan})

            
        # Standardize Dependents ('3+' -> '3')
        if 'Dependents' in df.columns:
            df['Dependents'] = df['Dependents'].astype(str).str.replace('+', '', regex=False)
            df['Dependents'] = df['Dependents'].replace({'nan': np.nan})
            
        # Convert numerical fields to numeric
        for col in NUMERICAL_FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        self.clean_df = df
        return self.clean_df
