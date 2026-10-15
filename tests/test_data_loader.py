import os
import pytest
import pandas as pd
from src.data_loader import DataLoader
from config import DATA_PATH

def test_data_loader_initialization():
    loader = DataLoader(DATA_PATH)
    assert loader.data_path == DATA_PATH

def test_load_data_returns_dataframe():
    loader = DataLoader(DATA_PATH)
    df = loader.load_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 614
    assert "Loan_ID" in df.columns
    assert "Loan_Status" in df.columns

def test_schema_summary():
    loader = DataLoader(DATA_PATH)
    summary = loader.get_schema_summary()
    assert summary["total_rows"] == 614
    assert summary["total_columns"] == 13
    assert "missing_summary" in summary
    assert "outliers_iqr" in summary
    assert "ApplicantIncome" in summary["outliers_iqr"]

def test_clean_data():
    loader = DataLoader(DATA_PATH)
    clean_df = loader.clean_data()
    assert "Dependents" in clean_df.columns
    # Check that '3+' was replaced with '3'
    assert '3+' not in clean_df['Dependents'].values
