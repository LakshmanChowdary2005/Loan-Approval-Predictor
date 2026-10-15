"""
Preprocessing Pipeline for Loan Approval Prediction.
Constructs consistent, leak-free scikit-learn transformers for numerical
and categorical variables with imputation and scaling.
"""
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES

def build_preprocessor(cat_features=None, num_features=None):
    """
    Creates an unfitted ColumnTransformer:
    - Numerical: Median imputation + StandardScaler
    - Categorical: Most Frequent imputation + OneHotEncoder(drop='first', handle_unknown='ignore')
    """
    cat_cols = cat_features or CATEGORICAL_FEATURES
    num_cols = num_features or NUMERICAL_FEATURES

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

    return preprocessor

def get_transformed_feature_names(preprocessor, cat_features=None, num_features=None):
    """Extracts output feature names from a fitted preprocessor."""
    cat_cols = cat_features or CATEGORICAL_FEATURES
    num_cols = num_features or NUMERICAL_FEATURES
    
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    encoded_cat_names = cat_encoder.get_feature_names_out(cat_cols)
    return list(num_cols) + list(encoded_cat_names)
