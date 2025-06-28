import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import (StandardScaler, OneHotEncoder, 
                                  FunctionTransformer, LabelEncoder)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from xverse.transformer import MonotonicBinning
from woe import WOE
import warnings
warnings.filterwarnings('ignore')

class FeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts temporal features from datetime columns"""
    def __init__(self, date_col='transaction_date'):
        self.date_col = date_col
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Convert to datetime if not already
        X[self.date_col] = pd.to_datetime(X[self.date_col])
        
        # Extract temporal features
        X['transaction_hour'] = X[self.date_col].dt.hour
        X['transaction_day'] = X[self.date_col].dt.day
        X['transaction_month'] = X[self.date_col].dt.month
        X['transaction_year'] = X[self.date_col].dt.year
        X['transaction_dayofweek'] = X[self.date_col].dt.dayofweek
        X['transaction_weekofyear'] = X[self.date_col].dt.isocalendar().week
        
        return X.drop(columns=[self.date_col])

class AggregateTransformer(BaseEstimator, TransformerMixin):
    """Creates aggregate features at customer level"""
    def __init__(self, customer_id_col='customer_id', amount_col='amount'):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
        self.agg_features_ = None
        
    def fit(self, X, y=None):
        # Calculate aggregate features
        agg_df = X.groupby(self.customer_id_col)[self.amount_col].agg(
            ['sum', 'mean', 'count', 'std', 'min', 'max']
        ).rename(columns={
            'sum': 'total_transaction_amount',
            'mean': 'avg_transaction_amount',
            'count': 'transaction_count',
            'std': 'std_transaction_amount',
            'min': 'min_transaction_amount',
            'max': 'max_transaction_amount'
        })
        
        self.agg_features_ = agg_df
        return self
    
    def transform(self, X):
        # Merge aggregate features back to original data
        X = X.merge(self.agg_features_, 
                   how='left', 
                   on=self.customer_id_col)
        return X

class TypeConverter(BaseEstimator, TransformerMixin):
    """Converts columns to specified data types"""
    def __init__(self, numeric_cols=None, categorical_cols=None):
        self.numeric_cols = numeric_cols or []
        self.categorical_cols = categorical_cols or []
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.numeric_cols:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
        for col in self.categorical_cols:
            if col in X.columns:
                X[col] = X[col].astype('category')
        return X

def create_feature_engineering_pipeline(config):
    """
    Creates a complete feature engineering pipeline based on config
    
    Args:
        config (dict): Configuration dictionary with column names and parameters
        
    Returns:
        sklearn.Pipeline: Complete feature engineering pipeline
    """
    # Define columns
    numeric_cols = config.get('numeric_cols', [])
    categorical_cols = config.get('categorical_cols', [])
    date_col = config.get('date_col', 'transaction_date')
    customer_id_col = config.get('customer_id_col', 'customer_id')
    amount_col = config.get('amount_col', 'amount')
    target_col = config.get('target_col', None)
    
    # Type conversion pipeline
    type_pipeline = Pipeline([
        ('type_converter', TypeConverter(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols
        ))
    ])
    
    # Feature extraction pipeline
    extraction_pipeline = Pipeline([
        ('feature_extractor', FeatureExtractor(date_col=date_col)),
        ('aggregate_transformer', AggregateTransformer(
            customer_id_col=customer_id_col,
            amount_col=amount_col
        ))
    ])
    
    # Numeric pipeline
    numeric_pipeline = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    
    # WOE encoding for categorical features (if target is provided)
    if target_col:
        woe_pipeline = Pipeline([
            ('woe_encoder', WOE())
        ])
    
    # Monotonic binning for numeric features (if target is provided)
    if target_col:
        monotonic_pipeline = Pipeline([
            ('monotonic_binning', MonotonicBinning())
        ])
    
    # Column transformers
    transformers = [
        ('numeric', numeric_pipeline, numeric_cols),
        ('categorical', categorical_pipeline, categorical_cols)
    ]
    
    if target_col:
        transformers.extend([
            ('woe', woe_pipeline, categorical_cols),
            ('monotonic', monotonic_pipeline, numeric_cols)
        ])
    
    preprocessing_pipeline = ColumnTransformer(
        transformers,
        remainder='passthrough'
    )
    
    # Complete pipeline
    full_pipeline = Pipeline([
        ('type_conversion', type_pipeline),
        ('feature_extraction', extraction_pipeline),
        ('preprocessing', preprocessing_pipeline)
    ])
    
    return full_pipeline