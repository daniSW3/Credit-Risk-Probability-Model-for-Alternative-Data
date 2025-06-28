import pandas as pd
from .feature_engineering import create_feature_engineering_pipeline
from .utils import get_feature_engineering_config
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    # Load data
    data_path = r"C:\Users\Daniel.Temesgen\Desktop\KIAM-Rsc\week5\Data\data.csv"
    df = pd.read_csv(data_path)
    
    # Get configuration
    config = get_feature_engineering_config()
    
    # Create pipeline
    pipeline = create_feature_engineering_pipeline(config)
    
    # Fit and transform data
    if config['target_col'] and config['target_col'] in df.columns:
        X = df.drop(columns=[config['target_col']])
        y = df[config['target_col']]
        processed_data = pipeline.fit_transform(X, y)
    else:
        processed_data = pipeline.fit_transform(df)
    
    # Save processed data
    processed_df = pd.DataFrame(processed_data, 
                              columns=pipeline.get_feature_names_out())
    processed_df.to_csv('processed_data.csv', index=False)
    print("Feature engineering completed. Processed data saved to processed_data.csv")

if __name__ == "__main__":
    main()