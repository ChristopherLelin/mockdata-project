"""
Export data script to handle illegal characters in Excel export
"""

import os
import pandas as pd
import pickle
import sys
import re

def clean_string_values(df):
    """Clean string values to remove illegal Excel characters."""
    for col in df.columns:
        if df[col].dtype == 'object':
            # Replace any control characters with empty string
            df[col] = df[col].apply(lambda x: re.sub(r'[\x00-\x1F\x7F]', '', str(x)) if isinstance(x, str) else x)
    return df

def export_to_excel(pickle_path, excel_path):
    """Export pickled DataFrames to Excel safely."""
    print(f"Loading data from {pickle_path}")
    with open(pickle_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    print(f"Saving data to {excel_path}")
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for table_name, data in data_dict.items():
            print(f"Processing table {table_name} with {len(data)} rows")
            # Clean data before export
            clean_data = clean_string_values(data)
            
            # Truncate very large tables for Excel (which has row limits)
            if len(clean_data) > 1000000:
                print(f"Table {table_name} has {len(clean_data)} rows. Truncating to 1,000,000 for Excel output.")
                clean_data = clean_data.iloc[:1000000].copy()
            
            clean_data.to_excel(writer, sheet_name=table_name, index=False)
    
    print(f"Successfully saved {len(data_dict)} tables to {excel_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python export_data.py [pickle_path] [excel_path]")
        sys.exit(1)
    
    pickle_path = sys.argv[1]
    excel_path = sys.argv[2]
    
    export_to_excel(pickle_path, excel_path)
