import os
import pandas as pd
import numpy as np

def analyze_dataset(dataset_path, output_dir):
    try:
        _, file_extension = os.path.splitext(dataset_path)
        
        # Load data based on file type
        if file_extension == '.csv':
            df = pd.read_csv(dataset_path)
        elif file_extension in ['.xls', '.xlsx']:
            df = pd.read_excel(dataset_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}. Please provide CSV or Excel file.")
        
        # Ensure dataset is not empty
        if df.empty:
            raise ValueError("The dataset is empty. Please provide a non-empty dataset.")
        
        # Initial analysis
        print("Dataset Overview:")
        print(f"Shape of dataset: {df.shape}")
        print(f"Columns in dataset: {df.columns.tolist()}")
        print("\nSample data:")
        print(df.head())
        
        # Identify column types
        column_types = df.dtypes
        print("\nColumn Types:")
        print(column_types)
        
        # Checking for missing values
        missing_data = df.isnull().sum()
        print("\nMissing Values per Column:")
        print(missing_data[missing_data > 0])  # Show only columns with missing values

        # Basic statistics for numerical columns
        print("\nBasic Statistics for Numerical Columns:")
        print(df.describe())

        # Save a report with these details in the outputs directory
        report_path = os.path.join(output_dir, "dataset_context_report.txt")
        with open(report_path, 'w') as f:
            f.write("Dataset Overview\n")
            f.write(f"Shape: {df.shape}\n")
            f.write(f"Columns: {df.columns.tolist()}\n\n")
            f.write("Column Types\n")
            f.write(column_types.to_string() + "\n\n")
            f.write("Missing Values\n")
            f.write(missing_data.to_string() + "\n\n")
            f.write("Basic Statistics for Numerical Columns\n")
            f.write(df.describe().to_string() + "\n")
        
        print(f"\nContext report saved at: {report_path}")
        return df
    
    except Exception as e:
        print(f"Error processing the dataset: {str(e)}")
        return None
