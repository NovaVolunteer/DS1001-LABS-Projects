#!/usr/bin/env python3
"""
LABS-3: Systems Project Script
Consolidated script from LABS-03_Systems.ipynb

This script contains the code from the Systems Project notebook,
designed to ensure your environment is properly set up.
"""

import pandas as pd


# Basic environment check
print("Hello, World!")


# Data loading utilities

def load_data(filepath):
    """
    Load data from a CSV file.
    
    Args:
        filepath (str): Path to the CSV file
    
    Returns:
        pandas.DataFrame: Loaded data
    """
    data = pd.read_csv(filepath)
    return data

# Example usage:
# data = load_data('your_data_file.csv')
# print(data.head())

if __name__ == "__main__":
    # Run the basic checks
    print("Environment setup check complete!")
    print(f"Pandas version: {pd.__version__}")
