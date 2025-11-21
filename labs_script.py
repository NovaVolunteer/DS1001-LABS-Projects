#!/usr/bin/env python3
"""
LABS-3: Systems Project Script
Consolidated script from LABS-03_Systems.ipynb

This script contains the code from the Systems Project notebook,
designed to ensure your environment is properly set up.
"""

## First cell
# run this for step 11 in your instructions

print("Hello, World!")

## Second cell
# run this for step 14 in your instructions

import pandas as pd

## Part 3: Accessing your data
# edit this code to load your data into your workspace

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
