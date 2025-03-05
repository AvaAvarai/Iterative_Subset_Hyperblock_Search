"""
Dataset Splitting Utility

This script creates multiple train-test splits of a dataset for cross-validation purposes.
It takes a CSV file as input and creates a specified number of splits (default: 10),
with each split having 1/3 of the data for training and 2/3 for testing.

For each split, the script:
1. Creates a directory structure in the 'splits' folder
2. Randomly divides the data using different random seeds for each split
3. Saves the resulting train and test sets as CSV files

Usage:
    python test3.py

The script expects a CSV file named 'fisher_iris_2class.csv' in the current directory,
but this can be modified by changing the 'data_path' variable.

Requirements:
    - pandas
    - numpy
    - scikit-learn
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def create_splits(data_path, num_splits=10):
    # Create base output directory
    os.makedirs('splits', exist_ok=True)
    
    # Load the dataset
    df = pd.read_csv(data_path)
    
    for split_num in range(num_splits):
        # Create directory for this split iteration
        split_dir = os.path.join('splits', f'split_{split_num}')
        os.makedirs(split_dir, exist_ok=True)
        
        # Split into 1/3 train and 2/3 test
        train, test = train_test_split(df, test_size=2/3, random_state=split_num)
        
        # Save the splits
        train.to_csv(os.path.join(split_dir, 'train.csv'), index=False)
        test.to_csv(os.path.join(split_dir, 'test.csv'), index=False)

if __name__ == "__main__":
    data_path = "fisher_iris_2class.csv"  # Update with your dataset path
    create_splits(data_path)
    print("Created 10 sets of 1/3 train - 2/3 test splits in the 'splits' directory")
