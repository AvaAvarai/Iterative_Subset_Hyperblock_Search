"""
Dataset Splitting Utility

This script creates multiple train-test splits of a dataset for cross-validation purposes.
It takes a CSV file as input and creates a specified number of splits (default: 10),
with each split having 1/3 of the data for training and 2/3 for testing.

For each split, the script:
1. Creates a directory structure in the 'splits' folder
2. Randomly divides the data using different random seeds for each split
3. Saves the resulting train and test sets as CSV files

By default, stratified sampling is used to maintain the same class distribution
in both training and test sets.

Usage:
    python data_splitter.py

The script will open a file dialog to select the CSV file to split.

Requirements:
    - pandas
    - numpy
    - scikit-learn
    - tkinter (for file dialog)
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tkinter import Tk, filedialog

def create_splits(data_path, num_splits=10, stratified=True):
    # Create base output directory
    os.makedirs('splits', exist_ok=True)
    
    # Load the dataset
    df = pd.read_csv(data_path)
    
    for split_num in range(num_splits):
        # Create directory for this split iteration
        split_dir = os.path.join('splits', f'split_{split_num}')
        os.makedirs(split_dir, exist_ok=True)
        
        # Split into 1/3 train and 2/3 test
        if stratified:
            # Assume the last column is the class label
            y = df.iloc[:, -1]
            train, test = train_test_split(df, test_size=2/3, random_state=split_num, stratify=y)
        else:
            train, test = train_test_split(df, test_size=2/3, random_state=split_num)
        
        # Save the splits
        train.to_csv(os.path.join(split_dir, 'train.csv'), index=False)
        test.to_csv(os.path.join(split_dir, 'test.csv'), index=False)

if __name__ == "__main__":
    # Initialize Tkinter and hide the main window
    root = Tk()
    root.withdraw()
    
    # Open file dialog to select the CSV file
    data_path = filedialog.askopenfilename(
        title="Select CSV file to split",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    
    if data_path:
        create_splits(data_path, stratified=True)
        print(f"Created 10 sets of 1/3 train - 2/3 test splits in the 'splits' directory for {os.path.basename(data_path)}")
    else:
        print("No file selected. Operation cancelled.")
