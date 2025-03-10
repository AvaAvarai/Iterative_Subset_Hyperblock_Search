# Required packages:
# pip install pandas numpy matplotlib seaborn

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
import seaborn as sns

def load_data():
    """Open file picker dialog and load CSV data"""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not file_path:
        raise ValueError("No file selected")
    return pd.read_csv(file_path)

def normalize_data(df, label_col):
    """Normalize numerical columns to [0,1] range using min-max scaling"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = numeric_cols[numeric_cols != label_col]
    
    df_norm = df.copy()
    for col in numeric_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        df_norm[col] = (df[col] - min_val) / (max_val - min_val)
    
    return df_norm

def find_pure_intervals(df, label_col):
    """Find intervals where consecutive values have same class label"""
    intervals = []
    
    for col in df.columns:
        if col == label_col:
            continue
            
        # Sort by column and get consecutive ranges
        df_sorted = df.sort_values(by=col)
        
        current_class = df_sorted[label_col].iloc[0]
        start_val = df_sorted[col].iloc[0]
        count = 1
        
        for i in range(1, len(df_sorted)):
            if df_sorted[label_col].iloc[i] != current_class:
                if count > 1:
                    intervals.append({
                        'attribute': col,
                        'start': start_val,
                        'end': df_sorted[col].iloc[i-1],
                        'class': current_class,
                        'count': count
                    })
                current_class = df_sorted[label_col].iloc[i]
                start_val = df_sorted[col].iloc[i]
                count = 1
            else:
                count += 1
                
        # Handle last interval
        if count > 1:
            intervals.append({
                'attribute': col,
                'start': start_val,
                'end': df_sorted[col].iloc[-1],
                'class': current_class,
                'count': count
            })
    
    return intervals

def plot_parallel_coordinates(df, intervals, label_col):
    """Plot parallel coordinates with pure intervals highlighted"""
    # Get unique classes and assign colors
    classes = df[label_col].unique()
    colors = sns.color_palette("Set2", n_colors=len(classes))
    class_colors = dict(zip(classes, [rgb2hex(c) for c in colors]))
    
    # Create parallel coordinates plot
    plt.figure(figsize=(12, 6))
    pd.plotting.parallel_coordinates(df, label_col, colormap=plt.cm.get_cmap("Set2"))
    
    # Plot intervals
    for interval in intervals:
        y_pos = df.columns.get_loc(interval['attribute'])
        plt.plot(
            [y_pos, y_pos],
            [interval['start'], interval['end']],
            color=class_colors[interval['class']],
            linewidth=5,
            alpha=0.5
        )
    
    plt.title("Parallel Coordinates with Pure Intervals")
    plt.tight_layout()
    plt.show()

def main():
    # Load data
    df = load_data()
    
    # Find label column
    label_col = next(col for col in df.columns if col.lower() == 'class')
    
    # Normalize data
    df_norm = normalize_data(df, label_col)
    
    # Find pure intervals
    intervals = find_pure_intervals(df_norm, label_col)
    
    # Print intervals
    print("\nPure Intervals Found:")
    for interval in intervals:
        print(f"\nAttribute: {interval['attribute']}")
        print(f"Range: [{interval['start']:.3f}, {interval['end']:.3f}]")
        print(f"Class: {interval['class']}")
        print(f"Count: {interval['count']}")
    
    # Plot results
    plot_parallel_coordinates(df_norm, intervals, label_col)

if __name__ == "__main__":
    main()
