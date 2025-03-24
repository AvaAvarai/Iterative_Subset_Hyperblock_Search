import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import tkinter as tk
from tkinter import filedialog
import re
import os
import time
import random
import string

# Debug mode control - set to False to suppress debug prints
DEBUG = False

class Hyperblock:
    def __init__(self, min_bounds, max_bounds, points, dominant_class):
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds
        self.points = points
        self.dominant_class = dominant_class
        self.num_cases = len(points)
        self.num_misclassified = sum(1 for p in points if p[-1] != dominant_class)

    def __repr__(self):
        return (f"Min Bounds: {self.min_bounds}, Max Bounds: {self.max_bounds}, "
                f"Dominant Class: {self.dominant_class}, "
                f"Cases Contained: {self.num_cases}, "
                f"Misclassifications: {self.num_misclassified}")

def load_and_normalize_data(file_path=None):
    """
    Load a CSV file and normalize all numerical features to [0,1]
    
    Args:
        file_path: Path to CSV file (if None, will open file dialog)
        
    Returns:
        df: Normalized DataFrame
        features: List of feature names
        class_col: Name of the class column
    """
    if file_path is None:
        # Open file dialog to select CSV
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not file_path:
            raise ValueError("No file selected")
    
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Find the class column (case-insensitive)
    class_col = None
    for col in df.columns:
        if re.match(r'^class$', col, re.IGNORECASE):
            class_col = col
            break
    
    if class_col is None:
        raise ValueError("No 'class' column found in the dataset")
    
    # Separate features from the class column
    features = [col for col in df.columns if col != class_col]
    
    # Apply Min-Max normalization to features
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    
    if DEBUG:
        print(f"Loaded dataset with {len(df)} samples, {len(features)} features")
        print(f"Class column: {class_col}")
        print(f"Features: {', '.join(features)}")
    
    return df, features, class_col

def ihyper_algorithm(df, class_col, purity_threshold=0.95):
    """
    Implements the IHyper algorithm for hyperblock generation.
    
    Args:
        df: DataFrame containing the data
        class_col: Name of the class column
        purity_threshold: Minimum purity threshold (default: 0.95)
    
    Returns:
        List of Hyperblock objects
    """
    if DEBUG:
        print(f"Running IHyper algorithm with purity threshold = {purity_threshold}")
    
    # Extract features and class
    attributes = [col for col in df.columns if col != class_col]
    X = df[attributes].values
    
    # Convert class labels to numeric indices for internal processing
    y_original = df[class_col].values
    unique_classes = list(df[class_col].unique())
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    y = np.array([class_to_idx[cls] for cls in y_original])
    
    # Compute LDF values using Linear Discriminant Analysis
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    ldf_values = lda.decision_function(X)  # LDF values
    
    # Initialize result
    hyperblocks = []
    remaining_points = set(range(len(df)))
    
    iteration = 0
    while remaining_points and iteration < 100:  # Limit iterations to prevent infinite loop
        iteration += 1
        if DEBUG:
            print(f"  IHyper iteration {iteration}, {len(remaining_points)} points remaining")
        
        best_interval = None
        best_interval_size = 0
        best_attribute = None
        best_points_indices = None
        
        # Step 1: For each attribute, create sorted array
        for attr_idx, attr_name in enumerate(attributes):
            # Create sorted array for attribute
            sorted_indices = np.argsort(X[:, attr_idx])
            sorted_values = X[sorted_indices, attr_idx]
            
            # Only consider points still in remaining_points
            valid_indices = [i for i in sorted_indices if i in remaining_points]
            if not valid_indices:
                continue
                
            # Step 2: Process each potential seed value
            for seed_idx in valid_indices:
                # Steps 3-4: Initialize boundaries for current seed point
                a_point = X[seed_idx]
                a_class = y[seed_idx]
                
                # Initialize b_i = a_i = d_i
                current_interval = [sorted_values[list(sorted_indices).index(seed_idx)], 
                                   sorted_values[list(sorted_indices).index(seed_idx)]]
                
                # Initialize points in the interval
                interval_points = [seed_idx]
                
                # Step 5-6: Try to expand the interval
                # Start looking for next values in the sorted array after the seed
                seed_pos = list(sorted_indices).index(seed_idx)
                
                # Try expanding upper bound
                for next_pos in range(seed_pos + 1, len(sorted_indices)):
                    next_idx = sorted_indices[next_pos]
                    if next_idx not in remaining_points:
                        continue
                        
                    # Check if adding this point would maintain purity
                    test_interval_points = interval_points + [next_idx]
                    test_classes = y[test_interval_points]
                    
                    # Count occurrences of each class manually
                    class_counts = {}
                    for c in test_classes:
                        if c not in class_counts:
                            class_counts[c] = 0
                        class_counts[c] += 1
                    
                    dominant_class = max(class_counts.items(), key=lambda x: x[1])[0]
                    purity = class_counts[dominant_class] / len(test_interval_points)
                    
                    if purity >= purity_threshold:
                        # Add this point to the interval
                        interval_points.append(next_idx)
                        current_interval[1] = sorted_values[next_pos]
                    else:
                        # Stop expanding if purity drops below threshold
                        break
                
                # Try expanding lower bound
                for next_pos in range(seed_pos - 1, -1, -1):
                    next_idx = sorted_indices[next_pos]
                    if next_idx not in remaining_points:
                        continue
                        
                    # Check if adding this point would maintain purity
                    test_interval_points = interval_points + [next_idx]
                    test_classes = y[test_interval_points]
                    
                    # Count occurrences of each class manually
                    class_counts = {}
                    for c in test_classes:
                        if c not in class_counts:
                            class_counts[c] = 0
                        class_counts[c] += 1
                    
                    dominant_class = max(class_counts.items(), key=lambda x: x[1])[0]
                    purity = class_counts[dominant_class] / len(test_interval_points)
                    
                    if purity >= purity_threshold:
                        # Add this point to the interval
                        interval_points.append(next_idx)
                        current_interval[0] = sorted_values[next_pos]
                    else:
                        # Stop expanding if purity drops below threshold
                        break
                
                # Step 7: Save this interval if it's the largest for this attribute
                if len(interval_points) > best_interval_size:
                    best_interval = current_interval
                    best_interval_size = len(interval_points)
                    best_attribute = attr_idx
                    best_points_indices = interval_points
        
        # Step 9-10: Select the attribute with the largest interval and create hyperblock
        if best_interval is None:
            # No more pure intervals can be found
            if DEBUG:
                print("  No more pure intervals can be found, stopping IHyper")
            break
        
        # Create hyperblock from best interval
        points_in_hb = df.iloc[best_points_indices].values
        numeric_classes = y[best_points_indices]
        
        # Count occurrences of each class manually
        class_counts = {}
        for c in numeric_classes:
            if c not in class_counts:
                class_counts[c] = 0
            class_counts[c] += 1
        
        dominant_class_idx = max(class_counts.items(), key=lambda x: x[1])[0]
        dominant_class = unique_classes[dominant_class_idx]  # Convert back to original class label
        
        # Create bounds for all attributes
        min_bounds = {attr: np.min(X[best_points_indices, i]) for i, attr in enumerate(attributes)}
        max_bounds = {attr: np.max(X[best_points_indices, i]) for i, attr in enumerate(attributes)}
        
        hyperblock = Hyperblock(min_bounds, max_bounds, points_in_hb, dominant_class)
        hyperblocks.append(hyperblock)
        
        if DEBUG:
            print(f"  Created hyperblock with {len(best_points_indices)} points, dominant class: {dominant_class}")
        
        # Step 11: Remove points in hyperblock from remaining_points
        remaining_points -= set(best_points_indices)
    
    if DEBUG:
        print(f"IHyper created {len(hyperblocks)} hyperblocks")
    return hyperblocks

def mhyper_algorithm(df, class_col, impurity_threshold=0.1):
    """
    Implements the MHyper algorithm for hyperblock generation.
    
    Args:
        df: DataFrame containing the data
        class_col: Name of the class column
        impurity_threshold: Maximum allowed impurity (default: 0.1)
    
    Returns:
        List of Hyperblock objects
    """
    if DEBUG:
        print(f"Running MHyper algorithm with impurity threshold = {impurity_threshold}")
    
    attributes = [col for col in df.columns if col != class_col]
    
    # Step 1: Initialize pure hyperblocks with single points
    hyperblocks = []
    for idx, row in df.iterrows():
        point_class = row[class_col]
        
        # Create min/max bounds (same as the point values)
        min_bounds = {attr: row[attr] for attr in attributes}
        max_bounds = {attr: row[attr] for attr in attributes}
        
        # Create a hyperblock with a single point
        hb = {
            'min_bounds': min_bounds,
            'max_bounds': max_bounds,
            'points': [idx],  # Store the DataFrame index
            'class': point_class
        }
        hyperblocks.append(hb)
    
    if DEBUG:
        print(f"  Initialized {len(hyperblocks)} single-point hyperblocks")
    
    # Steps 2-5: Merge pure hyperblocks
    remaining_hbs = list(range(len(hyperblocks)))
    result_hbs = []
    
    # Pure merging phase
    pure_merges = 0
    while remaining_hbs:
        # Step 2: Select a hyperblock
        hb_x_idx = remaining_hbs.pop(0)
        hb_x = hyperblocks[hb_x_idx]
        merged = False
        
        # Step 3: Try to merge with other hyperblocks
        i = 0
        while i < len(remaining_hbs):
            hb_i_idx = remaining_hbs[i]
            hb_i = hyperblocks[hb_i_idx]
            
            # Step 3a: If same class, try to merge
            if hb_x['class'] == hb_i['class']:
                # Create joint hyperblock (envelope)
                joint_min_bounds = {
                    attr: min(hb_x['min_bounds'][attr], hb_i['min_bounds'][attr]) 
                    for attr in attributes
                }
                joint_max_bounds = {
                    attr: max(hb_x['max_bounds'][attr], hb_i['max_bounds'][attr]) 
                    for attr in attributes
                }
                
                # Step 3b: Find points that belong in the envelope
                joint_points = set(hb_x['points'] + hb_i['points'])
                
                # Check all points in the dataframe to see if they belong in the envelope
                for idx in df.index:
                    if idx in joint_points:
                        continue
                        
                    row = df.loc[idx]
                    # Check if point is in envelope
                    in_envelope = True
                    for attr in attributes:
                        if (row[attr] < joint_min_bounds[attr] or 
                            row[attr] > joint_max_bounds[attr]):
                            in_envelope = False
                            break
                    
                    if in_envelope:
                        joint_points.add(idx)
                
                # Step 3c: Check if joint hyperblock is pure
                joint_classes = [df.loc[idx][class_col] for idx in joint_points]
                is_pure = all(cls == hb_x['class'] for cls in joint_classes)
                
                if is_pure:
                    # Create new pure hyperblock
                    joint_hb = {
                        'min_bounds': joint_min_bounds,
                        'max_bounds': joint_max_bounds,
                        'points': list(joint_points),
                        'class': hb_x['class']
                    }
                    
                    # Remove hb_i from remaining, add joint to result
                    remaining_hbs.pop(i)
                    hb_x = joint_hb  # Update hb_x to be the joint hyperblock
                    merged = True
                    pure_merges += 1
                    # Don't increment i, as we removed an element
                else:
                    i += 1
            else:
                i += 1
        
        # If we couldn't merge hb_x with anything, add it to result
        result_hbs.append(hb_x)
    
    if DEBUG:
        print(f"  Completed {pure_merges} pure merges, resulting in {len(result_hbs)} pure hyperblocks")
    
    # Step 6-9: Merge hyperblocks with limited impurity
    # Continue as long as we can find valid merges
    impure_merges = 0
    while True:
        best_merge = None
        lowest_impurity = float('inf')
        
        # Try all pairs of hyperblocks
        for i in range(len(result_hbs)):
            for j in range(i+1, len(result_hbs)):
                hb_i = result_hbs[i]
                hb_j = result_hbs[j]
                
                # Step 8a: Create joint hyperblock
                joint_min_bounds = {
                    attr: min(hb_i['min_bounds'][attr], hb_j['min_bounds'][attr]) 
                    for attr in attributes
                }
                joint_max_bounds = {
                    attr: max(hb_i['max_bounds'][attr], hb_j['max_bounds'][attr]) 
                    for attr in attributes
                }
                
                # Step 8b: Find points that belong in the envelope
                joint_points = set(hb_i['points'] + hb_j['points'])
                
                # Check all points in the dataframe
                for idx in df.index:
                    if idx in joint_points:
                        continue
                    
                    row = df.loc[idx]
                    # Check if point is in envelope
                    in_envelope = True
                    for attr in attributes:
                        if (row[attr] < joint_min_bounds[attr] or 
                            row[attr] > joint_max_bounds[attr]):
                            in_envelope = False
                            break
                    
                    if in_envelope:
                        joint_points.add(idx)
                
                # Step 8c: Compute impurity
                joint_classes = [df.loc[idx][class_col] for idx in joint_points]
                
                # Count occurrences of each class manually
                class_counts = {}
                for cls in joint_classes:
                    if cls not in class_counts:
                        class_counts[cls] = 0
                    class_counts[cls] += 1
                
                dominant_class = max(class_counts.items(), key=lambda x: x[1])[0]
                impurity = 1 - (class_counts[dominant_class] / len(joint_points))
                
                # Step 8d: If impurity is acceptable, consider this merge
                if impurity < impurity_threshold and impurity < lowest_impurity:
                    best_merge = (i, j, {
                        'min_bounds': joint_min_bounds,
                        'max_bounds': joint_max_bounds,
                        'points': list(joint_points),
                        'class': dominant_class
                    })
                    lowest_impurity = impurity
        
        # If no valid merge found, we're done
        if best_merge is None:
            break
            
        # Perform the best merge
        i, j, joint_hb = best_merge
        # Remove merged hyperblocks and add joint hyperblock
        if i < j:
            result_hbs.pop(j)
            result_hbs.pop(i)
        else:
            result_hbs.pop(i)
            result_hbs.pop(j)
        result_hbs.append(joint_hb)
        impure_merges += 1
    
    if DEBUG:
        print(f"  Completed {impure_merges} impure merges")
    
    # Convert to Hyperblock objects
    hyperblocks = []
    for hb in result_hbs:
        # Use loc instead of iloc to access by index values
        points = [df.loc[idx].values for idx in hb['points']]
        hb_obj = Hyperblock(hb['min_bounds'], hb['max_bounds'], points, hb['class'])
        hyperblocks.append(hb_obj)
    
    if DEBUG:
        print(f"MHyper created {len(hyperblocks)} hyperblocks")
    return hyperblocks

def imhyper_algorithm(df, class_col, purity_threshold=0.95, impurity_threshold=0.1):
    """
    Implements the IMHyper algorithm combining IHyper and MHyper.
    
    Args:
        df: DataFrame containing the data
        class_col: Name of the class column
        purity_threshold: Minimum purity threshold for IHyper (default: 0.95)
        impurity_threshold: Maximum allowed impurity for MHyper (default: 0.1)
    
    Returns:
        List of Hyperblock objects
    """
    if DEBUG:
        print(f"Running IMHyper algorithm (purity: {purity_threshold}, impurity: {impurity_threshold})")
    
    # Step 1: Run IHyper algorithm
    ihyper_blocks = ihyper_algorithm(df, class_col, purity_threshold)
    
    # Step 2: Find points not covered by IHyper blocks
    # Convert all points to hashable tuples for set operations
    all_points_tuples = [tuple(row.values) for _, row in df.iterrows()]
    covered_points_tuples = []
    
    for hb in ihyper_blocks:
        for point in hb.points:
            covered_points_tuples.append(tuple(point))
    
    # Find which points are not covered
    remaining_tuples = set(all_points_tuples) - set(covered_points_tuples)
    
    # If all points are covered, return just the IHyper blocks
    if not remaining_tuples:
        if DEBUG:
            print("All points covered by IHyper, skipping MHyper")
        return ihyper_blocks
    
    # Create DataFrame with remaining points
    remaining_rows = []
    for idx, row in df.iterrows():
        if tuple(row.values) in remaining_tuples:
            remaining_rows.append(row)
    
    remaining_df = pd.DataFrame(remaining_rows, columns=df.columns)
    if DEBUG:
        print(f"IHyper left {len(remaining_df)} points uncovered, running MHyper on them")
    
    # Step 3: Run MHyper on remaining points
    mhyper_blocks = mhyper_algorithm(remaining_df, class_col, impurity_threshold)
    
    # Combine results
    all_blocks = ihyper_blocks + mhyper_blocks
    if DEBUG:
        print(f"IMHyper created a total of {len(all_blocks)} hyperblocks")
    return all_blocks

def calculate_dataset_coverage(df, hyperblocks):
    """
    Calculate how many points from the dataset are covered by the hyperblocks.
    
    Args:
        df: DataFrame containing the data
        hyperblocks: List of Hyperblock objects
    
    Returns:
        covered_count: Number of points covered
        coverage_percentage: Percentage of dataset covered
        uncovered_points: DataFrame containing uncovered points
    """
    # Convert all data points to hashable tuples for set operations
    all_points = set(tuple(row) for row in df.values)
    
    # Collect all points covered by hyperblocks
    covered_points = set()
    for hb in hyperblocks:
        for point in hb.points:
            covered_points.add(tuple(point))
    
    # Calculate uncovered points
    uncovered_points = all_points - covered_points
    
    # Calculate coverage statistics
    total_points = len(all_points)
    covered_count = len(covered_points)
    coverage_percentage = (covered_count / total_points) * 100
    
    # Create DataFrame of uncovered points if needed
    uncovered_df = None
    if uncovered_points:
        uncovered_rows = [list(p) for p in uncovered_points]
        uncovered_df = pd.DataFrame(uncovered_rows, columns=df.columns)
    
    return covered_count, coverage_percentage, uncovered_df

def visualize_hyperblocks(df, features, class_col, hyperblocks, title="Hyperblocks in Parallel Coordinates", save_path=None):
    """
    Visualize hyperblocks in parallel coordinates and print coverage statistics.
    
    Args:
        df: DataFrame containing the data
        features: List of feature names
        class_col: Name of the class column
        hyperblocks: List of Hyperblock objects
        title: Title for the plot
        save_path: Path to save the figure (if None, just display)
    """
    # Calculate dataset coverage
    covered_count, coverage_percentage, uncovered_df = calculate_dataset_coverage(df, hyperblocks)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set up the axes
    x = list(range(len(features)))
    
    # Get unique classes for color mapping
    unique_classes = df[class_col].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
    class_color_map = dict(zip(unique_classes, colors))
    
    # Plot the data points first
    for _, row in df.iterrows():
        point_values = [row[feat] for feat in features]
        point_class = row[class_col]
        ax.plot(x, point_values, color=class_color_map[point_class], alpha=0.3, linewidth=0.8)
    
    # Generate distinct colors for each hyperblock
    block_colors = plt.cm.tab20(np.linspace(0, 1, len(hyperblocks)))
    
    # Plot each hyperblock
    for i, hb in enumerate(hyperblocks):
        block_color = block_colors[i]
        
        # Plot the hyperblock bounds as a shaded region
        for j in range(len(features)-1):
            # Get min and max values for current and next feature
            y1_min = hb.min_bounds[features[j]]
            y1_max = hb.max_bounds[features[j]]
            y2_min = hb.min_bounds[features[j+1]]
            y2_max = hb.max_bounds[features[j+1]]
            
            # Create polygon vertices for the shaded region
            xs = [x[j], x[j], x[j+1], x[j+1]]
            ys = [y1_min, y1_max, y2_max, y2_min]
            
            # Plot the polygon
            ax.fill(xs, ys, alpha=0.2, color=block_color, edgecolor=block_color, 
                   linewidth=1, label=f"HB {i+1} ({hb.dominant_class})" if j == 0 else None)
    
    # Create legend for classes
    class_legend_elements = [Line2D([0], [0], color=color, lw=2, label=f"Class {cls}")
                           for cls, color in class_color_map.items()]
    
    # Add dual legends
    ax.legend(title="Hyperblocks", loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    
    # Create a second legend for classes and add it
    ax.figure.legend(handles=class_legend_elements, title="Classes", 
                    loc='upper left', bbox_to_anchor=(1.05, 0.5), borderaxespad=0.)
    
    # Set the x-axis ticks and labels
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45)
    ax.set_ylim(0, 1)  # Set y-axis from 0 to 1 (normalized data)
    
    # Set title and labels
    ax.set_title(f"{title}\nDataset Coverage: {covered_count}/{len(df)} points ({coverage_percentage:.2f}%)")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Print complete hyperblock summary with all attributes - this is a summary table that should always print
    print("\nHyperblock Summary (Complete Bounds):")
    print("-" * 100)
    print(f"{'#':<3} {'Class':<10} {'Cases':<6} {'Misclass':<9} {'Bounds'}")
    print("-" * 100)
    
    for i, hb in enumerate(hyperblocks):
        print(f"{i+1:<3} {str(hb.dominant_class):<10} {hb.num_cases:<6} {hb.num_misclassified:<9}")
        
        # Print bounds for each feature on separate lines
        for f in features:
            print(f"    {f}: [{hb.min_bounds[f]:.4f}, {hb.max_bounds[f]:.4f}]")
        
        # Add a separator between hyperblocks
        if i < len(hyperblocks) - 1:
            print("-" * 50)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # Make room for the legend
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if DEBUG:
            print(f"Figure saved to: {save_path}")
    
    # Comment out plt.show() to avoid displaying the figure
    # plt.show()
    plt.close()  # Close the figure to free memory

def get_hyperblock_statistics(hyperblocks, total_points):
    """
    Generate statistics for hyperblocks.
    
    Args:
        hyperblocks: List of Hyperblock objects
        total_points: Total number of points in the dataset
    
    Returns:
        Dictionary with hyperblock statistics
    """
    total_hyperblocks = len(hyperblocks)
    total_covered = sum(hb.num_cases for hb in hyperblocks)
    coverage_percentage = (total_covered / total_points) * 100 if total_points > 0 else 0
    
    # Count misclassifications
    total_misclassified = sum(hb.num_misclassified for hb in hyperblocks)
    misclassification_rate = (total_misclassified / total_covered) * 100 if total_covered > 0 else 0
    
    # Class distribution
    class_distribution = {}
    for hb in hyperblocks:
        if hb.dominant_class not in class_distribution:
            class_distribution[hb.dominant_class] = 0
        class_distribution[hb.dominant_class] += 1
    
    # Calculate average hyperblock size (number of points)
    avg_size = total_covered / total_hyperblocks if total_hyperblocks > 0 else 0
    
    return {
        'total_hyperblocks': total_hyperblocks,
        'total_covered': total_covered,
        'coverage_percentage': coverage_percentage,
        'total_misclassified': total_misclassified,
        'misclassification_rate': misclassification_rate,
        'class_distribution': class_distribution,
        'avg_size': avg_size
    }

def visualize_single_hyperblock(df, features, class_col, hyperblock, hb_index, total_hbs, save_path=None):
    """
    Visualize a single hyperblock in parallel coordinates.
    
    Args:
        df: DataFrame containing the data
        features: List of feature names
        class_col: Name of the class column
        hyperblock: A single Hyperblock object
        hb_index: Index of this hyperblock
        total_hbs: Total number of hyperblocks
        save_path: Path to save the figure
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set up the axes
    x = list(range(len(features)))
    
    # Get unique classes for color mapping
    unique_classes = df[class_col].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
    class_color_map = dict(zip(unique_classes, colors))
    
    # Plot the background data points with low alpha
    for _, row in df.iterrows():
        point_values = [row[feat] for feat in features]
        point_class = row[class_col]
        ax.plot(x, point_values, color='gray', alpha=0.1, linewidth=0.5)
    
    # Generate a specific color for this hyperblock
    block_color = plt.cm.tab20(hb_index / total_hbs)
    
    # Plot the hyperblock bounds as a shaded region
    for j in range(len(features)-1):
        # Get min and max values for current and next feature
        y1_min = hyperblock.min_bounds[features[j]]
        y1_max = hyperblock.max_bounds[features[j]]
        y2_min = hyperblock.min_bounds[features[j+1]]
        y2_max = hyperblock.max_bounds[features[j+1]]
        
        # Create polygon vertices for the shaded region
        xs = [x[j], x[j], x[j+1], x[j+1]]
        ys = [y1_min, y1_max, y2_max, y2_min]
        
        # Plot the polygon
        ax.fill(xs, ys, alpha=0.3, color=block_color, edgecolor=block_color, linewidth=1)
    
    # Plot the points contained in this hyperblock with high alpha
    for point in hyperblock.points:
        # Extract feature values
        point_values = point[:-1]  # Exclude the class value
        point_class = point[-1]
        
        # Determine if this point is misclassified
        is_misclassified = point_class != hyperblock.dominant_class
        
        # Draw misclassified points in bright red, correctly classified points in their class color
        if is_misclassified:
            ax.plot(x, point_values, color='#FF0000', alpha=1.0, linewidth=3.0, zorder=3, 
                  marker='o', markersize=4, markeredgecolor='black', markeredgewidth=0.8)
        else:
            ax.plot(x, point_values, color=class_color_map[point_class], alpha=0.8, linewidth=1, zorder=2)
    
    # Set the x-axis ticks and labels
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45)
    ax.set_ylim(0, 1)  # Set y-axis from 0 to 1 (normalized data)
    
    # Set title
    correct_points = hyperblock.num_cases - hyperblock.num_misclassified
    ax.set_title(f"Hyperblock {hb_index+1}/{total_hbs}: Class {hyperblock.dominant_class}\n"
                f"Contains {hyperblock.num_cases} points ({correct_points} correct, {hyperblock.num_misclassified} misclassified)")
    
    # Add a legend for misclassified points if any exist
    if hyperblock.num_misclassified > 0:
        ax.plot([], [], color='#FF0000', linewidth=3.0, marker='o', markersize=4,
              markeredgecolor='black', markeredgewidth=0.8, label='Misclassified')
        ax.legend()
    
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()  # Close the figure to free memory
    return save_path

def create_final_hyperblocks_collage(df, features, class_col, hyperblocks, output_dir):
    """
    Creates a collage of individual hyperblock visualizations from the final iteration.
    
    Args:
        df: DataFrame containing the data
        features: List of feature names
        class_col: Name of the class column
        hyperblocks: List of Hyperblock objects from the final iteration
        output_dir: Directory to save images
    """
    if DEBUG:
        print("\nCreating individual hyperblock collage from final iteration...")
    
    # Create a subdirectory for individual hyperblock visualizations
    individual_dir = os.path.join(output_dir, "individual_hyperblocks")
    os.makedirs(individual_dir, exist_ok=True)
    
    # Generate individual hyperblock visualizations
    image_paths = []
    for i, hb in enumerate(hyperblocks):
        save_path = os.path.join(individual_dir, f"hyperblock_{i+1}_class_{hb.dominant_class}.png")
        visualize_single_hyperblock(df, features, class_col, hb, i, len(hyperblocks), save_path)
        image_paths.append(save_path)
    
    # Determine grid size for the collage
    num_hbs = len(hyperblocks)
    cols = min(5, num_hbs)
    rows = (num_hbs + cols - 1) // cols  # Ceiling division
    
    # Create collage figure
    fig = plt.figure(figsize=(5*cols, 4*rows))
    
    # Add each hyperblock visualization to the collage
    for i, img_path in enumerate(image_paths):
        img = plt.imread(img_path)
        
        # Add subplot
        ax = fig.add_subplot(rows, cols, i+1)
        ax.imshow(img)
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save the collage
    collage_path = os.path.join(output_dir, "final_hyperblocks_collage.png")
    plt.savefig(collage_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if DEBUG:
        print(f"Final hyperblocks collage saved to: {collage_path}")
        print(f"Individual hyperblock visualizations saved to: {individual_dir}")

def create_hyperblock_bounds_collage(df, features, class_col, hyperblocks, output_dir):
    """
    Creates a collage with each hyperblock showing upper and lower bound polylines.
    """
    if DEBUG:
        print("\nCreating hyperblock bounds collage...")
    
    # Determine grid size
    n_blocks = len(hyperblocks)
    cols = min(3, n_blocks)
    rows = (n_blocks + cols - 1) // cols
    
    # Create figure
    fig = plt.figure(figsize=(5*cols, 4*rows))
    
    # Create subplot for each hyperblock
    for i, hb in enumerate(hyperblocks):
        ax = plt.subplot(rows, cols, i+1)
        
        # X-values for features
        x = range(len(features))
        
        # Get upper and lower bounds for this hyperblock
        upper_bounds = [hb.max_bounds[feat] for feat in features]
        lower_bounds = [hb.min_bounds[feat] for feat in features]
        
        # Plot data points in gray
        for _, row in df.iterrows():
            values = [row[feat] for feat in features]
            plt.plot(x, values, color='gray', alpha=0.1, linewidth=0.5)
        
        # Plot the hyperblock UPPER bound in RED with thicker line
        plt.plot(x, upper_bounds, 'r-', linewidth=3, label='Upper Bound')
        
        # Plot the hyperblock LOWER bound in BLUE with thicker line
        plt.plot(x, lower_bounds, 'b-', linewidth=3, label='Lower Bound')
        
        # Set title
        plt.title(f"Hyperblock {i+1} (Class {hb.dominant_class})")
        
        # Format axes
        plt.xticks(x, features, rotation=45)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        # Add legend to first plot only
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    
    # Save the collage
    collage_path = os.path.join(output_dir, "hyperblock_bounds_collage.png")
    plt.savefig(collage_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if DEBUG:
        print(f"Hyperblock bounds collage saved to: {collage_path}")

def visualize_hyperblocks_with_bounds(df, features, class_col, hyperblocks, output_dir):
    """
    Create a collage showing each hyperblock with:
    1. The points IN the hyperblock with correctly classified in their class colors and misclassified in RED
    2. A RED line for the upper bound
    3. A BLUE line for the lower bound
    """
    # Setup grid layout
    n_blocks = len(hyperblocks)
    cols = min(3, n_blocks)
    rows = (n_blocks + cols - 1) // cols
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Get unique classes for color mapping
    unique_classes = df[class_col].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
    class_color_map = dict(zip(unique_classes, colors))
    
    for i, hb in enumerate(hyperblocks):
        # Create subplot
        plt.subplot(rows, cols, i+1)
        
        # Get x positions for features
        x = range(len(features))
        
        # Get the bounds for this hyperblock
        upper_bounds = [hb.max_bounds[f] for f in features]
        lower_bounds = [hb.min_bounds[f] for f in features]
        
        # Draw points that are IN THIS HYPERBLOCK with correctly classified in their class colors and misclassified in red
        misclassified_count = 0
        for point in hb.points:
            values = point[:-1]  # All except class
            point_class = point[-1]
            
            # Determine if point is misclassified
            is_misclassified = point_class != hb.dominant_class
            
            if is_misclassified:
                plt.plot(x, values, color='#FF0000', alpha=1.0, linewidth=3.0, zorder=3, 
                        marker='o', markersize=4, markeredgecolor='black', markeredgewidth=0.8)
                misclassified_count += 1
            else:
                plt.plot(x, values, color=class_color_map[point_class], alpha=0.7, linewidth=1, zorder=2)
        
        # 2. Draw RED line for UPPER bound - thick and prominent
        plt.plot(x, upper_bounds, 'r-', linewidth=3, label='Upper Bound', zorder=1)
        
        # 3. Draw BLUE line for LOWER bound - thick and prominent
        plt.plot(x, lower_bounds, 'b-', linewidth=3, label='Lower Bound', zorder=1)
        
        # Title and formatting
        correct_points = hb.num_cases - hb.num_misclassified
        plt.title(f"HB #{i+1}: Class {hb.dominant_class}\n"
                 f"{hb.num_cases} points ({correct_points} correct, {hb.num_misclassified} misclassified)")
        plt.xticks(x, features, rotation=90)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        # Create custom legend
        legend_elements = [
            Line2D([0], [0], color='blue', lw=3, label='Lower Bound'),
            Line2D([0], [0], color='red', lw=3, label='Upper Bound')
        ]
        
        # Add misclassified entry to legend if there are any
        if misclassified_count > 0:
            legend_elements.append(Line2D([0], [0], color='#FF0000', lw=3.0, marker='o', markersize=4,
                                         markeredgecolor='black', markeredgewidth=0.8,
                                         label=f'Misclassified ({misclassified_count})'))
        
        # Add legend to each plot
        plt.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hyperblock_bounds.png'), dpi=300)
    plt.close()
    
    if DEBUG:
        print(f"Saved hyperblock bounds visualization to {output_dir}/hyperblock_bounds.png")

def classify_with_hyperblocks(point, hyperblocks, features):
    """
    Classify a point using the nearest hyperblock centroid.
    
    Args:
        point: Data point to classify (array-like)
        hyperblocks: List of Hyperblock objects
        features: List of feature names
    
    Returns:
        predicted_class: Predicted class label
    """
    if not hyperblocks:
        return None
        
    min_distance = float('inf')
    predicted_class = None
    
    # Extract feature values from the point (exclude class at the end if present)
    if len(point) > len(features):
        point_features = point[:len(features)]
    else:
        point_features = point
        
    # Calculate centroid for each hyperblock and find the nearest
    for hb in hyperblocks:
        # Calculate centroid (mean of min and max bounds)
        centroid = [(hb.min_bounds[feat] + hb.max_bounds[feat]) / 2 for feat in features]
        
        # Calculate Euclidean distance
        distance = sum((p - c) ** 2 for p, c in zip(point_features, centroid)) ** 0.5
        
        # Update if this is the nearest hyperblock
        if distance < min_distance:
            min_distance = distance
            predicted_class = hb.dominant_class
            
    return predicted_class

def incremental_hyperblock_generation(df, features, class_col, rows_per_iteration):
    """
    Incrementally generate hyperblocks by gradually adding data.
    
    Args:
        df: DataFrame containing the data
        features: List of feature names
        class_col: Name of the class column
        rows_per_iteration: Number of rows to add in each iteration
    """
    # Create output directory for images
    output_dir = 'hyperblock_progression'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a unique run identifier
    import time
    import random
    import string
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    random_suffix = ''.join(random.choices(string.ascii_lowercase, k=4))
    run_id = f"{timestamp}_{random_suffix}"
    
    if DEBUG:
        print(f"Run ID: {run_id}")
    
    # Create a subdirectory for this specific run
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Get total dataset size
    total_rows = len(df)
    
    # Initial subset: take exactly 1/3 of the data as a completely random sample
    initial_subset_size = total_rows // 3
    
    # Create a fresh copy of the DataFrame and shuffle it completely
    shuffled_indices = np.random.permutation(df.index)
    
    # Select initial subset from the shuffled indices
    initial_indices = shuffled_indices[:initial_subset_size]
    current_df = df.loc[initial_indices].copy()
    
    # The rest of the indices will be used for incremental additions
    remaining_indices = shuffled_indices[initial_subset_size:]
    
    # Prepare to track statistics
    stats_records = []
    
    # Start with the initial subset
    current_size = initial_subset_size
    current_indices = list(initial_indices)
    
    # Calculate how many iterations we'll need
    remaining_rows = total_rows - current_size
    iterations = (remaining_rows + rows_per_iteration - 1) // rows_per_iteration + 1  # Ceiling division
    
    if DEBUG:
        print(f"Starting with {current_size} rows ({current_size/total_rows*100:.1f}% of the dataset)")
        print(f"Initial subset is a completely random sample")
        print(f"Will add {rows_per_iteration} rows at each step")
        print(f"Total {iterations} iterations needed to process all {total_rows} rows")
    
    # Save the sequence of indices for reproducibility
    indices_file = os.path.join(run_dir, f"{run_id}_indices_sequence.txt")
    with open(indices_file, 'w') as f:
        f.write("Initial indices:\n")
        f.write(",".join(map(str, initial_indices)) + "\n\n")
        f.write("Remaining indices (in order of addition):\n")
        f.write(",".join(map(str, remaining_indices)))
    
    # Create a fixed color palette - using a distinctive color set
    color_palette = [
        plt.cm.tab20(0),   # blue
        plt.cm.tab20(2),   # green
        plt.cm.tab20(4),   # red
        plt.cm.tab20(6),   # purple
        plt.cm.tab20(8),   # orange
        plt.cm.tab20(10),  # yellow
        plt.cm.tab20(12),  # teal
        plt.cm.tab20(14),  # pink
        plt.cm.tab20(16),  # light blue
        plt.cm.tab20(18),  # light green
        plt.cm.tab20(1),   # dark blue
        plt.cm.tab20(3),   # dark green
        plt.cm.tab20(5),   # dark red
        plt.cm.tab20(7),   # dark purple
        plt.cm.tab20(9),   # dark orange
        plt.cm.tab20(11),  # dark yellow
        plt.cm.tab20(13),  # dark teal
        plt.cm.tab20(15),  # dark pink
        plt.cm.tab20(17),  # medium blue
        plt.cm.tab20(19),  # medium green
    ]
    
    # Store ALL hyperblock centroids from ALL previous iterations with their colors
    # Format: [(centroid_vector, class, color_index), ...]
    all_previous_hbs = []
    
    # Function to calculate hyperblock centroid
    def calculate_centroid(hb):
        return [
            (hb.min_bounds[feat] + hb.max_bounds[feat]) / 2
            for feat in features
        ]
    
    # Function to compute similarity between two hyperblocks using centroids
    def compute_similarity(centroid1, centroid2):
        return 1.0 / (1.0 + sum((a - b) ** 2 for a, b in zip(centroid1, centroid2)) ** 0.5)
    
    # Function to find best color match based on centroid similarity
    def find_best_color_match(hb):
        centroid = calculate_centroid(hb)
        hb_class = hb.dominant_class
        
        if not all_previous_hbs:
            # First hyperblock ever, assign first color
            return 0, centroid
        
        # Find previously seen hyperblocks with same class
        same_class_hbs = [(prev_centroid, color_idx) 
                          for prev_centroid, prev_class, color_idx in all_previous_hbs 
                          if prev_class == hb_class]
        
        if not same_class_hbs:
            # First hyperblock of this class, find unused color
            used_colors = set(color_idx for _, _, color_idx in all_previous_hbs)
            for i in range(len(color_palette)):
                if i not in used_colors:
                    return i, centroid
            # If all colors used, use the least used one
            color_counts = {}
            for _, _, c in all_previous_hbs:
                color_counts[c] = color_counts.get(c, 0) + 1
            least_used = min(color_counts.items(), key=lambda x: x[1])[0]
            return least_used, centroid
        
        # Calculate similarity to all previous hyperblocks of same class
        similarities = [(compute_similarity(centroid, prev_centroid), color_idx) 
                        for prev_centroid, color_idx in same_class_hbs]
        
        # Return the color of the most similar hyperblock
        best_match = max(similarities, key=lambda x: x[0])
        # Only use this match if similarity is above threshold (0.7)
        if best_match[0] > 0.7:
            return best_match[1], centroid
        
        # Otherwise use a new color
        used_colors = set(color_idx for _, _, color_idx in all_previous_hbs)
        for i in range(len(color_palette)):
            if i not in used_colors:
                return i, centroid
        
        # If all colors used, use the least used one
        color_counts = {}
        for _, _, c in all_previous_hbs:
            color_counts[c] = color_counts.get(c, 0) + 1
        least_used = min(color_counts.items(), key=lambda x: x[1])[0]
        return least_used, centroid
    
    iteration = 1
    previous_hyperblocks = None
    
    # Save statistics for initial iteration before the while loop
    percentage = current_size / total_rows * 100
    if DEBUG:
        print(f"\n{'='*80}")
        print(f"Initial Iteration {iteration}/{iterations}: Processing {current_size} rows ({percentage:.1f}% of the dataset)")
        print(f"{'='*80}")
    
    # Generate hyperblocks for initial dataset
    hyperblocks = imhyper_algorithm(df.loc[current_indices], class_col, purity_threshold=0.9999, impurity_threshold=0.0001)
    
    # Sort hyperblocks by size (larger ones first) for more stable color assignment
    hyperblocks.sort(key=lambda hb: hb.num_cases, reverse=True)
    
    # Assign colors based on similarity to previous hyperblocks
    hb_colors = []
    for hb in hyperblocks:
        color_idx, centroid = find_best_color_match(hb)
        hb_colors.append(color_idx)
        # Add this hyperblock to our history
        all_previous_hbs.append((centroid, hb.dominant_class, color_idx))
    
    # Get statistics
    stats = get_hyperblock_statistics(hyperblocks, current_size)
    stats['iteration'] = iteration
    stats['rows_processed'] = current_size
    stats['percentage_of_total'] = percentage
    stats['misclassifications'] = "N/A"  # Default for first iteration
    stats_records.append(stats)
    
    # Create proper, safe filename for saving the figure with unique identifiers
    filename = f"{run_id}_iter_{iteration:02d}_rows_{current_size}.png"
    save_path = os.path.join(run_dir, filename)
    
    # Visualization for initial iteration
    fig, ax = plt.subplots(figsize=(14, 8))
    x = list(range(len(features)))
    
    # Get unique classes for color mapping (for data points)
    unique_classes = df[class_col].unique()
    class_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
    class_color_map = dict(zip(unique_classes, class_colors))
    
    # Plot the data points first - still using class colors for the data points
    for idx in current_indices:
        row = df.loc[idx]
        point_values = [row[feat] for feat in features]
        point_class = row[class_col]
        ax.plot(x, point_values, color=class_color_map[point_class], alpha=0.3, linewidth=0.8)
    
    # Create legend entries for hyperblocks only
    hyperblock_legend_entries = []
    
    # Plot each hyperblock using our assigned colors
    for i, hb in enumerate(hyperblocks):
        block_color = color_palette[hb_colors[i]]
        
        # Create legend entry with more detail
        legend_entry = Line2D([0], [0], color=block_color, lw=4, 
                              label=f"HB {i+1}: Class {hb.dominant_class}, {hb.num_cases} points")
        hyperblock_legend_entries.append(legend_entry)
        
        # Plot the hyperblock bounds as a shaded region
        for j in range(len(features)-1):
            # Get min and max values for current and next feature
            y1_min = hb.min_bounds[features[j]]
            y1_max = hb.max_bounds[features[j]]
            y2_min = hb.min_bounds[features[j+1]]
            y2_max = hb.max_bounds[features[j+1]]
            
            # Create polygon vertices for the shaded region
            xs = [x[j], x[j], x[j+1], x[j+1]]
            ys = [y1_min, y1_max, y2_max, y2_min]
            
            # Plot the polygon
            ax.fill(xs, ys, alpha=0.2, color=block_color, edgecolor=block_color, linewidth=1)
    
    # Add only the hyperblock legend - positioned to the right of the plot
    ax.legend(handles=hyperblock_legend_entries, title="Hyperblocks", 
             loc='center left', bbox_to_anchor=(1.05, 0.5), borderaxespad=0.)
    
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45)
    ax.set_ylim(0, 1)
    
    # Calculate coverage
    covered_count, coverage_percentage, _ = calculate_dataset_coverage(df.loc[current_indices], hyperblocks)
    ax.set_title(f"IMHyper: Data with Hyperblock Visualization (Iteration {iteration}: {current_size}/{total_rows} rows)\n"
                f"Dataset Coverage: {covered_count}/{len(current_indices)} points ({coverage_percentage:.2f}%)")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # Still need space for the hyperblock legend
    
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if DEBUG:
            print(f"Figure saved to: {save_path}")
    except Exception as e:
        if DEBUG:
            print(f"Error saving figure: {e}")
        # Try saving with an even simpler filename as fallback
        fallback_path = os.path.join(run_dir, f"{run_id}_iter_{iteration}.png")
        plt.savefig(fallback_path, dpi=300, bbox_inches='tight')
        if DEBUG:
            print(f"Saved to fallback path: {fallback_path}")
    
    plt.close()
    
    # Save the current hyperblocks for next iteration
    previous_hyperblocks = hyperblocks
    iteration += 1
    
    # Continue with additional iterations
    while current_size <= total_rows:
        percentage = current_size / total_rows * 100
        if DEBUG:
            print(f"\n{'='*80}")
            print(f"Iteration {iteration}/{iterations}: Processing {current_size} rows ({percentage:.1f}% of the dataset)")
            print(f"{'='*80}")
        
        # Generate hyperblocks using the current subset
        hyperblocks = imhyper_algorithm(df.loc[current_indices], class_col, purity_threshold=0.9999, impurity_threshold=0.0001)
        
        # Sort hyperblocks by size (larger ones first) for more stable color assignment
        hyperblocks.sort(key=lambda hb: hb.num_cases, reverse=True)
        
        # Assign colors based on similarity to previous hyperblocks
        hb_colors = []
        for hb in hyperblocks:
            color_idx, centroid = find_best_color_match(hb)
            hb_colors.append(color_idx)
            # Add this hyperblock to our history
            all_previous_hbs.append((centroid, hb.dominant_class, color_idx))
        
        # Get statistics
        stats = get_hyperblock_statistics(hyperblocks, current_size)
        stats['iteration'] = iteration
        stats['rows_processed'] = current_size
        stats['percentage_of_total'] = percentage
        
        # Track misclassifications of next batch if not the first iteration
        next_size = min(current_size + rows_per_iteration, total_rows)
        misclassified_str = "N/A"  # Default for first iteration
        
        # Track misclassified points for visualization
        misclassified_indices = []
        
        if iteration > 1 and previous_hyperblocks:
            # Get the points that were just added in this iteration
            if current_size <= len(remaining_indices) + initial_subset_size:
                # Calculate start and end indices for the recently added points
                start_idx = current_size - rows_per_iteration - initial_subset_size
                end_idx = current_size - initial_subset_size
                # Make sure we don't go out of bounds
                start_idx = max(0, start_idx)
                end_idx = min(len(remaining_indices), end_idx)
                
                # Get the indices of the recently added points
                recently_added_indices = remaining_indices[start_idx:end_idx]
                added_points = df.loc[recently_added_indices]
                
                # Count how many were misclassified by previous hyperblocks
                misclassified = 0
                for idx, row in added_points.iterrows():
                    true_class = row[class_col]
                    predicted_class = classify_with_hyperblocks(row[features].values, previous_hyperblocks, features)
                    if predicted_class != true_class:
                        misclassified += 1
                        misclassified_indices.append(idx)  # Store for visualization
                
                # Format as "X/Y" as requested
                added_count = len(added_points)
                misclassified_str = f"{misclassified}/{added_count}"
                stats['misclassifications'] = misclassified_str
                
                if DEBUG:
                    print(f"Of the {added_count} points just added:")
                    print(f"  {misclassified} would be misclassified by the previous hyperblocks")
            else:
                stats['misclassifications'] = "0/0"  # No points added in final iteration
        else:
            stats['misclassifications'] = misclassified_str
        
        stats_records.append(stats)
        
        # Create proper, safe filename for saving the figure with unique identifiers
        filename = f"{run_id}_iter_{iteration:02d}_rows_{current_size}.png"
        save_path = os.path.join(run_dir, filename)
        
        # Modified version of visualize_hyperblocks to use consistent colors
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Set up the axes
        x = list(range(len(features)))
        
        # Get unique classes for color mapping (for data points)
        unique_classes = df[class_col].unique()
        class_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
        class_color_map = dict(zip(unique_classes, class_colors))
        
        # Plot the data points first - still using class colors for the data points
        for idx in current_indices:
            row = df.loc[idx]
            point_values = [row[feat] for feat in features]
            point_class = row[class_col]
            ax.plot(x, point_values, color=class_color_map[point_class], alpha=0.3, linewidth=0.8)
        
        # Create legend entries for hyperblocks only
        hyperblock_legend_entries = []
        
        # Plot each hyperblock using our assigned colors
        for i, hb in enumerate(hyperblocks):
            block_color = color_palette[hb_colors[i]]
            
            # Create legend entry with more detail
            legend_entry = Line2D([0], [0], color=block_color, lw=4, 
                                  label=f"HB {i+1}: Class {hb.dominant_class}, {hb.num_cases} points")
            hyperblock_legend_entries.append(legend_entry)
            
            # Plot the hyperblock bounds as a shaded region
            for j in range(len(features)-1):
                # Get min and max values for current and next feature
                y1_min = hb.min_bounds[features[j]]
                y1_max = hb.max_bounds[features[j]]
                y2_min = hb.min_bounds[features[j+1]]
                y2_max = hb.max_bounds[features[j+1]]
                
                # Create polygon vertices for the shaded region
                xs = [x[j], x[j], x[j+1], x[j+1]]
                ys = [y1_min, y1_max, y2_max, y2_min]
                
                # Plot the polygon
                ax.fill(xs, ys, alpha=0.2, color=block_color, edgecolor=block_color, linewidth=1)
        
        # Add only the hyperblock legend - positioned to the right of the plot
        ax.legend(handles=hyperblock_legend_entries, title="Hyperblocks", 
                 loc='center left', bbox_to_anchor=(1.05, 0.5), borderaxespad=0.)
        
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=45)
        ax.set_ylim(0, 1)
        
        # Calculate coverage
        covered_count, coverage_percentage, _ = calculate_dataset_coverage(df.loc[current_indices], hyperblocks)
        ax.set_title(f"IMHyper: Data with Hyperblock Visualization (Iteration {iteration}: {current_size}/{total_rows} rows)\n"
                    f"Dataset Coverage: {covered_count}/{len(current_indices)} points ({coverage_percentage:.2f}%)")
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Draw misclassified points from the summary table
        if misclassified_indices:
            for idx in misclassified_indices:
                if idx in current_indices:  # Make sure the point is in the current dataset
                    row = df.loc[idx]
                    point_values = [row[feat] for feat in features]
                    ax.plot(x, point_values, color='#FF0000', alpha=1.0, linewidth=3.5, zorder=200, 
                           marker='o', markersize=5, markeredgecolor='black', markeredgewidth=1.0)
            
            # Add misclassified legend entry
            misclassified_legend = Line2D([0], [0], color='#FF0000', linewidth=3.5, marker='o', 
                                         markersize=5, markeredgecolor='black', markeredgewidth=1.0,
                                         label=f'Misclassified ({len(misclassified_indices)})')
            
            # Add to legend
            handles, labels = ax.get_legend().legend_handles, ax.get_legend().get_texts()
            handles.append(misclassified_legend)
            labels = [label.get_text() for label in labels]
            labels.append(misclassified_legend.get_label())
            ax.legend(handles=handles, labels=labels, title="Hyperblocks", 
                     loc='center left', bbox_to_anchor=(1.05, 0.5), borderaxespad=0.)
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.75)  # Still need space for the hyperblock legend
        
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if DEBUG:
                print(f"Figure saved to: {save_path}")
        except Exception as e:
            if DEBUG:
                print(f"Error saving figure: {e}")
            # Try saving with an even simpler filename as fallback
            fallback_path = os.path.join(run_dir, f"{run_id}_iter_{iteration}.png")
            plt.savefig(fallback_path, dpi=300, bbox_inches='tight')
            if DEBUG:
                print(f"Saved to fallback path: {fallback_path}")
        
        plt.close()
        
        # Save the current hyperblocks for next iteration
        previous_hyperblocks = hyperblocks
        
        # Add more data if not at the end
        if next_size > current_size and len(current_indices) < total_rows:
            # Calculate how many indices to add
            indices_to_add = next_size - current_size
            # Calculate start position in remaining_indices
            start_pos = current_size - initial_subset_size
            
            # Check how many indices we have available
            available_indices = len(remaining_indices) - start_pos
            
            if available_indices > 0:
                # Use as many indices as available, up to the desired amount
                actual_indices_to_add = min(indices_to_add, available_indices)
                
                next_batch_indices = remaining_indices[
                    start_pos:
                    start_pos + actual_indices_to_add
                ]
                current_indices.extend(next_batch_indices)
                current_df = df.loc[current_indices].copy()
                current_size = len(current_indices)
                iteration += 1
                
                if DEBUG:
                    print(f"Added {actual_indices_to_add} indices, now at {current_size}/{total_rows} rows")
                    
                # Check if we've used all available indices
                if start_pos + actual_indices_to_add >= len(remaining_indices):
                    if DEBUG:
                        print("Used all available indices, final iteration will be next")
            else:
                # No more indices available but we haven't reached total_rows
                # This should only happen if there are duplicate indices
                if DEBUG:
                    print(f"No more indices available. Used {len(current_indices)}/{total_rows} rows.")
                break
        else:
            break
    
    # Create a summary table
    print("\n\nSummary of Hyperblock Generation Progression:")
    print("-" * 80)
    print(f"{'Iter':<8} {'Rows':<12} {'%Total':<12} {'#HBs':<10} {'Avg Size':<15} {'Misclassified':<15}")
    print("-" * 80)
    
    for stats in stats_records:
        print(f"{stats['iteration']:<8} {stats['rows_processed']:<10} {stats['percentage_of_total']:>6.1f}%      {stats['total_hyperblocks']:<10} "
              f"{stats['avg_size']:>9.2f}        {stats['misclassifications']:<15}")
              
    # Plot progression statistics
    iterations = [stats['iteration'] for stats in stats_records]
    row_counts = [stats['rows_processed'] for stats in stats_records]
    hb_counts = [stats['total_hyperblocks'] for stats in stats_records]
    coverage = [stats['coverage_percentage'] for stats in stats_records]
    misclass = [stats['misclassification_rate'] for stats in stats_records]
    
    # Create a figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Number of hyperblocks vs. iteration
    axs[0, 0].plot(iterations, hb_counts, 'bo-')
    axs[0, 0].set_title('Number of Hyperblocks')
    axs[0, 0].set_xlabel('Iteration')
    axs[0, 0].set_ylabel('Count')
    axs[0, 0].grid(True)
    
    # Plot 2: Coverage percentage vs. iteration
    axs[0, 1].plot(iterations, coverage, 'go-')
    axs[0, 1].set_title('Coverage Percentage')
    axs[0, 1].set_xlabel('Iteration')
    axs[0, 1].set_ylabel('Percentage')
    axs[0, 1].grid(True)
    
    # Plot 3: Misclassification rate vs. iteration
    axs[1, 0].plot(iterations, misclass, 'ro-')
    axs[1, 0].set_title('Misclassification Rate')
    axs[1, 0].set_xlabel('Iteration')
    axs[1, 0].set_ylabel('Percentage')
    axs[1, 0].grid(True)
    
    # Plot 4: Average hyperblock size vs. iteration
    avg_sizes = [stats['avg_size'] for stats in stats_records]
    axs[1, 1].plot(iterations, avg_sizes, 'mo-')
    axs[1, 1].set_title('Average Hyperblock Size')
    axs[1, 1].set_xlabel('Iteration')
    axs[1, 1].set_ylabel('Avg. Points per Hyperblock')
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"{run_id}_statistics_summary.png"), dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
    # Save a summary text file with details about this run
    summary_file = os.path.join(run_dir, f"{run_id}_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Hyperblock Generation Run: {run_id}\n")
        f.write(f"Date and Time: {timestamp}\n")
        f.write(f"Total Dataset Size: {total_rows}\n")
        f.write(f"Initial Subset Size: {initial_subset_size}\n")
        f.write(f"Increment Size: {rows_per_iteration}\n")
        f.write(f"Total Iterations: {iterations}\n\n")
        
        f.write("Summary of Hyperblock Generation Progression:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Iter':<8} {'Rows':<12} {'%Total':<12} {'#HBs':<10} {'Avg Size':<15} {'Misclassified':<15}\n")
        f.write("-" * 80 + "\n")
        
        for stats in stats_records:
            f.write(f"{stats['iteration']:<8} {stats['rows_processed']:<10} {stats['percentage_of_total']:>6.1f}%      {stats['total_hyperblocks']:<10} "
                  f"{stats['avg_size']:>9.2f}        {stats['misclassifications']:<15}\n")
    
    if DEBUG:
        print(f"\nAll visualizations and summary saved to directory: {run_dir}")
    return run_dir  # Return the directory for this run

def decremental_hyperblock_generation(df, features, class_col, rows_per_iteration):
    """
    Decrementally generate hyperblocks by gradually removing data.
    
    Args:
        df: DataFrame containing the data
        features: List of feature names
        class_col: Name of the class column
        rows_per_iteration: Number of rows to remove in each iteration
    """
    # Create output directory for images
    output_dir = 'hyperblock_progression'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a unique run identifier
    import time
    import random
    import string
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    random_suffix = ''.join(random.choices(string.ascii_lowercase, k=4))
    run_id = f"decremental_{timestamp}_{random_suffix}"
    
    if DEBUG:
        print(f"\n{'='*80}")
        print(f"Starting Decremental Hyperblock Generation")
        print(f"Run ID: {run_id}")
        print(f"{'='*80}")
    
    # Create a subdirectory for this specific run
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Get total dataset size and number of unique classes
    total_rows = len(df)
    num_classes = len(df[class_col].unique())
    min_samples_needed = num_classes + 1  # Need at least one more sample than classes
    
    if DEBUG:
        print(f"Number of unique classes: {num_classes}")
        print(f"Minimum samples needed: {min_samples_needed}")
    
    # Prepare to track statistics
    stats_records = []
    
    # Start with the full dataset
    current_size = total_rows
    current_indices = list(df.index)
    
    # Calculate how many iterations we'll need
    iterations = (total_rows - min_samples_needed) // rows_per_iteration + 1  # Ceiling division
    
    if DEBUG:
        print(f"Starting with {current_size} rows (100% of the dataset)")
        print(f"Will remove {rows_per_iteration} rows at each step")
        print(f"Total {iterations} iterations needed (will stop at {min_samples_needed} samples)")
    
    # Save the sequence of indices for reproducibility
    indices_file = os.path.join(run_dir, f"{run_id}_indices_sequence.txt")
    with open(indices_file, 'w') as f:
        f.write("Initial indices (full dataset):\n")
        f.write(",".join(map(str, current_indices)) + "\n\n")
    
    # Create a fixed color palette - using a distinctive color set
    color_palette = [
        plt.cm.tab20(0),   # blue
        plt.cm.tab20(2),   # green
        plt.cm.tab20(4),   # red
        plt.cm.tab20(6),   # purple
        plt.cm.tab20(8),   # orange
        plt.cm.tab20(10),  # yellow
        plt.cm.tab20(12),  # teal
        plt.cm.tab20(14),  # pink
        plt.cm.tab20(16),  # light blue
        plt.cm.tab20(18),  # light green
        plt.cm.tab20(1),   # dark blue
        plt.cm.tab20(3),   # dark green
        plt.cm.tab20(5),   # dark red
        plt.cm.tab20(7),   # dark purple
        plt.cm.tab20(9),   # dark orange
        plt.cm.tab20(11),  # dark yellow
        plt.cm.tab20(13),  # dark teal
        plt.cm.tab20(15),  # dark pink
        plt.cm.tab20(17),  # medium blue
        plt.cm.tab20(19),  # medium green
    ]
    
    # Store ALL hyperblock centroids from ALL previous iterations with their colors
    all_previous_hbs = []
    
    # Function to calculate hyperblock centroid
    def calculate_centroid(hb):
        return [
            (hb.min_bounds[feat] + hb.max_bounds[feat]) / 2
            for feat in features
        ]
    
    # Function to compute similarity between two hyperblocks using centroids
    def compute_similarity(centroid1, centroid2):
        return 1.0 / (1.0 + sum((a - b) ** 2 for a, b in zip(centroid1, centroid2)) ** 0.5)
    
    # Function to find best color match based on centroid similarity
    def find_best_color_match(hb):
        centroid = calculate_centroid(hb)
        hb_class = hb.dominant_class
        
        if not all_previous_hbs:
            # First hyperblock ever, assign first color
            return 0, centroid
        
        # Find previously seen hyperblocks with same class
        same_class_hbs = [(prev_centroid, color_idx) 
                          for prev_centroid, prev_class, color_idx in all_previous_hbs 
                          if prev_class == hb_class]
        
        if not same_class_hbs:
            # First hyperblock of this class, find unused color
            used_colors = set(color_idx for _, _, color_idx in all_previous_hbs)
            for i in range(len(color_palette)):
                if i not in used_colors:
                    return i, centroid
            # If all colors used, use the least used one
            color_counts = {}
            for _, _, c in all_previous_hbs:
                color_counts[c] = color_counts.get(c, 0) + 1
            least_used = min(color_counts.items(), key=lambda x: x[1])[0]
            return least_used, centroid
        
        # Calculate similarity to all previous hyperblocks of same class
        similarities = [(compute_similarity(centroid, prev_centroid), color_idx) 
                        for prev_centroid, color_idx in same_class_hbs]
        
        # Return the color of the most similar hyperblock
        best_match = max(similarities, key=lambda x: x[0])
        # Only use this match if similarity is above threshold (0.7)
        if best_match[0] > 0.7:
            return best_match[1], centroid
        
        # Otherwise use a new color
        used_colors = set(color_idx for _, _, color_idx in all_previous_hbs)
        for i in range(len(color_palette)):
            if i not in used_colors:
                return i, centroid
        
        # If all colors used, use the least used one
        color_counts = {}
        for _, _, c in all_previous_hbs:
            color_counts[c] = color_counts.get(c, 0) + 1
        least_used = min(color_counts.items(), key=lambda x: x[1])[0]
        return least_used, centroid
    
    iteration = 1
    previous_hyperblocks = None
    
    # Save statistics for initial iteration before the while loop
    percentage = current_size / total_rows * 100
    if DEBUG:
        print(f"\n{'='*80}")
        print(f"Initial Iteration {iteration}/{iterations}: Processing {current_size} rows ({percentage:.1f}% of the dataset)")
        print(f"{'='*80}")
    
    # Generate hyperblocks for initial dataset
    hyperblocks = imhyper_algorithm(df.loc[current_indices], class_col, purity_threshold=0.9999, impurity_threshold=0.0001)
    
    # Sort hyperblocks by size (larger ones first) for more stable color assignment
    hyperblocks.sort(key=lambda hb: hb.num_cases, reverse=True)
    
    # Assign colors based on similarity to previous hyperblocks
    hb_colors = []
    for hb in hyperblocks:
        color_idx, centroid = find_best_color_match(hb)
        hb_colors.append(color_idx)
        # Add this hyperblock to our history
        all_previous_hbs.append((centroid, hb.dominant_class, color_idx))
    
    # Get statistics
    stats = get_hyperblock_statistics(hyperblocks, current_size)
    stats['iteration'] = iteration
    stats['rows_processed'] = current_size
    stats['percentage_of_total'] = percentage
    stats['misclassifications'] = "N/A"  # Default for first iteration
    stats_records.append(stats)
    
    # Create proper, safe filename for saving the figure with unique identifiers
    filename = f"{run_id}_iter_{iteration:02d}_rows_{current_size}.png"
    save_path = os.path.join(run_dir, filename)
    
    # Visualization for initial iteration
    fig, ax = plt.subplots(figsize=(14, 8))
    x = list(range(len(features)))
    
    # Get unique classes for color mapping (for data points)
    unique_classes = df[class_col].unique()
    class_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
    class_color_map = dict(zip(unique_classes, class_colors))
    
    # Plot the data points first - still using class colors for the data points
    for idx in current_indices:
        row = df.loc[idx]
        point_values = [row[feat] for feat in features]
        point_class = row[class_col]
        ax.plot(x, point_values, color=class_color_map[point_class], alpha=0.3, linewidth=0.8)
    
    # Create legend entries for hyperblocks only
    hyperblock_legend_entries = []
    
    # Plot each hyperblock using our assigned colors
    for i, hb in enumerate(hyperblocks):
        block_color = color_palette[hb_colors[i]]
        
        # Create legend entry with more detail
        legend_entry = Line2D([0], [0], color=block_color, lw=4, 
                              label=f"HB {i+1}: Class {hb.dominant_class}, {hb.num_cases} points")
        hyperblock_legend_entries.append(legend_entry)
        
        # Plot the hyperblock bounds as a shaded region
        for j in range(len(features)-1):
            # Get min and max values for current and next feature
            y1_min = hb.min_bounds[features[j]]
            y1_max = hb.max_bounds[features[j]]
            y2_min = hb.min_bounds[features[j+1]]
            y2_max = hb.max_bounds[features[j+1]]
            
            # Create polygon vertices for the shaded region
            xs = [x[j], x[j], x[j+1], x[j+1]]
            ys = [y1_min, y1_max, y2_max, y2_min]
            
            # Plot the polygon
            ax.fill(xs, ys, alpha=0.2, color=block_color, edgecolor=block_color, linewidth=1)
    
    # Add only the hyperblock legend - positioned to the right of the plot
    ax.legend(handles=hyperblock_legend_entries, title="Hyperblocks", 
             loc='center left', bbox_to_anchor=(1.05, 0.5), borderaxespad=0.)
    
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45)
    ax.set_ylim(0, 1)
    
    # Calculate coverage
    covered_count, coverage_percentage, _ = calculate_dataset_coverage(df.loc[current_indices], hyperblocks)
    ax.set_title(f"IMHyper: Data with Hyperblock Visualization (Iteration {iteration}: {current_size}/{total_rows} rows)\n"
                f"Dataset Coverage: {covered_count}/{len(current_indices)} points ({coverage_percentage:.2f}%)")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # Still need space for the hyperblock legend
    
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if DEBUG:
            print(f"Figure saved to: {save_path}")
    except Exception as e:
        if DEBUG:
            print(f"Error saving figure: {e}")
        # Try saving with an even simpler filename as fallback
        fallback_path = os.path.join(run_dir, f"{run_id}_iter_{iteration}.png")
        plt.savefig(fallback_path, dpi=300, bbox_inches='tight')
        if DEBUG:
            print(f"Saved to fallback path: {fallback_path}")
    
    plt.close()
    
    # Save the current hyperblocks for next iteration
    previous_hyperblocks = hyperblocks
    iteration += 1
    
    # Continue with additional iterations
    while current_size > min_samples_needed:  # Changed condition to ensure we have enough samples
        percentage = current_size / total_rows * 100
        if DEBUG:
            print(f"\n{'='*80}")
            print(f"Iteration {iteration}/{iterations}: Processing {current_size} rows ({percentage:.1f}% of the dataset)")
            print(f"{'='*80}")
        
        # Generate hyperblocks using the current subset
        hyperblocks = imhyper_algorithm(df.loc[current_indices], class_col, purity_threshold=0.9999, impurity_threshold=0.0001)
        
        # Sort hyperblocks by size (larger ones first) for more stable color assignment
        hyperblocks.sort(key=lambda hb: hb.num_cases, reverse=True)
        
        # Assign colors based on similarity to previous hyperblocks
        hb_colors = []
        for hb in hyperblocks:
            color_idx, centroid = find_best_color_match(hb)
            hb_colors.append(color_idx)
            # Add this hyperblock to our history
            all_previous_hbs.append((centroid, hb.dominant_class, color_idx))
        
        # Get statistics
        stats = get_hyperblock_statistics(hyperblocks, current_size)
        stats['iteration'] = iteration
        stats['rows_processed'] = current_size
        stats['percentage_of_total'] = percentage
        
        # Track misclassifications of removed batch if not the first iteration
        next_size = current_size - rows_per_iteration
        misclassified_str = "N/A"  # Default for first iteration
        
        # Track misclassified points for visualization
        misclassified_indices = []
        
        if iteration > 1 and previous_hyperblocks:
            # Get the points that will be removed in the next iteration
            removed_indices = current_indices[-rows_per_iteration:]
            removed_points = df.loc[removed_indices]
            
            # Count how many would be misclassified by current hyperblocks
            misclassified = 0
            for idx, row in removed_points.iterrows():
                true_class = row[class_col]
                predicted_class = classify_with_hyperblocks(row[features].values, hyperblocks, features)
                if predicted_class != true_class:
                    misclassified += 1
                    misclassified_indices.append(idx)  # Store for visualization
            
            # Format as "X/Y" as requested
            removed_count = len(removed_points)
            misclassified_str = f"{misclassified}/{removed_count}"
            stats['misclassifications'] = misclassified_str
            
            if DEBUG:
                print(f"Of the {removed_count} points to be removed:")
                print(f"  {misclassified} would be misclassified by the current hyperblocks")
        else:
            stats['misclassifications'] = misclassified_str
        
        stats_records.append(stats)
        
        # Create proper, safe filename for saving the figure with unique identifiers
        filename = f"{run_id}_iter_{iteration:02d}_rows_{current_size}.png"
        save_path = os.path.join(run_dir, filename)
        
        # Modified version of visualize_hyperblocks to use consistent colors
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Set up the axes
        x = list(range(len(features)))
        
        # Get unique classes for color mapping (for data points)
        unique_classes = df[class_col].unique()
        class_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
        class_color_map = dict(zip(unique_classes, class_colors))
        
        # Plot the data points first - still using class colors for the data points
        for idx in current_indices:
            row = df.loc[idx]
            point_values = [row[feat] for feat in features]
            point_class = row[class_col]
            ax.plot(x, point_values, color=class_color_map[point_class], alpha=0.3, linewidth=0.8)
        
        # Create legend entries for hyperblocks only
        hyperblock_legend_entries = []
        
        # Plot each hyperblock using our assigned colors
        for i, hb in enumerate(hyperblocks):
            block_color = color_palette[hb_colors[i]]
            
            # Create legend entry with more detail
            legend_entry = Line2D([0], [0], color=block_color, lw=4, 
                                  label=f"HB {i+1}: Class {hb.dominant_class}, {hb.num_cases} points")
            hyperblock_legend_entries.append(legend_entry)
            
            # Plot the hyperblock bounds as a shaded region
            for j in range(len(features)-1):
                # Get min and max values for current and next feature
                y1_min = hb.min_bounds[features[j]]
                y1_max = hb.max_bounds[features[j]]
                y2_min = hb.min_bounds[features[j+1]]
                y2_max = hb.max_bounds[features[j+1]]
                
                # Create polygon vertices for the shaded region
                xs = [x[j], x[j], x[j+1], x[j+1]]
                ys = [y1_min, y1_max, y2_max, y2_min]
                
                # Plot the polygon
                ax.fill(xs, ys, alpha=0.2, color=block_color, edgecolor=block_color, linewidth=1)
        
        # Add only the hyperblock legend - positioned to the right of the plot
        ax.legend(handles=hyperblock_legend_entries, title="Hyperblocks", 
                 loc='center left', bbox_to_anchor=(1.05, 0.5), borderaxespad=0.)
        
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=45)
        ax.set_ylim(0, 1)
        
        # Calculate coverage
        covered_count, coverage_percentage, _ = calculate_dataset_coverage(df.loc[current_indices], hyperblocks)
        ax.set_title(f"IMHyper: Data with Hyperblock Visualization (Iteration {iteration}: {current_size}/{total_rows} rows)\n"
                    f"Dataset Coverage: {covered_count}/{len(current_indices)} points ({coverage_percentage:.2f}%)")
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Draw misclassified points from the summary table
        if misclassified_indices:
            for idx in misclassified_indices:
                if idx in current_indices:  # Make sure the point is in the current dataset
                    row = df.loc[idx]
                    point_values = [row[feat] for feat in features]
                    ax.plot(x, point_values, color='#FF0000', alpha=1.0, linewidth=3.5, zorder=200, 
                           marker='o', markersize=5, markeredgecolor='black', markeredgewidth=1.0)
            
            # Add misclassified legend entry
            misclassified_legend = Line2D([0], [0], color='#FF0000', linewidth=3.5, marker='o', 
                                         markersize=5, markeredgecolor='black', markeredgewidth=1.0,
                                         label=f'Misclassified ({len(misclassified_indices)})')
            
            # Add to legend
            handles, labels = ax.get_legend().legend_handles, ax.get_legend().get_texts()
            handles.append(misclassified_legend)
            labels = [label.get_text() for label in labels]
            labels.append(misclassified_legend.get_label())
            ax.legend(handles=handles, labels=labels, title="Hyperblocks", 
                     loc='center left', bbox_to_anchor=(1.05, 0.5), borderaxespad=0.)
        
        plt.tight_layout()
        plt.subplots_adjust(right=0.75)  # Still need space for the hyperblock legend
        
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if DEBUG:
                print(f"Figure saved to: {save_path}")
        except Exception as e:
            if DEBUG:
                print(f"Error saving figure: {e}")
            # Try saving with an even simpler filename as fallback
            fallback_path = os.path.join(run_dir, f"{run_id}_iter_{iteration}.png")
            plt.savefig(fallback_path, dpi=300, bbox_inches='tight')
            if DEBUG:
                print(f"Saved to fallback path: {fallback_path}")
        
        plt.close()
        
        # Save the current hyperblocks for next iteration
        previous_hyperblocks = hyperblocks
        
        # Remove data for next iteration
        if next_size > min_samples_needed:  # Only remove if we'll still have enough samples
            # Calculate how many rows we can actually remove
            max_to_remove = current_size - min_samples_needed
            actual_rows_to_remove = min(rows_per_iteration, max_to_remove)
            
            if actual_rows_to_remove > 0:
                # Randomly select indices to remove
                indices_to_remove = np.random.choice(current_indices, size=actual_rows_to_remove, replace=False)
                # Remove the selected indices
                current_indices = [idx for idx in current_indices if idx not in indices_to_remove]
                current_size = len(current_indices)
                iteration += 1
                
                if DEBUG:
                    print(f"Removed {actual_rows_to_remove} rows, now at {current_size}/{total_rows} rows")
                    if current_size <= min_samples_needed:
                        print(f"Reached minimum required samples ({min_samples_needed}), this was the final iteration")
            else:
                if DEBUG:
                    print(f"Cannot remove more rows without going below minimum required ({min_samples_needed})")
                break
        else:
            if DEBUG:
                print(f"\nStopping at {current_size} samples (minimum required: {min_samples_needed})")
            break
    
    # Create a summary table - this should always print
    print("\n\nSummary of Decremental Hyperblock Generation Progression:")
    print("-" * 80)
    print(f"{'Iter':<8} {'Rows':<12} {'%Total':<12} {'#HBs':<10} {'Avg Size':<15} {'Misclassified':<15}")
    print("-" * 80)
    
    for stats in stats_records:
        print(f"{stats['iteration']:<8} {stats['rows_processed']:<10} {stats['percentage_of_total']:>6.1f}%      {stats['total_hyperblocks']:<10} "
              f"{stats['avg_size']:>9.2f}        {stats['misclassifications']:<15}")
              
    # Plot progression statistics
    iterations = [stats['iteration'] for stats in stats_records]
    row_counts = [stats['rows_processed'] for stats in stats_records]
    hb_counts = [stats['total_hyperblocks'] for stats in stats_records]
    coverage = [stats['coverage_percentage'] for stats in stats_records]
    misclass = [stats['misclassification_rate'] for stats in stats_records]
    
    # Create a figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Number of hyperblocks vs. iteration
    axs[0, 0].plot(iterations, hb_counts, 'bo-')
    axs[0, 0].set_title('Number of Hyperblocks')
    axs[0, 0].set_xlabel('Iteration')
    axs[0, 0].set_ylabel('Count')
    axs[0, 0].grid(True)
    
    # Plot 2: Coverage percentage vs. iteration
    axs[0, 1].plot(iterations, coverage, 'go-')
    axs[0, 1].set_title('Coverage Percentage')
    axs[0, 1].set_xlabel('Iteration')
    axs[0, 1].set_ylabel('Percentage')
    axs[0, 1].grid(True)
    
    # Plot 3: Misclassification rate vs. iteration
    axs[1, 0].plot(iterations, misclass, 'ro-')
    axs[1, 0].set_title('Misclassification Rate')
    axs[1, 0].set_xlabel('Iteration')
    axs[1, 0].set_ylabel('Percentage')
    axs[1, 0].grid(True)
    
    # Plot 4: Average hyperblock size vs. iteration
    avg_sizes = [stats['avg_size'] for stats in stats_records]
    axs[1, 1].plot(iterations, avg_sizes, 'mo-')
    axs[1, 1].set_title('Average Hyperblock Size')
    axs[1, 1].set_xlabel('Iteration')
    axs[1, 1].set_ylabel('Avg. Points per Hyperblock')
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"{run_id}_statistics_summary.png"), dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
    # Save a summary text file with details about this run
    summary_file = os.path.join(run_dir, f"{run_id}_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Decremental Hyperblock Generation Run: {run_id}\n")
        f.write(f"Date and Time: {timestamp}\n")
        f.write(f"Total Dataset Size: {total_rows}\n")
        f.write(f"Number of Classes: {num_classes}\n")
        f.write(f"Minimum Samples Required: {min_samples_needed}\n")
        f.write(f"Decrement Size: {rows_per_iteration}\n")
        f.write(f"Total Iterations: {iterations}\n\n")
        
        f.write("Summary of Hyperblock Generation Progression:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Iter':<8} {'Rows':<12} {'%Total':<12} {'#HBs':<10} {'Avg Size':<15} {'Misclassified':<15}\n")
        f.write("-" * 80 + "\n")
        
        for stats in stats_records:
            f.write(f"{stats['iteration']:<8} {stats['rows_processed']:<10} {stats['percentage_of_total']:>6.1f}%      {stats['total_hyperblocks']:<10} "
                  f"{stats['avg_size']:>9.2f}        {stats['misclassifications']:<15}\n")
    
    if DEBUG:
        print(f"\nAll visualizations and summary saved to directory: {run_dir}")
    return run_dir  # Return the directory for this run

def main():
    # Load and normalize data
    df, features, class_col = load_and_normalize_data()
    
    # Ask user which experiment to run
    print("\nSelect experiment to run:")
    print("1. Single run of incremental and decremental hyperblock generation")
    print("2. Multiple seed experiment (misclassification distribution analysis)")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        # Get the number of rows to add/remove per iteration from the user
        while True:
            try:
                rows_per_iteration = int(input("\nEnter the number of rows to add/remove per iteration: "))
                if rows_per_iteration <= 0:
                    print("Please enter a positive number.")
                    continue
                if rows_per_iteration >= len(df):
                    print("Please enter a number smaller than the total dataset size.")
                    continue
                break
            except ValueError:
                print("Please enter a valid number.")
        
        # Run the incremental analysis
        if DEBUG:
            print("\nRunning Incremental Hyperblock Generation...")
        incremental_dir = incremental_hyperblock_generation(df, features, class_col, rows_per_iteration)
        if DEBUG:
            print(f"Incremental analysis results saved to: {incremental_dir}")
        
        # Run the decremental analysis
        if DEBUG:
            print("\nRunning Decremental Hyperblock Generation...")
        decremental_dir = decremental_hyperblock_generation(df, features, class_col, rows_per_iteration)
        if DEBUG:
            print(f"Decremental analysis results saved to: {decremental_dir}")
    
    elif choice == "2":
        # Get parameters for the multiple seed experiment
        while True:
            try:
                rows_per_iteration = int(input("\nEnter the number of rows to add/remove per iteration: "))
                if rows_per_iteration <= 0:
                    print("Please enter a positive number.")
                    continue
                if rows_per_iteration >= len(df):
                    print("Please enter a number smaller than the total dataset size.")
                    continue
                break
            except ValueError:
                print("Please enter a valid number.")
        
        while True:
            try:
                num_seeds = int(input("\nEnter the number of random seeds to use (1-1000, recommended 100): "))
                if num_seeds <= 0:
                    print("Please enter a positive number.")
                    continue
                if num_seeds > 1000:
                    print("Warning: Using more than 1000 seeds may take a very long time.")
                    confirm = input("Continue with more than 1000 seeds? (y/n): ")
                    if confirm.lower() != 'y':
                        continue
                break
            except ValueError:
                print("Please enter a valid number.")
        
        # Run the multiple seed experiment
        print(f"\nRunning multiple seed experiment with {num_seeds} seeds...")
        output_dir = run_multiple_seed_experiment(df, features, class_col, rows_per_iteration, num_seeds)
        print(f"Multiple seed experiment results saved to: {output_dir}")
    
    else:
        print("Invalid choice. Exiting.")

def create_hyperblock_collage(df, features, class_col):
    """
    Creates a collage with one visualization per hyperblock from the final iteration.
    """
    # Run the algorithm once more on the full dataset to get final hyperblocks
    if DEBUG:
        print("\nGenerating final hyperblocks for collage...")
    hyperblocks = imhyper_algorithm(df, class_col, purity_threshold=0.9999, impurity_threshold=0.0001)
    
    # Create output directory
    output_dir = 'hyperblock_collage'
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine grid size
    n_blocks = len(hyperblocks)
    cols = min(3, n_blocks)
    rows = (n_blocks + cols - 1) // cols
    
    # Create a figure for the collage
    plt.figure(figsize=(6*cols, 5*rows))
    
    # Get unique classes for color mapping
    unique_classes = df[class_col].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
    class_color_map = dict(zip(unique_classes, colors))
    
    # Create and add individual hyperblock plots
    for i, hb in enumerate(hyperblocks):
        # Create subplot
        plt.subplot(rows, cols, i+1)
        
        # Plot all data points in gray
        for _, row in df.iterrows():
            values = [row[feat] for feat in features]
            plt.plot(features, values, color='lightgray', alpha=0.2, linewidth=0.5)
        
        # Highlight points in this hyperblock
        misclassified_count = 0
        for point in hb.points:
            # Get feature values (all except last which is class)
            values = point[:-1]
            # Get class
            point_class = point[-1]
            
            # Determine if this point is misclassified
            is_misclassified = point_class != hb.dominant_class
            
            # Plot with original class colors for correctly classified points, prominent bright red for misclassified
            if is_misclassified:
                plt.plot(features, values, color='#FF0000', alpha=1.0, linewidth=3.0, zorder=3, 
                        marker='o', markersize=4, markeredgecolor='black', markeredgewidth=0.8)
                misclassified_count += 1
            else:
                plt.plot(features, values, color=class_color_map[point_class], alpha=0.7, linewidth=1, zorder=2)
        
        # Set title with hyperblock info
        accuracy = (hb.num_cases - hb.num_misclassified) / hb.num_cases * 100
        plt.title(f"HB #{i+1}: Class {hb.dominant_class}\n{hb.num_cases} points, {accuracy:.1f}% accuracy")
        
        # Create custom legend for classes
        class_legend_elements = []
        classes_in_hb = set(point[-1] for point in hb.points)
        for cls in classes_in_hb:
            class_legend_elements.append(Line2D([0], [0], color=class_color_map[cls], lw=1, 
                                               label=f"Class {cls}"))
        
        # Add misclassified entry to legend if any exist
        if misclassified_count > 0:
            class_legend_elements.append(Line2D([0], [0], color='#FF0000', lw=3.0, marker='o', markersize=4,
                                               markeredgecolor='black', markeredgewidth=0.8,
                                               label=f'Misclassified ({misclassified_count})'))
            
        plt.legend(handles=class_legend_elements, loc='best')
        
        # Adjust formatting
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    # Save the collage
    collage_path = os.path.join(output_dir, "hyperblocks_collage.png")
    plt.savefig(collage_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    if DEBUG:
        print(f"Hyperblock collage saved to: {collage_path}")

def mark_misclassified_points_in_hyperblocks(df, features, class_col, hyperblocks, ax, x):
    """
    Helper function to mark misclassified points in hyperblocks with prominent red polylines.
    
    Args:
        df: DataFrame containing the data
        features: List of feature names
        class_col: Name of the class column
        hyperblocks: List of Hyperblock objects
        ax: Matplotlib axis to plot on
        x: List of x coordinates for features
    """
    # Count misclassified points 
    total_misclassified = 0
    
    # Plot misclassified points in bright red with thicker polylines for high visibility
    for hb in hyperblocks:
        for point in hb.points:
            point_values = point[:-1]  # All except class
            point_class = point[-1]
            
            # Check if point is misclassified
            if point_class != hb.dominant_class:
                # Draw misclassified points with MUCH more prominent bright red polylines
                ax.plot(x, point_values, color='#FF0000', alpha=1.0, linewidth=3.0, zorder=100, 
                       marker='o', markersize=4, markeredgecolor='black', markeredgewidth=0.8)
                total_misclassified += 1
                
    # Add misclassified to legend if any exist
    if total_misclassified > 0:
        return Line2D([0], [0], color='#FF0000', linewidth=3.0, marker='o', markersize=4,
                     markeredgecolor='black', markeredgewidth=0.8,
                     label=f'Misclassified ({total_misclassified})')
    return None

def run_multiple_seed_experiment(df, features, class_col, rows_per_iteration, num_seeds=1000):
    """
    Run the incremental and decremental hyperblock generation experiments with multiple random seeds
    and create a visualization of misclassification distributions.
    
    Args:
        df: DataFrame containing the data
        features: List of feature names
        class_col: Name of the class column
        rows_per_iteration: Number of rows to add/remove per iteration
        num_seeds: Number of different random seeds to use (default: 1000)
    """
    # Create output directory for results
    output_dir = 'multiple_seed_experiment'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for this experiment run
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"multiseed_{timestamp}"
    
    # Create specific output directory for this run
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    if DEBUG:
        print(f"\n{'='*80}")
        print(f"Starting Multiple Seed Experiment with {num_seeds} seeds")
        print(f"Output directory: {run_dir}")
        print(f"{'='*80}")
    
    # Create data structures to store results
    incremental_results = []
    decremental_results = []
    
    # Calculate total dataset size and expected number of iterations
    total_rows = len(df)
    initial_subset_size = total_rows // 3
    remaining_rows = total_rows - initial_subset_size
    incremental_iterations = (remaining_rows + rows_per_iteration - 1) // rows_per_iteration + 1
    
    # For decremental, need to ensure we stop at minimum required samples
    num_classes = len(df[class_col].unique())
    min_samples_needed = num_classes + 1
    max_removable = total_rows - min_samples_needed
    decremental_iterations = (max_removable + rows_per_iteration - 1) // rows_per_iteration + 1
    
    if DEBUG:
        print(f"Expected incremental iterations: {incremental_iterations}")
        print(f"Expected decremental iterations: {decremental_iterations}")
    
    # Run experiments with multiple seeds
    for seed_idx in range(num_seeds):
        seed = seed_idx + 1  # Use seed values 1-1000 instead of 0-999
        
        if DEBUG and seed_idx % 10 == 0:
            print(f"Processing seed {seed}/{num_seeds}...")
        elif seed_idx % 100 == 0:
            print(f"Processing seed {seed}/{num_seeds}...")
        
        # Set the random seed
        np.random.seed(seed)
        random.seed(seed)
        
        # Capture the original random state to restore between experiments
        random_state = np.random.get_state()
        
        # Run incremental experiment and collect misclassification data
        np.random.set_state(random_state)  # Reset random state for consistency
        
        # Create a modified version of incremental_hyperblock_generation that returns misclassification data
        inc_misclass_data = []
        
        # Get fresh shuffled indices
        shuffled_indices = np.random.permutation(df.index)
        initial_indices = shuffled_indices[:initial_subset_size]
        remaining_indices = shuffled_indices[initial_subset_size:]
        
        # Start with initial subset
        current_size = initial_subset_size
        current_indices = list(initial_indices)
        current_df = df.loc[current_indices].copy()
        
        # Save initial hyperblocks
        hyperblocks = imhyper_algorithm(current_df, class_col, purity_threshold=0.9999, impurity_threshold=0.0001)
        previous_hyperblocks = hyperblocks
        
        # Track initial iteration (N/A misclassifications for first iteration)
        inc_misclass_data.append({
            'seed': seed,
            'iteration': 1,
            'rows': current_size,
            'misclassified': 0,
            'total_points': 0,
            'misclass_rate': 0.0
        })
        
        # Process iterations
        iteration = 2  # Start from the second iteration
        while current_size < total_rows:
            next_size = min(current_size + rows_per_iteration, total_rows)
            
            # Calculate indices to add
            start_pos = current_size - initial_subset_size
            available_indices = len(remaining_indices) - start_pos
            
            if available_indices <= 0:
                break
                
            # Use as many indices as available, up to the desired amount
            actual_indices_to_add = min(rows_per_iteration, available_indices)
            
            next_batch_indices = remaining_indices[
                start_pos:
                start_pos + actual_indices_to_add
            ]
            
            # Count misclassifications in the batch
            misclassified = 0
            for idx in next_batch_indices:
                row = df.loc[idx]
                true_class = row[class_col]
                predicted_class = classify_with_hyperblocks(row[features].values, previous_hyperblocks, features)
                if predicted_class != true_class:
                    misclassified += 1
            
            # Save misclassification data
            misclass_rate = misclassified / len(next_batch_indices) if len(next_batch_indices) > 0 else 0
            inc_misclass_data.append({
                'seed': seed,
                'iteration': iteration,
                'rows': current_size + actual_indices_to_add,
                'misclassified': misclassified,
                'total_points': len(next_batch_indices),
                'misclass_rate': misclass_rate
            })
            
            # Update for next iteration
            current_indices.extend(next_batch_indices)
            current_df = df.loc[current_indices].copy()
            current_size = len(current_indices)
            
            # Generate new hyperblocks
            hyperblocks = imhyper_algorithm(current_df, class_col, purity_threshold=0.9999, impurity_threshold=0.0001)
            previous_hyperblocks = hyperblocks
            
            iteration += 1
        
        # Add to overall results
        incremental_results.extend(inc_misclass_data)
        
        # Run decremental experiment and collect misclassification data
        np.random.set_state(random_state)  # Reset random state for consistency
        
        # Create a modified version of decremental_hyperblock_generation that returns misclassification data
        dec_misclass_data = []
        
        # Start with full dataset
        current_size = total_rows
        current_indices = list(df.index)
        
        # Save initial hyperblocks for decremental
        hyperblocks = imhyper_algorithm(df.loc[current_indices], class_col, purity_threshold=0.9999, impurity_threshold=0.0001)
        previous_hyperblocks = hyperblocks
        
        # Track initial iteration (N/A misclassifications for first iteration)
        dec_misclass_data.append({
            'seed': seed,
            'iteration': 1,
            'rows': current_size,
            'misclassified': 0,
            'total_points': 0,
            'misclass_rate': 0.0
        })
        
        # Process iterations
        iteration = 2  # Start from the second iteration
        while current_size > min_samples_needed:
            # Calculate how many rows we can actually remove
            max_to_remove = current_size - min_samples_needed
            actual_rows_to_remove = min(rows_per_iteration, max_to_remove)
            
            if actual_rows_to_remove <= 0:
                break
                
            # Randomly select indices to remove
            indices_to_remove = np.random.choice(current_indices, size=actual_rows_to_remove, replace=False)
            
            # Count how many would be misclassified by current hyperblocks
            misclassified = 0
            for idx in indices_to_remove:
                row = df.loc[idx]
                true_class = row[class_col]
                predicted_class = classify_with_hyperblocks(row[features].values, previous_hyperblocks, features)
                if predicted_class != true_class:
                    misclassified += 1
            
            # Save misclassification data
            misclass_rate = misclassified / len(indices_to_remove) if len(indices_to_remove) > 0 else 0
            dec_misclass_data.append({
                'seed': seed,
                'iteration': iteration,
                'rows': current_size - actual_rows_to_remove,
                'misclassified': misclassified,
                'total_points': len(indices_to_remove),
                'misclass_rate': misclass_rate
            })
            
            # Update for next iteration
            current_indices = [idx for idx in current_indices if idx not in indices_to_remove]
            current_size = len(current_indices)
            
            # Generate new hyperblocks
            hyperblocks = imhyper_algorithm(df.loc[current_indices], class_col, purity_threshold=0.9999, impurity_threshold=0.0001)
            previous_hyperblocks = hyperblocks
            
            iteration += 1
        
        # Add to overall results
        decremental_results.extend(dec_misclass_data)
    
    # Convert results to DataFrames
    inc_df = pd.DataFrame(incremental_results)
    dec_df = pd.DataFrame(decremental_results)
    
    # Save raw results
    inc_df.to_csv(os.path.join(run_dir, f"{run_id}_incremental_results.csv"), index=False)
    dec_df.to_csv(os.path.join(run_dir, f"{run_id}_decremental_results.csv"), index=False)
    
    # Create summary statistics by iteration
    inc_summary = inc_df.groupby('iteration').agg({
        'misclass_rate': ['mean', 'std', 'min', 'max', 'count'],
        'misclassified': ['sum', 'mean'],
        'total_points': ['sum', 'mean']
    }).reset_index()
    
    dec_summary = dec_df.groupby('iteration').agg({
        'misclass_rate': ['mean', 'std', 'min', 'max', 'count'],
        'misclassified': ['sum', 'mean'],
        'total_points': ['sum', 'mean']
    }).reset_index()
    
    # Save summary statistics
    inc_summary.to_csv(os.path.join(run_dir, f"{run_id}_incremental_summary.csv"))
    dec_summary.to_csv(os.path.join(run_dir, f"{run_id}_decremental_summary.csv"))
    
    # Create visualization of misclassification distributions
    plot_misclassification_distribution(inc_df, dec_df, run_dir, run_id)
    
    if DEBUG:
        print(f"\nExperiment completed. Results saved to: {run_dir}")
    
    return run_dir

def plot_misclassification_distribution(inc_df, dec_df, output_dir, run_id):
    """
    Create visualizations of misclassification distributions from multiple seed experiments.
    
    Args:
        inc_df: DataFrame with incremental experiment results
        dec_df: DataFrame with decremental experiment results
        output_dir: Directory to save visualizations
        run_id: Unique identifier for this experiment run
    """
    # Create a super plot with multiple visualizations
    plt.figure(figsize=(20, 15))
    
    # 1. Boxplot of misclassification rates by iteration for incremental
    plt.subplot(2, 2, 1)
    iteration_data = [inc_df[inc_df['iteration'] == i]['misclass_rate'].values 
                      for i in sorted(inc_df['iteration'].unique())]
    plt.boxplot(iteration_data)
    plt.title('Incremental Hyperblock Generation\nMisclassification Rate Distribution by Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Misclassification Rate')
    plt.grid(True, alpha=0.3)
    
    # 2. Boxplot of misclassification rates by iteration for decremental
    plt.subplot(2, 2, 2)
    iteration_data = [dec_df[dec_df['iteration'] == i]['misclass_rate'].values 
                      for i in sorted(dec_df['iteration'].unique())]
    plt.boxplot(iteration_data)
    plt.title('Decremental Hyperblock Generation\nMisclassification Rate Distribution by Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Misclassification Rate')
    plt.grid(True, alpha=0.3)
    
    # 3. Heatmap showing distribution density for incremental
    plt.subplot(2, 2, 3)
    iterations = sorted(inc_df['iteration'].unique())
    
    # Create a 2D histogram
    heatmap_data = np.zeros((len(iterations), 10))  # 10 bins from 0 to 1
    for i, iteration in enumerate(iterations):
        iter_data = inc_df[inc_df['iteration'] == iteration]['misclass_rate'].values
        if len(iter_data) > 0:
            hist, _ = np.histogram(iter_data, bins=10, range=(0, 1))
            heatmap_data[i] = hist / len(iter_data)  # Normalize by count
    
    plt.imshow(heatmap_data, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label='Density')
    plt.title('Incremental Misclassification Rate Distribution Density')
    plt.xlabel('Misclassification Rate Bin')
    plt.ylabel('Iteration')
    plt.yticks(range(len(iterations)), iterations)
    plt.xticks(range(10), [f'{i/10:.1f}-{(i+1)/10:.1f}' for i in range(10)])
    
    # 4. Heatmap showing distribution density for decremental
    plt.subplot(2, 2, 4)
    iterations = sorted(dec_df['iteration'].unique())
    
    # Create a 2D histogram
    heatmap_data = np.zeros((len(iterations), 10))  # 10 bins from 0 to 1
    for i, iteration in enumerate(iterations):
        iter_data = dec_df[dec_df['iteration'] == iteration]['misclass_rate'].values
        if len(iter_data) > 0:
            hist, _ = np.histogram(iter_data, bins=10, range=(0, 1))
            heatmap_data[i] = hist / len(iter_data)  # Normalize by count
    
    plt.imshow(heatmap_data, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label='Density')
    plt.title('Decremental Misclassification Rate Distribution Density')
    plt.xlabel('Misclassification Rate Bin')
    plt.ylabel('Iteration')
    plt.yticks(range(len(iterations)), iterations)
    plt.xticks(range(10), [f'{i/10:.1f}-{(i+1)/10:.1f}' for i in range(10)])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{run_id}_misclassification_distribution.png"), dpi=300, bbox_inches='tight')
    
    # Create violin plots for more detailed distribution visualization
    plt.figure(figsize=(20, 10))
    
    # Violin plot for incremental
    plt.subplot(1, 2, 1)
    # Convert to long form data for seaborn
    inc_data = []
    for iteration in sorted(inc_df['iteration'].unique()):
        for rate in inc_df[inc_df['iteration'] == iteration]['misclass_rate'].values:
            inc_data.append({'Iteration': iteration, 'Misclassification Rate': rate})
    inc_long_df = pd.DataFrame(inc_data)
    
    # Use seaborn for violin plot if available
    try:
        import seaborn as sns
        sns.violinplot(x='Iteration', y='Misclassification Rate', data=inc_long_df)
    except ImportError:
        # Fallback to matplotlib violin plot
        iteration_data = [inc_df[inc_df['iteration'] == i]['misclass_rate'].values 
                        for i in sorted(inc_df['iteration'].unique())]
        plt.violinplot(iteration_data, showmeans=True)
        plt.xticks(range(1, len(sorted(inc_df['iteration'].unique()))+1), sorted(inc_df['iteration'].unique()))
    
    plt.title('Incremental Hyperblock Generation\nMisclassification Rate Distribution (Violin Plot)')
    plt.grid(True, alpha=0.3)
    
    # Violin plot for decremental
    plt.subplot(1, 2, 2)
    # Convert to long form data for seaborn
    dec_data = []
    for iteration in sorted(dec_df['iteration'].unique()):
        for rate in dec_df[dec_df['iteration'] == iteration]['misclass_rate'].values:
            dec_data.append({'Iteration': iteration, 'Misclassification Rate': rate})
    dec_long_df = pd.DataFrame(dec_data)
    
    # Use seaborn for violin plot if available
    try:
        import seaborn as sns
        sns.violinplot(x='Iteration', y='Misclassification Rate', data=dec_long_df)
    except ImportError:
        # Fallback to matplotlib violin plot
        iteration_data = [dec_df[dec_df['iteration'] == i]['misclass_rate'].values 
                        for i in sorted(dec_df['iteration'].unique())]
        plt.violinplot(iteration_data, showmeans=True)
        plt.xticks(range(1, len(sorted(dec_df['iteration'].unique()))+1), sorted(dec_df['iteration'].unique()))
    
    plt.title('Decremental Hyperblock Generation\nMisclassification Rate Distribution (Violin Plot)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{run_id}_violin_plots.png"), dpi=300, bbox_inches='tight')
    
    if DEBUG:
        print(f"Distribution plots saved to {output_dir}")

if __name__ == "__main__":
    main()
