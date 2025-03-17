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
        
        print(f"  Created hyperblock with {len(best_points_indices)} points, dominant class: {dominant_class}")
        
        # Step 11: Remove points in hyperblock from remaining_points
        remaining_points -= set(best_points_indices)
    
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
    
    print(f"  Completed {impure_merges} impure merges")
    
    # Convert to Hyperblock objects
    hyperblocks = []
    for hb in result_hbs:
        # Use loc instead of iloc to access by index values
        points = [df.loc[idx].values for idx in hb['points']]
        hb_obj = Hyperblock(hb['min_bounds'], hb['max_bounds'], points, hb['class'])
        hyperblocks.append(hb_obj)
    
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
        print("All points covered by IHyper, skipping MHyper")
        return ihyper_blocks
    
    # Create DataFrame with remaining points
    remaining_rows = []
    for idx, row in df.iterrows():
        if tuple(row.values) in remaining_tuples:
            remaining_rows.append(row)
    
    remaining_df = pd.DataFrame(remaining_rows, columns=df.columns)
    print(f"IHyper left {len(remaining_df)} points uncovered, running MHyper on them")
    
    # Step 3: Run MHyper on remaining points
    mhyper_blocks = mhyper_algorithm(remaining_df, class_col, impurity_threshold)
    
    # Combine results
    all_blocks = ihyper_blocks + mhyper_blocks
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

def visualize_hyperblocks(df, features, class_col, hyperblocks, title="Hyperblocks in Parallel Coordinates"):
    """
    Visualize hyperblocks in parallel coordinates and print coverage statistics.
    
    Args:
        df: DataFrame containing the data
        features: List of feature names
        class_col: Name of the class column
        hyperblocks: List of Hyperblock objects
        title: Title for the plot
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
    
    # Print dataset coverage information
    print("\nDataset Coverage Information:")
    print("-" * 60)
    print(f"Total data points: {len(df)}")
    print(f"Points covered by hyperblocks: {covered_count}")
    print(f"Coverage percentage: {coverage_percentage:.2f}%")
    
    if uncovered_df is not None and not uncovered_df.empty:
        print(f"Uncovered points: {len(uncovered_df)}")
        
        # Print class distribution of uncovered points
        if class_col in uncovered_df.columns:
            uncovered_class_counts = uncovered_df[class_col].value_counts()
            print("\nUncovered points class distribution:")
            for cls, count in uncovered_class_counts.items():
                print(f"  Class {cls}: {count} points")
    else:
        print("All points are covered by hyperblocks!")
    
    # Print complete hyperblock summary with all attributes
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
    plt.show()

def main():
    # Load and normalize data
    df, features, class_col = load_and_normalize_data()
    
    # Generate hyperblocks using IMHyper (which combines IHyper and MHyper)
    hyperblocks = imhyper_algorithm(df, class_col, purity_threshold=0.9999, impurity_threshold=0.0001)
    
    # Visualize the results
    visualize_hyperblocks(df, features, class_col, hyperblocks, 
                         title="IMHyper: Data with Hyperblock Visualization")

if __name__ == "__main__":
    main()
