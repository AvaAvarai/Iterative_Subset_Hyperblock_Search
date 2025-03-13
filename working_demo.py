import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.cm as cm
from sklearn.preprocessing import MinMaxScaler
import re

def load_and_preprocess_data(file_path, random_seed=42):
    """
    Load CSV data and apply Min-Max normalization to numerical features.
    
    Args:
        file_path: Path to the CSV file
        random_seed: Random seed for reproducibility
        
    Returns:
        normalized_df: Normalized DataFrame
        features: List of feature names
        class_column: Name of the class column
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Find the class column (case-insensitive)
    class_column = None
    for col in df.columns:
        if re.match(r'^class$', col, re.IGNORECASE):
            class_column = col
            break
    
    if class_column is None:
        raise ValueError("No 'class' column found in the dataset")
    
    # Separate features from the class column
    features = [col for col in df.columns if col != class_column]
    
    # Apply Min-Max normalization to numerical features
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    
    return df, features, class_column

def visualize_parallel_coordinates(df, features, class_column, title, ax=None):
    """
    Create a parallel coordinates plot for the given data.
    
    Args:
        df: DataFrame containing the data
        features: List of feature names
        class_column: Name of the class column
        title: Title for the plot
        ax: Matplotlib axis (optional)
        
    Returns:
        ax: Matplotlib axis with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get unique classes and assign colors
    classes = df[class_column].unique()
    colors = cm.tab10(np.linspace(0, 1, len(classes)))
    color_dict = dict(zip(classes, colors))
    
    # Create the parallel coordinates plot
    for cls in classes:
        class_data = df[df[class_column] == cls]
        for i in range(len(class_data)):
            row = class_data.iloc[i]
            coords = [(j, row[feature]) for j, feature in enumerate(features)]
            xs, ys = zip(*coords)
            ax.plot(xs, ys, color=color_dict[cls], alpha=0.5)
    
    # Set the x-ticks and labels
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45)
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    
    # Add a title
    ax.set_title(title)
    
    # Add a legend
    for cls, color in zip(classes, colors):
        ax.plot([], [], color=color, label=f'Class {cls}')
    ax.legend(loc='upper right')
    
    return ax

def create_hyperblock_envelopes(df, features, class_column):
    """
    Create hyperblock envelopes (pure clusters) for each class.
    
    Args:
        df: DataFrame containing the data
        features: List of feature names
        class_column: Name of the class column
        
    Returns:
        hyperblocks: Dictionary mapping class labels to their hyperblock boundaries
    """
    hyperblocks = {}
    classes = df[class_column].unique()
    
    for cls in classes:
        class_data = df[df[class_column] == cls]
        
        # For each feature, find the min and max values
        feature_bounds = {}
        for feature in features:
            feature_bounds[feature] = {
                'min': class_data[feature].min(),
                'max': class_data[feature].max()
            }
        
        hyperblocks[cls] = feature_bounds
    
    return hyperblocks

def visualize_hyperblocks(ax, hyperblocks, features):
    """
    Visualize hyperblock envelopes on the parallel coordinates plot.
    
    Args:
        ax: Matplotlib axis with the parallel coordinates plot
        hyperblocks: Dictionary mapping class labels to their hyperblock boundaries
        features: List of feature names
    """
    classes = list(hyperblocks.keys())
    colors = cm.tab10(np.linspace(0, 1, len(classes)))
    
    for cls, color in zip(classes, colors):
        for i, feature in enumerate(features):
            bounds = hyperblocks[cls][feature]
            min_val = bounds['min']
            max_val = bounds['max']
            
            # Use facecolor and edgecolor instead of color
            rect = patches.Rectangle((i-0.2, min_val), 0.4, max_val-min_val, 
                                    alpha=0.2, facecolor=color, edgecolor=color, 
                                    linewidth=1.5)
            ax.add_patch(rect)

def check_purity_violation(new_data, hyperblocks, features, class_column):
    """
    Check if new data points violate the purity of existing hyperblocks.
    
    Args:
        new_data: DataFrame containing new data points
        hyperblocks: Dictionary mapping class labels to their hyperblock boundaries
        features: List of feature names
        class_column: Name of the class column
        
    Returns:
        violations: DataFrame containing data points that violate purity
        non_violations: DataFrame containing data points that don't violate purity
    """
    violations_list = []
    non_violations_list = []
    
    for idx, row in new_data.iterrows():
        cls = row[class_column]
        is_violation = False
        
        # Check if the point falls within a hyperblock of a different class
        for hb_cls, feature_bounds in hyperblocks.items():
            if hb_cls == cls:
                continue
                
            point_in_hb = True
            for feature in features:
                bounds = feature_bounds[feature]
                if row[feature] < bounds['min'] or row[feature] > bounds['max']:
                    point_in_hb = False
                    break
            
            if point_in_hb:
                is_violation = True
                break
        
        if is_violation:
            violations_list.append(row)
        else:
            non_violations_list.append(row)
    
    # Create DataFrames from the lists
    violations = pd.DataFrame(violations_list, columns=new_data.columns) if violations_list else pd.DataFrame(columns=new_data.columns)
    non_violations = pd.DataFrame(non_violations_list, columns=new_data.columns) if non_violations_list else pd.DataFrame(columns=new_data.columns)
    
    return violations, non_violations

def update_hyperblocks(hyperblocks, new_data, features, class_column):
    """
    Update hyperblock envelopes with new non-violating data.
    
    Args:
        hyperblocks: Dictionary mapping class labels to their hyperblock boundaries
        new_data: DataFrame containing new data points
        features: List of feature names
        class_column: Name of the class column
        
    Returns:
        updated_hyperblocks: Updated hyperblock boundaries
    """
    updated_hyperblocks = hyperblocks.copy()
    
    for idx, row in new_data.iterrows():
        cls = row[class_column]
        
        # If this is a new class, create a new hyperblock
        if cls not in updated_hyperblocks:
            updated_hyperblocks[cls] = {feature: {'min': row[feature], 'max': row[feature]} 
                                       for feature in features}
        else:
            # Update existing hyperblock bounds
            for feature in features:
                if row[feature] < updated_hyperblocks[cls][feature]['min']:
                    updated_hyperblocks[cls][feature]['min'] = row[feature]
                if row[feature] > updated_hyperblocks[cls][feature]['max']:
                    updated_hyperblocks[cls][feature]['max'] = row[feature]
    
    return updated_hyperblocks

def calculate_hyperblock_area(hyperblocks, features):
    """
    Calculate the area (volume) of each hyperblock and the total area.
    
    Args:
        hyperblocks: Dictionary mapping class labels to their hyperblock boundaries
        features: List of feature names
        
    Returns:
        class_areas: Dictionary mapping class labels to their hyperblock areas
        total_area: Total area of all hyperblocks
    """
    class_areas = {}
    total_area = 0
    
    for cls, feature_bounds in hyperblocks.items():
        # Calculate the volume of the hyperblock (product of ranges for each feature)
        area = 1.0
        for feature in features:
            bounds = feature_bounds[feature]
            feature_range = bounds['max'] - bounds['min']
            area *= feature_range
        
        class_areas[cls] = area
        total_area += area
    
    return class_areas, total_area

def print_hyperblock_bounds(hyperblocks, features):
    """
    Print the min and max bounds for each feature in each hyperblock.
    
    Args:
        hyperblocks: Dictionary mapping class labels to their hyperblock boundaries
        features: List of feature names
    """
    print("\nHyperblock Bounds:")
    print("-" * 80)
    
    # Calculate the maximum width needed for each column
    cls_width = max(len(str(cls)) for cls in hyperblocks.keys()) + 2
    feature_width = max(len(feature) for feature in features) + 2
    value_width = 10  # Assuming float values with reasonable precision
    
    # Print header
    header = f"{'Class':{cls_width}} {'Feature':{feature_width}} {'Min':{value_width}} {'Max':{value_width}}"
    print(header)
    print("-" * len(header))
    
    # Print bounds for each class and feature
    for cls, feature_bounds in hyperblocks.items():
        for feature in features:
            bounds = feature_bounds[feature]
            print(f"{cls:{cls_width}} {feature:{feature_width}} {bounds['min']:{value_width}.4f} {bounds['max']:{value_width}.4f}")
    
    print("-" * 80)

def main(file_path, random_seed=42, num_trials=5):
    """
    Main function to execute the visual analytics and incremental data selection process.
    
    Args:
        file_path: Path to the CSV file
        random_seed: Random seed for reproducibility
        num_trials: Number of different initial 1/3 subsets to try
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Step 1: Load and preprocess data
    df, features, class_column = load_and_preprocess_data(file_path, random_seed)
    n_samples = len(df)
    initial_size = n_samples // 3
    
    # Store results from all trials
    trial_results = []
    
    # Try different initial 1/3 subsets
    for trial in range(num_trials):
        print(f"\nTrial {trial+1}/{num_trials}:")
        
        # Generate a new random permutation for each trial
        trial_seed = random_seed + trial
        np.random.seed(trial_seed)
        indices = np.random.permutation(n_samples)
        initial_indices = indices[:initial_size]
        remaining_indices = indices[initial_size:]
        
        initial_data = df.iloc[initial_indices].copy()
        remaining_data = df.iloc[remaining_indices].copy()
        
        # Build pure clusters (hyperblock envelopes)
        hyperblocks = create_hyperblock_envelopes(initial_data, features, class_column)
        
        # Incremental data addition and purity checking
        current_data = initial_data.copy()
        increment_size = int(n_samples * 0.05)  # 5% of the dataset
        
        # Track data points that violate purity
        all_violations = pd.DataFrame(columns=df.columns)
        
        # Process remaining data in 5% increments
        remaining_indices_list = remaining_indices.tolist()
        
        iteration = 1
        violations_occurred = False
        
        while len(remaining_indices_list) > 0 and not violations_occurred:
            # Get the next increment
            next_increment_size = min(increment_size, len(remaining_indices_list))
            next_indices = remaining_indices_list[:next_increment_size]
            remaining_indices_list = remaining_indices_list[next_increment_size:]
            
            next_data = df.iloc[next_indices].copy()
            
            # Check for purity violations
            violations, non_violations = check_purity_violation(next_data, hyperblocks, features, class_column)
            
            # If violations occur, we're done
            if len(violations) > 0:
                violations_occurred = True
                all_violations = pd.concat([all_violations, violations], ignore_index=True)
                print(f"  Violations detected in iteration {iteration}. Stopping.")
            
            # Add non-violating points to current data
            if len(non_violations) > 0:
                current_data = pd.concat([current_data, non_violations], ignore_index=True)
                
                # Update hyperblocks with non-violating data
                hyperblocks = update_hyperblocks(hyperblocks, non_violations, features, class_column)
            
            iteration += 1
            
            # Check stopping condition
            if len(remaining_indices_list) == 0:
                print(f"  All data processed after {iteration-1} increments without violations.")
        
        # Calculate data usage metrics
        used_data_count = len(current_data)  # Only count non-violating data as "used"
        data_usage_percentage = (used_data_count / n_samples) * 100
        
        # Calculate hyperblock areas
        class_areas, total_area = calculate_hyperblock_area(hyperblocks, features)
        
        # Print summary for this trial
        print(f"  Initial subset size: {initial_size}")
        print(f"  Final dataset size: {used_data_count}")
        print(f"  Number of purity violations: {len(all_violations)}")
        print(f"  Data usage: {used_data_count} samples ({data_usage_percentage:.2f}%)")
        
        # Print hyperblock bounds and areas
        print("\n  Hyperblock Areas:")
        for cls, area in class_areas.items():
            print(f"    Class {cls}: {area:.6f}")
        print(f"    Total Area: {total_area:.6f}")
        
        # Print detailed hyperblock bounds
        print("\n  Hyperblock Bounds:")
        for cls, feature_bounds in hyperblocks.items():
            print(f"    Class {cls}:")
            for feature in features:
                bounds = feature_bounds[feature]
                print(f"      {feature}: [{bounds['min']:.4f}, {bounds['max']:.4f}]")
        
        # Store results
        trial_results.append({
            'trial': trial + 1,
            'seed': trial_seed,
            'initial_data': initial_data,
            'current_data': current_data,
            'violations': all_violations,
            'hyperblocks': hyperblocks,
            'violation_count': len(all_violations),
            'used_count': used_data_count,
            'data_usage_percentage': data_usage_percentage,
            'iterations': iteration - 1,
            'class_areas': class_areas,
            'total_area': total_area
        })
    
    # Create a results table
    results_table = pd.DataFrame({
        'Trial': [r['trial'] for r in trial_results],
        'Seed': [r['seed'] for r in trial_results],
        'Data Used': [r['used_count'] for r in trial_results],
        'Data Usage (%)': [f"{r['data_usage_percentage']:.2f}%" for r in trial_results],
        'Total HB Area': [f"{r['total_area']:.6f}" for r in trial_results],
        'Iterations': [r['iterations'] for r in trial_results],
        'Violations': [r['violation_count'] for r in trial_results]
    })
    
    # Sort by data usage (ascending)
    results_table = results_table.sort_values('Data Used')
    
    # Print the table
    print("\n" + "="*80)
    print("Trial Results (Sorted by Data Usage):")
    print("="*80)
    print(results_table.to_string(index=False))
    print("="*80)
    
    # Find the best trial (with minimum data usage)
    best_trial = min(trial_results, key=lambda x: x['used_count'])
    
    print("\nBest Trial (Minimum Data Usage):")
    print(f"Trial {best_trial['trial']} (Seed: {best_trial['seed']})")
    print(f"  Data used: {best_trial['used_count']} samples ({best_trial['data_usage_percentage']:.2f}%)")
    print(f"  Iterations completed: {best_trial['iterations']}")
    print(f"  Violations: {best_trial['violation_count']}")
    
    # Print detailed information about the best trial
    print("\nBest Trial Hyperblock Details:")
    print_hyperblock_bounds(best_trial['hyperblocks'], features)
    
    # Print hyperblock areas for the best trial
    print("\nBest Trial Hyperblock Areas:")
    for cls, area in best_trial['class_areas'].items():
        print(f"  Class {cls}: {area:.6f}")
    print(f"  Total Area: {best_trial['total_area']:.6f}")
    
    # Visualize the best trial
    visualize_best_trial(df, features, class_column, best_trial)
    
    return best_trial, results_table

def visualize_best_trial(df, features, class_column, best_trial):
    """
    Visualize the results of the best trial with enhanced pure region highlighting.
    
    Args:
        df: Full DataFrame
        features: List of feature names
        class_column: Name of the class column
        best_trial: Dictionary containing the results of the best trial
    """
    # Create a figure with subplots for visualization
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # Visualize initial data
    ax1 = visualize_parallel_coordinates(best_trial['initial_data'], features, class_column, 
                                        "Best Trial: Initial Data Subset (1/3 of Dataset)", axes[0, 0])
    
    # Visualize hyperblocks on the initial data
    visualize_hyperblocks(ax1, best_trial['hyperblocks'], features)
    
    # Visualize remaining data
    remaining_indices = ~df.index.isin(best_trial['initial_data'].index)
    remaining_data = df[remaining_indices].copy()
    ax2 = visualize_parallel_coordinates(remaining_data, features, class_column, 
                                        "Best Trial: Remaining Data", axes[0, 1])
    
    # Visualize final data with hyperblocks
    ax3 = visualize_parallel_coordinates(best_trial['current_data'], features, class_column, 
                                       f"Best Trial: Final Data", axes[1, 0])
    
    # Clear any existing legend before adding hyperblocks to avoid duplicate labels
    if ax3.get_legend() is not None:
        ax3.get_legend().remove()
    
    # Visualize hyperblocks with overlaps
    visualize_hyperblocks_with_overlaps(ax3, best_trial['hyperblocks'], features)
    
    # Visualize violations or pure regions
    if len(best_trial['violations']) > 0:
        ax4 = visualize_parallel_coordinates(best_trial['violations'], features, class_column, 
                                           "Best Trial: Purity Violations", axes[1, 1])
    else:
        axes[1, 1].text(0.5, 0.5, "No purity violations found", 
                       horizontalalignment='center', verticalalignment='center',
                       transform=axes[1, 1].transAxes, fontsize=14)
    
    plt.tight_layout()
    plt.show()


def identify_overlapping_regions(hyperblocks, features):
    """
    Identify regions where hyperblocks from different classes overlap.
    
    Args:
        hyperblocks: Dictionary mapping class labels to their hyperblock boundaries
        features: List of feature names
        
    Returns:
        overlaps: Dictionary mapping feature pairs to lists of overlapping class pairs
    """
    classes = list(hyperblocks.keys())
    overlaps = {}
    
    # Check each pair of classes for overlaps in each feature
    for i, cls1 in enumerate(classes):
        for cls2 in classes[i+1:]:  # Only check each pair once
            # Check if the hyperblocks overlap in all features
            overlap_in_all_features = True
            
            for feature in features:
                bounds1 = hyperblocks[cls1][feature]
                bounds2 = hyperblocks[cls2][feature]
                
                # Check if ranges overlap
                if bounds1['max'] < bounds2['min'] or bounds2['max'] < bounds1['min']:
                    overlap_in_all_features = False
                    break
            
            if overlap_in_all_features:
                # These classes have overlapping hyperblocks
                if 'full_overlap' not in overlaps:
                    overlaps['full_overlap'] = []
                overlaps['full_overlap'].append((cls1, cls2))
            else:
                # Check for partial overlaps (feature by feature)
                for feature in features:
                    bounds1 = hyperblocks[cls1][feature]
                    bounds2 = hyperblocks[cls2][feature]
                    
                    # Check if ranges overlap for this feature
                    if not (bounds1['max'] < bounds2['min'] or bounds2['max'] < bounds1['min']):
                        if feature not in overlaps:
                            overlaps[feature] = []
                        overlaps[feature].append((cls1, cls2))
    
    return overlaps

def visualize_hyperblocks_with_overlaps(ax, hyperblocks, features):
    """
    Visualize hyperblock envelopes on the parallel coordinates plot, 
    highlighting pure regions and showing overlaps.
    
    Args:
        ax: Matplotlib axis with the parallel coordinates plot
        hyperblocks: Dictionary mapping class labels to their hyperblock boundaries
        features: List of feature names
    """
    classes = list(hyperblocks.keys())
    colors = cm.tab10(np.linspace(0, 1, len(classes)))
    color_dict = dict(zip(classes, colors))
    
    # Find overlapping regions
    overlaps = identify_overlapping_regions(hyperblocks, features)
    
    # First, draw the pure regions for each class
    for cls, color in zip(classes, colors):
        for i, feature in enumerate(features):
            bounds = hyperblocks[cls][feature]
            min_val = bounds['min']
            max_val = bounds['max']
            
            # Use facecolor and edgecolor instead of color
            rect = patches.Rectangle((i-0.2, min_val), 0.4, max_val-min_val, 
                                    alpha=0.25, facecolor=color, edgecolor=color, 
                                    linewidth=1.5)
            ax.add_patch(rect)
    
    # Then, highlight overlapping regions with a different pattern/color
    for feature, class_pairs in overlaps.items():
        if feature == 'full_overlap':
            # Handle full overlaps (across all features)
            for cls1, cls2 in class_pairs:
                for i, feat in enumerate(features):
                    bounds1 = hyperblocks[cls1][feat]
                    bounds2 = hyperblocks[cls2][feat]
                    
                    # Calculate the overlap region
                    overlap_min = max(bounds1['min'], bounds2['min'])
                    overlap_max = min(bounds1['max'], bounds2['max'])
                    
                    # Draw the overlap with a hatched pattern
                    overlap_rect = patches.Rectangle((i-0.2, overlap_min), 0.4, overlap_max-overlap_min, 
                                                   alpha=0.5, hatch='///', fill=False, 
                                                   edgecolor='red', linewidth=1.5)
                    ax.add_patch(overlap_rect)
        else:
            # Handle partial overlaps (specific features)
            i = features.index(feature)
            for cls1, cls2 in class_pairs:
                bounds1 = hyperblocks[cls1][feature]
                bounds2 = hyperblocks[cls2][feature]
                
                # Calculate the overlap region
                overlap_min = max(bounds1['min'], bounds2['min'])
                overlap_max = min(bounds1['max'], bounds2['max'])
                
                # Draw the overlap with a hatched pattern
                overlap_rect = patches.Rectangle((i-0.2, overlap_min), 0.4, overlap_max-overlap_min, 
                                               alpha=0.5, hatch='///', fill=False, 
                                               edgecolor='orange', linewidth=1.5)
                ax.add_patch(overlap_rect)
    
    # Calculate pure areas (excluding overlaps)
    pure_areas, total_pure_area = calculate_pure_areas(hyperblocks, features, overlaps)
    
    # Add a legend with class information and overlap pattern
    legend_elements = []
    
    # Add class patches with area information
    for cls, color in zip(classes, colors):
        full_area = calculate_hyperblock_area(hyperblocks, features)[0][cls]
        pure_area = pure_areas[cls]
        
        # Create a patch for the legend
        class_patch = patches.Patch(
            facecolor=color, alpha=0.25,  # Use facecolor instead of color
            label=f'Class {cls} (Area: {pure_area:.4f})'
        )
        legend_elements.append(class_patch)
    
    # Add overlap patch
    if any(overlaps.values()):
        overlap_patch = patches.Patch(
            facecolor='white', edgecolor='red', alpha=0.5, hatch='///',
            label='Overlapping (Non-Pure) Regions'
        )
        legend_elements.append(overlap_patch)
    
    # Add total area information
    total_area = sum(calculate_hyperblock_area(hyperblocks, features)[0].values())
    total_patch = patches.Patch(
        facecolor='none', edgecolor='none', label=f'Total Pure Area: {total_pure_area:.4f}'
    )
    legend_elements.append(total_patch)
    
    # Add the legend
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

def calculate_pure_areas(hyperblocks, features, overlaps):
    """
    Calculate the area of pure regions (excluding overlaps) for each class.
    
    Args:
        hyperblocks: Dictionary mapping class labels to their hyperblock boundaries
        features: List of feature names
        overlaps: Dictionary mapping feature pairs to lists of overlapping class pairs
        
    Returns:
        pure_areas: Dictionary mapping class labels to their pure area
        total_pure_area: Total area of all pure regions
    """
    # Start with the full hyperblock areas
    class_areas, total_area = calculate_hyperblock_area(hyperblocks, features)
    pure_areas = class_areas.copy()
    
    # Subtract overlapping areas
    if 'full_overlap' in overlaps:
        for cls1, cls2 in overlaps['full_overlap']:
            # Calculate the volume of the overlap
            overlap_volume = 1.0
            for feature in features:
                bounds1 = hyperblocks[cls1][feature]
                bounds2 = hyperblocks[cls2][feature]
                
                overlap_min = max(bounds1['min'], bounds2['min'])
                overlap_max = min(bounds1['max'], bounds2['max'])
                overlap_volume *= (overlap_max - overlap_min)
            
            # Subtract half of the overlap from each class
            # (this is a simplification - in reality, the overlap belongs to neither class)
            pure_areas[cls1] -= overlap_volume / 2
            pure_areas[cls2] -= overlap_volume / 2
    
    # Calculate total pure area
    total_pure_area = sum(pure_areas.values())
    
    return pure_areas, total_pure_area

if __name__ == "__main__":
    # Replace 'your_dataset.csv' with the actual file path
    # You can also specify a different random seed if needed
    main('fisher_iris.csv', random_seed=42, num_trials=50)
