import pandas as pd
import numpy as np
from tabulate import tabulate
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import tkinter as tk
from tkinter import filedialog
import os
import datetime

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


def load_data(file_path):
    df = pd.read_csv(file_path)
    class_column = [col for col in df.columns if col.lower() == 'class'][0]
    return df, class_column


def generate_test_data(n_samples=100, n_features=2):
    """
    Generate test data with two Gaussian distributions.
    
    :param n_samples: Number of samples to generate
    :param n_features: Number of features
    :return: DataFrame with features and class labels
    """
    # Generate class 0 samples from first Gaussian
    n_class0 = n_samples // 2
    mean0 = np.zeros(n_features)
    cov0 = np.eye(n_features)
    X0 = np.random.multivariate_normal(mean0, cov0, n_class0)
    y0 = np.zeros(n_class0)
    
    # Generate class 1 samples from second Gaussian
    n_class1 = n_samples - n_class0
    mean1 = np.ones(n_features) * 3
    cov1 = np.eye(n_features) * 1.5
    X1 = np.random.multivariate_normal(mean1, cov1, n_class1)
    y1 = np.ones(n_class1)
    
    # Combine the data
    X = np.vstack((X0, X1))
    y = np.hstack((y0, y1))
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Create DataFrame
    data = np.column_stack((X, y))
    df = pd.DataFrame(data, columns=feature_names + ['class'])
    df['class'] = df['class'].astype(int)
    
    return df


class IHyper:
    def __init__(self, data, labels, purity_threshold=0.9):
        """
        Initialize the IHyper algorithm.

        :param data: numpy array of shape (n_samples, n_features), dataset features.
        :param labels: numpy array of shape (n_samples,), class labels.
        :param purity_threshold: float, minimum purity threshold.
        """
        self.data = data
        self.labels = labels
        self.purity_threshold = purity_threshold
        self.lda = LinearDiscriminantAnalysis()
        self.lda.fit(data, labels)
        self.ldf_values = self.lda.decision_function(data)  # Compute LDF values
        self.n_features = data.shape[1]
        self.hyperblocks = []

    def compute_purity(self, indices):
        """
        Compute the purity of a subset of indices.

        :param indices: List of indices corresponding to dataset points.
        :return: Purity score (float).
        """
        subset_labels = self.labels[indices]
        counts = np.bincount(subset_labels)
        max_count = np.max(counts)
        purity = max_count / len(subset_labels)
        return purity

    def find_largest_pure_interval(self, feature_idx):
        """
        Find the largest pure interval for a given feature.

        :param feature_idx: Index of the feature being analyzed.
        :return: (start, end) of the largest pure interval.
        """
        sorted_indices = np.argsort(self.data[:, feature_idx])
        sorted_values = self.data[sorted_indices, feature_idx]
        sorted_ldf = self.ldf_values[sorted_indices]
        sorted_labels = self.labels[sorted_indices]

        best_interval = None
        best_size = 0
        start = 0

        while start < len(sorted_values):
            end = start
            while end < len(sorted_values):
                indices = sorted_indices[start:end + 1]
                purity = self.compute_purity(indices)

                if purity < self.purity_threshold:
                    break  # Stop if purity threshold is violated

                end += 1

            interval_size = end - start
            if interval_size > best_size:
                best_size = interval_size
                best_interval = (sorted_values[start], sorted_values[end - 1])

            start = end

        return best_interval

    def construct_hyperblocks(self):
        """
        Construct hyperblocks using the IHyper algorithm.
        """
        remaining_points = set(range(self.data.shape[0]))

        while remaining_points:
            best_intervals = {}

            for feature_idx in range(self.n_features):
                interval = self.find_largest_pure_interval(feature_idx)
                if interval:
                    best_intervals[feature_idx] = interval

            if not best_intervals:
                break  # No more intervals can be formed

            # Select the attribute with the largest interval
            best_feature = max(best_intervals, key=lambda k: best_intervals[k][1] - best_intervals[k][0])
            best_interval = best_intervals[best_feature]

            # Create the hyperblock
            hyperblock = {
                'feature': best_feature,
                'interval': best_interval
            }
            self.hyperblocks.append(hyperblock)

            # Remove covered points
            min_val, max_val = best_interval
            covered_indices = {i for i in remaining_points if min_val <= self.data[i, best_feature] <= max_val}
            remaining_points -= covered_indices

        return self.hyperblocks


def ihyper(df, class_col, purity_threshold=1.0):
    hyperblocks = []
    attributes = [col for col in df.columns if col != class_col]
    remaining_points = df.copy()
    
    while not remaining_points.empty:
        best_hb = None
        for attr in attributes:
            sorted_values = sorted(remaining_points[attr].unique())
            for val in sorted_values:
                sub_df = remaining_points[remaining_points[attr] >= val]
                # Determine dominant class by finding the most frequent label
                dominant_class = sub_df[class_col].value_counts().idxmax()
                purity = (sub_df[class_col] == dominant_class).mean()
                if purity >= purity_threshold:
                    min_bounds = {a: sub_df[a].min() for a in attributes}
                    max_bounds = {a: sub_df[a].max() for a in attributes}
                    hb = Hyperblock(min_bounds, max_bounds, sub_df.values, dominant_class)
                    if best_hb is None or hb.num_cases > best_hb.num_cases:
                        best_hb = hb
        
        if best_hb:
            hyperblocks.append(best_hb)
            remaining_points = remaining_points.drop(index=[i for i in range(len(df)) if tuple(df.iloc[i]) in best_hb.points])
        else:
            break
    
    return hyperblocks


class MHyper:
    def __init__(self, data, labels, impurity_threshold=0.1):
        """
        Initialize the MHyper algorithm.

        :param data: numpy array of shape (n_samples, n_features), dataset features.
        :param labels: numpy array of shape (n_samples,), class labels.
        :param impurity_threshold: float, maximum allowed impurity for merging.
        """
        self.data = data
        self.labels = labels
        self.impurity_threshold = impurity_threshold
        self.n_features = data.shape[1]
        self.hyperblocks = self.initialize_hyperblocks()

    def initialize_hyperblocks(self):
        """
        Step 1: Initialize pure hyperblocks with a single n-D point each.
        """
        return [{'points': [i], 'class': self.labels[i], 
                 'bounds': [(self.data[i, j], self.data[i, j]) for j in range(self.n_features)]} 
                for i in range(len(self.data))]

    def compute_envelope(self, hb1, hb2):
        """
        Step 3a: Create a joint hyperblock as an envelope around hb1 and hb2.

        :param hb1: First hyperblock.
        :param hb2: Second hyperblock.
        :return: New hyperblock encompassing both.
        """
        new_bounds = [(min(hb1['bounds'][j][0], hb2['bounds'][j][0]), 
                       max(hb1['bounds'][j][1], hb2['bounds'][j][1])) 
                      for j in range(self.n_features)]
        new_points = list(set(hb1['points'] + hb2['points']))
        
        # Determine dominant class by finding the most frequent label in the combined points
        class_counts = {}
        for idx in new_points:
            label = self.labels[idx]
            class_counts[label] = class_counts.get(label, 0) + 1
        dominant_class = max(class_counts.items(), key=lambda x: x[1])[0]
        
        return {'points': new_points, 'class': dominant_class, 'bounds': new_bounds}

    def is_pure(self, hb):
        """
        Step 3c: Check if all points in a hyperblock have the same class.

        :param hb: Hyperblock to check.
        :return: True if pure, False otherwise.
        """
        unique_classes = set(self.labels[i] for i in hb['points'])
        return len(unique_classes) == 1

    def merge_pure_hyperblocks(self):
        """
        Step 2-5: Merge hyperblocks of the same class that remain pure.
        """
        merged_hbs = []

        while self.hyperblocks:
            hb_x = self.hyperblocks.pop(0)
            merged = False

            for i in range(len(self.hyperblocks)):
                hb_i = self.hyperblocks[i]

                if hb_x['class'] == hb_i['class']:  # Step 3
                    joint_hb = self.compute_envelope(hb_x, hb_i)

                    # Step 3b: Check if any other points belong in the envelope
                    joint_hb['points'] = [idx for idx in range(len(self.data)) 
                                          if all(joint_hb['bounds'][j][0] <= self.data[idx, j] <= joint_hb['bounds'][j][1] 
                                                 for j in range(self.n_features))]
                    
                    # Recalculate dominant class based on all points in the envelope
                    class_counts = {}
                    for idx in joint_hb['points']:
                        label = self.labels[idx]
                        class_counts[label] = class_counts.get(label, 0) + 1
                    joint_hb['class'] = max(class_counts.items(), key=lambda x: x[1])[0]

                    if self.is_pure(joint_hb):  # Step 3c
                        self.hyperblocks.pop(i)
                        merged_hbs.append(joint_hb)
                        merged = True
                        break

            if not merged:
                merged_hbs.append(hb_x)

        self.hyperblocks = merged_hbs

    def compute_impurity(self, hb):
        """
        Step 8c: Compute the impurity of a hyperblock.

        :param hb: Hyperblock.
        :return: Impurity value (percentage of opposite-class points).
        """
        class_counts = {}
        for idx in hb['points']:
            label = self.labels[idx]
            class_counts[label] = class_counts.get(label, 0) + 1
        
        # Determine dominant class by finding the most frequent label
        dominant_class = max(class_counts.items(), key=lambda x: x[1])[0]
        hb['class'] = dominant_class  # Update the dominant class
        
        dominant_class_count = class_counts[dominant_class]
        total_points = sum(class_counts.values())

        return 1 - (dominant_class_count / total_points)

    def merge_impure_hyperblocks(self):
        """
        Step 6-9: Merge hyperblocks with lowest impurity.
        """
        while True:
            best_merge = None
            lowest_impurity = float('inf')

            for i in range(len(self.hyperblocks)):
                for j in range(i + 1, len(self.hyperblocks)):
                    hb_x, hb_i = self.hyperblocks[i], self.hyperblocks[j]

                    joint_hb = self.compute_envelope(hb_x, hb_i)
                    joint_hb['points'] = [idx for idx in range(len(self.data)) 
                                          if all(joint_hb['bounds'][k][0] <= self.data[idx, k] <= joint_hb['bounds'][k][1] 
                                                 for k in range(self.n_features))]

                    impurity = self.compute_impurity(joint_hb)

                    if impurity < self.impurity_threshold and impurity < lowest_impurity:
                        best_merge = (i, j, joint_hb)
                        lowest_impurity = impurity

            if best_merge is None:
                break  # No valid merges left

            # Merge best pair
            i, j, joint_hb = best_merge
            self.hyperblocks.pop(j)
            self.hyperblocks.pop(i)
            self.hyperblocks.append(joint_hb)

    def run(self):
        """
        Execute the full MHyper algorithm.
        """
        self.merge_pure_hyperblocks()
        self.merge_impure_hyperblocks()
        return self.hyperblocks


def mhyper(df, class_col, impurity_threshold=0.1):
    # Convert DataFrame to numpy arrays
    attributes = [col for col in df.columns if col != class_col]
    X = df[attributes].values
    y = df[class_col].factorize()[0]  # Convert class labels to numeric indices
    
    # Run MHyper algorithm
    mhyper_instance = MHyper(X, y, impurity_threshold)
    hyperblocks = mhyper_instance.run()
    
    # Convert results back to Hyperblock objects
    result_hyperblocks = []
    for hb in hyperblocks:
        points = [df.iloc[i].values for i in hb['points']]
        # Determine dominant class by finding the most frequent label
        class_counts = {}
        for i in hb['points']:
            label = df[class_col].iloc[i]
            class_counts[label] = class_counts.get(label, 0) + 1
        dominant_class = max(class_counts.items(), key=lambda x: x[1])[0]
        
        min_bounds = {attr: hb['bounds'][i][0] for i, attr in enumerate(attributes)}
        max_bounds = {attr: hb['bounds'][i][1] for i, attr in enumerate(attributes)}
        result_hyperblocks.append(Hyperblock(min_bounds, max_bounds, points, dominant_class))
    
    return result_hyperblocks


def imhyper(df, class_col, purity_threshold=1.0, impurity_threshold=0.1):
    ihyper_blocks = ihyper(df, class_col, purity_threshold)
    remaining_points = df[~df.apply(tuple, axis=1).isin([tuple(p) for hb in ihyper_blocks for p in hb.points])]
    
    if not remaining_points.empty:
        mhyper_blocks = mhyper(remaining_points, class_col, impurity_threshold)
        return ihyper_blocks + mhyper_blocks
    else:
        return ihyper_blocks


def plot_hyperblock(hyperblock, df, class_col, block_num, title=None):
    """
    Plot a single hyperblock using parallel coordinates.
    
    :param hyperblock: A single Hyperblock object
    :param df: Original DataFrame
    :param class_col: Name of the class column
    :param block_num: Number of the hyperblock (for title)
    :param title: Optional title override
    :return: Figure and axis objects
    """
    attributes = [col for col in df.columns if col != class_col]
    n_attrs = len(attributes)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set up the axes
    x = list(range(n_attrs))
    
    # Get unique classes for color mapping
    unique_classes = df[class_col].unique()
    class_colors = list(mcolors.TABLEAU_COLORS)[:len(unique_classes)]
    class_color_map = dict(zip(unique_classes, class_colors))
    
    # Get color for this hyperblock
    block_color = plt.cm.tab20(block_num / 20)  # Use a color from tab20 colormap
    class_color = class_color_map[hyperblock.dominant_class]
    
    # Plot the min-max bounds as a shaded region
    for j in range(n_attrs-1):
        # Get min and max values for current and next attribute
        y1_min = hyperblock.min_bounds[attributes[j]]
        y1_max = hyperblock.max_bounds[attributes[j]]
        y2_min = hyperblock.min_bounds[attributes[j+1]]
        y2_max = hyperblock.max_bounds[attributes[j+1]]
        
        # Create polygon vertices for the shaded region
        xs = [x[j], x[j], x[j+1], x[j+1]]
        ys = [y1_min, y1_max, y2_max, y2_min]
        
        # Plot the polygon
        ax.fill(xs, ys, alpha=0.3, color=block_color, edgecolor=block_color, 
               linewidth=1, label="Hyperblock Region" if j == 0 else None)
    
    # Plot the points in this hyperblock
    for point in hyperblock.points:
        point_values = point[:-1]  # Feature values
        point_class = point[-1]    # Class value
        
        # Plot the line connecting the points
        ax.plot(x, point_values, color=class_color_map[point_class], alpha=0.7, linewidth=0.8,
               label=f"Class {point_class}" if point_class not in ax.get_legend_handles_labels()[1] else None)
    
    # Set the x-axis ticks and labels
    ax.set_xticks(x)
    ax.set_xticklabels(attributes)
    
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    # Set title and labels
    if title is None:
        title = f"Hyperblock #{block_num+1} (Dominant Class: {hyperblock.dominant_class})"
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig, ax


def plot_parallel_coordinates(hyperblocks, df, class_col, title="Hyperblocks in Parallel Coordinates"):
    """
    Plot all hyperblocks together using parallel coordinates.
    
    :param hyperblocks: List of Hyperblock objects
    :param df: Original DataFrame
    :param class_col: Name of the class column
    :param title: Title for the plot
    """
    attributes = [col for col in df.columns if col != class_col]
    n_attrs = len(attributes)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set up the axes
    x = list(range(n_attrs))
    
    # Get unique classes for color mapping
    unique_classes = df[class_col].unique()
    class_colors = list(mcolors.TABLEAU_COLORS)[:len(unique_classes)]
    class_color_map = dict(zip(unique_classes, class_colors))
    
    # Generate distinct colors for each hyperblock
    block_colors = plt.cm.tab20(np.linspace(0, 1, len(hyperblocks)))
    
    # Plot each hyperblock as a distinct shaded region
    for i, hb in enumerate(hyperblocks):
        # Get the color for this hyperblock
        block_color = block_colors[i]
        class_color = class_color_map[hb.dominant_class]
        
        # Plot the min-max bounds as a shaded region
        for j in range(n_attrs-1):
            # Get min and max values for current and next attribute
            y1_min = hb.min_bounds[attributes[j]]
            y1_max = hb.max_bounds[attributes[j]]
            y2_min = hb.min_bounds[attributes[j+1]]
            y2_max = hb.max_bounds[attributes[j+1]]
            
            # Create polygon vertices for the shaded region
            xs = [x[j], x[j], x[j+1], x[j+1]]
            ys = [y1_min, y1_max, y2_max, y2_min]
            
            # Plot the polygon with distinct color and label for the first segment only
            label = f"Block {i+1} ({hb.dominant_class})" if j == 0 else None
            ax.fill(xs, ys, alpha=0.3, color=block_color, edgecolor=block_color, 
                   linewidth=1, label=label)
        
        # Plot the points in this hyperblock
        for point in hb.points:
            point_values = point[:-1]  # Feature values
            point_class = point[-1]    # Class value
            
            # Plot the line connecting the points
            ax.plot(x, point_values, color=class_color_map[point_class], alpha=0.5, linewidth=0.5)
    
    # Set the x-axis ticks and labels
    ax.set_xticks(x)
    ax.set_xticklabels(attributes)
    
    # Create legend for classes
    class_legend_elements = [Line2D([0], [0], color=color, lw=2, label=class_name) 
                           for class_name, color in class_color_map.items()]
    
    # Add both legends
    ax.legend(title="Hyperblocks", loc='upper left')
    
    # Add a second legend for classes
    second_legend = ax.legend(handles=class_legend_elements, title="Classes", loc='upper right')
    ax.add_artist(second_legend)
    
    # Set title and labels
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig, ax


def select_file():
    """
    Open a file dialog to select a CSV file.
    
    :return: Path to the selected file
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Make sure the root window is sized to 1x1 and centered
    root.geometry("1x1+{}+{}".format(
        root.winfo_screenwidth() // 2,
        root.winfo_screenheight() // 2
    ))
    
    # Make the root window appear on top of all windows
    root.attributes('-topmost', True)
    
    # Ensure root is initialized and positioned before creating the dialog
    root.update_idletasks()
    root.deiconify()
    root.focus_force()
    root.update()
    
    # Now create the file dialog
    file_path = filedialog.askopenfilename(
        title="Select a CSV file",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
        parent=root
    )
    
    # Destroy the root window
    root.destroy()
    
    return file_path


def plot_gaussian_data(df, class_col):
    """
    Plot the generated Gaussian data.
    
    :param df: DataFrame with features and class labels
    :param class_col: Name of the class column
    """
    attributes = [col for col in df.columns if col != class_col]
    
    if len(attributes) == 2:
        # For 2D data, create a scatter plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get unique classes for color mapping
        unique_classes = df[class_col].unique()
        class_colors = list(mcolors.TABLEAU_COLORS)[:len(unique_classes)]
        
        # Plot each class with a different color
        for cls in unique_classes:
            subset = df[df[class_col] == cls]
            ax.scatter(subset[attributes[0]], subset[attributes[1]], 
                      label=f'Class {cls}', alpha=0.7, 
                      edgecolors='w', linewidth=0.5)
        
        ax.set_xlabel(attributes[0])
        ax.set_ylabel(attributes[1])
        ax.set_title('Gaussian Test Data')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
    else:
        # For higher dimensions, use parallel coordinates
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get unique classes for color mapping
        unique_classes = df[class_col].unique()
        class_colors = list(mcolors.TABLEAU_COLORS)[:len(unique_classes)]
        class_color_map = dict(zip(unique_classes, class_colors))
        
        # Plot each data point
        for _, row in df.iterrows():
            x = list(range(len(attributes)))
            y = [row[attr] for attr in attributes]
            ax.plot(x, y, color=class_color_map[row[class_col]], alpha=0.5)
        
        # Set the x-axis ticks and labels
        ax.set_xticks(range(len(attributes)))
        ax.set_xticklabels(attributes)
        
        # Create legend for classes
        class_legend_elements = [Line2D([0], [0], color=color, lw=2, label=f'Class {cls}') 
                               for cls, color in class_color_map.items()]
        ax.legend(handles=class_legend_elements, title="Classes")
        
        ax.set_title('Gaussian Test Data (Parallel Coordinates)')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()


def save_tabulation_to_file(data, headers, filename, mode='w', algorithm_name=None):
    """
    Save tabulation results to a file.
    
    :param data: List of lists containing the data to tabulate
    :param headers: List of headers for the table
    :param filename: Name of the file to save to
    :param mode: File mode ('w' for write, 'a' for append)
    :param algorithm_name: Name of the algorithm to include in the output
    """
    with open(filename, mode) as f:
        if mode == 'w':
            f.write("=" * 80 + "\n\n")  # Top border
        
        # Always print algorithm name if provided, regardless of mode
        if algorithm_name is not None:
            f.write(f"\nAlgorithm: {algorithm_name}\n\n")
            
        f.write(tabulate(data, headers=headers, tablefmt='grid'))
        f.write("\n" + "=" * 80 + "\n\n")  # Bottom border and spacing


def main():
    # Ask user if they want to use generated data or load from file
    print("Do you want to use generated Gaussian test data or load from a file?")
    print("1. Generate Gaussian test data")
    print("2. Load from file")
    choice = input("Enter your choice (1 or 2): ")
    
    # Get timestamp for filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if choice == '1':
        # Generate test data with two Gaussian distributions
        n_samples = int(input("Enter number of samples (default: 100): ") or "100")
        n_features = int(input("Enter number of features (default: 2): ") or "2")
        df = generate_test_data(n_samples, n_features)
        class_col = 'class'
        dataset_name = f"gaussian_{n_samples}samples_{n_features}features"
        
        # Plot the generated data
        plot_gaussian_data(df, class_col)
    else:
        # Use file picker to select CSV file
        file_path = select_file()
        if not file_path:
            print("No file selected. Exiting.")
            return
        
        df, class_col = load_data(file_path)
        # Get dataset name from file path
        dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    
    attributes = [col for col in df.columns if col != class_col]
    
    # Create output directory if it doesn't exist
    output_dir = "tabulation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the main results file
    results_file = os.path.join(output_dir, f"{dataset_name}_results_{timestamp}.txt")
    
    print("Processing IHyper...")
    ih_blocks = ihyper(df, class_col)
    ih_data = [[i+1, hb.dominant_class, hb.num_cases, hb.num_misclassified] + 
               [f"{hb.min_bounds[a]:.1f}-{hb.max_bounds[a]:.1f}" for a in attributes]
               for i, hb in enumerate(ih_blocks)]
    
    # Save IHyper results
    save_tabulation_to_file(ih_data, ['Block #', 'Class', 'Cases', 'Misclassified'] + attributes,
                          results_file, mode='w', algorithm_name="IHyper")
    
    # Save individual IHyper block cases
    for i, hb in enumerate(ih_blocks):
        case_data = []
        for point in hb.points:
            case_values = point[:-1]  # Feature values
            case_class = point[-1]    # Class value
            case_data.append([*case_values, case_class])
        save_tabulation_to_file(case_data, attributes + [class_col],
                              results_file, mode='a', algorithm_name="IHyper")
    
    print("Processing MHyper...")
    mh_blocks = mhyper(df, class_col)
    mh_data = [[i+1, hb.dominant_class, hb.num_cases, hb.num_misclassified] + 
               [f"{hb.min_bounds[a]:.1f}-{hb.max_bounds[a]:.1f}" for a in attributes]
               for i, hb in enumerate(mh_blocks)]
    
    # Save MHyper results
    save_tabulation_to_file(mh_data, ['Block #', 'Class', 'Cases', 'Misclassified'] + attributes,
                          results_file, mode='a', algorithm_name="MHyper")
    
    # Save individual MHyper block cases
    for i, hb in enumerate(mh_blocks):
        case_data = []
        for point in hb.points:
            case_values = point[:-1]  # Feature values
            case_class = point[-1]    # Class value
            case_data.append([*case_values, case_class])
        save_tabulation_to_file(case_data, attributes + [class_col],
                              results_file, mode='a', algorithm_name="MHyper")
    
    print("Processing IMHyper...")
    imh_blocks = imhyper(df, class_col)
    imh_data = [[i+1, hb.dominant_class, hb.num_cases, hb.num_misclassified] + 
                [f"{hb.min_bounds[a]:.1f}-{hb.max_bounds[a]:.1f}" for a in attributes]
                for i, hb in enumerate(imh_blocks)]
    
    # Save IMHyper results
    save_tabulation_to_file(imh_data, ['Block #', 'Class', 'Cases', 'Misclassified'] + attributes,
                          results_file, mode='a', algorithm_name="IMHyper")
    
    # Save individual IMHyper block cases
    for i, hb in enumerate(imh_blocks):
        case_data = []
        for point in hb.points:
            case_values = point[:-1]  # Feature values
            case_class = point[-1]    # Class value
            case_data.append([*case_values, case_class])
        save_tabulation_to_file(case_data, attributes + [class_col],
                              results_file, mode='a', algorithm_name="IMHyper")
    
    print(f"\nResults have been saved to '{results_file}'")
    
    # Plot all hyperblocks together
    fig, ax = plot_parallel_coordinates(imh_blocks, df, class_col, "IMHyper Blocks in Parallel Coordinates")
    plt.show()
    
    # Plot each hyperblock individually
    for i, hb in enumerate(imh_blocks):
        fig, ax = plot_hyperblock(hb, df, class_col, i)
        plt.show()

if __name__ == "__main__":
    main()
