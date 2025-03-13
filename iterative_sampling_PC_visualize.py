import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import random

def load_and_preprocess_data(file_path):
    """
    Load CSV data and preprocess numerical features with Min-Max normalization
    """
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Find the class column (case-insensitive)
    class_col = None
    for col in df.columns:
        if col.lower() == 'class':
            class_col = col
            break
    
    if class_col is None:
        raise ValueError("No 'class' column found in the dataset")
    
    # Separate features and class
    features = df.drop(columns=[class_col])
    classes = df[class_col]
    
    # Apply Min-Max normalization to numerical features
    normalized_features = (features - features.min()) / (features.max() - features.min())
    
    # Combine normalized features with original class labels
    normalized_df = pd.concat([normalized_features, classes], axis=1)
    
    return normalized_df, class_col

def parallel_coordinates_plot(data, class_column, title, filename=None):
    """
    Create a parallel coordinates plot for the given data
    """
    # Get feature names (excluding class column)
    features = [col for col in data.columns if col != class_column]
    
    # Get unique classes and assign colors
    classes = data[class_column].unique()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(classes)))
    color_dict = dict(zip(classes, colors))
    
    # Create figure and axis
    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()
    
    # Set up the axes
    x = list(range(len(features)))
    ax.set_xlim([x[0]-0.5, x[-1]+0.5])
    ax.set_ylim([0, 1])
    
    # Set the tick positions and labels
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45)
    
    # Plot each data point as a line
    for i, row in data.iterrows():
        y = [row[feature] for feature in features]
        ax.plot(x, y, color=color_dict[row[class_column]], linewidth=1, alpha=0.5)
    
    # Add a legend
    legend_elements = [plt.Line2D([0], [0], color=color_dict[cls], lw=2, label=str(cls)) 
                      for cls in classes]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Set title and labels
    plt.title(title)
    plt.tight_layout()
    
    # Save the figure if filename is provided
    if filename:
        plt.savefig(filename)
        
    plt.show()

def incremental_visualization(data, class_column, seed=42):
    """
    Incrementally add data and visualize using parallel coordinates in the same window
    """
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Get total number of samples
    total_samples = len(data)
    
    # Start with 1/3 of the data
    initial_size = int(total_samples / 3)
    
    # Calculate 5% increment size
    increment_size = int(total_samples * 0.05)
    
    # Create a random permutation of indices
    indices = np.random.permutation(total_samples)
    
    # Create a figure that will be reused
    plt.figure(figsize=(12, 6))
    
    # Start with initial subset
    current_size = initial_size
    
    # Incrementally add data and visualize (including initial subset)
    while current_size <= total_samples:
        # Get current subset of data
        current_indices = indices[:current_size]
        current_data = data.iloc[current_indices]
        
        # Clear previous plot
        plt.clf()
        ax = plt.gca()
        
        # Get feature names (excluding class column)
        features = [col for col in current_data.columns if col != class_column]
        
        # Get unique classes and assign colors
        classes = current_data[class_column].unique()
        colors = plt.cm.rainbow(np.linspace(0, 1, len(classes)))
        color_dict = dict(zip(classes, colors))
        
        # Set up the axes
        x = list(range(len(features)))
        ax.set_xlim([x[0]-0.5, x[-1]+0.5])
        ax.set_ylim([0, 1])
        
        # Set the tick positions and labels
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=45)
        
        # Plot each data point as a line
        for i, row in current_data.iterrows():
            y = [row[feature] for feature in features]
            ax.plot(x, y, color=color_dict[row[class_column]], linewidth=1, alpha=0.5)
        
        # Add a legend
        legend_elements = [plt.Line2D([0], [0], color=color_dict[cls], lw=2, label=str(cls)) 
                          for cls in classes]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Set title and labels
        title = f"Parallel Coordinates Plot - {current_size}/{total_samples} samples ({current_size/total_samples:.1%})"
        plt.title(title)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(f"parallel_coords_{current_size}.png")
        
        # Display the plot and pause to show it
        plt.draw()
        plt.pause(1)  # Pause for 1 second to show the plot
        
        # Break if we've reached the total number of samples
        if current_size == total_samples:
            break
            
        # Add 5% more data (or whatever remains if less than 5%)
        next_size = min(current_size + increment_size, total_samples)
        current_size = next_size
    
    # Keep the final plot open
    plt.show()

def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # File path to the CSV dataset
    # Replace with your actual file path
    file_path = "fisher_iris.csv"
    
    try:
        # Load and preprocess the data
        print("Loading and preprocessing data...")
        normalized_data, class_column = load_and_preprocess_data(file_path)
        
        # Run incremental visualization
        print("Starting incremental visualization...")
        incremental_visualization(normalized_data, class_column)
        
        print("Visualization complete!")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
