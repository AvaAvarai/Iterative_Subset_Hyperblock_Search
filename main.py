# This experiment visualizes datasets using an animated parallel coordinates plot.
# The visualization randomly samples 1/k of the data points in each frame, creating a dynamic view
# of the relationships between features. Each class is assigned a distinct color, and the features 
# are normalized to a 0-1 scale for better comparison. The animation updates every second, showing 
# different random samples to help identify patterns and relationships in the data across multiple views.
#
# Required packages:
# pip install pandas numpy matplotlib scikit-learn
#
# Or install from requirements.txt:
# pip install -r requirements.txt
#
# requirements.txt contents:
# pandas>=1.0.0
# numpy>=1.19.0 
# matplotlib>=3.3.0
# scikit-learn>=0.24.0

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.preprocessing import MinMaxScaler
import matplotlib.colors as mcolors
from tkinter import filedialog
from tkinter import *

def generate_color_palette(unique_classes):
    """Generate a color palette for the unique classes."""
    color_map = {}
    for cls in unique_classes:
        if str(cls).lower() == 'malignant':
            color_map[cls] = 'red'
        elif str(cls).lower() == 'benign':
            color_map[cls] = 'green'
        else:
            # Use default color map for other classes
            base_colors = list(mcolors.TABLEAU_COLORS.values())
            remaining_classes = [c for c in unique_classes if c not in color_map]
            remaining_colors = base_colors[:len(remaining_classes)]
            color_map.update(dict(zip(remaining_classes, remaining_colors)))
    return color_map

def load_and_process_data(file_path, class_column=None):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None, None, None
    
    # If class column not specified, try to identify it
    if class_column is None:
        class_col = [col for col in df.columns if col.lower() == 'class']
        class_column = class_col[0] if class_col else df.columns[-1]
    
    # Normalize features
    features = df.drop(columns=[class_column])
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(
        scaler.fit_transform(features),
        columns=features.columns
    )
    
    # Retain the class column
    df_normalized[class_column] = df[class_column]
    
    # Generate color palette for unique classes
    color_palette = generate_color_palette(df[class_column].unique())

    return df_normalized, class_column, color_palette

def update_plot(frame, df, class_col, color_palette, ax, legend_handles, visit_counts, k):
    ax.clear()
    
    # Select random 1/k of data
    sample_df = df.sample(n=len(df) // k, random_state=frame)

    # Update visit counts - increment for visited points and decay others
    visited_indices = sample_df.index
    visit_counts[visited_indices] += 1
    visit_counts[~visit_counts.index.isin(visited_indices)] *= 0.95  # Decay factor
    
    # Manually assign colors and line widths based on visit counts
    color_series = sample_df[class_col].map(color_palette)
    # Make lines thicker for more frequently visited points
    line_widths = 1 + 4 * (visit_counts[sample_df.index] / visit_counts.max())
    
    # Plot each row manually with varying line width
    for i, row in sample_df.iterrows():
        ax.plot(sample_df.columns[:-1], row[:-1], 
                color=color_series[i], 
                alpha=0.8,
                linewidth=line_widths[i])
    
    ax.set_title(f"Parallel Coordinates Plot (1/{k} Random Sample)")
    ax.set_xticks(range(len(sample_df.columns[:-1])))  # Set fixed number of ticks
    ax.set_xticklabels(sample_df.columns[:-1], rotation=45, fontsize=10)
    ax.set_ylim(0, 1)  # Set fixed y-limits since data is normalized
    ax.set_xlim(-0.5, len(sample_df.columns[:-1]) - 0.5)  # Set fixed x-limits
    ax.grid(True)
    # space out to read labels
    ax.tick_params(axis='x', which='major', pad=0)
    # make more space for the labels
    ax.set_xlabel('Features', fontsize=10)
    ax.set_ylabel('Normalized Values', fontsize=10)
    # Adjust top y margin to be smaller
    ax.set_position([0.025, 0.15, 0.95, 0.825])  # [left, bottom, width, height]
    # Add legend
    ax.legend(handles=legend_handles, loc='upper right', title="Classes")

def visualize_dataset(file_path, class_column=None, n_frames=100, interval=1000, k=3):
    # Set up the figure and animation
    fig, ax = plt.subplots(figsize=(10, 4))
    df, class_col, color_palette = load_and_process_data(file_path, class_column)
    
    if df is None:
        print("Data loading failed. Exiting.")
        return

    # Initialize visit counts for all data points
    visit_counts = pd.Series(0.0, index=df.index)

    # Create legend handles (dynamic colors)
    import matplotlib.patches as mpatches
    legend_handles = [mpatches.Patch(color=color, label=cls) 
                     for cls, color in color_palette.items()]

    ani = animation.FuncAnimation(
        fig, 
        update_plot,
        frames=range(n_frames),  
        fargs=(df, class_col, color_palette, ax, legend_handles, visit_counts, k),
        interval=interval  
    )

    plt.tight_layout()
    plt.show()

def open_file_picker():
    root = Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename()
    return file_path

if __name__ == "__main__":
    file_path = open_file_picker()
    k = int(input("Enter the value of k: "))
    visualize_dataset(file_path, k=k)