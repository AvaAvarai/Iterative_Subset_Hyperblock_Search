# This experiment visualizes the Fisher Iris dataset using an animated parallel coordinates plot.
# The visualization randomly samples 1/3 of the data points in each frame, creating a dynamic view
# of the relationships between features. Each iris class (Setosa, Versicolor, and Virginica) is
# assigned a distinct color, and the features are normalized to a 0-1 scale for better comparison.
# The animation updates every second, showing different random samples to help identify patterns
# and relationships in the data across multiple views.
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

# Define a fixed color palette for all iterations
FIXED_COLORS = {
    'Setosa': '#FF0000',      # Red
    'Versicolor': '#00FF00',  # Green  
    'Virginica': '#0000FF'    # Blue
}

def load_and_process_data(file_path):
    df = pd.read_csv(file_path)
    
    # Identify class column
    class_col = [col for col in df.columns if col.lower() == 'class'][0]
    
    # Normalize features
    features = df.drop(columns=[class_col])
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(
        scaler.fit_transform(features),
        columns=features.columns
    )
    
    # Retain the class column
    df_normalized[class_col] = df[class_col]

    return df_normalized, class_col

def update_plot(frame, df, class_col, ax, legend_handles):
    ax.clear()
    
    # Select random 1/3 of data
    sample_df = df.sample(n=len(df) // 3, random_state=frame)

    # Manually assign colors
    color_series = sample_df[class_col].map(FIXED_COLORS)
    
    # Plot each row manually
    for i, row in sample_df.iterrows():
        ax.plot(sample_df.columns[:-1], row[:-1], color=color_series[i], alpha=0.8)
    
    ax.set_title("Parallel Coordinates Plot (1/3 Random Sample)")
    ax.set_xticklabels(sample_df.columns[:-1], rotation=45)
    ax.set_ylim(0, 1)  # Set fixed y-limits since data is normalized
    ax.set_xlim(-0.5, len(sample_df.columns[:-1]) - 0.5)  # Set fixed x-limits
    ax.grid(True)

    # Add legend
    ax.legend(handles=legend_handles, loc='upper right', title="Classes")

# Set up the figure and animation
fig, ax = plt.subplots(figsize=(10, 6))
df, class_col = load_and_process_data("fisher_iris.csv")

# Create legend handles (fixed colors)
import matplotlib.patches as mpatches
legend_handles = [mpatches.Patch(color=color, label=cls) for cls, color in FIXED_COLORS.items()]

ani = animation.FuncAnimation(
    fig, 
    update_plot,
    frames=range(100),  
    fargs=(df, class_col, ax, legend_handles),
    interval=1000  
)

plt.tight_layout()
plt.show()
