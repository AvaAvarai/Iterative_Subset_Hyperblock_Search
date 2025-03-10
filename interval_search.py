# Required packages:
# pip install pandas numpy matplotlib seaborn

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
    
    if len(df) == 0:
        return intervals
        
    for col in df.columns:
        if col == label_col:
            continue
            
        # Sort by column and get consecutive ranges
        df_sorted = df.sort_values(by=col)
        
        current_class = df_sorted[label_col].iloc[0]
        start_val = df_sorted[col].iloc[0]
        count = 1
        total_cases = len(df[df[label_col] == current_class])
        
        for i in range(1, len(df_sorted)):
            if df_sorted[label_col].iloc[i] != current_class:
                if count > 1:
                    intervals.append({
                        'attribute': col,
                        'start': start_val,
                        'end': df_sorted[col].iloc[i-1],
                        'class': current_class,
                        'count': count,
                        'total_cases': total_cases,
                        'coverage_ratio': count/total_cases,
                        'indices': df_sorted.index[i-count:i].tolist()
                    })
                current_class = df_sorted[label_col].iloc[i]
                start_val = df_sorted[col].iloc[i]
                count = 1
                total_cases = len(df[df[label_col] == current_class])
            else:
                count += 1
                
        # Handle last interval
        if count > 1:
            intervals.append({
                'attribute': col,
                'start': start_val,
                'end': df_sorted[col].iloc[-1],
                'class': current_class,
                'count': count,
                'total_cases': total_cases,
                'coverage_ratio': count/total_cases,
                'indices': df_sorted.index[-count:].tolist()
            })
    
    return intervals

def plot_parallel_coordinates(fig, df, intervals, label_col, highlight_largest=False, class_colors=None):
    """Plot parallel coordinates with pure intervals highlighted"""
    # Clear previous plot
    fig.clear()
    ax = fig.add_subplot(111)
    
    if len(df) == 0:
        ax.set_title("No data points remaining")
        fig.tight_layout()
        return
        
    # Create parallel coordinates plot with original class colors
    pd.plotting.parallel_coordinates(df, label_col, color=[class_colors[c] for c in df[label_col].unique()], ax=ax)
    
    # Find largest interval if highlighting
    largest_interval = None
    if highlight_largest and intervals:
        largest_interval = max(intervals, key=lambda x: x['coverage_ratio'])
        
        # Highlight only the cases in the largest interval
        for idx in largest_interval['indices']:
            row = df.loc[idx]
            for i in range(len(df.columns)-1):
                if i > 0:
                    ax.plot([i-1, i], [row.iloc[i-1], row.iloc[i]], color='yellow', linewidth=2, zorder=5)
    
    # Plot intervals on top of everything else
    ax.set_zorder(1)
    ax.patch.set_visible(False)
    for interval in intervals:
        y_pos = df.columns.get_loc(interval['attribute'])
        alpha = 0.8 if interval == largest_interval else 0.5
        linewidth = 8 if interval == largest_interval else 5
        color = 'yellow' if interval == largest_interval else class_colors[interval['class']]
        ax.plot(
            [y_pos, y_pos],
            [interval['start'], interval['end']],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=10  # Ensure intervals are drawn on top
        )
    
    ax.set_title("Parallel Coordinates with Pure Intervals")
    fig.tight_layout()

def create_control_window(df, label_col):
    """Create window with control buttons"""
    control_window = tk.Tk()
    control_window.title("Interval Controls")
    
    # Create figure and canvas
    fig = plt.Figure(figsize=(12, 6))
    canvas = FigureCanvasTkAgg(fig, master=control_window)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    
    # Create consistent color mapping for all classes at start
    all_classes = sorted(df[label_col].unique())  # Sort to ensure consistent order
    colors = sns.color_palette("Set2", n_colors=len(all_classes))
    class_colors = dict(zip(all_classes, [rgb2hex(c) for c in colors]))
    
    df_current = df.copy()
    intervals = find_pure_intervals(df_current, label_col)
    
    def highlight_largest():
        plot_parallel_coordinates(fig, df_current, intervals, label_col, highlight_largest=True, class_colors=class_colors)
        canvas.draw()
        
    def remove_largest():
        nonlocal df_current, intervals
        if intervals:
            largest_interval = max(intervals, key=lambda x: x['coverage_ratio'])
            df_current = df_current.drop(largest_interval['indices'])
            intervals = find_pure_intervals(df_current, label_col)
            plot_parallel_coordinates(fig, df_current, intervals, label_col, class_colors=class_colors)
            canvas.draw()
            
    def on_closing():
        control_window.quit()
        control_window.destroy()
    
    button_frame = tk.Frame(control_window)
    button_frame.pack(side=tk.BOTTOM)
    
    tk.Button(button_frame, text="Highlight Largest Interval", command=highlight_largest).pack(side=tk.LEFT)
    tk.Button(button_frame, text="Remove Largest Interval", command=remove_largest).pack(side=tk.LEFT)
    
    plot_parallel_coordinates(fig, df_current, intervals, label_col, class_colors=class_colors)
    canvas.draw()
    
    control_window.protocol("WM_DELETE_WINDOW", on_closing)
    control_window.mainloop()

def main():
    # Load data
    df = load_data()
    
    # Find label column
    label_col = next(col for col in df.columns if col.lower() == 'class')
    
    # Normalize data
    df_norm = normalize_data(df, label_col)
    
    # Create control window and show initial plot
    create_control_window(df_norm, label_col)

if __name__ == "__main__":
    main()
