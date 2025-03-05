import pandas as pd
import numpy as np
from tabulate import tabulate

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
                dominant_class = sub_df[class_col].mode()[0]
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


def mhyper(df, class_col, impurity_threshold=0.1):
    # Create initial hyperblocks with proper column names
    hyperblocks = []
    attributes = [col for col in df.columns if col != class_col]
    
    for _, row in df.iterrows():
        min_bounds = {attr: row[attr] for attr in attributes}
        max_bounds = {attr: row[attr] for attr in attributes}
        hyperblocks.append(Hyperblock(min_bounds, max_bounds, [tuple(row)], row[class_col]))
        
    merged = True
    
    while merged:
        merged = False
        new_hyperblocks = []
        used = set()
        
        for i, hb1 in enumerate(hyperblocks):
            if i in used:
                continue
            best_hb = hb1
            for j, hb2 in enumerate(hyperblocks):
                if j in used or i == j or hb1.dominant_class != hb2.dominant_class:
                    continue
                
                min_bounds = {a: min(hb1.min_bounds[a], hb2.min_bounds[a]) for a in attributes}
                max_bounds = {a: max(hb1.max_bounds[a], hb2.max_bounds[a]) for a in attributes}
                combined_points = hb1.points + hb2.points
                
                impurity = sum(1 for p in combined_points if p[-1] != hb1.dominant_class) / len(combined_points)
                if impurity <= impurity_threshold:
                    best_hb = Hyperblock(min_bounds, max_bounds, combined_points, hb1.dominant_class)
                    merged = True
                    used.add(j)
            
            new_hyperblocks.append(best_hb)
            used.add(i)
        
        hyperblocks = new_hyperblocks
    
    return hyperblocks


def imhyper(df, class_col, purity_threshold=1.0, impurity_threshold=0.1):
    ihyper_blocks = ihyper(df, class_col, purity_threshold)
    remaining_points = df[~df.apply(tuple, axis=1).isin([tuple(p) for hb in ihyper_blocks for p in hb.points])]
    
    if not remaining_points.empty:
        mhyper_blocks = mhyper(remaining_points, class_col, impurity_threshold)
        return ihyper_blocks + mhyper_blocks
    else:
        return ihyper_blocks


def main():
    file_path = 'fisher_iris.csv'  # Replace with your CSV file
    df, class_col = load_data(file_path)
    attributes = [col for col in df.columns if col != class_col]
    
    print("IHyper Results:")
    ih_blocks = ihyper(df, class_col)
    ih_data = [[i+1, hb.dominant_class, hb.num_cases, hb.num_misclassified] + 
               [f"{hb.min_bounds[a]:.1f}-{hb.max_bounds[a]:.1f}" for a in attributes]
               for i, hb in enumerate(ih_blocks)]
    print(tabulate(ih_data, headers=['Block #', 'Class', 'Cases', 'Misclassified'] + attributes, 
                  tablefmt='grid'))
    
    print("\nMHyper Results:")
    mh_blocks = mhyper(df, class_col)
    mh_data = [[i+1, hb.dominant_class, hb.num_cases, hb.num_misclassified] + 
               [f"{hb.min_bounds[a]:.1f}-{hb.max_bounds[a]:.1f}" for a in attributes]
               for i, hb in enumerate(mh_blocks)]
    print(tabulate(mh_data, headers=['Block #', 'Class', 'Cases', 'Misclassified'] + attributes,
                  tablefmt='grid'))
    
    print("\nIMHyper Results:")
    imh_blocks = imhyper(df, class_col)
    imh_data = [[i+1, hb.dominant_class, hb.num_cases, hb.num_misclassified] + 
                [f"{hb.min_bounds[a]:.1f}-{hb.max_bounds[a]:.1f}" for a in attributes]
                for i, hb in enumerate(imh_blocks)]
    print(tabulate(imh_data, headers=['Block #', 'Class', 'Cases', 'Misclassified'] + attributes,
                  tablefmt='grid'))

if __name__ == "__main__":
    main()
