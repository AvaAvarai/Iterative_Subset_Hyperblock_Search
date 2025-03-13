import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import concurrent.futures
from collections import defaultdict
import pandas as pd
import os
import matplotlib.pyplot as plt

@dataclass
class DataPoint:
    values: np.ndarray
    class_label: int

@dataclass
class HyperBlock:
    min_bounds: np.ndarray
    max_bounds: np.ndarray
    class_label: int
    accuracy: float = 100.0
    num_points: int = 0  # Added field to track number of points
    
class HyperBlockGenerator:
    def __init__(self, accuracy_threshold: float = 100.0):
        self.accuracy_threshold = accuracy_threshold
        self.hyperblocks: List[HyperBlock] = []
        
    def generate_hyperblocks(self, data: List[DataPoint]) -> List[HyperBlock]:
        """Main method to generate hyperblocks using interval-based approach followed by merging"""
        # Convert data to numpy array for faster processing
        data_array = np.array([d.values for d in data])
        labels = np.array([d.class_label for d in data])
        
        # Min-max normalize the data
        data_min = np.min(data_array, axis=0)
        data_max = np.max(data_array, axis=0)
        data_array = (data_array - data_min) / (data_max - data_min)
        
        # Step 1: Generate initial hyperblocks using interval-based approach
        initial_blocks = self._generate_interval_hyperblocks(data_array, labels)
        
        # Step 2: Merge overlapping hyperblocks
        merged_blocks = self._merge_hyperblocks(initial_blocks, data_array, labels)
        
        self.hyperblocks = merged_blocks
        return merged_blocks
    
    def _generate_interval_hyperblocks(self, data: np.ndarray, labels: np.ndarray) -> List[HyperBlock]:
        """Generate initial hyperblocks using interval-based approach"""
        n_dims = data.shape[1]
        blocks = []
        
        # Process each dimension/attribute in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_dim = {
                executor.submit(self._find_intervals, data[:, dim], labels, dim, n_dims): dim 
                for dim in range(n_dims)
            }
            
            for future in concurrent.futures.as_completed(future_to_dim):
                dim_blocks = future.result()
                blocks.extend(dim_blocks)
                
        return blocks
    
    def _find_intervals(self, dim_data: np.ndarray, labels: np.ndarray, dim: int, n_dims: int) -> List[HyperBlock]:
        """Find intervals in a single dimension that meet accuracy threshold"""
        blocks = []
        sorted_indices = np.argsort(dim_data)
        sorted_data = dim_data[sorted_indices]
        sorted_labels = labels[sorted_indices]
        
        start_idx = 0
        while start_idx < len(sorted_data):
            current_label = sorted_labels[start_idx]
            end_idx = start_idx + 1
            
            # Extend interval while maintaining class purity
            while (end_idx < len(sorted_data) and 
                   sorted_labels[end_idx] == current_label):
                end_idx += 1
                
            if end_idx - start_idx > 1:  # Minimum 2 points for an interval
                # Create hyperblock for this interval
                min_bounds = np.full(n_dims, -np.inf)
                max_bounds = np.full(n_dims, np.inf)
                min_bounds[dim] = sorted_data[start_idx]
                max_bounds[dim] = sorted_data[end_idx - 1]
                
                blocks.append(HyperBlock(
                    min_bounds=min_bounds,
                    max_bounds=max_bounds,
                    class_label=current_label,
                    num_points=end_idx - start_idx
                ))
                
            start_idx = end_idx
            
        return blocks
    
    def _merge_hyperblocks(self, blocks: List[HyperBlock], data: np.ndarray, 
                          labels: np.ndarray) -> List[HyperBlock]:
        """Merge overlapping hyperblocks while maintaining accuracy"""
        merged = blocks.copy()
        changes_made = True
        
        while changes_made:
            changes_made = False
            
            for i in range(len(merged)):
                for j in range(i + 1, len(merged)):
                    if merged[i].class_label != merged[j].class_label:
                        continue
                        
                    # Try merging blocks i and j
                    merged_min = np.maximum(merged[i].min_bounds, merged[j].min_bounds)
                    merged_max = np.minimum(merged[i].max_bounds, merged[j].max_bounds)
                    
                    # Check if merge is valid
                    if np.all(merged_min <= merged_max):
                        # Calculate accuracy and points of merged block
                        accuracy, num_points = self._calculate_accuracy_and_points(
                            merged_min, merged_max, 
                            merged[i].class_label, 
                            data, labels
                        )
                        
                        if accuracy >= self.accuracy_threshold:
                            # Create new merged block
                            new_block = HyperBlock(
                                min_bounds=merged_min,
                                max_bounds=merged_max,
                                class_label=merged[i].class_label,
                                accuracy=accuracy,
                                num_points=num_points
                            )
                            
                            # Replace blocks i and j with merged block
                            merged[i] = new_block
                            merged.pop(j)
                            changes_made = True
                            break
                            
                if changes_made:
                    break
                    
        return merged
    
    def _calculate_accuracy_and_points(self, min_bounds: np.ndarray, max_bounds: np.ndarray, 
                          class_label: int, data: np.ndarray, 
                          labels: np.ndarray) -> Tuple[float, int]:
        """Calculate accuracy and number of points in a hyperblock"""
        # Find points within the hyperblock bounds
        mask = np.all((data >= min_bounds) & (data <= max_bounds), axis=1)
        points_inside = data[mask]
        labels_inside = labels[mask]
        
        if len(points_inside) == 0:
            return 0.0, 0
            
        # Calculate accuracy
        correct = np.sum(labels_inside == class_label)
        return (correct / len(points_inside)) * 100, len(points_inside)

def visualize_hyperblocks(hyperblocks: List[HyperBlock], feature_names: List[str]):
    """Visualize hyperblocks using parallel coordinates"""
    plt.figure(figsize=(12, 6))
    
    # Create parallel coordinates
    n_dims = len(feature_names)
    dims = range(n_dims)
    
    # Plot each hyperblock
    colors = ['r', 'b', 'g', 'c', 'm', 'y']  # Add more colors if needed
    for i, block in enumerate(hyperblocks):
        color = colors[block.class_label % len(colors)]
        
        # Plot min bounds
        plt.plot(dims, block.min_bounds, color=color, alpha=0.5)
        # Plot max bounds
        plt.plot(dims, block.max_bounds, color=color, alpha=0.5)
        # Fill between bounds
        plt.fill_between(dims, block.min_bounds, block.max_bounds, color=color, alpha=0.2)

    plt.xticks(dims, feature_names, rotation=45)
    plt.ylabel('Normalized Values')
    plt.title('Hyperblocks Visualization (Parallel Coordinates)')
    plt.grid(True)
    plt.tight_layout()

def main():
    try:
        # Try to load data, first checking if file exists
        file_path = "fisher_iris_2class.csv"  # Change this to actual dataset path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
        # Load the data, keeping strings as strings
        df = pd.read_csv(file_path)
        
        # Find the class column, case-insensitive
        class_col = next((col for col in df.columns if col.lower() == 'class'), None)
        if class_col is None:
            # If no 'class' column, try the last column as a fallback
            class_col = df.columns[-1]
            print(f"Warning: No 'class' column found, using last column '{class_col}' as class labels")
        
        # Convert only the feature columns to numeric, leave class labels as strings
        attributes = [col for col in df.columns if col != class_col]
        df[attributes] = df[attributes].apply(pd.to_numeric, errors='coerce')
        
        # Check for any non-numeric data in feature columns
        if df[attributes].isna().any().any():
            raise ValueError("Non-numeric data found in feature columns after conversion")
            
        X = df[attributes].to_numpy()
        y = pd.factorize(df[class_col])[0]  # Convert string labels to numeric indices
        
        data_points = [DataPoint(values=x, class_label=y_) for x, y_ in zip(X, y)]
        
        generator = HyperBlockGenerator(accuracy_threshold=100.0)
        hyperblocks = generator.generate_hyperblocks(data_points)
        
        print(f"\nGenerated {len(hyperblocks)} hyperblocks:")
        for i, block in enumerate(hyperblocks):
            print(f"\nHyperblock {i + 1}:")
            print(f"Class: {block.class_label}")
            print(f"Number of points: {block.num_points}")
            print(f"Accuracy: {block.accuracy:.2f}%")
            print(f"Min bounds: {block.min_bounds}")
            print(f"Max bounds: {block.max_bounds}")
            
        # Visualize the hyperblocks
        visualize_hyperblocks(hyperblocks, attributes)
        plt.show()
            
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
    except ValueError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
