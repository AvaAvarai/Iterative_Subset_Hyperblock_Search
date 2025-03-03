import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class Hyperblock:
    min_bounds: np.ndarray
    max_bounds: np.ndarray
    class_label: str
    num_cases: int

class HyperblockGenerator:
    def __init__(self, purity_threshold: float = 100.0):
        self.purity_threshold = purity_threshold
        self.hyperblocks: List[Hyperblock] = []
        self.data_min = None
        self.data_max = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        # MinMax normalize the data
        self.data_min = np.min(X, axis=0)
        self.data_max = np.max(X, axis=0)
        X_normalized = (X - self.data_min) / (self.data_max - self.data_min)
        
        remaining_indices = np.arange(len(y))  # Track original dataset indices
        all_original_indices = set(remaining_indices.copy())  # Track full dataset

        while len(remaining_indices) > 0:
            best_block = self._interval_hyper(X_normalized[remaining_indices], y[remaining_indices])
            if best_block is None:
                break
            self.hyperblocks.append(best_block)
            
            # Find covered indices in global index space
            covered_global_indices = self._covered_indices(X_normalized, best_block.min_bounds, best_block.max_bounds)
            covered_indices = np.intersect1d(remaining_indices, covered_global_indices)
            
            # Remove covered points from remaining_indices
            remaining_indices = np.setdiff1d(remaining_indices, covered_indices)

        self._merger_hyper(X_normalized, y)
        self._assign_remaining_cases(X_normalized, y, all_original_indices)

    def _interval_hyper(self, X: np.ndarray, y: np.ndarray) -> Hyperblock:
        num_attributes = X.shape[1]
        best_block = None
        best_count = 0
        
        for attr_index in range(num_attributes):
            sorted_indices = np.argsort(X[:, attr_index])
            sorted_X = X[sorted_indices]
            sorted_y = y[sorted_indices]

            start_idx = 0
            while start_idx < len(sorted_y):
                end_idx = start_idx
                while end_idx < len(sorted_y) and sorted_y[end_idx] == sorted_y[start_idx]:
                    end_idx += 1

                if end_idx - start_idx > best_count:
                    min_bounds = np.min(sorted_X[start_idx:end_idx], axis=0)
                    max_bounds = np.max(sorted_X[start_idx:end_idx], axis=0)
                    best_block = Hyperblock(min_bounds, max_bounds, sorted_y[start_idx], end_idx - start_idx)
                    best_count = end_idx - start_idx

                start_idx = end_idx

        return best_block

    def _covered_indices(self, X: np.ndarray, min_bounds: np.ndarray, max_bounds: np.ndarray) -> np.ndarray:
        """Finds indices of all points in the dataset covered by the given bounds."""
        mask = np.all((X >= min_bounds) & (X <= max_bounds), axis=1)
        return np.where(mask)[0]

    def _merger_hyper(self, X: np.ndarray, y: np.ndarray):
        merged = True
        while merged:
            merged = False
            for i in range(len(self.hyperblocks)):
                for j in range(i + 1, len(self.hyperblocks)):
                    if self.hyperblocks[i].class_label == self.hyperblocks[j].class_label:
                        merged_block = self._attempt_merge(self.hyperblocks[i], self.hyperblocks[j], X, y)
                        if merged_block:
                            self.hyperblocks[i] = merged_block
                            del self.hyperblocks[j]
                            merged = True
                            break
                if merged:
                    break

    def _attempt_merge(self, hb1: Hyperblock, hb2: Hyperblock, X: np.ndarray, y: np.ndarray) -> Hyperblock:
        min_bounds = np.minimum(hb1.min_bounds, hb2.min_bounds)
        max_bounds = np.maximum(hb1.max_bounds, hb2.max_bounds)
        covered_indices = self._covered_indices(X, min_bounds, max_bounds)
        covered_classes = y[covered_indices]

        if np.all(covered_classes == hb1.class_label):
            return Hyperblock(min_bounds, max_bounds, hb1.class_label, len(covered_indices))
        return None

    def _assign_remaining_cases(self, X: np.ndarray, y: np.ndarray, all_indices: set):
        """Ensure ALL points are assigned to a hyperblock."""
        covered_indices = set()
        for hb in self.hyperblocks:
            covered_indices.update(self._covered_indices(X, hb.min_bounds, hb.max_bounds))

        uncovered_indices = list(all_indices - covered_indices)
        for idx in uncovered_indices:
            min_bounds = X[idx].copy()
            max_bounds = X[idx].copy()
            class_label = y[idx]
            self.hyperblocks.append(Hyperblock(min_bounds, max_bounds, class_label, 1))

def load_dataset(file_path: str):
    df = pd.read_csv(file_path)
    class_col = next((col for col in df.columns if col.lower() == 'class'), None)
    if class_col is None:
        raise ValueError("No 'class' column found in dataset.")

    X = df.drop(columns=[class_col]).to_numpy()
    y = df[class_col].to_numpy()
    return X, y

def main():
    file_path = 'fisher_iris_2class.csv'  # Update this path
    X, y = load_dataset(file_path)

    generator = HyperblockGenerator(purity_threshold=100.0)
    generator.fit(X, y)

    # Verify all data points are covered by hyperblocks
    all_indices = set(range(len(X)))
    
    # Normalize X for consistency with the hyperblocks
    X_normalized = (X - generator.data_min) / (generator.data_max - generator.data_min)
    
    covered_indices = set()
    for hb in generator.hyperblocks:
        covered_indices.update(generator._covered_indices(X_normalized, hb.min_bounds, hb.max_bounds))
    
    uncovered = all_indices - covered_indices
    if uncovered:
        print(f"Warning: {len(uncovered)} data points are not covered by any hyperblock")
    else:
        print("All data points are successfully allocated to hyperblocks")

    print(f"\nGenerated {len(generator.hyperblocks)} hyperblocks:")
    for i, hb in enumerate(generator.hyperblocks, 1):
        print(f"\nHyperblock {i}:")
        print(f"Class: {hb.class_label}")
        
        # Get all cases covered by this hyperblock, regardless of class
        covered_indices = generator._covered_indices(X_normalized, hb.min_bounds, hb.max_bounds)
        print(f"Cases Captured: {len(covered_indices)}")
        
        print(f"Min Bounds (normalized): {hb.min_bounds}")
        print(f"Max Bounds (normalized): {hb.max_bounds}")
        
        # Print all cases in this hyperblock
        print(f"Cases in this hyperblock:")
        for idx in covered_indices:
            print(f"  Case {idx}: {X[idx]} (normalized: {X_normalized[idx]}) - Class: {y[idx]}")

if __name__ == "__main__":
    main()