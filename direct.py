import pandas as pd
import numpy as np
from tabulate import tabulate
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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
    
    # Print all cases for each IHyper block
    for i, hb in enumerate(ih_blocks):
        print(f"\nIHyper Block #{i+1} Cases (Dominant Class: {hb.dominant_class}):")
        case_data = []
        for point in hb.points:
            case_values = point[:-1]  # Feature values
            case_class = point[-1]    # Class value
            case_data.append([*case_values, case_class])
        print(tabulate(case_data, headers=attributes + [class_col], tablefmt='grid'))
    
    print("\nMHyper Results:")
    mh_blocks = mhyper(df, class_col)
    mh_data = [[i+1, hb.dominant_class, hb.num_cases, hb.num_misclassified] + 
               [f"{hb.min_bounds[a]:.1f}-{hb.max_bounds[a]:.1f}" for a in attributes]
               for i, hb in enumerate(mh_blocks)]
    print(tabulate(mh_data, headers=['Block #', 'Class', 'Cases', 'Misclassified'] + attributes,
                  tablefmt='grid'))
    
    # Print all cases for each MHyper block
    for i, hb in enumerate(mh_blocks):
        print(f"\nMHyper Block #{i+1} Cases (Dominant Class: {hb.dominant_class}):")
        case_data = []
        for point in hb.points:
            case_values = point[:-1]  # Feature values
            case_class = point[-1]    # Class value
            case_data.append([*case_values, case_class])
        print(tabulate(case_data, headers=attributes + [class_col], tablefmt='grid'))
    
    print("\nIMHyper Results:")
    imh_blocks = imhyper(df, class_col)
    imh_data = [[i+1, hb.dominant_class, hb.num_cases, hb.num_misclassified] + 
                [f"{hb.min_bounds[a]:.1f}-{hb.max_bounds[a]:.1f}" for a in attributes]
                for i, hb in enumerate(imh_blocks)]
    print(tabulate(imh_data, headers=['Block #', 'Class', 'Cases', 'Misclassified'] + attributes,
                  tablefmt='grid'))
    
    # Print all cases for each IMHyper block
    for i, hb in enumerate(imh_blocks):
        print(f"\nIMHyper Block #{i+1} Cases (Dominant Class: {hb.dominant_class}):")
        case_data = []
        for point in hb.points:
            case_values = point[:-1]  # Feature values
            case_class = point[-1]    # Class value
            case_data.append([*case_values, case_class])
        print(tabulate(case_data, headers=attributes + [class_col], tablefmt='grid'))

if __name__ == "__main__":
    main()
