import pandas as pd
import numpy as np
from itertools import permutations
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import sys


def separation_distance(candidate, others, dim, axis_value):
    diffs = np.abs(others[dim] - candidate[dim])
    crossings = (others[axis_value] - candidate[axis_value]) * (candidate[axis_value] - others[axis_value]) <= 0
    if crossings.any():
        return diffs[crossings].min()
    else:
        return 0


def process_permutation(args):
    pi, df, class_col, feature_columns, classes = args
    d = len(feature_columns)
    local_indices = {c: set() for c in classes}

    for c in classes:
        class_group = df[df[class_col] == c]
        other_group = df[df[class_col] != c]

        for j in range(d):
            fj = pi[j]
            fL = pi[max(0, j - 1)]
            fR = pi[min(d - 1, j + 1)]

            for t in ['min', 'max']:
                val = class_group[fj].min() if t == 'min' else class_group[fj].max()
                candidates = class_group[class_group[fj] == val]

                best_L_score = -1
                best_R_score = -1
                best_L_idx = None
                best_R_idx = None

                for idx, candidate in candidates.iterrows():
                    deltaL = separation_distance(candidate, other_group, fL, fj)
                    deltaR = separation_distance(candidate, other_group, fR, fj)

                    if deltaL > best_L_score:
                        best_L_score = deltaL
                        best_L_idx = idx
                    if deltaR > best_R_score:
                        best_R_score = deltaR
                        best_R_idx = idx

                if best_L_idx is not None:
                    local_indices[c].add(best_L_idx)
                if best_R_idx is not None:
                    local_indices[c].add(best_R_idx)

    return local_indices


def extract_envelope_cases(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    class_col = next((col for col in df.columns if col.lower() == 'class'), None)
    if class_col is None:
        raise ValueError("No column named 'class' found (case-insensitive).")

    feature_columns = [col for col in df.columns if col != class_col]
    classes = df[class_col].unique()

    scaler = MinMaxScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])

    dimension_permutations = list(permutations(feature_columns))
    args_list = [(pi, df, class_col, feature_columns, classes) for pi in dimension_permutations]

    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(process_permutation, args_list), total=len(args_list), desc="Processing permutations"))

    envelope_indices = {c: set() for c in classes}
    for result in results:
        for c in result:
            envelope_indices[c].update(result[c])

    selected_indices = set()
    for idx_set in envelope_indices.values():
        selected_indices.update(idx_set)

    filtered_df = df.loc[sorted(selected_indices)]
    filtered_df.to_csv(output_csv, index=False)
    print(f"Saved {len(filtered_df)} envelope cases to {output_csv}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python filter.py input.csv output.csv")
        sys.exit(1)
    extract_envelope_cases(sys.argv[1], sys.argv[2])