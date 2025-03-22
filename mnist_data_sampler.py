import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

# Load MNIST
print("Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=True)
X = mnist.data
y = mnist.target

# Number word mapping
number_words = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
}

# Prepare DataFrame
df = X.copy()
df['class'] = pd.Series(y).astype(str).map(number_words)

# Sample 100 random rows
sampled_df = df.sample(n=100, random_state=42)

# Save to CSV
output_file = "mnist_100_samples.csv"
sampled_df.to_csv(output_file, index=False)
print(f"Saved {len(sampled_df)} samples to {output_file}")
