# Iterative_Subset_Hyperblock_Search

Finding representative hyperblocks by iterating on subsets of the training data for repeating patterns.

Video of picking up completely randomly 1/3 subsets of the Fisher Iris training data iteratively with replacement: <https://www.youtube.com/watch?v=HR7-bnE_b64>

## How to use

1. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

   Or install packages individually:

   ```bash
   pip install pandas numpy matplotlib scikit-learn
   ```

2. Prepare your dataset:
   - The data should be in CSV format
   - One column should contain the class labels (by default, looks for a column named "class" or uses the last column)
   - All other columns should be numeric features
   - Example datasets: breast cancer wisconsin diagnostic dataset, iris dataset

3. Run the visualization:

   ```bash
   python main.py
   ```

   By default, it looks for "fisher_iris.csv" in the current directory.

4. Custom dataset visualization:

   ```python
   from main import visualize_dataset
   
   # Basic usage with default parameters
   visualize_dataset("your_dataset.csv")
   
   # Advanced usage with custom parameters
   visualize_dataset(
       file_path="your_dataset.csv",
       class_column="your_class_column",  # Specify the class column name
       n_frames=200,                      # Number of animation frames
       interval=500                       # Milliseconds between frames
   )
   ```

5. Interpreting the visualization:
   - Each line represents a data point
   - Features are shown as vertical axes
   - Colors indicate different classes
   - Line thickness increases for frequently sampled points
   - Each frame shows a random 1/3 of the dataset
   - The animation updates every second by default

## License

This project is licensed for free and commercial use under the MIT License. See the [LICENSE](LICENSE) file for details.
