# SPY SVM Classifier

This project implements an SVM (Support Vector Machines) classifier in Python to predict positive entry days for the SPY stock index. The classifier uses MinMax normalization and hyperparameter tuning via RandomizedSearchCV with a polynomial kernel.

## Features

- **Data Normalization:**  
  Applies MinMaxScaler to normalize the feature data.

- **Hyperparameter Tuning:**  
  Uses RandomizedSearchCV to explore various hyperparameter configurations for the SVM classifier.

- **Model Evaluation:**  
  Generates a classification report and a confusion matrix to assess the model's performance.

- **Support for Polynomial Kernel:**  
  The classifier uses a polynomial kernel, with tuning for parameters such as degree, C, tolerance, and coef0.

## Requirements

- Python 3.6 or higher
- Required Python packages:
  - pandas
  - numpy
  - scikit-learn
  - scipy
  - matplotlib
  - seaborn

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/runciter2078/Classification_SVM.git
   ```

2. (Optional) Rename the repository folder to `SPY_SVM_Classifier` for clarity.

3. Navigate to the project directory:

   ```bash
   cd Classification_SVM
   ```

## Usage

1. Place your CSV file (e.g., `SPYV3.csv`) in the project directory.

2. Run the script:

   ```bash
   python svm_spyv3.py
   ```

The script will:
- Load and normalize the dataset.
- Split the data into training and testing sets.
- Perform hyperparameter tuning for the SVM classifier.
- Train the final SVM model using the chosen parameters.
- Evaluate the model by printing a classification report and a confusion matrix.

## Notes

- **Hyperparameter Tuning:**  
  The hyperparameter search uses 4196 iterations by default. Adjust `n_iter_search` as needed based on your dataset size and available computational resources.

- **Parameter Example:**  
  The final model is instantiated with example parameters (e.g., degree = 11, C ≈ 56.48, tol ≈ 0.45, coef0 ≈ 0.744). Use the tuning results to choose appropriate values for your dataset.

## License

This project is distributed under the [MIT License](LICENSE).
