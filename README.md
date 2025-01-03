# Shopping Prediction Classifier

## Overview

This project implements a K-Nearest Neighbors (KNN) classifier to predict whether a user intends to make a purchase based on their browsing session data. The classifier processes information such as the number of pages visited, the duration spent on different pages, and other session characteristics to predict user behavior.

## Features

- **Data Loading**: Parses user session data from a CSV file.
- **Data Preprocessing**: Converts categorical data into numerical representations for compatibility with machine learning algorithms.
- **Training and Prediction**: Implements a KNN classifier to predict purchase intent.
- **Evaluation**: Measures performance using sensitivity (true positive rate) and specificity (true negative rate).

## How It Works

### Input

The input dataset `shopping.csv`.

### Output

The program outputs:

- **Correct Predictions**: Number of correctly classified sessions.
- **Incorrect Predictions**: Number of incorrectly classified sessions.
- **True Positive Rate (Sensitivity)**: Percentage of purchasers correctly identified.
- **True Negative Rate (Specificity)**: Percentage of non-purchasers correctly identified.

## Usage

### Prerequisites

Ensure the following are installed:

- Python 3.x
- Required libraries: `scikit-learn`

Install dependencies using:

```bash
pip install scikit-learn
```

### Running the Program

Run the script with the input CSV file:

```bash
python shopping.py shopping.csv
```

### Example Output

```text
Correct: 4075
Incorrect: 857
True Positive Rate: 37.17%
True Negative Rate: 90.95%
```

## Code Structure

- **`main`**: Manages the overall flow of the program.
- **`load_data(filename)`**: Loads and preprocesses the CSV data into evidence and labels.
- **`train_model(evidence, labels)`**: Trains a KNN classifier using the provided evidence and labels.
- **`evaluate(labels, predictions)`**: Evaluates the model's performance in terms of sensitivity and specificity.

## Evaluation Metrics

- **Sensitivity (True Positive Rate)**:
  Measures the proportion of actual positive cases (purchasers) correctly identified.

  Formula:

  ```
  Sensitivity = True Positives / (True Positives + False Negatives)
  ```

- **Specificity (True Negative Rate)**:
  Measures the proportion of actual negative cases (non-purchasers) correctly identified.

  Formula:

  ```
  Specificity = True Negatives / (True Negatives + False Positives)
  ```

## Improvements

To enhance performance:

1. Address class imbalance by oversampling the minority class or undersampling the majority class.
2. Experiment with different `k` values and distance metrics for KNN.
3. Explore alternative models such as decision trees or logistic regression.

## Dependencies

- Python 3.x
- Libraries:
  - `csv`
  - `sys`
  - `scikit-learn`

