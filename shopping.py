import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensi, speci = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensi:.2f}%")
    print(f"True Negative Rate: {100 * speci:.2f}%")


def load_data(filename):
    
    month_map = {
        "January": 0, "Jan": 0,
        "February": 1, "Feb": 1,
        "March": 2, "Mar": 2,
        "April": 3, "Apr": 3,
        "May": 4,
        "June": 5, "Jun": 5,
        "July": 6, "Jul": 6,
        "August": 7, "Aug": 7,
        "September": 8, "Sep": 8,
        "October": 9, "Oct": 9,
        "November": 10, "Nov": 10,
        "December": 11, "Dec": 11
    }

    evidence = []
    labels = []

    with open(filename, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            evidence.append([
                int(row["Administrative"]),
                float(row["Administrative_Duration"]),
                int(row["Informational"]),
                float(row["Informational_Duration"]),
                int(row["ProductRelated"]),
                float(row["ProductRelated_Duration"]),
                float(row["BounceRates"]),
                float(row["ExitRates"]),
                float(row["PageValues"]),
                float(row["SpecialDay"]),
                month_map[row["Month"]],
                int(row["OperatingSystems"]),
                int(row["Browser"]),
                int(row["Region"]),
                int(row["TrafficType"]),
                1 if row["VisitorType"] == "Returning_Visitor" else 0,
                1 if row["Weekend"] == "TRUE" else 0,
            ])
            labels.append(1 if row["Revenue"] == "TRUE" else 0)

    return (evidence, labels)


def train_model(evidence, labels):
    
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
   
    true_p = sum(1 for actual, predicted in zip(labels, predictions) if actual == 1 and predicted == 1)
    true_n = sum(1 for actual, predicted in zip(labels, predictions) if actual == 0 and predicted == 0)

    total_p = sum(1 for actual in labels if actual == 1)
    total_n = sum(1 for actual in labels if actual == 0)

    sensi = true_p / total_p if total_p else 0
    speci = true_n / total_n if total_n else 0

    return sensi, speci


if __name__ == "__main__":
    main()
