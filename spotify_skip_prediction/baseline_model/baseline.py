# stdlib
import csv
import time

# external
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import train_test_split


def load_data():
    # Load data
    X_train = pd.read_csv(
        "datahandler/trimmed_merged_no_track_id_or_session_id.csv"
    )  # load data from csv

    drop_columns = ["mode"]  # drop unimportant columns
    X_train.drop(labels=drop_columns, axis=1, inplace=True)

    y_train = X_train["skip"]
    X_train.drop(labels="skip", axis=1, inplace=True)  # ground truth from train_data

    # split data 60% train, 20% validation, 20% test
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=1
    )

    print("X_train:", len(X_train))
    print("y_train:", len(y_train))
    print("X_val:", len(X_val))
    print("y_val:", len(y_val))

    return X_train, y_train, X_val, y_val


# Hyperparameters (96 combinations)
n_estimators = [1, 10, 20, 50, 100, 200]  # 6 parameters
lr = [0.001, 0.01, 0.5, 1]  # 4 parameters
max_depth = [1, 2, 5, 10]  # 4 parameters


def baseline(n_estimators, lr, max_depth, model_num):
    # Gradient boosted trees
    baseline_model = GradientBoostingClassifier(
        n_estimators=n_estimators, learning_rate=lr, max_depth=max_depth, random_state=1
    )

    # Fit the model to our training data
    start_time = time.time()
    classifier = baseline_model.fit(X_train, y_train)
    end_time = time.time()

    # Accuracy
    train_acc = baseline_model.score(X_train, y_train)
    val_acc = baseline_model.score(X_val, y_val)

    print("Model #", model_num)
    print(
        "number of estimators: %s, learning rate: %s, max_depth: %s"
        % (n_estimators, lr, max_depth)
    )
    print("Training accuracy: {0:.3f}".format(train_acc))
    print("Validation accuracy: {0:.3f}".format(val_acc))
    print("Total time: {0:.3f} seconds".format(end_time - start_time))

    return train_acc, val_acc, baseline_model


def tune_parameters(n_estimators, lr, max_depth):
    model_num = 0
    cur_train_acc = 0
    cur_val_acc = 0
    parameters = np.zeros(3)
    for estimator in n_estimators:
        for learn_rate in lr:
            for depth in max_depth:
                model_num += 1
                train_acc, val_acc, model = baseline(
                    estimator, learn_rate, depth, model_num
                )

                if (train_acc > cur_train_acc) & (val_acc > cur_val_acc):
                    cur_train_acc = train_acc
                    cur_val_acc = val_acc
                    parameters[0] = estimator
                    parameters[1] = learn_rate
                    parameters[2] = depth

    print("Optimized parameters [n_estimators, learning_rate, max_depth]", parameters)
    print("Optimized training accuracy:", cur_train_acc)
    print("Optimized validation accuracy:", cur_val_acc)
    return cur_train_acc, cur_val_acc, parameters


# Final test accuracy
def get_test_accuracy(model, X_test, y_test):
    """
    Returns accuracy on test set.
    """
    # Make predictions - test accuracy
    test_pred = model.predict(X_test)
    score = accuracy_score(test_pred, y_test)
    print("Test Accuracy:", score)

    return test_pred


def confusion_matrix(model, X_test, y_test, test_pred):
    # Confusion matrix
    class_names = ["Skipped", "Not Skipped"]
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, test_pred))

    disp = plot_confusion_matrix(
        model,
        X_test,
        y_test,
        display_labels=class_names,
        cmap=plt.cm.Blues,
        normalize="true",
    )
    disp.ax_.set_title("Confusion Matrix (Normalized)")

    print("Confusion Matrix (Normalized)")
    print(disp.confusion_matrix)

    plt.show()


# Main code to run
X_train, y_train, X_val, y_val = load_data()
best_train_acc, best_val_acc, parameters = tune_parameters(
    n_estimators, lr, max_depth
)  # tune to get best parameters

# Save parameters to csv
headers = ["n_estimators", "lr", "max_depth"]
with open("baseline_model/baseline_model_parameters.csv", "w") as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(headers)

    # write the data
    writer.writerow(parameters)
