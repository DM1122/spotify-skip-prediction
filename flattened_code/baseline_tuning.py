import csv
from spotify_skip_prediction.baseline_model import baseline

# Hyperparameters (96 combinations)
n_estimators = [1, 10, 20, 50, 100, 200]  # 6 parameters
lr = [0.001, 0.01, 0.5, 1]  # 4 parameters
max_depth = [1, 2, 5, 10]  # 4 parameters

# Main code to run for optimizing baseline model
X_train, y_train, X_val, y_val, _, _ = baseline.load_data()
best_train_acc, best_val_acc, parameters = baseline.tune_parameters(
    n_estimators, lr, max_depth, X_train, y_train, X_val, y_val
)  # tune to get best parameters

# Save parameters to csv
headers = ["n_estimators", "lr", "max_depth", "train acc", "val acc"]
with open("spotify_skip_prediction/baseline_model/baseline_model_parameters.csv", "w") as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(headers)

    # write the data
    writer.writerow(parameters)