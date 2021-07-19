# external
import numpy as np
import pandas as pd
from baseline_model.baseline import *
from core.gym import *
from core.models import *
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix

# Get test data
X_test = pd.read_csv("datahandler/test_data.csv")  # placeholder for test data
y_test = X_test["skip"]  # ground truth
X_test.drop(labels="skip", axis=1, inplace=True)

# Baseline model
X_train, y_train, X_val, y_val = load_data()
parameters = pd.read_csv("baseline_model/baseline_model_parameters.csv")

baseline_model = GradientBoostingClassifier(
    n_estimators=int(parameters[0]),
    learning_rate=parameters[1],
    max_depth=int(parameters[2]),
    random_state=1,
)
classifier = baseline_model.fit(X_train, y_train)

# get test accuracy & confusion matrix
baseline_test_pred = get_test_accuracy(baseline_model, X_test, y_test)
confusion_matrix(baseline_model, X_test, y_test, baseline_test_pred)


# Our model
best_model = RNN()  # use optimized parameters
state = torch.load("best_model")  # placeholder for trained model
best_model.load_state_dict(state)

# Data loader for testing
_, _, test_loader = Tuner_RNN_Test._get_dataloaders()

# Get loss and accuracy of model using test data
loss, acc = Trainer.test(best_model, test_loader)

# Plot confusion matrix
confusion_matrix(baseline_model, X_test, y_test, baseline_test_pred)
