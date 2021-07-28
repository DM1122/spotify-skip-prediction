# external
import numpy as np
import pandas as pd
from spotify_skip_prediction.baseline_model import baseline
from spotify_skip_prediction.core import gym, models
#from spotify_skip_prediction.datahandler import data_loaders
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
import torch

# Baseline model
X_train, y_train, X_val, y_val, X_test, y_test = baseline.load_data()
parameters = pd.read_csv("spotify_skip_prediction/baseline_model/baseline_model_parameters.csv")
parameters = np.array(parameters.values)

baseline_model = GradientBoostingClassifier(
    n_estimators=int(parameters[0][0]),
    learning_rate=parameters[0][1],
    max_depth=int(parameters[0][2]),
    random_state=1,
)
classifier = baseline_model.fit(X_train, y_train)

# get test accuracy & confusion matrix
baseline_test_pred = baseline.get_test_accuracy(baseline_model, X_test, y_test)
baseline.create_confusion_matrix(baseline_model, X_test, y_test, baseline_test_pred, title="Baseline_Model")


# Our model
best_model = models.RNN()  #TODO:  use optimized parameters
state = torch.load("rnn_spotify")  # load trained model
best_model.load_state_dict(state)   

# TODO: Data loader for testing
_, _, dataloader_valid = data_loaders.get_rnn_dataloaders()

# Get loss and accuracy of model using test data
loss, acc = gym.Trainer.test(best_model, dataloader_valid)

# Plot confusion matrix
samples = len(dataloader_valid)
predictions = np.zeros(samples)
i = 0
with torch.no_grad():
    for inputs, labels in dataloader_valid and (i < samples):
        # forward pass
        logits = best_model(inputs)
        predictions[i] = torch.argmax(input=logits, dim=1, keepdim=False)
        i += 1
confusion_matrix(best_model, dataloader_valid, labels, predictions, title="RNN Model")

