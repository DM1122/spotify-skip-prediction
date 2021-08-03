# external
import numpy as np
import pandas as pd
from torch.nn.modules import rnn
from spotify_skip_prediction.baseline_model import baseline
from flattened_code import gym, models, rnn_data_loader
from flattened_code.models import RNN
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import torch


# Baseline model
def baseline_validation():
    X_train, y_train, _, _, X_test, y_test = baseline.load_data()
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
    print("Baseline model testing complete")

#########################
#  Our model
def RNN_validation():
    best_model = RNN(
        input_size=5,
        hidden_size=16,
        num_rnn_layers=1,
        output_size=1)
    best_model = torch.load("flattened_code/models/rnn_spotify.pt", map_location=torch.device('cpu'))  # load trained model
    best_model.eval()

    # Data loader for testing
    dataloader_train = rnn_data_loader.read_rnn_dataloaders(
            features="encoded_features", 
            labels="labels", 
            dataset_type="train", 
            batch_size=1)
    dataloader_valid = rnn_data_loader.read_rnn_dataloaders(
            features="encoded_features", 
            labels="labels", 
            dataset_type="valid", 
            batch_size=1)

    optimizer = torch.optim.Adam(params=best_model.parameters(), lr=0.13832)
    criterion = torch.nn.BCELoss(reduction="sum")

    # training
    trainer = gym.Trainer(
        model=best_model,
        dataloader_train=dataloader_train,
        dataloader_test=dataloader_valid,
        optimizer=optimizer,
        criterion=criterion,
        device="cpu",
        logname="test_time_series",
    )

    # Get loss and accuracy of model using test data
    loss, acc = trainer.test(dataloader_valid)
    print("Loss: %s" % loss.item())
    print("Accuracy: %s" % acc.item())

    # Plot confusion matrix
    samples = len(dataloader_valid)
    predictions = []
    ground_truth = []

    i = 0
    with torch.no_grad():
        for inputs, labels in dataloader_valid:
            ground_truth.append(np.squeeze(np.array(labels)))
            # forward pass
            logits = best_model(inputs)
            prediction = (logits>0.5).float()
            predictions.append(np.squeeze(np.array(prediction)))
            i += 1
            if i > samples:
                break
    # transform to np arrays
    ground_truth_np = np.array(ground_truth)
    predictions_np = np.array(predictions)

    ground_truth_np = np.hstack(ground_truth_np)
    predictions_np = np.hstack(predictions_np)

    # Confusion matrix
    title="RNN_Model"

    class_names = ["Skipped", "Not Skipped"]
    print("Confusion Matrix: %s" % title)
    cm = confusion_matrix(ground_truth_np, predictions_np)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=class_names)

    disp = disp.plot(cmap=plt.cm.Blues)
    plt.title(title)

    plt.show()

    print("RNN model testing complete")

#baseline_validation()
RNN_validation()
