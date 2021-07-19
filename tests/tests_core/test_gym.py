"""Tests for gym training routines."""

# stdlib
import logging
import math

# external
import numpy as np
import pandas as pd
import pytest
import sklearn
import torch
import torchinfo
from pmdarima import datasets as pmd
from sklearn import datasets as skd

# project
from spotify_skip_prediction.core import gym, models
from spotify_skip_prediction.libs import datalib, plotlib

LOG = logging.getLogger(__name__)


@pytest.mark.skip("Not implemented")
def test_trainer_regression():
    """Test trainer at regression task using Boston house-prices dataset and dense
    model."""
    device = gym.get_device()
    LOG.info(f"Using {device}")


@pytest.mark.skip("Not implemented")
def test_trainer_binary_classification():
    """Test trainer at binary classification task using breast cancer dataset and dense
    model."""
    device = gym.get_device()
    LOG.info(f"Using {device}")


def test_trainer_unsupervised():
    """Test trainer at unsupervised task using wine dataset and autoencoder model."""
    device = gym.get_device()
    LOG.info(f"Using {device}")

    # region dataloading
    # datasets
    features, _ = skd.load_wine(return_X_y=True)  # should not be scaling entire dataset

    scaler = sklearn.preprocessing.StandardScaler()
    features = scaler.fit_transform(X=features)

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(features, dtype=torch.float),
        torch.tensor(features, dtype=torch.float),
    )
    dataset_train, dataset_test, dataset_valid = torch.utils.data.random_split(
        dataset=dataset, lengths=(100, 39, 39)
    )

    # dataloaders
    dataloader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=32,
        shuffle=True,
        pin_memory=True,
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
    )
    dataloader_valid = torch.utils.data.DataLoader(
        dataset=dataset_valid,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
    )
    # endregion

    # region model definiton
    model = models.AutoEncoder(input_size=13, embed_size=4, radius=1).to(device)
    summary = torchinfo.summary(
        model=model,
        input_data=next(iter(dataloader_train))[0],
        col_names=("input_size", "output_size", "num_params"),
        verbose=0,
    )
    LOG.info(f"Model:\n{summary}")

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss(reduction="sum")
    # endregion

    trainer = gym.Trainer(
        model=model,
        dataloader_train=dataloader_train,
        dataloader_test=dataloader_test,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        logname="test_unsupervised",
    )
    tb = trainer.train(iterations=1000)
    tb.close()

    loss_valid, acc_valid = trainer.test(dataloader=dataloader_valid)
    LOG.info(
        f"Validation loss:\t{loss_valid:.3f}\tValidation acc:\t{acc_valid*100:.2f}%"
    )


@pytest.mark.plot
def test_trainer_timeseries_regression():
    """Test trainer at time-series forecasting task using MSFT stock dataset and
    rnn model."""
    device = gym.get_device()
    LOG.info(f"Using {device}")
    # region data preprocessing

    # import
    LOG.info("Importing data")
    data = pmd.stocks.load_msft()
    LOG.info(f"Data:\n{data}")

    # filter
    LOG.info("Filtering data")
    data = data[["Open", "High", "Low", "Close", "Volume"]]

    # create labels
    LOG.info("Creating labels")
    data["Close_Future"] = data["Close"].shift(-1)
    data = data[:-1]
    LOG.debug(f"Data preprocessed:\n{data}")

    # split
    LOG.info("Splitting data")
    data_train = data[0:6000]
    LOG.debug(f"Data train:\n{data_train}")
    data_test = data[6000:7000]
    LOG.debug(f"Data test:\n{data_test}")
    data_valid = data[7000:7983]
    LOG.debug(f"Data valid:\n{data_valid}")

    # features and labels
    LOG.info("Extracting features and labels")
    features_train = data_train.drop("Close_Future", axis=1)
    LOG.debug(f"Features train:\n{features_train}")
    labels_train = data_train.loc[:, ["Close_Future"]]
    LOG.debug(f"Labels train:\n{labels_train}")

    features_test = data_test.drop("Close_Future", axis=1)
    labels_test = data_test.loc[:, ["Close_Future"]]

    features_valid = data_valid.drop("Close_Future", axis=1)
    labels_valid = data_valid.loc[:, ["Close_Future"]]

    # scaling
    LOG.info("Scaling data")
    scaler_features = sklearn.preprocessing.StandardScaler(
        with_mean=False, with_std=True
    )
    scaler_labels = sklearn.preprocessing.StandardScaler(with_mean=False, with_std=True)

    scaler_features.fit(features_train)
    LOG.debug(
        f"Feature scaler stats:\n"
        f"Scale:\t{scaler_features.scale_}\nMean:\t{scaler_features.mean_}\n"
        f"Var:\t{scaler_features.var_}\nSamples:\t{scaler_features.n_samples_seen_}\n"
    )
    scaler_labels.fit(labels_train)
    LOG.debug(
        f"Label scaler stats:\n"
        f"Scale:\t{scaler_labels.scale_}\nMean:\t{scaler_labels.mean_}\n"
        f"Var:\t{scaler_labels.var_}\nSamples:\t{scaler_labels.n_samples_seen_}\n"
    )

    features_train = scaler_features.transform(features_train)
    LOG.debug(f"Features train scaled {features_train.shape}:\n{features_train}")
    labels_train = scaler_labels.transform(labels_train)
    LOG.debug(f"Labels train scaled {labels_train.shape}:\n{labels_train}")

    features_test = scaler_features.transform(features_test)
    labels_test = scaler_labels.transform(labels_test)

    features_valid = scaler_features.transform(features_valid)
    labels_valid = scaler_labels.transform(labels_valid)

    # reshaping
    LOG.info("Reshaping features and labels")
    features_train = datalib.split_sequences(sequences=features_train, n_steps=7)
    LOG.debug(f"Features train reshaped {features_train.shape}:\n{features_train}")
    labels_train = datalib.split_sequences(sequences=labels_train, n_steps=7)
    LOG.debug(f"Labels train reshaped {labels_train.shape}:\n{labels_train}")

    features_test = datalib.split_sequences(sequences=features_test, n_steps=7)
    labels_test = datalib.split_sequences(sequences=labels_test, n_steps=7)

    features_valid = datalib.split_sequences(sequences=features_valid, n_steps=7)
    labels_valid = datalib.split_sequences(sequences=labels_valid, n_steps=7)
    # endregion

    # region datasets
    LOG.info("Creating datasets")
    features_train = torch.tensor(features_train, dtype=torch.float)
    labels_train = torch.tensor(labels_train, dtype=torch.float)

    features_test = torch.tensor(features_test, dtype=torch.float)
    labels_test = torch.tensor(labels_test, dtype=torch.float)

    features_valid = torch.tensor(features_valid, dtype=torch.float)
    labels_valid = torch.tensor(labels_valid, dtype=torch.float)

    dataset_train = torch.utils.data.TensorDataset(
        features_train,
        labels_train,
    )
    LOG.debug(f"Dataset train:\n{dataset_train}")

    dataset_test = torch.utils.data.TensorDataset(
        features_test,
        labels_test,
    )

    dataset_valid = torch.utils.data.TensorDataset(
        features_valid,
        labels_valid,
    )
    # endregion

    # region dataloaders
    LOG.info("Creating dataloaders")
    dataloader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=32,
        shuffle=True,
        pin_memory=True,
    )
    LOG.debug(f"Dataloader train:\n{dataloader_train}")
    dataloader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
    )
    dataloader_valid = torch.utils.data.DataLoader(
        dataset=dataset_valid,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
    )
    # endregion

    # region model definiton
    LOG.info("Instantiating model")
    model = models.RNN(
        input_size=5, hidden_size=16, num_rnn_layers=4, output_size=1
    ).to(device)
    summary = torchinfo.summary(
        model=model,
        input_data=next(iter(dataloader_train))[0],
        col_names=("input_size", "output_size", "num_params"),
        verbose=0,
    )
    LOG.info(f"Model:\n{summary}")

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.03)
    criterion = torch.nn.MSELoss(reduction="sum")
    # endregion

    # region training
    trainer = gym.Trainer(
        model=model,
        dataloader_train=dataloader_train,
        dataloader_test=dataloader_test,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        logname="test_time_series",
    )
    tb = trainer.train(iterations=100)
    tb.close()

    loss_valid, acc_valid = trainer.test(dataloader=dataloader_valid)
    LOG.info(
        f"Validation loss:\t{loss_valid:.3f}\tValidation acc:\t{acc_valid*100:.2f}%"
    )
    # endregion

    # region inference
    LOG.info("Running inference")
    with torch.no_grad():
        logits_train = model(features_train)
        logits_test = model(features_test)
        logits_valid = model(features_valid)

    labels_train = scaler_labels.inverse_transform(labels_train)
    logits_train = scaler_labels.inverse_transform(logits_train)

    labels_test = scaler_labels.inverse_transform(labels_test)
    logits_test = scaler_labels.inverse_transform(logits_test)

    labels_valid = scaler_labels.inverse_transform(labels_valid)
    logits_valid = scaler_labels.inverse_transform(logits_valid)

    stack_train = np.stack((labels_train.flatten(), logits_train.flatten()), axis=1)
    df_train = pd.DataFrame(
        data=stack_train, index=None, columns=["Labels_Train", "Logits_Train"]
    )

    stack_test = np.stack((labels_test.flatten(), logits_test.flatten()), axis=1)
    df_test = pd.DataFrame(
        data=stack_test, index=None, columns=["Labels_Test", "Logits_Test"]
    )

    stack_valid = np.stack((labels_valid.flatten(), logits_valid.flatten()), axis=1)
    df_valid = pd.DataFrame(
        data=stack_valid, index=None, columns=["Labels_Valid", "Logits_Valid"]
    )

    plotlib.plot_series(
        df=df_train,
        title="Training Logits vs Labels",
        traces=["Labels_Train", "Logits_Train"],
        dark_mode=True,
    )

    plotlib.plot_series(
        df=df_test,
        title="Testing Logits vs Labels",
        traces=["Labels_Test", "Logits_Test"],
        dark_mode=True,
    )

    plotlib.plot_series(
        df=df_valid,
        title="Validation Logits vs Labels",
        traces=["Labels_Valid", "Logits_Valid"],
        dark_mode=True,
    )
    # endregion


def test_trainer_classification():
    """Test trainer at classification task using iris plant dataset and dense model.
    Tests saving and restoration of model once training is complete."""
    device = gym.get_device()
    LOG.info(f"Using {device}")

    # region dataloading
    # datasets
    LOG.info("Getting dataset")
    features, labels = skd.load_iris(return_X_y=True, as_frame=False)
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(features, dtype=torch.float),
        torch.tensor(labels, dtype=torch.long),
    )
    dataset_train, dataset_test, dataset_valid = torch.utils.data.random_split(
        dataset=dataset, lengths=(100, 25, 25)
    )

    # dataloaders
    LOG.info("Creating dataloaders")
    dataloader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=8,
        shuffle=True,
        pin_memory=True,
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
    )
    dataloader_valid = torch.utils.data.DataLoader(
        dataset=dataset_valid,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
    )
    # endregion

    # region model definiton
    LOG.info("Instantiating model")
    model = models.Dense(
        input_size=4, hidden_size=32, num_hidden_layers=4, output_size=3
    ).to(device)
    summary = torchinfo.summary(
        model=model,
        input_data=next(iter(dataloader_train))[0],
        col_names=("input_size", "output_size", "num_params"),
        verbose=0,
    )
    LOG.info(f"Model:\n{summary}")

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    # endregion

    trainer = gym.Trainer(
        model=model,
        dataloader_train=dataloader_train,
        dataloader_test=dataloader_test,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        logname="test_classification_save_model",
    )
    tb = trainer.train(iterations=100)
    tb.close()

    loss_valid, acc_valid = trainer.test(dataloader=dataloader_valid)
    LOG.info(
        f"Validation loss:\t{loss_valid:.3f}\tValidation acc:\t{acc_valid*100:.2f}%"
    )

    trainer.save_model("model_dense_test")

    LOG.debug("Restoring model from disk")
    model_restored = torch.load("models/model_dense_test.pt")

    trainer = gym.Trainer(
        model=model_restored,
        dataloader_train=dataloader_train,
        dataloader_test=dataloader_test,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        logname="test_classification_save_model",
    )
    loss_valid_restored, acc_valid_restored = trainer.test(dataloader=dataloader_valid)
    LOG.info(
        f"Restored validation loss:\t{loss_valid_restored:.3f}\t"
        f"Restored validation acc:\t{acc_valid_restored*100:.2f}%"
    )

    assert math.isclose(acc_valid, acc_valid_restored), (
        "Validation accuracy "
        f"({acc_valid}) and restored validation accuracy ({acc_valid_restored}) "
        "do not match"
    )


@pytest.mark.skip("Not implemented")
def test_trainer_timeseries_classification():
    """Test trainer at timeseries classification task using x dataset and RNN model."""
    device = gym.get_device()
    LOG.info(f"Using {device}")


@pytest.mark.slow
def test_tuner_unsupervised_explore():
    """Test tuner exploration at unsupervised task using wine dataset and autoencoder
    model."""

    tuner = gym.Tuner_Autoencoder_Test()
    tuner.explore(n_calls=3)


@pytest.mark.slow
def test_tuner_unsupervised_tune():
    """Test tuner tuning at unsupervised task using wine dataset and autoencoder
    model."""

    tuner = gym.Tuner_Autoencoder_Test()
    tuner.tune(n_calls=3)


@pytest.mark.slow
def test_tuner_explore_timeseries_regression():
    """Test tuner exploration at timeseries regression task using MSFT stock dataset and
    rnn model."""

    tuner = gym.Tuner_RNN_Test()
    tuner.explore(n_calls=3)
