"""Tests for gym training routines."""

# stdlib
import logging
import math

# external
import pytest
import sklearn
import torch
import torchinfo
from pmdarima import datasets as pmd
from sklearn import datasets as skd

# project
from core import gym, models
from libs import datalib

LOG = logging.getLogger(__name__)


@pytest.mark.skip(reason="WIP")
def test_trainer_regression():
    """Test trainer at regression task using Boston house-prices dataset and dense
    model."""
    device = gym.get_device()
    LOG.info(f"Using {device}")


@pytest.mark.skip(reason="WIP")
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


def test_trainer_timeseries_regression():
    """Test trainer at time-series forecasting task using MSFT stock dataset and
    rnn model."""
    device = gym.get_device()
    LOG.info(f"Using {device}")

    # region dataloading
    # datasets
    data = pmd.stocks.load_msft()
    LOG.info(f"Dataset:\n{data}")
    features = data[["Open", "High", "Low", "Close"]].values
    labels = data["Close"].shift(-1).values  # predict close one day in advance

    LOG.info(f"Features:\n{features}")
    LOG.info(f"Labels:\n{labels}")

    # reshape to samples of sequence length 7 (one week)
    features_reshaped = datalib.split_sequences(sequences=features, n_steps=7)
    labels_reshaped = datalib.split_sequences(sequences=labels, n_steps=7)

    dataset_train = torch.utils.data.TensorDataset(
        torch.tensor(features_reshaped[0:4000], dtype=torch.float),
        torch.tensor(labels_reshaped[0:4000], dtype=torch.float).unsqueeze(-1),
    )

    dataset_test = torch.utils.data.TensorDataset(
        torch.tensor(features_reshaped[4000:6000], dtype=torch.float),
        torch.tensor(labels_reshaped[4000:6000], dtype=torch.float).unsqueeze(-1),
    )

    dataset_valid = torch.utils.data.TensorDataset(
        torch.tensor(features_reshaped[6000:7977], dtype=torch.float),
        torch.tensor(labels_reshaped[6000:7977], dtype=torch.float).unsqueeze(-1),
    )

    # dataloaders
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
    model = models.RNN(input_size=4, hidden_size=8, num_rnn_layers=2, output_size=1).to(
        device
    )
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
        logname="test_time_series",
    )
    tb = trainer.train(iterations=100)
    tb.close()

    loss_valid, acc_valid = trainer.test(dataloader=dataloader_valid)
    LOG.info(
        f"Validation loss:\t{loss_valid:.3f}\tValidation acc:\t{acc_valid*100:.2f}%"
    )


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


def test_trainer_timeseries_classification():
    """Test trainer at timeseries classification task using x dataset and RNN model."""
    device = gym.get_device()
    LOG.info(f"Using {device}")


@pytest.mark.slow
def test_tuner_unsupervised_explore():
    """Test tuner exploration at unsupervised task using wine dataset and autoencoder
    model."""

    tuner = gym.Tuner_Autoencoder()
    tuner.explore(n_calls=3)


@pytest.mark.slow
def test_tuner_unsupervised_tune():
    """Test tuner tuning at unsupervised task using wine dataset and autoencoder
    model."""

    tuner = gym.Tuner_Autoencoder()
    tuner.tune(n_calls=3)
