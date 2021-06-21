"""Tests for gym training routines."""

# external
import pytest
import sklearn
import torch
import torchinfo
from sklearn import datasets as skd

# project
from core import gym, models


@pytest.mark.skip(reason="WIP")
def test_trainer_regression():
    """Test trainer at regression task using Boston house-prices dataset and dense
    model."""
    device = gym.get_device()
    print(device)


def test_trainer_classification():
    """Test trainer at classification task using iris plant dataset and dense model."""
    device = gym.get_device()

    # region dataloading
    # datasets
    features, labels = skd.load_iris(return_X_y=True, as_frame=False)
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(features, dtype=torch.float),
        torch.tensor(labels, dtype=torch.long),
    )
    dataset_train, dataset_test, dataset_valid = torch.utils.data.random_split(
        dataset=dataset, lengths=(100, 25, 25)
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
    model = models.Dense(
        input_size=4, hidden_size=32, num_hidden_layers=4, output_size=3
    ).to(device)
    torchinfo.summary(
        model=model,
        input_data=next(iter(dataloader_train))[0],
        col_names=("input_size", "output_size", "num_params"),
    )

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
        logname="test_classification",
    )
    trainer.train(iterations=100)

    loss_valid, acc_valid = trainer.test(dataloader=dataloader_valid)
    print(
        f"\nValidation loss:\t{loss_valid:.3f}\tValidation acc:\t{acc_valid*100:.2f}%"
    )


@pytest.mark.skip(reason="WIP")
def test_trainer_binary_classification():
    """Test trainer at binary classification task using breast cancer dataset and dense
    model."""
    device = gym.get_device()
    print(device)


def test_trainer_unsupervised():
    """Test trainer at unsupervised task using wine dataset and autoencoder model."""
    device = gym.get_device()

    # region dataloading
    # datasets
    features, _ = skd.load_wine(return_X_y=True)

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
    torchinfo.summary(
        model=model,
        input_data=next(iter(dataloader_train))[0],
        col_names=("input_size", "output_size", "num_params"),
    )

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
    trainer.train(iterations=100)

    loss_valid, acc_valid = trainer.test(dataloader=dataloader_valid)
    print(
        f"\nValidation loss:\t{loss_valid:.3f}\tValidation acc:\t{acc_valid*100:.2f}%"
    )
