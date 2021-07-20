"""The set of all neccesary dataloaders."""

# stdlib
import logging
import pathlib
from io import StringIO

# external
import pandas as pd
import sklearn
import torch

LOG = logging.getLogger(__name__)

# region path config
data_path = pathlib.Path("data/trimmed_merged_no_track_id_or_session_id.csv")
# endregion


def get_autoencoder_dataloaders(batch_size):
    """Builds dataloaders for autoencoder model using Pytorch Tensor dataloader.

    Features and labels are the same for the purposes of autoencoder reconstruction.
    Does not shuffle data prior to split.

    Args:
        batch_size (int): Batch size for training dataloader.

    Returns:
        tuple: A tuple of torch.utils.data.dataloader.DataLoader objects consisting of
            (train dataloader, test dataloader, validation dataloader)
    """

    # import
    LOG.info(f"Importing data from: {data_path}")
    data = pd.read_csv(
        filepath_or_buffer=data_path,
        true_values=["True"],
        false_values=["False"],
    )
    LOG.info(f"Data ({type(data)}):\n{data}")
    info_buf = StringIO()
    data.info(buf=info_buf)
    LOG.info(f"Data info:\n{info_buf.getvalue()}")
    data_desc = data.describe(include="all")
    LOG.info(f"Data stats:\n{data_desc}")

    # filter
    LOG.info("Filtering data")
    data = data.drop(columns=["session_position", "session_length", "mode"])
    LOG.info(f"Data filtered ({type(data)}):\n{data}")

    # split
    LOG.info("Splitting data")
    data_train = data[0:134304]  # 80%
    LOG.debug(f"Data train:\n{data_train}")
    data_test = data[134304:151092]  # 10%
    LOG.debug(f"Data test:\n{data_test}")
    data_valid = data[151092:167880]  # 10%
    LOG.debug(f"Data valid:\n{data_valid}")

    # features and labels
    LOG.info("Extracting features and labels")
    features_train = data_train.drop("skip", axis=1)
    LOG.debug(f"Features train:\n{features_train}")
    labels_train = data_train.loc[:, ["skip"]]
    LOG.debug(f"Labels train:\n{labels_train}")

    features_test = data_test.drop("skip", axis=1)
    labels_test = data_test.loc[:, ["skip"]]

    features_valid = data_valid.drop("skip", axis=1)
    labels_valid = data_valid.loc[:, ["skip"]]

    # scaling
    LOG.info("Scaling data")
    scaler_features = sklearn.preprocessing.StandardScaler(
        with_mean=False, with_std=True
    )

    scaler_features.fit(
        features_train
    )  # only fit to train dataset, do not scale labels
    LOG.debug(
        f"Feature scaler stats:\n"
        f"Scale:\t{scaler_features.scale_}\nMean:\t{scaler_features.mean_}\n"
        f"Var:\t{scaler_features.var_}\n"
        f"Samples:\t{scaler_features.n_samples_seen_}\n"
    )

    features_train = scaler_features.transform(features_train)
    LOG.debug(f"Features train scaled {features_train.shape}:\n{features_train}")

    features_test = scaler_features.transform(features_test)

    features_valid = scaler_features.transform(features_valid)

    # numpy
    LOG.info("Converting data to numpy array")
    labels_train = labels_train.to_numpy()
    LOG.info(
        "Labels train numpy "
        f"({labels_train.shape}, {labels_train.dtype}):\n{labels_train}"
    )
    labels_test = labels_test.to_numpy()
    labels_valid = labels_valid.to_numpy()

    # region datasets
    LOG.info("Creating datasets")

    features_train = torch.tensor(features_train, dtype=torch.float)
    labels_train = torch.tensor(labels_train, dtype=torch.float)

    features_test = torch.tensor(features_test, dtype=torch.float)
    labels_test = torch.tensor(labels_test, dtype=torch.float)

    features_valid = torch.tensor(features_valid, dtype=torch.float)
    labels_valid = torch.tensor(labels_valid, dtype=torch.float)

    dataset_train = torch.utils.data.TensorDataset(  # only care about features here
        features_train,
        features_train,
    )
    LOG.debug(f"Dataset train:\n{dataset_train}")

    dataset_test = torch.utils.data.TensorDataset(
        features_test,
        features_test,
    )

    dataset_valid = torch.utils.data.TensorDataset(
        features_valid,
        features_valid,
    )
    # endregion

    # region dataloaders
    LOG.info("Creating dataloaders")
    dataloader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
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

    return dataloader_train, dataloader_test, dataloader_valid
