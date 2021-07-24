"""The set of all neccesary dataloaders."""

# stdlib
import logging
import pathlib
from io import StringIO

# external
import numpy as np
import pandas as pd
import sklearn
import torch

LOG = logging.getLogger(__name__)

# region path config
tracklist_path = pathlib.Path("../../data/track_list.csv")
features_path = pathlib.Path("../../data/track_features.csv")

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

    # import tracklist
    LOG.info(f"Importing tracklist from: {tracklist_path}")
    tracklist = pd.read_csv(
        filepath_or_buffer=tracklist_path,
        true_values=["True"],
        false_values=["False"],
    )
    LOG.info(f"Tracklist data ({type(tracklist)}):\n{tracklist}")
    info_buf = StringIO()  # call with an argument? Not sure how this works
    tracklist.info(buf=info_buf)
    LOG.info(f"Tracklist info:\n{info_buf.getvalue()}")
    data_desc = tracklist.describe(include="all")
    LOG.info(f"Tracklist stats:\n{data_desc}")

    # change column names and filter, I tried tracklist.rename(columns={'track_id_clean':'track_id'} but it didn't work
    to_keep = [
        "session_id",
        "session_position",
        "session_length",
        "track_id_clean",
        "skip_2",
    ]
    tracklist = tracklist[to_keep]
    tracklist.columns = [
        "session_id",
        "session_position",
        "session_length",
        "track_id",
        "skip",
    ]

    # import features
    LOG.info(f"Importing features from: {features_path}")
    features = pd.read_csv(
        filepath_or_buffer=features_path,
        true_values=["True"],
        false_values=["False"],
    )
    LOG.info(f"Features ({type(features)}):\n{features}")
    info_buf = StringIO()  # call with an argument? Not sure how this works
    features.info(buf=info_buf)
    LOG.info(f"Features info:\n{info_buf.getvalue()}")
    data_desc = features.describe(include="all")
    LOG.info(f"Features stats:\n{data_desc}")

    # filter
    LOG.info("Filtering data")
    features_columns_to_drop = ["mode"]
    data = data.drop(columns=features_columns_to_drop)
    LOG.info(f"Filtered ({type(data)}):\n{data}")

    # merge
    LOG.info("Merging data")

    data = mergeLeftInOrder(tracklist, features)

    # calcualte ranges to split
    data_train = 0.8
    data_test = 0.1
    data_valid = 0.1

    LOG.info(
        f"Splitting data into {data_train} for training, {data_test} for testing, {data_valid} for validation."
    )

    total_length = len(data)

    split_1 = int(total_length * data_train)
    split_2 = int(total_length * (data_train + data_test))

    # split
    data_train = data[0:split_1]
    LOG.debug(f"Data train:\n{data_train}")
    data_test = data[split_1 + 1 : split_2]
    LOG.debug(f"Data test:\n{data_test}")
    data_valid = data[split_2 + 1 : -1]
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


def mergeLeftInOrder(x, y, on=None):
    x = x.copy()
    x["Order"] = np.arange(len(x))
    z = x.merge(y, how="left", on=on).set_index("Order").loc[np.arange(len(x)), :]
    return z
