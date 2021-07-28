"""The set of all neccesary dataloaders."""

# stdlib
import logging
import pathlib
from io import StringIO
from os import read

# external
import numpy as np
from numpy.lib.shape_base import split
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

LOG = logging.getLogger(__name__)
 
# region path config venv
# tracklist_path = pathlib.Path("data/track_list.csv")
# features_path = pathlib.Path("data/track_features.csv")
# sample_data_path = pathlib.Path("data/trimmed_merged_no_track_id_or_session_id.csv")
# endregion

# region path config local test
#tracklist_path = "../../data/track_list.csv"
#features_path = "../../data/track_features.csv"
# sample_data_path = "../../data/trimmed_merged_no_track_id_or_session_id.csv"
# endregion

# region path config local real data
tracklist_path = "../../data/log_0_20180715_000000000000.csv"
features_path_1 = "../../data/tf_000000000000.csv"
features_path_2 = "../../data/tf_000000000001.csv"
#tracklist_path = "../../data/track_list.csv"
#features_path_1 = "../../data/track_features.csv"
#features_path_2 = "../../data/empty.csv"


def get_autoencoder_dataloaders(batch_size):
    """Builds dataloaders for autoencoder model using Pytorch Tensor dataloader.

    Features and labels are the same for the purposes of autoencoder reconstruction.

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
    LOG.info(f"Importing features from: {features_path_1} and {features_path_2}")
    features_1 = pd.read_csv(
        filepath_or_buffer=features_path_1,
        true_values=["True"],
        false_values=["False"],
    )
    features_2 = pd.read_csv(
        filepath_or_buffer=features_path_2,
        true_values=["True"],
        false_values=["False"],
    )
    features = pd.concat([features_1, features_2])

    LOG.info(f"Features ({type(features)}):\n{features}")
    info_buf = StringIO()  # call with an argument? Not sure how this works
    features.info(buf=info_buf)
    LOG.info(f"Features info:\n{info_buf.getvalue()}")
    data_desc = features.describe(include="all")
    LOG.info(f"Features stats:\n{data_desc}")

    # filter
    LOG.info("Filtering data")
    features_columns_to_drop = ["mode"]
    features = features.drop(columns=features_columns_to_drop)
    LOG.info(f"Filtered ({type(features)}):\n{features}")



    # merge
    LOG.info("Merging data")

    data = mergeLeftInOrder(tracklist, features)

    # remove listening sessions which are not 20 songs long
    data=data[np.any(data == 20, axis = 1)]

    data = data.drop(
        columns=[
            "session_position",
            "session_id",
            "session_position",
            "session_length",
            "track_id",
        ]
    )  # these can't be scaled
    LOG.info(f"TEST:\n{data}")

    # calcualte ranges to split
    data_train = 0.8
    data_test = 0.1
    data_valid = 0.1

    LOG.info(
        f"Splitting data into {data_train} for training, {data_test} for testing, {data_valid} for validation."
    )

    total_length = len(data)

    split_index_1 = int(round_to_20(total_length * data_train))
    split_index_2 = int(round_to_20(total_length * (data_train + data_test)))
    

    # split
    data_train = data[0:split_index_1]
    LOG.debug(f"Data train:\n{data_train}")
    data_test = data[split_index_1  : split_index_2-1]
    LOG.debug(f"Data test:\n{data_test}")
    data_valid = data[split_index_2  : -1]
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
    scaler_features = StandardScaler(with_mean=False, with_std=True)

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

    # save files to disc
    np.savetxt("../../data/features_train.csv", features_train, delimiter=",")
    np.savetxt("../../data/labels_train.csv", labels_train, fmt="%.0d", delimiter=",")
    np.savetxt("../../data/features_valid.csv", features_valid, delimiter=",")
    np.savetxt("../../data/labels_valid.csv", labels_valid, fmt="%.0d", delimiter=",")
    np.savetxt("../../data/features_test.csv", features_test, delimiter=",")
    np.savetxt("../../data/labels_test.csv", labels_test, fmt="%.0d", delimiter=",")

    LOG.info(f"Features train saved at ../../data/features_train.csv")
    LOG.info(f"Labels train saved at ../../data/labels_train.csv")
    LOG.info(f"Features validation saved at ../../data/features_valid.csv")
    LOG.info(f"Labels validation saved at ../../data/labels_valid.csv")
    LOG.info(f"Features testing saved at ../../data/features_test.csv")
    LOG.info(f"Labels testing saved at ../../data/labels_test.csv")
    

    #features_train.to_csv ("../../data/let_me_sleeptrain.csv", sep=",")
    #features_test.to_csv ("../../data/let_me_sleeptest.csv", sep=",")
    #features_valid.to_csv ("../../data/let_me_sleepvalid.csv", sep=",")

    return True


def read_autoencoder_dataloaders(
    batch_size,
    features_train_csv="../../data/features_train.csv",
    labels_train_csv="../../data/labels_train.csv",
    features_valid_csv="../../data/features_valid.csv",
    labels_valid_csv="../../data/labels_valid.csv",
    features_test_csv="../../data/features_test.csv",
    labels_test_csv="../../data/labels_test.csv",
):

    LOG.info(f"Features training set is {features_train_csv}")
    LOG.info(f"Labels training set is {labels_train_csv}")
    LOG.info(f"Features validation set is {features_valid_csv}")
    LOG.info(f"Labels validation set is {labels_valid_csv}")
    LOG.info(f"Features testing set is {features_test_csv}")
    LOG.info(f"Labels testing set is {labels_test_csv}")

    # read files to pandas then numpy
    features_train = (pd.read_csv(features_train_csv)).to_numpy()
    features_test = (pd.read_csv(features_test_csv)).to_numpy()
    features_valid = (pd.read_csv(features_valid_csv)).to_numpy()
    labels_train = (pd.read_csv(labels_train_csv)).to_numpy()
    labels_test = (pd.read_csv(labels_test_csv)).to_numpy()
    labels_valid = (pd.read_csv(labels_valid_csv)).to_numpy()

    LOG.info(
        "Labels train numpy "
        f"({labels_train.shape}, {labels_train.dtype}):\n{labels_train}"
    )

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

'''
def get_autoencoder_dataloaders_no_split(batch_size):
    """Builds dataloaders for autoencoder model using Pytorch Tensor dataloader.

    Features and labels are the same for the purposes of autoencoder reconstruction.

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
    features = features.drop(columns=features_columns_to_drop)
    LOG.info(f"Filtered ({type(features)}):\n{features}")

    # merge
    LOG.info("Merging data")

    data = mergeLeftInOrder(tracklist, features)
    data = data.drop(
        columns=[
            "session_position",
            "session_id",
            "session_position",
            "session_length",
            "track_id",
        ]
    )  # these can't be scaled
    LOG.info(f"TEST:\n{data}")

    # features and labels
    LOG.info("Extracting features and labels")
    features_train = data.drop("skip", axis=1)
    LOG.debug(f"Features train:\n{features_train}")
    labels_train = data.loc[:, ["skip"]]
    LOG.debug(f"Labels train:\n{labels_train}")

    # scaling
    LOG.info("Scaling data")
    scaler_features = StandardScaler(with_mean=False, with_std=True)

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

    # numpy
    LOG.info("Converting data to numpy array")
    labels_train = labels_train.to_numpy()
    LOG.info(
        "Labels train numpy "
        f"({labels_train.shape}, {labels_train.dtype}):\n{labels_train}"
    )

    # region datasets
    LOG.info("Creating datasets")

    features_train = torch.tensor(features_train, dtype=torch.float)
    labels_train = torch.tensor(labels_train, dtype=torch.float)

    dataset_train = torch.utils.data.TensorDataset(  # only care about features here
        features_train,
        features_train,
    )
    LOG.debug(f"Dataset train:\n{dataset_train}")
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
    # endregion

    return dataloader_train
'''

'''
def get_rnn_dataloaders(batch_size):
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
    LOG.info(f"Importing data from: {sample_data_path}")
    data = pd.read_csv(
        filepath_or_buffer=sample_data_path,
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
    scaler_features = StandardScaler(with_mean=False, with_std=True)

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

'''

def mergeLeftInOrder(x, y, on=None):
    x = x.copy()
    x["Order"] = np.arange(len(x))
    z = x.merge(y, how="left", on=on).set_index("Order").loc[np.arange(len(x)), :]
    return z

def session_counter(df):
    df = df.drop_duplicates(subset=["session_id"], keep="first")
    df = df[["session_length"]]
    df = df.values.tolist()
    return pd.DataFrame(flatten(df))

def flatten(list):
    flat_list = []
    for sublist in list:
        for item in sublist:
            flat_list.append(item)
    return flat_list

def round_to_20(x, base=20):
    return base * round(x/base)

get_autoencoder_dataloaders(20)