# external
import numpy as np
import pandas as pd
import torch


def features_merger(tracklist, featureslist):
    # same as in data_merger
    old_file = pd.read_csv(tracklist)
    features = pd.read_csv(featureslist)

    to_keep = [
        "session_id",
        "session_position",
        "session_length",
        "track_id_clean",
        "skip_2",
    ]

    new_file = old_file[to_keep]

    new_file.columns = [
        "session_id",
        "session_position",
        "session_length",
        "track_id",
        "skip",
    ]

    merged = mergeLeftInOrder(new_file, features)

    return merged


def raw_to_encoder(df):
    """
    Splits the input data into a format able to be taken by the autoencoder.
    for i in range(output):
        encoded=AutoEncoder(InputSize=29, EmbedSize=?, Radius=?)
        encoded.forward(i)

    Takes input dataframe from features_merger() and checks the session id of every row.
    If it is the same as the previous, group those rows into a single dataframe and remove the first few columns about session id, session length, session number, track id.
    outputs python list where each element is a df of one listening session.
    session length determined by output[session_number].shape[0]
    """

    # prep output, temp dataframe, first session_id
    output = [pd.DataFrame([])]
    prev_session = df.loc[0, "session_id"]
    count = 0
    # iterate through the input df
    for i in range(df.shape[0]):
        curr_session = df.loc[i, "session_id"]
        # check if current row's session is the same as the previous. If so, append the useful columns
        if prev_session == curr_session:
            output[count] = output[count].append(df.loc[i, 4:])

        # if not the same session, make a new df
        else:
            count += 1
            output += [pd.DataFrame([])]
            prev_session = curr_session
    return output


def encoder_to_model(encoded_data, raw_data):

    """
    Inputs: encoded data from the autoencoder, filepath to track_list.csv
    Outpus: torch tensor of data split by session_id with -1 appended to end for shorter encoded_datasets
    """
    # prep output, temp tensors
    output = []
    temp = []

    # find lengths of each session
    sess_length = session_counter(raw_data)

    # flatten encoded input data
    encoded_data = flatten(encoded_data.to_numpy())

    # counters
    j = 0
    k = 0

    print(sess_length)
    print(encoded_data)

    # iterate through all the sessions
    for data in encoded_data:
        k += 1
        temp.append(data)
        if k == sess_length[j]:
            temp = temp + zero_list(20 - k)
            output.append(temp)
            temp = []
            k = 0
            j += 1

        # concatenate the temp tensor onto the output tensor

    print(output)
    return torch.tensor(output)


## Helper Functions
def mergeLeftInOrder(x, y, on=None):
    x = x.copy()
    x["Order"] = np.arange(len(x))
    z = x.merge(y, how="left", on=on).set_index("Order").loc[np.arange(len(x)), :]
    return z


def session_counter(file):
    df = file  # pd.read_csv(file)
    df = df.drop_duplicates(subset=["session_id"], keep="first")
    df = df[["session_length"]]
    df = df.values.tolist()
    return flatten(df)


def flatten(list):
    flat_list = []
    for sublist in list:
        for item in sublist:
            flat_list.append(item)
    return flat_list


def zero_list(length):
    out = []
    for i in range(length):
        out.append(0)
    return out
