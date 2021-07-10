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


def encoder_to_model(data):
    """
    Calls the autoencoder and encodes the input to one value
    Outputs a 2D array of values, with -1 appended to end for shorter datasets
    """
    # prep output, temp tensors
    output = torch.Tensor()
    temp = torch.ones(1, 20)
    # iterate through all the sessions
    for i in range(len(data)):
        sess_length = len(data[i])
        # iterate through each song in each session
        for j in range(20):
            # Put in the encoded values into the tensor
            # if songs in session than 20 (the max), put in -1 to pad
            if j < sess_length:
                temp[0, j] = encoder_to_rnn_collector(data[i, j])
            else:
                temp[0, j] = -1 * torch.ones(1)

        # concatenate the temp tensor onto the output tensor
        output = torch.cat((output, temp), 0)

    return output


## Helper Functions
def mergeLeftInOrder(x, y, on=None):
    x = x.copy()
    x["Order"] = np.arange(len(x))
    z = x.merge(y, how="left", on=on).set_index("Order").loc[np.arange(len(x)), :]
    return z


# Optional Args TBD
def encoder_to_rnn_collector(data, InputSize=29, EmbedSize=10, Radius=10):
    encoded = AutoEncoder(InputSize, EmbedSize, Radius)

    return torch.Tensor(encoded.forward(data))
