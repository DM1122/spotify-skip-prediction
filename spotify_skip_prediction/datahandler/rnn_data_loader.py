def get_rnn_dataloaders(encoded_data, raw_data):

    """
    Inputs: encoded data from the autoencoder, filepath to track_list.csv
    Outpus: torch tensor of data split by session_id with -1 appended to end for shorter encoded_datasets
    """
    # prep output, temp tensors
    output = []
    temp = [] # holds data for a session to be appended to output

    # find lengths of each session from track_list.csv since this data was dropped when encoding occured
    # session_counter returns an array where each element is the length of a session in the encoded data.
    # e.g. [20, 20, 15, 20] would be four listening sessions (batches) of length 20, 20, 15, 20 tracks
    sess_length = session_counter(raw_data)

    # flatten encoded input data since it comes in [[0], [1], ...] format
    encoded_data = flatten(encoded_data.to_numpy())

    # counters
    j = 0 #tracks position in the session counter
    k = 0 #tracks position in a batch size of 20

    # iterate through all the sessions in encoded data
    for data in encoded_data:
        k += 1
        temp.append(data)
        if k == sess_length[j]: # append songs to temp until number of songs in that session is reached
            temp = temp + empty_pad(k) # pad -1 to keep consistent batch size
            output.append(temp) #append temp to output list
            temp = [] # reset variables
            k = 0
            j += 1

    return torch.tensor(output)


## Helper Functions
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


def empty_pad(length_in):
    length=20-length_in
    out = []
    for i in range(length):
        out.append(-1)
    return out
