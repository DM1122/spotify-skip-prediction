# from spotify_skip_prediction.datahandler.autoencoder_data_loaders import get_autoencoder_dataloaders
# external
import numpy as np
import pandas as pd
import torch
from numpy.lib.shape_base import split
from sklearn.preprocessing import StandardScaler


def get_rnn_dataloaders(
    encoded_data, sess_length=20, feature_width=4, dataset_type="train"
):

    """
    Inputs: encoded data from the autoencoder as a 2D torch tensor with song as dim 1, features as dim 2; filepath to sesssion_lengths.csv
    Outputs: 3D numpy tensor of data split by session_id with -1 appended to end for shorter encoded_datasets
    Read with torch.load()
        dim 1: batches of listening sessions
        dim 2: each song in a session
        dim 3: features of that song
    """

    encoded_data = pd.read_csv(encoded_data, header=None)
    # scale the data
    scaler_features = StandardScaler(with_mean=False, with_std=True)

    scaler_features.fit(encoded_data)

    encoded_data = scaler_features.transform(encoded_data)

    # reshape data as specified in docstring
    encoded_data = encoded_data.reshape(1, -1)
    encoded_data = encoded_data.squeeze()
    encoded_data = encoded_data.reshape(-1, sess_length, feature_width)

    output = encoded_data

    # begin region
    """
    # snippet to save a 3d tensor with numpy from https://stackoverflow.com/a/3685339, but this is just for us humans to look at. Use torch.save on line 61 for saving to file.
    print(output)
    with open("../../data/encoded_features_"+dataset_type+".csv", 'w') as outfile:
    # I'm writing a header here just for the sake of readability
    # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# Array shape: {0}\n'.format(output.shape))
        
        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        for data_slice in output:

            # The formatting string indicates that I'm writing out
            # the values in left-justified columns 7 characters in width
            # with 2 decimal places.  
            np.savetxt(outfile, data_slice, fmt='%-7.2f')

            # Writing out a break to indicate different slices...
            outfile.write('# New slice\n')
    #end region
    """
    filename = "../../data/encoded_features_" + dataset_type + ".tensor"
    torch.save(output, filename)
    return output


# get_rnn_dataloaders("../../data/features_train.csv")
