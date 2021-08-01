# from spotify_skip_prediction.datahandler.autoencoder_data_loaders import get_autoencoder_dataloaders
import logging
import numpy as np
from numpy.lib.shape_base import split
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


LOG = logging.getLogger(__name__)

def get_rnn_dataloaders(encoded_data, dataset_type, iterator='', sess_length = 20, feature_width = 4):


    """
    Inputs: encoded data from the autoencoder as a 2D torch tensor with song as dim 1, features as dim 2
    Outputs: 3D numpy tensor of data split by session_id with -1 appended to end for shorter encoded_datasets
    Read with torch.load()
        dim 1: batches of listening sessions
        dim 2: each song in a session
        dim 3: features of that song 
    """
    rnn_dir="../data/for_rnn/" # run outside env
    #rnn_dir="for_rnn/" #run inside env

    #encoded_data=pd.read_csv(encoded_data, header=None)


    #scale the data
    scaler_features = StandardScaler(with_mean=False, with_std=True)

    encoded_data=encoded_data.cpu()
    scaler_features.fit(encoded_data)

    encoded_data = scaler_features.transform(encoded_data)


    #reshape data as specified in docstring
    encoded_data = encoded_data.reshape(1, -1)
    encoded_data = encoded_data.squeeze()
    encoded_data = encoded_data.reshape(-1, sess_length, feature_width)

    output=encoded_data

    #begin region
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
    filename=rnn_dir+"encoded_features_"+dataset_type+iterator+".tensor"
    torch.save(output, filename)
    return output

def read_rnn_dataloaders(features, labels, dataset_type, iterator='', batch_size=20):
    """
    inputs:
    features: encoded data generated by get_rnn_dataloaders
    labels:   labels data generated by get_autoencoder_dataloader
    dataset_type: one of train, test, valid

    outputs:
    dataloader to be passed into rnn
    """
    rnn_dir="../data/for_rnn/" # run outside env
    encoded_dir="../data/for_encoder/"

    #rnn_dir="for_rnn/" #run inside env
    #encoded_dir="for_encoder/"

    features=rnn_dir+features+"_"+dataset_type+iterator+".tensor"
    labels=encoded_dir+labels+"_"+dataset_type+iterator+".csv"

    LOG.info(f"Features {dataset_type} set is {features}")
    LOG.info(f"Labels {dataset_type}set is {labels}")

    # read files to pandas then numpy
    features = torch.tensor((torch.load(features))).to('cuda:0')
    labels = (pd.read_csv(labels, header=None)).to_numpy()
    # region dataloaders
    LOG.info("Creating dataloaders")
    labels = torch.tensor(labels, dtype=torch.float).to('cuda:0')

    labels=labels.squeeze()
    labels=labels.squeeze()
    labels=labels.reshape(-1, batch_size, 1)
    #concatenated=torch.cat((features,labels), dim=2)
    #print(features)
    #print(concatenated)


    dataset = torch.utils.data.TensorDataset(
        features,
        labels
        )
    LOG.debug(f"Dataset {dataset_type}:\n{dataset}")
    if get_device() == torch.device("cuda:0"):
        pin=False
    else:
        pin=True

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin,
    )
    # endregion

    return dataloader

#get_rnn_dataloaders("../../data/for_encoder/features_test1.csv", "test", iterator='1')

#read_rnn_dataloaders("encoded_features_test", "labels_test", "test", iterator='1')

def get_device():
    """Gets the available device for training. Either CPU or cuda GPU.

    Returns:
        torch.device: The device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    return device