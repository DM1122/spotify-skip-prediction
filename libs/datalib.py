"""Data processing and manipulation functions."""

# external
import numpy as np


def split_sequences(sequences, n_steps):
    """Split a multivariate sequence into samples.

    Args:
        sequences (np.array): A multivariate numpy array. Rows for time steps and
            columns for parallel series.
        n_steps (int): Number of steps per sequence.

    Returns:
        np.array: (batch size, n_steps, parrallel series)
    """
    x = []
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        x.append(sequences[i:end_ix])
    return np.array(x)
