"""Tests for dataloaders."""

# stdlib
import logging

# external
import pytest

# project
from spotify_skip_prediction.datahandler import data_loaders

LOG = logging.getLogger(__name__)


@pytest.mark.star
def test_get_autoencoder_dataloaders():
    """Test the dahandler's dataloader creation for autoencoder."""
    dataloader_train, _, _ = data_loaders.get_autoencoder_dataloaders(batch_size=16)

    batch_train_features = next(iter(dataloader_train))[0]
    LOG.info(
        "Features batch from train dataloader "
        f"({batch_train_features.shape}, {batch_train_features.dtype}):\n"
        f"{batch_train_features}"
    )
    batch_train_labels = next(iter(dataloader_train))[1]
    LOG.info(
        "Labels batch from train dataloader "
        f"({batch_train_labels.shape}, {batch_train_labels.dtype}):\n{batch_train_labels}"
    )
