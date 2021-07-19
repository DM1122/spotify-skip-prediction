"""Tests for model architectures."""

# stdlib
import logging

# external
import pytest
import torch
import torchinfo

# project
from spotify_skip_prediction.core import models

LOG = logging.getLogger(__name__)


@pytest.mark.skip("Not implemented")
def test_linearparametric():
    """WIP."""


def test_autoencoder():
    """Test autoencoder model functionality."""
    model = models.AutoEncoder(input_size=32, embed_size=8, radius=4)

    data = torch.randn(32)
    LOG.info(f"Input ({data.size()}):\n{data}")

    summary = torchinfo.summary(
        model=model,
        input_data=data,
        col_names=("input_size", "output_size", "num_params"),
        verbose=0,
    )
    LOG.info(f"Model:\n{summary}")

    output = model(data)
    LOG.info(f"Output ({output.size()}):\n{output}")

    code = model.encoder(data)
    LOG.info(f"Encoding ({code.size()}):\n{code}")

    decode = model.decoder(code)
    LOG.info(f"Decoding ({decode.size()}):\n{decode}")

    assert (output == decode).all()


def test_rnn():
    """Test RNN model functionality."""
    data = torch.randn(4, 8, 16)  # (batch size, sequence length, features)
    LOG.info(f"Input ({data.size()}):\n{data}")

    LOG.info("Instantiating model")
    model = models.RNN(input_size=16, hidden_size=8, num_rnn_layers=2, output_size=1)

    summary = torchinfo.summary(
        model=model,
        input_data=data,
        col_names=("input_size", "output_size", "num_params"),
        verbose=0,
    )
    LOG.info(f"Model:\n{summary}")

    output = model(data)
    LOG.info(f"Output ({output.size()}):\n{output}")


def test_lstm():
    """Test LSTM model functionality."""
    data = torch.randn(4, 8, 16)  # (batch size, sequence length, features)
    LOG.info(f"Input ({data.size()}):\n{data}")

    LOG.info("Instantiating model")
    model = models.LSTM(input_size=16, hidden_size=8, num_rnn_layers=2, output_size=1)

    summary = torchinfo.summary(
        model=model,
        input_data=data,
        col_names=("input_size", "output_size", "num_params"),
        verbose=0,
    )
    LOG.info(f"Model:\n{summary}")

    output = model(data)
    LOG.info(f"Output ({output.size()}):\n{output}")


def test_gru():
    """Test GRU model functionality."""
    data = torch.randn(4, 8, 16)  # (batch size, sequence length, features)
    LOG.info(f"Input ({data.size()}):\n{data}")

    LOG.info("Instantiating model")
    model = models.GRU(input_size=16, hidden_size=8, num_rnn_layers=2, output_size=1)

    summary = torchinfo.summary(
        model=model,
        input_data=data,
        col_names=("input_size", "output_size", "num_params"),
        verbose=0,
    )
    LOG.info(f"Model:\n{summary}")

    output = model(data)
    LOG.info(f"Output ({output.size()}):\n{output}")
