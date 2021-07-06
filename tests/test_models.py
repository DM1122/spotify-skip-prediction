"""Tests for model architectures."""

# external
import pytest
import torch
import torchinfo

# project
from core import models


@pytest.mark.skip(reason="WIP")
def test_linearparametric():
    """WIP."""


def test_autoencoder():
    """Test autoencoder model functionality."""
    model = models.AutoEncoder(input_size=32, embed_size=8, radius=4)

    data = torch.randn(32)
    print(f"Input ({data.size()}):\n", data)

    torchinfo.summary(
        model=model,
        input_data=data,
        col_names=("input_size", "output_size", "num_params"),
    )

    output = model(data)
    print(f"Output ({output.size()}):\n", output)

    code = model.encoder(data)
    print(f"Encoding ({code.size()}):\n", code)

    decode = model.decoder(code)
    print(f"Decoding ({decode.size()}):\n", decode)

    assert (output == decode).all()


def test_rnn():
    """Test RNN model functionality."""
    model = models.RNN(input_size=16, hidden_size=8, num_rnn_layers=2, output_size=1)

    data = torch.randn(4, 8, 16)  # (batch size, sequence length, features)
    print(f"Input ({data.size()}):\n", data)

    torchinfo.summary(
        model=model,
        input_data=data,
        col_names=("input_size", "output_size", "num_params"),
    )

    output = model(data)
    print(f"Output ({output.size()}):\n", output)
