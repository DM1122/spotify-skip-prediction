"""Tests for models.py."""

# external
import numpy as np
import torch
import torchinfo

# project
from core import models


def test_autoencoder():
    """Test autoencoder functionality."""
    model = models.AutoEncoder(input_size=32, embed_size=8, radius=4)

    data = torch.Tensor(np.random.rand(32))
    print("Data:", data)

    torchinfo.summary(
        model=model,
        input_data=data,
        col_names=(
            "input_size",
            "output_size",
            "num_params",
            "kernel_size",
            "mult_adds",
        ),
        verbose=2,
    )

    output = model(data)
    print("Output:", output)
    code = model.encoder(data)
    print("Code:", code)
    decode = model.decoder(code)
    print("Decode:", decode)

    assert (output == decode).all()
