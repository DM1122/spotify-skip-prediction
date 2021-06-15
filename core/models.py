"""Collection of model architectures."""


# external
import numpy as np
import torch


class LinearParametric(torch.nn.Module):
    """A parametric linear architecture.

    Args:
        input_size (int): size of input to the model
        output_size (int): size of output from the model
        num_layers (int): number of hidden layers between input and output.
        Layer sizes are interpolated from input and output size

    """

    def __init__(self, input_size, output_size, num_layers):
        super().__init__()

        layer_sizes = np.linspace(
            start=input_size, stop=output_size, num=num_layers, endpoint=True, dtype=int
        )

        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    in_features=layer_sizes[i],
                    out_features=layer_sizes[i + 1],
                    bias=True,
                )
                for i in range(len(layer_sizes) - 1)
            ]
        )

    def forward(self, x):
        """Model forward pass.

        Args:
            x (torch.Tensor): input to model

        Returns:
            torch.Tensor: model output

        """
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


class AutoEncoder(torch.nn.Module):
    """An autoencoder model.

    Args:
        input_size (int): size of the input to the model
        embed_size (int): size of the embedding layer
        radius (int): specifies the number of hidden layers in the encoder and decoder,
        as measured from the embedding layer (total number of hidden
        layers = 2*(radius+1)). Hidden layer sizes are interpolated form the input and
        embedding layer sizes.

    """

    def __init__(self, input_size, embed_size, radius):
        super().__init__()

        self.encoder = LinearParametric(
            input_size=input_size, output_size=embed_size, num_layers=radius + 1
        )
        self.decoder = LinearParametric(
            input_size=embed_size, output_size=input_size, num_layers=radius + 1
        )

    def forward(self, x):
        """Model forward pass.

        Args:
            x (torch.Tensor): input to model

        Returns:
            torch.Tensor: model output

        """
        x = self.encoder(x)
        x = self.decoder(x)

        return x
