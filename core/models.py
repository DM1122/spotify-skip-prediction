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

    def __init__(
        self,
        input_size,
        output_size,
        num_layers=1,
        act_hidden=torch.relu,
        act_final=torch.relu,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.act_hidden = act_hidden
        self.act_final = act_final

        layer_sizes = list(
            np.linspace(
                start=input_size,
                stop=output_size,
                num=self.num_layers,
                endpoint=False,
                dtype=int,
            )
        )
        layer_sizes.append(output_size)

        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    in_features=layer_sizes[i], out_features=layer_sizes[i + 1]
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

        if self.num_layers == 1:
            x = self.layers[0](x)
        else:
            for layer in self.layers[:-1]:
                x = self.act_hidden(layer(x))

            if self.act_final is not None:
                x = self.act_final(self.layers[-1](x))
            else:
                x = self.layers[-1](x)

        return x


class Dense(torch.nn.Module):
    """A classic dense model architecture. Composed using three linear parametric
    modules.

    Args:
        input_size (int): Size of input
        hidden_size (int): Size of hidden layers.
        num_hidden_layers (int): Number of hidden layers.
        output_size (int): Size of model output.
    """

    def __init__(self, input_size, hidden_size, num_hidden_layers, output_size):
        super().__init__()

        self.input_layer = LinearParametric(
            input_size=input_size, output_size=hidden_size
        )

        self.hidden_layers = LinearParametric(
            input_size=hidden_size,
            output_size=hidden_size,
            num_layers=num_hidden_layers,
        )

        self.output_layer = LinearParametric(
            input_size=hidden_size, output_size=output_size
        )

    def forward(self, x):
        """Model forward pass.

        Args:
            x (torch.Tensor): input to model

        Returns:
            torch.Tensor: model output

        """
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)

        return x


class AutoEncoder(torch.nn.Module):
    """An autoencoder model.

    Args:
        input_size (int): size of the input to the model
        embed_size (int): size of the embedding layer
        radius (int): specifies the number of hidden layers in the encoder and decoder.
            Hidden layer sizes are interpolated form the input and embedding layer
            sizes.
    """

    def __init__(self, input_size, embed_size, radius):
        super().__init__()

        self.encoder = LinearParametric(
            input_size=input_size,
            output_size=embed_size,
            num_layers=radius,
            act_hidden=torch.nn.functional.elu,
            act_final=torch.nn.functional.elu,
        )
        self.decoder = LinearParametric(
            input_size=embed_size,
            output_size=input_size,
            num_layers=radius,
            act_hidden=torch.nn.functional.elu,
            act_final=torch.nn.functional.elu,
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
