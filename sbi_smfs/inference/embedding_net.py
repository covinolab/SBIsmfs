from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """Simple single layer CNN with ReLU activation

    Parameters
    ----------
    out_channels : int
        Number of output channels.
    kernel_size : int
        Size of the convolution kernel.
    stride : int
        Stride of the convolution.
    num_bins : int
        Number of bins for transition matrix.
    num_lags : int
        Number of lag times for which a transition matrix is generated.
    activation : torch.nn.Module
        Activation function.
    """

    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        stride: int,
        num_bins: int,
        num_lags: int,
        activation: Callable[[], nn.Module] = nn.ReLU,
    ):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=num_lags,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.num_bins = num_bins
        self.num_lags = num_lags
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check the original matrix size and lag times!!!
        x = x.view((-1, self.num_lags, self.num_bins, self.num_bins))
        x = self.activation(self.conv1(x))
        return x.flatten(start_dim=1)


class MultiLayerCNN(nn.Module):
    """Multi layer CNN with LeakyReLU activation

    Parameters
    ----------
    num_bins : int
        Number of bins for transition matrix.
    num_lags : int
        Number of lag times for which a transition matrix is generated.

    Returns
    -------
    torch.Tensor
        Flattened output of the CNN.
    """

    def __init__(
        self, num_bins: int, num_lags: int,
    ):
        super(MultiLayerCNN, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=num_lags,
                out_channels=num_lags * 16,
                kernel_size=3,
                stride=1,
            ),
            nn.LeakyReLU(0.1),
            nn.Conv2d(
                in_channels=num_lags * 16,
                out_channels=num_lags * 32,
                kernel_size=3,
                stride=2,
            ),
            nn.LeakyReLU(0.1),
            nn.Conv2d(
                in_channels=num_lags * 32,
                out_channels=num_lags * 64,
                kernel_size=3,
                stride=2,
            ),
            nn.LeakyReLU(0.1),
            nn.Conv2d(
                in_channels=num_lags * 64,
                out_channels=num_lags * 128,
                kernel_size=3,
                stride=2,
            ),
            nn.LeakyReLU(0.1),
        )

        self.num_bins = num_bins
        self.num_lags = num_lags

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check the original matrix size and lag times!!!
        x = x.view((-1, self.num_lags, self.num_bins, self.num_bins))
        x = self.cnn_layers(x)
        return x.flatten(start_dim=1)
