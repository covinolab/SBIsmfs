from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F


EMBEDDING_NETS = {}


def add_embedding(name: str):
    """
    Decorator to register an embedding network class.

    Args:
        name (str): Name of the embedding network.
    """

    def decorator(embedding_class):
        EMBEDDING_NETS[name] = embedding_class
        return embedding_class

    return decorator


@add_embedding("single_layer_cnn")
class SimpleCNN(nn.Module):
    """
    Simple single layer CNN with ReLU activation

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


@add_embedding("multi_layer_cnn")
class MultiLayerCNN(nn.Module):
    """
    Multi layer CNN with LeakyReLU activation

    Parameters
    ----------
    num_bins : int
        Number of bins for transition matrix.
    num_lags : int
        Number of lag times for which a transition matrix is generated.
    num_features : int
        Number of output features for the embedding.
    """

    def __init__(
        self,
        num_bins: int,
        num_lags: int,
        num_features: int,
    ):
        super(MultiLayerCNN, self).__init__()

        self.num_bins = num_bins
        self.num_lags = num_lags
        self.num_features = num_features

        self.cnn = nn.Sequential(
            nn.Conv2d(num_lags, num_lags * 2, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(num_lags * 2, num_lags * 2, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(num_lags * 2, num_lags * 4, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.AvgPool2d(2),
            nn.Conv2d(num_lags * 4, num_lags * 8, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(num_lags * 8, num_lags * 16, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(num_lags * 16, num_lags * 32, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.AvgPool2d(2),
            nn.Conv2d(num_lags * 32, num_lags * 64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(num_lags * 64, num_lags * 128, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(num_lags * 128, num_lags * 128, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_lags * 128, num_lags * 128),
            nn.LeakyReLU(0.1),
            nn.Linear(num_lags * 128, num_features),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        x = x.view((-1, self.num_lags, self.num_bins, self.num_bins))
        return self.cnn(x)
