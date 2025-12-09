import torch
import torch.nn as nn
import pytest
from sbi_smfs.inference.embedding_net import SimpleCNN, MultiLayerCNN, SingleLayerMLP

num_bins = 20
num_lags = 5
batch_size = 19
num_freq = 10


@pytest.fixture
def input_tensor():
    return torch.randn(batch_size, (num_lags * num_bins * num_bins) + num_freq)


def test_SimpleCNN(input_tensor):
    model = SimpleCNN(
        out_channels=num_lags,
        kernel_size=4,
        stride=2,
        num_bins=num_bins,
        num_lags=num_lags,
        num_freq=num_freq,
        activation=nn.ReLU,
    )
    output = model(input_tensor)
    assert output.shape == (batch_size, 410)


def test_SimpleCNN_activation(input_tensor):
    model = SimpleCNN(
        out_channels=num_lags,
        kernel_size=4,
        stride=2,
        num_bins=num_bins,
        num_lags=num_lags,
        num_freq=num_freq,
        activation=nn.ReLU,
    )
    output = model(input_tensor)
    assert torch.all(
        output >= 0
    )  # expected all outputs to be non-negative due to ReLU activation


def test_MultiLayerCNN(input_tensor):
    model = MultiLayerCNN(num_bins=num_bins, num_lags=num_lags, num_freq=num_freq, num_features=120)
    output = model(input_tensor)
    assert output.shape == (
        batch_size,
        125,
    )


def test_MultiLayerCNN_activation(input_tensor):
    model = MultiLayerCNN(num_bins=num_bins, num_lags=num_lags, num_freq=num_freq, num_features=120)
    output = model(input_tensor)
    assert (
        torch.all(output < 0) == False
    )  # expected at least one output to be positive due to LeakyReLU activation


def test_SingleLayerMLP(input_tensor):
    model = SingleLayerMLP(
        num_bins=num_bins,
        num_lags=num_lags,
        num_features=10,
        num_freq=num_freq,
        activation=nn.GELU,
    )       
    output = model(input_tensor)

    assert output .shape == (
        batch_size,
        10,
    )