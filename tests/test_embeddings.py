import torch
from sbi_smfs.inference.embedding_net import SimpleCNN


def test_simple_cnn():
    num_bins = 20
    num_lags = 5
    batch_size = 19

    data = torch.randn((batch_size, num_lags, num_bins, num_bins)).flatten(start_dim=1)
    cnn = SimpleCNN(num_lags, num_lags, 4, 2, num_bins, num_lags)
    out = cnn(data)

    assert out.shape[0] == batch_size
    assert out.shape[1] == 405
