import torch
from sbi_smfs.inference import (
    train_armortized_posterior,
    train_sequential_posterior,
    train_truncated_posterior,
)
from sbi.inference.posteriors import DirectPosterior


def test_armortized_training():
    data_set_size = 10
    test_data = (torch.randn((10, 13)), torch.randn((10, 2400)))
    test_config = "tests/config_files/test.config"
    posterior = train_armortized_posterior(test_config, test_data)
    assert isinstance(posterior, DirectPosterior)


def test_sequential_training():
    test_config = "tests/config_files/test.config"
    test_observation = torch.randn(2400)
    posterior = train_sequential_posterior(test_config, 2, 50, 1, test_observation)
    assert isinstance(posterior, DirectPosterior)


def test_sequential_training_with_pdd():
    test_config = "tests/config_files/test_pdd.config"
    test_observation = torch.randn(2400)
    posterior = train_sequential_posterior(test_config, 2, 50, 1, test_observation)
    assert isinstance(posterior, DirectPosterior)


def test_sequential_training_with_Dx():
    test_config = "tests/config_files/test_2.config"
    test_observation = torch.randn(2400)
    posterior = train_sequential_posterior(test_config, 2, 50, 1, test_observation)
    assert isinstance(posterior, DirectPosterior)


def test_truncated_training():
    test_config = "tests/config_files/test.config"
    test_observation = torch.randn(2400)
    posterior = train_truncated_posterior(test_config, 2, 50, 1, test_observation)
    assert isinstance(posterior, DirectPosterior)
