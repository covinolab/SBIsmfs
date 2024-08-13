import pytest
import torch
from sbi_smfs.inference.priors import SplinePrior

@pytest.fixture
def config():
    config = "your_config_file.config"
    return config

@pytest.fixture
def config_dx():
    config = "your_config_file.config"
    return config


def test_get_prior_from_config(config):
    num_samples = 1000000
    prior = SplinePrior(config)
    samples = prior.sample((num_samples,))

    assert torch.isclose(
        samples[:, 0].mean(), torch.tensor(-1.5), atol=0.1
    ).item(), "mean Dq"
    assert torch.isclose(
        samples[:, 0].std(), torch.tensor(0.866), atol=0.1
    ).item(), "std Dq"
    assert torch.isclose(
        samples[:, 1].mean(), torch.tensor(0.0), atol=0.1
    ).item(), "mean k"
    assert torch.isclose(
        samples[:, 1].std(), torch.tensor(1.0), atol=0.1
    ).item(), "std k"
    assert (
        torch.isclose(samples[:, 2:].mean(axis=1), torch.zeros(num_samples), atol=1e-5)
    ).all(), "spline mean"


def test_get_prior_from_config_with_dx(config_dx):
    num_samples = 1000000
    prior = SplinePrior(config_dx)
    samples = prior.sample((num_samples,))
    assert torch.isclose(
        samples[:, 0].mean(), torch.tensor(-3.0), atol=0.1
    ).item(), "mean Dx"
    assert torch.isclose(
        samples[:, 0].std(), torch.tensor(0.5), atol=0.1
    ).item(), "std Dq"
    assert torch.isclose(
        samples[:, 1].mean(), torch.tensor(-1.5), atol=0.1
    ).item(), "mean Dq"
    assert torch.isclose(
        samples[:, 1].std(), torch.tensor(0.866), atol=0.1
    ).item(), "std Dq"
    assert torch.isclose(
        samples[:, 2].mean(), torch.tensor(0.0), atol=0.1
    ).item(), "mean k"
    assert torch.isclose(
        samples[:, 2].std(), torch.tensor(1.0), atol=0.1
    ).item(), "std k"
    assert (
        torch.isclose(samples[:, 3:].mean(axis=1), torch.zeros(num_samples), atol=1e-5)
    ).all(), "spline mean"


def test_individual_spline_prior(config):
    num_samples = 1000000
    prior = SplinePrior(config)
    samples = prior.sample((num_samples,))
    true_mean = torch.tensor([0, 1, 0, 1, 0, -3], dtype=torch.float32)
    true_std = torch.tensor([1, 3, 1, 3, 1, 1], dtype=torch.float32)

    for i in range(6):
        assert torch.isclose(
            samples[:, i + 2].mean(), true_mean[i], atol=0.1
        ).item(), f"mean spline {i}"
        assert torch.isclose(
            samples[:, i + 2].std(), true_std[i], atol=0.1
        ).item(), f"std spline {i}"