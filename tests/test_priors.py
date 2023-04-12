import configparser
import torch
from sbi_smfs.inference.priors import get_priors_from_config


def test_get_prior_from_config():
    num_samples = 1000000
    prior = get_priors_from_config("tests/test.config")
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


def test_get_prior_from_config_with_dx():
    num_samples = 1000000
    prior = get_priors_from_config("tests/test_2.config")
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


# TODO : Add test for non spline prior
