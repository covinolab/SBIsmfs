import configparser
import torch
from sbi_smfs.inference.priors import get_priors_from_config


def test_get_prior_from_config():
    num_samples = 1000000
    prior = get_priors_from_config("tests/config_files/test.config")
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
    prior = get_priors_from_config("tests/config_files/test_2.config")
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


def test_individual_spline_prior():
    num_samples = 1000000
    prior = get_priors_from_config("tests/config_files/test_spline_prior.config")
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


def test_get_prior_from_config_with_pdd():
    num_samples = 1000000
    prior = get_priors_from_config("tests/config_files/test_pdd.config")
    samples = prior.sample((num_samples,))
    print(samples.std(axis=0))
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

    for i in range(10):
        assert torch.isclose(
            samples[:, i + 2].mean(), torch.tensor(-2.0), atol=0.1
        ).item(), f"mean spline Dx {i}"
        assert torch.isclose(
            samples[:, i + 2].std(), torch.tensor(0.5), atol=0.1
        ).item(), f"std spline Dx {i}"

    for i in range(6):
        assert torch.isclose(
            samples[:, i + 12].mean(), torch.tensor(0.0), atol=0.1
        ).item(), f"mean spline Gx {i + 12}"
        assert torch.isclose(
            samples[:, i + 12].std(), torch.tensor(2.0), atol=0.1
        ).item(), f"std spline Gx {i + 12}"
