import torch
from sbi_smfs.inference.priors import get_priors_from_config


def test_get_prior_from_config():

    prior = get_priors_from_config('tests/test.config')
    samples = prior.sample((1000000,))
    
    assert torch.isclose(samples[:, 0].mean(), torch.tensor(0.0), atol=0.1).item()
    assert torch.isclose(samples[:, 0].std(), torch.tensor(1.732), atol=0.1).item()
    assert torch.isclose(samples[:, 1].mean(), torch.tensor(3.0), atol=0.1).item()
    assert torch.isclose(samples[:, 1].std(), torch.tensor(1.0), atol=0.1).item()
    for i in range(3, 15):
        assert torch.isclose(samples[:, i].mean(), torch.tensor(0.0), atol=0.1).item()
        assert torch.isclose(samples[:, i].std(), torch.tensor(2.0), atol=0.1).item()