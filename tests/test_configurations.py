import pytest
import torch
import omegaconf
import sbi_smfs.utils.configurations as cfg_utils

# Assume all the necessary classes (Config, SimulatorConfig, PriorsConfig, etc.) are imported or defined here

@pytest.fixture
def sample_config():
    config = cfg_utils.load_config_yaml("tests/config_files/test_config.yaml")
    return config

@pytest.fixture
def sample_config_str():#
    return "tests/config_files/test_config.yaml"


def test_load_config_yaml(sample_config_str):
    config = cfg_utils.load_config_yaml(sample_config_str)
    assert isinstance(config, omegaconf.dictconfig.DictConfig)


def test_initialize_distribution(sample_config):
    dx_distribution = cfg_utils.initialize_distribution(sample_config.priors.dx)
    assert isinstance(dx_distribution, torch.distributions.Distribution)