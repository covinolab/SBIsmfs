import numpy as np
import pytest
import torch
from sbi_smfs.simulator.simulator import Simulator


@pytest.fixture
def config():
    config = "tests/config_files/test_config.yaml"
    return config


@pytest.fixture
def config_dx():
    config = "tests/config_files/test_config.yaml"
    return config


@pytest.mark.parametrize(
    "num_bins, lag_times", [(20, [1, 10, 100, 1000]), (50, [1, 10, 100])]
)
def test_simulator(config, num_bins: int, lag_times: list[int]):
    simulator = Simulator(config)
    params = torch.tensor(
        [
            0,
            3,
            6.94227994,
            -0.67676768,
            -4.23232323,
            -3.72438672,
            0.45021645,
            2.48196248,
            0.45021645,
            3.72438672,
            -4.23232323,
            -0.67676768,
            6.94227994,
        ]
    )
    summary_stats = simulator(
        parameters=params,
        dt=5e-4,
        N=1e6,
        saving_freq=2,
        Dx=1.0,
        N_knots=15,
        min_x=-6,
        max_x=6,
        max_G_0=70,
        max_G_1=30,
        init_xq_range=(-2, 2),
        min_bin=-5,
        max_bin=5,
        num_bins=num_bins,
        lag_times=lag_times,
    )

    assert summary_stats.shape[0] == len(lag_times) * (num_bins ** 2)


def test_simulator_from_config(config):
    params = torch.tensor(
        [
            0,
            3,
            6.94227994,
            -0.67676768,
            -4.23232323,
            -3.72438672,
            0.45021645,
            2.48196248,
            0.45021645,
            3.72438672,
            -4.23232323,
            -0.67676768,
            6.94227994,
        ]
    )

    simulator = Simulator(config)
    summary_stats = simulator(params)
    assert summary_stats.shape[0] == 6 * (20 ** 2)


def test_simulator_from_config_with_Dx(config_dx):
    params = torch.tensor(
        [
            0,
            0,
            3,
            6.94227994,
            -0.67676768,
            -4.23232323,
            -3.72438672,
            0.45021645,
            2.48196248,
            0.45021645,
            3.72438672,
            -4.23232323,
            -0.67676768,
            6.94227994,
        ]
    )

    simulator = Simulator(config_dx)
    summary_stats = simulator(params)
    assert summary_stats.shape[0] == 6 * (20 ** 2)


def test_simulator_from_config_with_no_Dx(config_no_dx):
    params = torch.tensor(
        [
            0,
            0,
            3,
            6.94227994,
            -0.67676768,
            -4.23232323,
            -3.72438672,
            0.45021645,
            2.48196248,
            0.45021645,
            3.72438672,
            -4.23232323,
            -0.67676768,
            6.94227994,
        ]
    )
    with pytest.raises(NotImplementedError):
        simulator = Simulator(config_no_dx)