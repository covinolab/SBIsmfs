import numpy as np
import pytest
import torch
from sbi_smfs.simulator.simulator import smfe_simulator_mm, get_simulator_from_config


def test_simulator():
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

    lag_times = [1, 10, 100, 1000]
    num_bins = 20
    summary_stats = smfe_simulator_mm(
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


def test_simulator_from_config():
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

    simulator = get_simulator_from_config("tests/config_files/test.config")
    summary_stats = simulator(params)
    assert summary_stats.shape[0] == 6 * (20 ** 2)


def test_simulator_from_config_with_Dx():
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

    simulator = get_simulator_from_config("tests/config_files/test_2.config")
    summary_stats = simulator(params)
    assert summary_stats.shape[0] == 6 * (20 ** 2)


def test_simulator_from_config_with_no_Dx():
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
        simulator = get_simulator_from_config("tests/config_files/no_Dx.config")
