import pytest
import torch
import os
from sbi_smfs.inference.generate_simulations import generate_simulations


def test_generate_simulations():
    observations = generate_simulations(
        "tests/config_files/test.config",
        num_sim=3,
        num_workers=1,
        simulation_batch_size=1,
        file_name=None,
        show_progressbar=False,
        save_as_file=False,
    )
    assert observations is not None
    assert observations[0].shape == (3, 13)
    assert observations[1].shape == (3, 2400)


def test_generate_simulations_with_Dx():
    observations = generate_simulations(
        "tests/config_files/test_2.config",
        num_sim=3,
        num_workers=1,
        simulation_batch_size=1,
        file_name=None,
        show_progressbar=False,
        save_as_file=False,
    )
    assert observations is not None
    assert observations[0].shape == (3, 14)
    assert observations[1].shape == (3, 2400)


def test_generate_simulations_wrong_input():
    with pytest.raises(AssertionError):
        observations = generate_simulations(
            "tests/config_files/test.config",
            num_sim=3,
            num_workers=1,
            simulation_batch_size=1,
            file_name=None,
            show_progressbar=False,
            save_as_file=True,
        )


def test_generate_simulations_save_file():
    observations = generate_simulations(
        "tests/config_files/test.config",
        num_sim=3,
        num_workers=1,
        simulation_batch_size=1,
        file_name="test",
        show_progressbar=False,
        save_as_file=True,
    )
    observation_x_shape = torch.load(f"test_x.pt").shape
    observation_theta_shape = torch.load(f"test_theta.pt").shape
    os.remove("test_x.pt")
    os.remove("test_theta.pt")
    assert observation_theta_shape == (3, 13)
    assert observation_x_shape == (3, 2400)
