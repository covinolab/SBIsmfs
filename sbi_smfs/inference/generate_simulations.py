import argparse
import torch
from typing import Union, Tuple

from sbi.inference import simulate_for_sbi
from sbi.utils.user_input_checks import get_batch_loop_simulator
from sbi_smfs.inference.priors import get_priors_from_config
from sbi_smfs.simulator import get_simulator_from_config


def generate_simulations(
    config_file: str,
    num_sim: int,
    num_workers: int,
    simulation_batch_size: int,
    file_name: Union[str, None] = None,
    show_progressbar: bool = False,
    save_as_file: bool = False,
) -> Union[Tuple[torch.Tensor, torch.Tensor], None]:
    """
    Run simulations with parameters from prior.

    Parameters
    ----------
    config_file: str
        Config file name.
    num_sim: int
        Number of simulations to run.
    num_workers: int
        Number of workers to use for simulation.
    file_name: str, None
        File name to save the simulations to.
    show_progressbar: bool
        Whether to show a progress bar.
    save_as_file: bool
        Whether to save the simulations to a file.

    Returns
    -------
    theta: torch.Tensor
        Parameters used for the simulations.
    x: torch.Tensor
        Simulations.
    """

    if save_as_file:
        assert isinstance(
            file_name, str
        ), "You need to specify a filename if save_as_file=True"

    prior = get_priors_from_config(config_file)
    simulator = get_batch_loop_simulator(get_simulator_from_config(config_file))

    theta, x = simulate_for_sbi(
        simulator,
        prior,
        num_simulations=num_sim,
        num_workers=num_workers,
        show_progress_bar=show_progressbar,
        simulation_batch_size=simulation_batch_size,
    )

    if save_as_file:
        x_file_name = f"{file_name}_x.pt"
        theta_file_name = f"{file_name}_theta.pt"
        torch.save(x, x_file_name)
        torch.save(theta, theta_file_name)
    else:
        return theta, x
