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
    file_name: Union[str, None] = None,
    show_progressbar: bool = False,
    save_as_file: bool = False,
):
    """Run simulations with parameters from prior.

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
    )

    if save_as_file:
        x_file_name = f"{file_name}_x.pt"
        theta_file_name = f"{file_name}_theta.pt"
        torch.save(x, x_file_name)
        torch.save(theta, theta_file_name)
    else:
        return theta, x


def main():
    cl_parser = argparse.ArgumentParser()
    cl_parser.add_argument("--config_file", action="store", type=str, required=True)
    cl_parser.add_argument("--num_sim", action="store", type=int, required=True)
    cl_parser.add_argument("--num_workers", action="store", type=int, required=True)
    cl_parser.add_argument("--file_name", action="store", type=str, required=True)
    args = cl_parser.parse_args()

    generate_simulations(
        args.config_file, args.num_sim, args.num_workers, True, args.file_name
    )


if __name__ == "__main__":
    main()
