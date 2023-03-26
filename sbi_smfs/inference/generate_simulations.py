import argparse
import torch

from sbi.inference import simulate_for_sbi
from sbi_smfs.inference.priors import get_priors_from_config
from sbi_smfs.simulator import get_simulator_from_config


def generate_simulation(
    config_file: str,
    num_sim: int,
    num_workers: int,
    file_name=None,
    show_progressbar=False,
    save_as_file=True,
):
    """Run simulations with parameters from prior."""

    if save_as_file:
        assert isinstance(
            file_name, str
        ), "You need to specify a filename if save_as_file=True"

    prior = get_priors_from_config(config_file)
    simulator = get_simulator_from_config(config_file)

    theta, x = simulate_for_sbi(
        simulator,
        prior,
        num_simulations=num_sim,
        num_workers=num_workers,
        show_progress_bar=False,
    )

    if save_as_file:
        x_file_name = f"{file_name}_x.pt"
        theta_file_name = f"{file_name}_theta.pt"
        torch.save(x, x_file_name)
        torch.save(theta, theta_file_name)
    else:
        return x, theta


if __name__ == "__main__":

    cl_parser = argparse.ArgumentParser()
    cl_parser.add_argument("--config_file", action="store", type=str, required=True)
    cl_parser.add_argument("--num_sim", action="store", type=int, required=True)
    cl_parser.add_argument("--num_workers", action="store", type=int, required=True)
    cl_parser.add_argument("--file_name", action="store", type=str, required=True)
    args = cl_parser.parse_args()

    generate_simulation(args.num_workers, args.num_sim, args.file_name)
