import argparse
import torch

from sbi.inference import prepare_for_sbi, simulate_for_sbi
import sbi.utils as utils

from simulator import smfe_simulator_mm
import config


def main(N_workers: int, N_sim: int, file_name: str):

    lower_limits = [
        config.logD_lims[0],
        config.k_lims[0],
        *(config.spline_lims[0] for i in range(config.N_knots_prior)),
    ]

    upper_limits = [
        config.logD_lims[1],
        config.k_lims[1],
        *(config.spline_lims[1] for i in range(config.N_knots_prior)),
    ]

    prior = utils.BoxUniform(
        low=torch.tensor(lower_limits), high=torch.tensor(upper_limits)
    )

    simulator, prior = prepare_for_sbi(smfe_simulator_mm, prior)

    theta, x = simulate_for_sbi(
        simulator,
        prior,
        num_simulations=N_sim,
        num_workers=N_workers,
        show_progress_bar=False,
    )

    x_file_name = f"{file_name}_x.pt"
    theta_file_name = f"{file_name}_theta.pt"
    torch.save(x, x_file_name)
    torch.save(theta, theta_file_name)


if __name__ == "__main__":

    cl_parser = argparse.ArgumentParser()
    cl_parser.add_argument("--num_workers", action="store", type=int, required=True)
    cl_parser.add_argument("--num_sim", action="store", type=int, required=True)
    cl_parser.add_argument("--file_name", action="store", type=str, required=True)
    args = cl_parser.parse_args()

    main(args.num_workers, args.num_sim, args.file_name)
