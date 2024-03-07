import argparse
from sbi_smfs.inference.truncated_posterior import train_truncated_posterior
from sbi_smfs.inference.sequential_posterior import train_sequential_posterior
from sbi_smfs.inference.armortized_posterior import train_armortized_posterior
from sbi_smfs.inference.generate_simulations import generate_simulations


def cmd_train_truncated_posterior():
    """
    Command line tool for training a truncated posterior.
    """

    cl_parser = argparse.ArgumentParser()
    cl_parser.add_argument("--config_file", action="store", type=str, required=True)
    cl_parser.add_argument("--num_rounds", action="store", type=int, required=True)
    cl_parser.add_argument(
        "--num_sim_per_round", action="store", type=int, required=True
    )
    cl_parser.add_argument("--num_workers", action="store", type=int, required=True)
    cl_parser.add_argument(
        "--observation_file", action="store", type=str, required=True
    )
    cl_parser.add_argument("--posterior_file", action="store", type=str, required=True)
    cl_parser.add_argument(
        "--device", action="store", type=str, required=False, default="cpu"
    )
    cl_parser.add_argument(
        "--save_interval", action="store", type=int, required=False, default=1
    )
    args = cl_parser.parse_args()

    train_truncated_posterior(
        args.config_file,
        args.num_rounds,
        args.num_sim_per_round,
        args.num_workers,
        args.observation_file,
        args.posterior_file,
        args.device,
        args.save_interval,
    )


def cmd_train_sequential_posterior():
    """
    Command line tool for training a sequential posterior.
    """

    cl_parser = argparse.ArgumentParser()
    cl_parser.add_argument("--config_file", action="store", type=str, required=True)
    cl_parser.add_argument("--num_rounds", action="store", type=int, required=True)
    cl_parser.add_argument(
        "--num_sim_per_round", action="store", type=int, required=True
    )
    cl_parser.add_argument("--num_workers", action="store", type=int, required=True)
    cl_parser.add_argument(
        "--observation_file", action="store", type=str, required=True
    )
    cl_parser.add_argument("--posterior_file", action="store", type=str, required=True)
    cl_parser.add_argument(
        "--device", action="store", type=str, required=False, default="cpu"
    )
    cl_parser.add_argument(
        "--save_interval", action="store", type=int, required=False, default=1
    )
    args = cl_parser.parse_args()

    train_sequential_posterior(
        args.config_file,
        args.num_rounds,
        args.num_sim_per_round,
        args.num_workers,
        args.observation_file,
        args.posterior_file,
        args.device,
        args.save_interval,
    )


def cmd_train_armortized_posterior():
    """
    Command line tool for training an armortized posterior.
    """

    cl_parser = argparse.ArgumentParser()
    cl_parser.add_argument("--config_file", action="store", type=str, required=True)
    cl_parser.add_argument("--train_data", action="store", type=str, required=True)
    cl_parser.add_argument("--posterior_file", action="store", type=str, required=True)
    cl_parser.add_argument(
        "--device", action="store", type=str, required=False, default="cpu"
    )
    args = cl_parser.parse_args()
    train_armortized_posterior(
        args.config_file, args.train_data, args.posterior_file, args.device
    )


def cmd_generate_simulations():
    """
    Command line tool for generating simulations.
    """

    cl_parser = argparse.ArgumentParser()
    cl_parser.add_argument("--config_file", action="store", type=str, required=True)
    cl_parser.add_argument("--num_sim", action="store", type=int, required=True)
    cl_parser.add_argument("--num_workers", action="store", type=int, required=True)
    cl_parser.add_argument("--file_name", action="store", type=str, required=True)
    cl_parser.add_argument(
        "--progress_bar", action="store", type=bool, required=False, default=False
    )
    args = cl_parser.parse_args()

    generate_simulations(
        config_file=args.config_file,
        num_sim=args.num_sim,
        num_workers=args.num_workers,
        save_as_file=True,
        file_name=args.file_name,
        show_progressbar=True,
    )
