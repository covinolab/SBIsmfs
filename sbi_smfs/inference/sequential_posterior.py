import argparse
import pickle

import torch
from sbi.inference import SNPE

from sbi.utils.user_input_checks import get_batch_loop_simulator
from sbi.simulators.simutils import simulate_in_batches
from sbi.utils.get_nn_models import posterior_nn

from sbi_smfs.simulator import get_simulator_from_config
from sbi_smfs.inference.priors import get_priors_from_config
from sbi_smfs.inference.embedding_net import SimpleCNN
from sbi_smfs.utils.config_utils import get_config_parser


def train_sequential_posterior(
    config_file: str,
    num_rounds: int,
    num_sim_per_round: int,
    num_workers: int,
    observation,
    posterior_file=None,
    device="cpu",
):
    if isinstance(observation, str):
        observation = torch.load(observation)
    prior = get_priors_from_config(config_file)
    simulator = get_simulator_from_config(config_file)
    config = get_config_parser(config_file)

    print("Building neural network on :", device)
    cnn_net = SimpleCNN(
        len(config.getlistint("SUMMARY_STATS", "lag_times")),
        4,
        2,
        config.getint("SUMMARY_STATS", "num_bins"),
        len(config.getlistint("SUMMARY_STATS", "lag_times")),
    )

    kwargs_flow = {
        "num_blocks": 2,
        "dropout_probability": 0.0,
        "use_batch_norm": False,
    }

    neural_posterior = posterior_nn(
        model="nsf",
        hidden_features=100,
        num_transforms=5,
        num_bins=10,
        embedding_net=cnn_net,
        z_score_x="none",
        **kwargs_flow,
    )

    inference = SNPE(prior=prior, density_estimator=neural_posterior, device=device)
    simulator = get_batch_loop_simulator(simulator)
    proposal = prior

    for idx_round in range(num_rounds):
        theta = proposal.sample((num_sim_per_round,))

        x = simulate_in_batches(
            simulator, theta.cpu(), sim_batch_size=20, num_workers=num_workers
        )

        inference = inference.append_simulations(
            theta, x, proposal=proposal, data_device="cpu"
        )

        density_estimator = inference.train(
            show_train_summary=True,
            validation_fraction=0.15,
            training_batch_size=50,
            learning_rate=0.0005,
            stop_after_epochs=20,
        )

        posterior = inference.build_posterior(density_estimator)
        if isinstance(posterior_file, str):
            with open(f"{posterior_file}_round={idx_round}.pkl", "wb") as handle:
                pickle.dump(posterior, handle)

        proposal = posterior.set_default_x(observation)

    if not isinstance(posterior_file, str):
        return posterior


def main():
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
    args = cl_parser.parse_args()

    train_sequential_posterior(
        args.config_file,
        args.num_rounds,
        args.num_sim_per_round,
        args.num_workers,
        args.observation_file,
        args.posterior_file,
        args.device,
    )


if __name__ == "__main__":
    main()
