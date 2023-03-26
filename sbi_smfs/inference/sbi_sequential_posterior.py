import argparse
import pickle
import dill
import torch

import sbi.utils as utils
from sbi.inference import SNPE
import sbi.utils as utils
from sbi.utils.user_input_checks import get_batch_loop_simulator
from sbi.simulators.simutils import simulate_in_batches
from sbi.utils.get_nn_models import posterior_nn

from simulator import smfe_simulator_mm
from embedding_net import CNN
import config as config


def main(
    N_rounds: int,
    N_sim_per_round: int,
    N_workers: int,
    observation_file: str,
    posterior_file: str,
):

    device = "cuda"

    observation = torch.load(observation_file)

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
        low=torch.tensor(lower_limits), high=torch.tensor(upper_limits), device=device
    )

    print("Building neural network ...", device)
    cnn_net = CNN()

    neural_posterior = posterior_nn(
        model="nsf",
        hidden_features=100,
        num_transforms=5,
        num_bins=10,
        embedding_net=cnn_net,
    )

    inference = SNPE(prior=prior, density_estimator=neural_posterior, device=device)

    simulator = get_batch_loop_simulator(smfe_simulator_mm)
    proposal = prior

    for idx_round in range(N_rounds):

        theta = proposal.sample((N_sim_per_round,))

        x = simulate_in_batches(
            simulator,
            theta.cpu(),
            sim_batch_size=N_sim_per_round // (N_workers - 1),
            num_workers=N_workers,
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

        with open(f"{posterior_file}_round={idx_round}.pkl", "wb") as handle:
            pickle.dump(posterior, handle)

        proposal = posterior.set_default_x(observation)


if __name__ == "__main__":

    cl_parser = argparse.ArgumentParser()
    cl_parser.add_argument("--N_rounds", action="store", type=int, required=True)
    cl_parser.add_argument("--N_sim_per_round", action="store", type=int, required=True)
    cl_parser.add_argument("--N_workers", action="store", type=int, required=True)
    cl_parser.add_argument(
        "--observation_file", action="store", type=str, required=True
    )
    cl_parser.add_argument("--posterior_file", action="store", type=str, required=True)
    args = cl_parser.parse_args()

    main(
        args.N_rounds,
        args.N_sim_per_round,
        args.N_workers,
        args.observation_file,
        args.posterior_file,
    )
