import argparse
import pickle
from typing import Union

import torch
from sbi.inference import SNPE

from sbi.utils.user_input_checks import get_batch_loop_simulator
from sbi.simulators.simutils import simulate_in_batches
from sbi_smfs.inference.build_nn_models import build_npe_model, get_train_parameter

from sbi_smfs.simulator import get_simulator_from_config
from sbi_smfs.inference.priors import get_priors_from_config
from sbi_smfs.utils.config_utils import get_config_parser
from sbi_smfs.utils.summary_stats import (
    compute_stats,
    check_if_observation_contains_features,
)


def train_sequential_posterior(
    config_file: str,
    num_rounds: int,
    num_sim_per_round: int,
    num_workers: int,
    observation: Union[str, torch.Tensor],
    posterior_file: Union[None, str] = None,
    device: str = "cpu",
    save_interval: int = 1,
) -> Union[None, SNPE]:
    """
    Trains a sequential posterior.

    Parameters
    ----------
    config_file: str
        Config file name.
    num_rounds: int
        Number of rounds to train the posterior.
    num_sim_per_round: int
        Number of simulations per round.
    num_workers: int
        Number of workers to use for simulation.
    observation: str, torch.Tensor
        Observation to use for conditioning the posterior.
    posterior_file: str, None
        File name to save the posterior to.
    device: str
        Device to use for training.

    Returns
    -------
    inference: sbi.inference.snpe.snpe_base.SNPE
        Trained posterior.
    """
    prior = get_priors_from_config(config_file, device=device)
    simulator = get_simulator_from_config(config_file)
    config = get_config_parser(config_file, validate=True)

    if isinstance(observation, str):
        observation = torch.load(observation)

    if not check_if_observation_contains_features(observation, config):
        observation = compute_stats(observation, config)

    print("Building neural network on :", device)
    neural_posterior = build_npe_model(config)
    train_parameters = get_train_parameter(config)

    inference = SNPE(prior=prior, density_estimator=neural_posterior, device=device)
    simulator = get_batch_loop_simulator(simulator)
    proposal = prior

    for idx_round in range(num_rounds):
        theta = proposal.sample((num_sim_per_round,))

        x = simulate_in_batches(
            simulator,
            theta.cpu(),
            sim_batch_size=num_sim_per_round // num_workers,
            num_workers=num_workers,
        )

        inference = inference.append_simulations(
            theta, x, proposal=proposal, data_device="cpu"
        )

        density_estimator = inference.train(
            show_train_summary=True,
            **train_parameters,
        )
        posterior = inference.build_posterior(
            density_estimator, direct_sampling_parameters={"enable_transform": False}
        )
        if (
            isinstance(posterior_file, str)
            and idx_round % save_interval == 0
            and idx_round > 0
        ):
            with open(f"{posterior_file}_round={idx_round}.pkl", "wb") as handle:
                pickle.dump(posterior, handle)

        proposal = posterior.set_default_x(observation)

    if not isinstance(posterior_file, str):
        return posterior
    elif isinstance(posterior_file, str):
        with open(f"{posterior_file}.pkl", "wb") as handle:
            pickle.dump(posterior, handle)
