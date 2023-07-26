import argparse
import pickle
from typing import Union

import torch
from sbi.inference import SNPE

import sbi.utils as utils
from sbi.utils.user_input_checks import get_batch_loop_simulator
from sbi.simulators.simutils import simulate_in_batches
from sbi.utils.get_nn_models import posterior_nn

from sbi_smfs.simulator import get_simulator_from_config
from sbi_smfs.inference.priors import get_priors_from_config
from sbi_smfs.inference.embedding_net import EMBEDDING_NETS
from sbi_smfs.utils.config_utils import get_config_parser
from sbi_smfs.utils.summary_stats import (
    compute_stats,
    check_if_observation_contains_features,
)


def train_truncated_posterior(
    config_file: str,
    num_rounds: int,
    num_sim_per_round: int,
    num_workers: int,
    observation: Union[str, torch.Tensor],
    posterior_file: Union[str, None] = None,
    device: str = "cpu",
) -> Union[None, SNPE]:
    """Trains a truncated posterior.

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

    if isinstance(observation, str):
        observation = torch.load(observation)
    prior = get_priors_from_config(config_file, device=device)
    simulator = get_simulator_from_config(config_file)
    config = get_config_parser(config_file)

    if isinstance(observation, str):
        observation = torch.load(observation)

    if not check_if_observation_contains_features(observation, config):
        observation = compute_stats(observation, config)

    print("Building neural network on :", device)

    if config.get("SUMMARY_STATS", "embedding_net") in EMBEDDING_NETS.keys():
        cnn_net = EMBEDDING_NETS[config.get("SUMMARY_STATS", "embedding_net")](
            config.getint("SUMMARY_STATS", "num_bins"),
            len(config.getlistint("SUMMARY_STATS", "lag_times")),
            100,
        )
        print("Using embedding net :", config.get("SUMMARY_STATS", "embedding_net"))
    else:
        print(
            f"only available embeddings are : {[key for key in EMBEDDING_NETS.keys()]}. Falling back to single_layer_cnn"
        )
        cnn_net = EMBEDDING_NETS["single_layer_cnn"](
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
        if idx_round > 0:
            theta = proposal.sample(
                (num_sim_per_round,), max_sampling_batch_size=500000
            )
        else:
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
            force_first_round_loss=True,
            show_train_summary=True,
            validation_fraction=0.15,
            training_batch_size=50,
            learning_rate=0.0005,
            stop_after_epochs=20,
        )

        posterior = inference.build_posterior(density_estimator).set_default_x(
            observation
        )
        if isinstance(posterior_file, str):
            with open(f"{posterior_file}_round={idx_round}.pkl", "wb") as handle:
                pickle.dump(posterior, handle)

        accept_reject_fn = utils.get_density_thresholder(posterior, quantile=1e-3)
        proposal = utils.RestrictedPrior(
            prior, accept_reject_fn, sample_with="rejection"
        )

    if not isinstance(posterior_file, str):
        return posterior
