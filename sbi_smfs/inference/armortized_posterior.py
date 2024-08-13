import argparse
import pickle
from typing import Union, Tuple

import torch
from sbi.inference import SNPE

from sbi.utils.user_input_checks import get_batch_loop_simulator
from sbi.simulators.simutils import simulate_in_batches
from sbi.utils.get_nn_models import posterior_nn

from sbi_smfs.simulator import get_simulator_from_config
from sbi_smfs.inference.priors import get_priors_from_config
from sbi_smfs.inference.embedding_net import SimpleCNN
from sbi_smfs.inference.build_nn_models import build_npe_model, get_train_parameter
from sbi_smfs.utils.config_utils import get_config_parser


def train_armortized_posterior(
    config_file: str,
    train_data: Union[str, Tuple[torch.Tensor, torch.Tensor]],
    posterior_file: Union[str, None] = None,
    device: str = "cpu",
) -> Union[None, SNPE]:
    """Trains a truncated posterior.

    Parameters
    ----------
    config_file: str
        Config file name.
    train_data: str, tuple
        Data to use for training the posterior. If str, it is assumed to be a file name
        containing the data. If tuple, it is assumed to be a tuple of torch.Tensor
        containing the data.
    posterior_file: str, None
        File name to save the posterior to.
    device: str
        Device to use for training.

    Returns
    -------
    inference: sbi.inference.snpe.snpe_base.SNPE
        Trained posterior.
    """
    config = get_config_parser(config_file, validate=True)

    print("Building neural network on :", device)
    neural_posterior = build_npe_model(config)
    train_parameters = get_train_parameter(config)

    inference = SNPE(
        density_estimator=neural_posterior,
        device=device,
    )

    if isinstance(train_data, str):
        x_file_name = f"{train_data}_x.pt"
        theta_file_name = f"{train_data}_theta.pt"
        theta = torch.load(theta_file_name)
        x = torch.load(x_file_name)
    elif isinstance(train_data, tuple):
        theta, x = train_data
    else:
        raise NotImplementedError

    inference = inference.append_simulations(theta, x, data_device="cuda")
    density_estimator = inference.train(
        show_train_summary=True,
        **train_parameters,
    )

    posterior = inference.build_posterior(density_estimator)

    if not isinstance(posterior_file, str):
        return posterior  # type: ignore
    elif isinstance(posterior_file, str):
        with open(f"{posterior_file}.pkl", "wb") as handle:
            pickle.dump(posterior, handle)
    else:
        raise NotImplementedError("posterior_file needs to be either None or a string!")
