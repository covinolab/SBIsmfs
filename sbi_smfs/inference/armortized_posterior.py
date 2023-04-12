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


def train_armortized_posterior(
    config_file: str, train_data, posterior_file=None, device="cpu",
):
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

    inference = SNPE(density_estimator=neural_posterior, device=device)

    if isinstance(train_data, str):
        x_file_name = f"{train_data}_x.pt"
        theta_file_name = f"{train_data}_theta.pt"
        theta = torch.load(theta_file_name)
        x = torch.load(x_file_name)
    elif isinstance(train_data, tuple):
        theta, x = train_data
    else:
        raise NotImplementedError

    inference = inference.append_simulations(theta, x, data_device="cpu")
    density_estimator = inference.train(
        show_train_summary=True,
        validation_fraction=0.15,
        training_batch_size=500,
        learning_rate=0.0005,
        stop_after_epochs=20,
    )

    posterior = inference.build_posterior(density_estimator)

    if not isinstance(posterior_file, str):
        return posterior
    elif isinstance(posterior_file, str):
        with open(f"{posterior_file}.pkl", "wb") as handle:
            pickle.dump(posterior, handle)
    else:
        raise NotImplementedError("posterior_file needs to be either None or a string!")


def main():
    cl_parser = argparse.ArgumentParser()
    cl_parser.add_argument("--config_file", action="store", type=str, required=True)
    cl_parser.add_argument("--train_data", action="store", type=str, required=True)
    cl_parser.add_argument("--posterior_file", action="store", type=str, required=True)
    cl_parser.add_argument(
        "--device", action="store", type=str, required=False, default="cpu"
    )
    args = cl_parser.parse_args()
    train_posterior(args.config_file, args.train_data, args.posterior_file, args.device)


if __name__ == "__main__":
    main()
