import argparse
import pickle
import dill
import torch

from sbi.inference import SNPE
import sbi.utils as utils
from sbi.utils.get_nn_models import posterior_nn

import config
from embedding_net import CNN


def main(training_file_name: str, posterior_file_name: str):

    device = 'cpu'

    lower_limits = [
        config.logD_lims[0], config.k_lims[0],
        *(config.spline_lims[0] for i in range(config.N_knots_prior))
    ]

    upper_limits = [
        config.logD_lims[1], config.k_lims[1],
        *(config.spline_lims[1] for i in range(config.N_knots_prior))
    ]

    prior = utils.BoxUniform(
        low=torch.tensor(lower_limits),
        high=torch.tensor(upper_limits),
        device=device
    )

    x_file_name = f'{training_file_name}_x.pt'
    theta_file_name = f'{training_file_name}_theta.pt'

    theta = torch.load(theta_file_name, map_location=torch.device(device))
    x = torch.load(x_file_name, map_location=torch.device(device))
    
    print("Buld Neural Network ...")
    cnn_net = CNN()

    neural_posterior = posterior_nn(
        model='nsf',
        hidden_features=100,
        num_transforms=5,
        num_bins=10,
        embedding_net=cnn_net
    )

    inference = SNPE(
        prior=prior,
        density_estimator=neural_posterior,
        device=device
    )

    inference = inference.append_simulations(theta, x)
    density_estimator = inference.train(
        show_train_summary=True,
        validation_fraction=0.15,
        training_batch_size=1500,
        learning_rate=0.0005,
        stop_after_epochs=20
    )

    posterior = inference.build_posterior(density_estimator)

    with open(f'{posterior_file_name}_NN.pkl', 'wb') as handle:
        dill.dump(inference, handle)

    with open(f'{posterior_file_name}.pkl', 'wb') as handle:
        pickle.dump(posterior, handle)


if __name__ == '__main__':

    cl_parser = argparse.ArgumentParser()
    cl_parser.add_argument(
        '--train_file',
        action='store',
        type=str,
        required=True
    )
    cl_parser.add_argument(
        '--posterior_file',
        action='store',
        type=str,
        required=True
    )
    args = cl_parser.parse_args()

    main(args.train_file, args.posterior_file)
