import argparse
import pickle
import dill
import torch

import sbi.utils as utils
from sbi.inference import SNPE
import sbi.utils as utils
from  sbi.utils.user_input_checks import get_batch_loop_simulator
from sbi.simulators.simutils import simulate_in_batches
from sbi.utils.get_nn_models import posterior_nn

from sbi.utils import MultipleIndependent
import torch.distributions as dists

from simulator import smfe_simulator_mm
from embedding_net import CNN
import config_real_data as config


class SplinePrior(MultipleIndependent):
    def __init__(self, dists):
        super().__init__(dists, validate_args=None, arg_constraints={})

    def sample(self, sample_shape=torch.Size([])):
        samples = super().sample(sample_shape)
        if sample_shape == torch.Size():
            samples[2:] = samples[2:] - torch.mean(samples[2:]).reshape(-1, 1)
        else:
            samples[:, 2:] = samples[:, 2:] - torch.mean(samples[:, 2:], dim=1).reshape(-1, 1)
        return samples


def main(N_rounds: int, N_sim_per_round: int, N_workers: int, observation_file: str, posterior_file: str):

    device = 'cuda'

    observation = torch.load(observation_file)

    priors = [
        dists.Uniform(torch.tensor([-0.5], device=device), torch.tensor([0.5], device=device)),
        dists.Uniform(torch.tensor([-1.5], device=device), torch.tensor([-0.5], device=device)),
        *(dists.Normal(torch.tensor([0.], device=device), torch.tensor([3.], device=device)) for i in range(config.N_knots_prior))
    ]

    prior = SplinePrior(priors)

    print('Building neural network ...', device)
    cnn_net = CNN()

    kwargs_flow = {
        'num_blocks': 2,
        'dropout_probability': 0.0,
        'use_batch_norm': False,
    }

    neural_posterior = posterior_nn(
        model='nsf',
        hidden_features=100,
        num_transforms=5,
        num_bins=10,
        embedding_net=cnn_net,
        z_score_x='none',
        **kwargs_flow
    )

    inference = SNPE(
        prior=prior,
        density_estimator=neural_posterior,
        device=device
    )

    simulator = get_batch_loop_simulator(smfe_simulator_mm)
    proposal = prior

    for idx_round in range(N_rounds):

        theta = proposal.sample((N_sim_per_round,))

        x = simulate_in_batches(
            simulator,
            theta.cpu(),
            sim_batch_size=20,
            num_workers=N_workers
        )

        inference = inference.append_simulations(
            theta,
            x, 
            proposal=proposal,
            data_device='cpu'
            )

        density_estimator = inference.train(
            show_train_summary=True,
            validation_fraction=0.15,
            training_batch_size=50,
            learning_rate=0.0005,
            stop_after_epochs=20
        )

        posterior = inference.build_posterior(density_estimator)

        with open(f'{posterior_file}_round={idx_round}.pkl', 'wb') as handle:
            pickle.dump(posterior, handle)

        proposal = posterior.set_default_x(observation)


if __name__ == '__main__':

    cl_parser = argparse.ArgumentParser()
    cl_parser.add_argument(
        '--N_rounds',
        action='store',
        type=int,
        required=True
    )
    cl_parser.add_argument(
        '--N_sim_per_round',
        action='store',
        type=int,
        required=True
    )
    cl_parser.add_argument(
        '--N_workers',
        action='store',
        type=int,
        required=True
    )
    cl_parser.add_argument(
        '--observation_file',
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
    
    main(
        args.N_rounds,
        args.N_sim_per_round,
        args.N_workers,
        args.observation_file,
        args.posterior_file
    )
