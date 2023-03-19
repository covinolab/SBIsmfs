import argparse
import pickle
import dill
import torch
import torch.distributions as dists

import sbi.utils as utils
from sbi.inference import SNPE
import sbi.utils as utils
from  sbi.utils.user_input_checks import get_batch_loop_simulator
from sbi.simulators.simutils import simulate_in_batches
from sbi.utils.get_nn_models import posterior_nn


from simulator import smfe_simulator_mm
from embedding_net import CNN
import config_real_data as config
from priors import SplinePrior

def main(N_rounds: int, N_sim_per_round: int, N_workers: int, observation_file: str, posterior_file: str):

    device = 'cuda'

    observation = torch.load(observation_file)

    priors = [
        dists.Uniform(torch.tensor([-0.5], device=device), torch.tensor([0.5], device=device)),
        dists.Uniform(torch.tensor([-1.5], device=device), torch.tensor([-0.5], device=device)),
        *(dists.Normal(torch.tensor([0.], device=device), torch.tensor([5.], device=device)) for i in range(config.N_knots_prior - 1))
    ]

    prior = SplinePrior(priors)

    print('Building neural network on :', device)
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

    dataloader_kwargs = {
        'num_workers': 10,
        'pin_memory': True
    }

    simulator = get_batch_loop_simulator(smfe_simulator_mm)
    proposal = prior

    for idx_round in range(N_rounds):
        
        if idx_round > 0:
            theta = proposal.sample((N_sim_per_round,), max_sampling_batch_size=500000)
        else:
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
            force_first_round_loss=True,
            show_train_summary=True,
            validation_fraction=0.15,
            training_batch_size=64,
            learning_rate=0.0005,
            stop_after_epochs=20,
            dataloader_kwargs=dataloader_kwargs
        )

        posterior = inference.build_posterior(
            density_estimator
        ).set_default_x(observation)

        with open(f'{posterior_file}_round={idx_round}.pkl', 'wb') as handle:
            pickle.dump(posterior, handle)

        accept_reject_fn = utils.get_density_thresholder(posterior, quantile=1e-3)
        proposal = utils.RestrictedPrior(prior, accept_reject_fn, sample_with="rejection")

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