import argparse
import pickle
import dill
import torch

from sbi.inference import prepare_for_sbi, simulate_for_sbi
import sbi.utils as utils
from sbi.inference import SNRE
import sbi.utils as utils
from sbi.utils.user_input_checks import get_batch_loop_simulator
from sbi.simulators.simutils import simulate_in_batches
from sbi.utils.get_nn_models import classifier_nn

from simulator import smfe_simulator_mm
from embedding_net import CNN, SplineNormalize
import config


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

    dx_mean = 10 ** config.logD_lims[1] - 10 ** config.logD_lims[0]
    dx_std = dx_mean
    k_mean = 10 ** config.k_lims[1] - 10 ** config.k_lims[0]
    k_std = k_mean
    spline_mean = config.spline_lims[1] - config.spline_lims[0]
    spline_std = spline_mean
    spline_net = SplineNormalize(
        dx_mean, dx_std, k_mean, k_std, spline_mean, spline_std
    )

    kwargs_classifier = {
        "num_blocks": 5,
        "dropout_probability": 0.2,
        "use_batch_norm": True,
    }
    neural_ratio = classifier_nn(
        model="resnet",
        z_score_x="structured",
        z_score_theta="none",
        hidden_features=256,
        embedding_net_x=cnn_net,
        embedding_net_theta=spline_net,
        **kwargs_classifier,
    )

    inference = SNRE(prior=prior, classifier=neural_ratio, device=device)

    simulator = get_batch_loop_simulator(smfe_simulator_mm)
    proposal = prior

    kwargs_mcmc = {"num_chains": 20, "num_workers": 3}

    for idx_round in range(N_rounds):

        theta = proposal.sample((N_sim_per_round,))

        x = simulate_in_batches(
            simulator, theta.cpu(), sim_batch_size=10, num_workers=N_workers
        )

        inference = inference.append_simulations(theta, x, data_device="cpu")

        ratio_estimator = inference.train(
            show_train_summary=True,
            validation_fraction=0.15,
            training_batch_size=50,
            learning_rate=0.0005,
            stop_after_epochs=20,
        )

        posterior = inference.build_posterior(
            ratio_estimator,
            sample_with="mcmc",
            mcmc_method="slice_np_vectorized",
            mcmc_parameters=kwargs_mcmc,
        )

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
