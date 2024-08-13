import pickle
from typing import Union, Optional, Tuple
import torch
from omegaconf import OmegaConf, DictConfig
from sbi.inference import SNPE
from sbi.utils.user_input_checks import get_batch_loop_simulator
from sbi.simulators.simutils import simulate_in_batches
from sbi_smfs.utils.configurations import load_config_yaml
from sbi_smfs.utils.observation import Observation
from sbi_smfs.simulator.simulator import Simulator
from sbi_smfs.inference.build_nn_models import build_npe_model
from sbi_smfs.inference.priors import SplinePrior



class PosteriorTrainer:
    def __init__(self, config: Union[str, DictConfig], device: str = "cpu") -> None:
        self.config = self.load_config_yaml(config)
        self.device = device
        self.neural_posterior = self.build_npe_model(config)

    def _prepare_observation(self, observation: Union[str, torch.Tensor]) -> torch.Tensor:
        return Observation(self.config, observation)

    def _save_posterior(self, posterior: SNPE, file_name: str) -> None:
        with open(file_name, "wb") as f:
            pickle.dump(posterior, f)

    def _load_training_data(self, theta: str, q: str) -> Tuple[torch.Tensor, torch.Tensor]:
        theta = torch.load(theta)
        q = torch.load(q)
        return theta, q

    def train_sequential(self, num_workers, save_interval, posterior_file) -> Optional[Tuple[SNPE, SNPE]]:
        self.prior = SplinePrior(self.config, self.device)
        self.simulator = Simulator(self.config)
        self.observation = self._prepare_observation(self.config)
        train_parameters = self.config.posterior.training
        inference = SNPE(
            prior=self.prior,
            density_estimator=self.neural_posterior,
            device=self.device
        )
        proposal = self.prior
        batched_simulator = get_batch_loop_simulator(self.simulator)

        for idx_round in range(self.config.training.num_rounds):
            theta = proposal.sample((self.config.training.num_sim_per_round,))
            x = simulate_in_batches(
                batched_simulator,
                theta.cpu(),
                sim_batch_size=self.config.training.num_sim_per_round // num_workers,
                num_workers=num_workers
            )
            inference = inference.append_simulations(
                theta,
                x,
                proposal=proposal,
                data_device="cpu"
            )
            inference = inference.append_simulations(theta, x, proposal=proposal, data_device="cpu")
            density_estimator = inference.train(show_train_summary=True, **train_parameters)
            posterior = inference.build_posterior(density_estimator)
            posterior.set_default_x(self.observation.q)
            if idx_round % save_interval == 0 and isinstance(posterior_file, str):
                self._save_posterior(posterior, f"{posterior_file}_{idx_round}.pkl")

        if isinstance(posterior_file, str):
            self._save_posterior(posterior, f"{posterior_file}.pkl")
            self._save_posterior(inference, f"{posterior_file}_inference.pkl")
        else:
            return posterior, inference

    def train_amortized(self, theta_train: str, q_train: str, posterior_file: Union[str, None] = None) -> Optional[Tupel[SNPE, SNPE]]:
        self.prior = SplinePrior(self.config, self.device)
        inference = SNPE(
            prior=self.prior,
            density_estimator=self.neural_posterior,
            device=self.device
        )
        train_parameters = self.config.posterior.training
        theta, x = self._load_training_data(theta_train, q_train)
        inference = inference.append_simulations(theta, x, data_device="cpu")
        density_estimator = inference.train(show_train_summary=True, **train_parameters)
        posterior = inference.build_posterior(density_estimator)
        if isinstance(posterior_file, str):
            self._save_posterior(posterior, f"{posterior_file}.pkl")
            self._save_posterior(inference, f"{posterior_file}_inference.pkl")
        else:
            return posterior, inference
