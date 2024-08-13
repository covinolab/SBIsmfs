from typing import Union, Tuple, List, Optional
import torch
import numpy as np
from omegaconf import DictConfig
from sbi.utils.user_input_checks import get_batch_loop_simulator
from sbi.inference import simulate_for_sbi
from sbi_smfs.utils.configurations import load_config_yaml
from sbi_smfs.utils.summary_stats import build_transition_matricies
from sbi_smfs.simulator.brownian_integrator import brownian_integrator
from sbi_smfs.inference.priors import SplinePrior

class Simulator:
    def __init__(self, config: Union[str, DictConfig]):
        self.config = load_config_yaml(config)
        self.simulator_params = self.config.simulator
        self.summary_stats_params = self.config.posterior.summary_statistics

    def __call__(self, parameters: torch.Tensor) -> Optional[torch.Tensor]:
        return self.run_simulation(parameters)

    def run_simulation(self, parameters: torch.Tensor) -> Optional[torch.Tensor]:
        Dx, Dq, k = self._process_diffusion_parameters(parameters)
        x_knots, y_knots = self._process_spline_parameters(parameters)
        x_init, q_init = self._get_initial_positions()

        q = brownian_integrator(
            x0=x_init,
            q0=q_init,
            Dx=Dx,
            Dq=Dq,
            x_knots=x_knots,
            y_knots=y_knots,
            k=k,
            N=self.simulator_params.num_steps,
            dt=self.simulator_params.dt,
            fs=self.simulator_params.saving_freq,
        )

        if q is None:
            return None

        return build_transition_matricies(
            q,
            self.summary_stats_params.lag_times,
            self.summary_stats_params.min_bin,
            self.summary_stats_params.max_bin,
            self.summary_stats_params.num_bins
        )

    def _process_diffusion_parameters(self, parameters: torch.Tensor) -> Tuple[float, float, float]:
        if self.simulator_params.log_dx is None:
            Dx, Dq, k = (10 ** param.item() for param in parameters[:3])
        else:
            Dx = 10 ** self.simulator_params.log_dx
            Dq = Dx * (10 ** parameters[0].item())
            k = 10 ** parameters[1].item()
        return Dx, Dq, k

    def _process_spline_parameters(self, parameters: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        num_ind_params = 3 if self.simulator_params.log_dx is None else 2
        spline_config = self.simulator_params.spline
        x_knots = np.linspace(spline_config.min_x, spline_config.max_x, spline_config.num_knots)
        y_knots = np.zeros(spline_config.num_knots)

        y_knots[0] = spline_config.max_G_0 + parameters[num_ind_params].item()
        y_knots[-1] = spline_config.max_G_0 + parameters[-1].item()
        y_knots[1] = spline_config.max_G_1 + parameters[num_ind_params].item()
        y_knots[-2] = spline_config.max_G_1 + parameters[-1].item()
        y_knots[2:-2] = parameters[num_ind_params:-1].numpy()

        return x_knots, y_knots

    def _get_initial_positions(self):
        init_range = self.simulator_params.init_xq_range
        x_init = np.random.uniform(low=init_range[0], high=init_range[1])
        q_init = np.random.uniform(low=init_range[0], high=init_range[1])
        return x_init, q_init
    

def generate_simulations(
    config_file: str,
    num_sim: int,
    num_workers: int,
    file_name: Union[str, None] = None,
    show_progressbar: bool = False,
    save_as_file: bool = False,
) -> Union[Tuple[torch.Tensor, torch.Tensor], None]:
    """Run simulations with parameters from prior.

    Parameters
    ----------
    config_file: str
        Config file name.
    num_sim: int
        Number of simulations to run.
    num_workers: int
        Number of workers to use for simulation.
    file_name: str, None
        File name to save the simulations to.
    show_progressbar: bool
        Whether to show a progress bar.
    save_as_file: bool
        Whether to save the simulations to a file.

    Returns
    -------
    theta: torch.Tensor
        Parameters used for the simulations.
    x: torch.Tensor
        Simulations.
    """

    if save_as_file:
        assert isinstance(
            file_name, str
        ), "You need to specify a filename if save_as_file=True"

    prior = SplinePrior(config_file)
    simulator = get_batch_loop_simulator(Simulator(config_file))    

    theta, x = simulate_for_sbi(
        simulator,
        prior,
        num_simulations=num_sim,
        num_workers=num_workers,
        show_progress_bar=show_progressbar,
    )

    if save_as_file:
        x_file_name = f"{file_name}_x.pt"
        theta_file_name = f"{file_name}_theta.pt"
        torch.save(x, x_file_name)
        torch.save(theta, theta_file_name)
    else:
        return theta, x