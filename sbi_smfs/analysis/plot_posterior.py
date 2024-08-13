from typing import List
import torch
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from typing import Union, Tuple
from sbi_smfs.utils.gsl_spline import c_spline
from sbi_smfs.utils.configurations import load_config_yaml
from dataclasses import dataclass

class SplinePlotter:
    def __init__(self, config: Union[str, DictConfig]):
        if isinstance(config, str):
            from sbi_smfs.utils.config_utils import load_config_yaml
            self.config = load_config_yaml(config)
        else:
            self.config = config
        
        self.x_axis = np.linspace(
            self.config.simulator.spline.min_x,
            self.config.simulator.spline.max_x,
            1000
        )
        self.x_knots = np.linspace(
            self.config.simulator.spline.min_x,
            self.config.simulator.spline.max_x,
            self.config.simulator.spline.num_knots
        )

    def plot_spline_ensemble(self, 
                             posterior_samples: torch.Tensor, 
                             num_splines: int, 
                             ylims: Tuple[float, float] = (-10, 10), 
                             line_alpha: float = 0.02) -> None:
        random_idx = torch.randperm(posterior_samples.shape[0])
        samples = posterior_samples[random_idx[:num_splines]]

        for sample in samples.cpu():
            y_knots = self._get_y_knots(sample)
            y_axis = c_spline(self.x_knots, y_knots, self.x_axis)
            plt.plot(self.x_axis, y_axis, alpha=line_alpha, color="blue")

        self._set_plot_properties(ylims)

    def plot_spline_mean_with_error(self, 
                                    posterior_samples: torch.Tensor, 
                                    alpha: float = 0.05, 
                                    ylims: Tuple[float, float] = (-10, 10), 
                                    line_alpha: float = 1.0) -> None:
        mean_posterior = torch.mean(posterior_samples.cpu(), dim=0)
        y_knots = self._get_y_knots(mean_posterior)
        y_axis = c_spline(self.x_knots, y_knots, self.x_axis)

        y_knots_err = np.zeros((2, self.config.simulator.spline.num_knots))
        y_knots_err[:, 2:-2] = np.abs(
            np.quantile(posterior_samples.cpu().numpy(), [alpha, 1 - alpha], axis=0)[:, self._num_ind_var():]
            - mean_posterior[self._num_ind_var():].numpy()
        )

        plt.plot(self.x_axis, y_axis - np.min(y_axis), alpha=line_alpha, color="blue")
        plt.errorbar(
            self.x_knots,
            y_knots - np.min(y_axis),
            yerr=y_knots_err,
            linestyle="",
            marker="o",
            color="blue",
            label="posterior mean",
            alpha=line_alpha,
        )

        self._set_plot_properties(ylims)

    def plot_spline(self, 
                    spline_nodes: torch.Tensor, 
                    ylims: Tuple[float, float] = (-10, 10), 
                    color: str = "red", 
                    line_alpha: float = 1.0) -> None:
        y_knots = self._get_y_knots(spline_nodes)
        y_axis = c_spline(self.x_knots, y_knots, self.x_axis)

        plt.plot(self.x_axis, y_axis, alpha=line_alpha, color=color)
        self._set_plot_properties(ylims)

    def plot_independent_marginals(self, samples: torch.Tensor, fig_kwargs: dict = {}) -> Tuple[plt.Figure, plt.Axes]:
        num_ind_var = self._num_ind_var()
        labels = self._get_labels()

        if isinstance(samples, torch.Tensor):
            samples = samples.cpu().numpy()

        fig, axes = plt.subplots(1, num_ind_var, **fig_kwargs)
        for i in range(num_ind_var):
            axes[i].hist(samples[:, i].flatten(), bins=100, density=True, histtype="step")
            axes[i].set_xlabel(labels[i])
            axes[i].set_yticks([])

        return fig, axes

    def _get_y_knots(self, sample: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        y_knots = np.ones(self.config.simulator.spline.num_knots)
        y_knots[1] = y_knots[-2] = self.config.simulator.spline.max_G_1
        y_knots[0] = y_knots[-1] = self.config.simulator.spline.max_G_0
        
        if isinstance(sample, torch.Tensor):
            y_knots[2:-2] = sample[self._num_ind_var():].cpu().numpy()
        else:
            y_knots[2:-2] = sample[self._num_ind_var():]
        
        return y_knots

    def _set_plot_properties(self, ylims: Tuple[float, float]) -> None:
        plt.ylim(ylims)
        plt.xlim(self.config.simulator.spline.min_x, self.config.simulator.spline.max_x)
        plt.xlabel(r"Molecular extension x", fontsize=18)
        plt.ylabel(r"$G_0(x)$", fontsize=18)
        plt.grid(True)

    def _num_ind_var(self) -> int:
        return 2 if self.config.simulator.log_dx is None else 3

    def _get_labels(self) -> List[str]:
        if self.config.simulator.log_dx is None:
            return [r"$log(D_q/D_x)$", r"$log(k_l)$"]
        else:
            return [r"$log(D_x)$", r"$log(D_q)$", r"$log(k_l)$"]