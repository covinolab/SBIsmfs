from typing import Tuple, Union
from configparser import ConfigParser
import torch
import numpy as np
import matplotlib.pyplot as plt
from sbi_smfs.utils.gsl_spline import c_spline
from sbi_smfs.utils.config_utils import get_config_parser


def evaluate_spline(
    x_eval: torch.Tensor, spline_nodes: torch.Tensor, config: Union[str, ConfigParser]
) -> torch.Tensor:
    """Evaluate the spline at x_eval.

    Parameters
    ----------
    x_eval: torch.Tensor
        Points at which to evaluate the spline.
    spline_nodes: torch.Tensor
        Spline nodes to construct the spline function.
    config: str, ConfigParser
        Config file with parameters.

    Returns
    -------
    y_eval: torch.Tensor
        Spline evaluated at x_eval.
    """
    if isinstance(x_eval, torch.Tensor):
        x_eval = x_eval.to(torch.float64).numpy()

    config = get_config_parser(config)
    x_axis = np.linspace(
        config.getfloat("SIMULATOR", "min_x"),
        config.getfloat("SIMULATOR", "max_x"),
        1000,
    )
    x_knots = np.linspace(
        config.getfloat("SIMULATOR", "min_x"),
        config.getfloat("SIMULATOR", "max_x"),
        config.getint("SIMULATOR", "num_knots"),
    )
    y_knots = np.ones(config.getint("SIMULATOR", "num_knots"))
    y_knots[1] = spline_nodes[0] + config.getint("SIMULATOR", "max_G_1")
    y_knots[-2] = spline_nodes[-1] + config.getint("SIMULATOR", "max_G_1")
    y_knots[0] = spline_nodes[0] + config.getint("SIMULATOR", "max_G_0")
    y_knots[-1] = spline_nodes[-1] + config.getint("SIMULATOR", "max_G_0")
    y_knots[2:-2] = spline_nodes
    y_eval = c_spline(x_knots, y_knots, x_eval)
    return y_eval


def plot_spline_ensemble(
    posterior_samples: torch.Tensor,
    num_splines: int,
    config: Union[str, ConfigParser],
    ylims: tuple = (-10, 10),
    label: Union[str, None] = None,
    **plot_kwargs,
) -> None:
    """Plot a ensemble of splines from the posterior samples.

    Parameters
    ----------
    posterior_samples: torch.Tensor
        Samples from the posterior distribution.
    num_splines: int
        Number of splines to plot.
    config: str, ConfigParser
        Config file with entries for simualtion.
    ylims: tuple
        Limits for y-axis.
    line_alpha: float
        Transparency of the splines.
    """
    config = get_config_parser(config)

    if not plot_kwargs:
        plot_kwargs = {"color": "blue", "alpha": 0.02}

    random_idx = torch.randperm(posterior_samples.shape[0])
    samples = posterior_samples[random_idx[:num_splines]]
    x_axis = np.linspace(
        config.getfloat("SIMULATOR", "min_x"),
        config.getfloat("SIMULATOR", "max_x"),
        1000,
    )
    x_knots = np.linspace(
        config.getfloat("SIMULATOR", "min_x"),
        config.getfloat("SIMULATOR", "max_x"),
        config.getint("SIMULATOR", "num_knots"),
    )

    for idx, sample in enumerate(samples.cpu()):
        if "Dx" in config["SIMULATOR"]:
            num_ind_var = 2
        else:
            num_ind_var = 3
        spline_nodes = sample[num_ind_var:].numpy()
        y_knots = np.ones(config.getint("SIMULATOR", "num_knots"))
        y_knots[1] = spline_nodes[0] + config.getint("SIMULATOR", "max_G_1")
        y_knots[-2] = spline_nodes[-1] + config.getint("SIMULATOR", "max_G_1")
        y_knots[0] = spline_nodes[0] + config.getint("SIMULATOR", "max_G_0")
        y_knots[-1] = spline_nodes[-1] + config.getint("SIMULATOR", "max_G_0")
        y_knots[2:-2] = spline_nodes
        y_axis = c_spline(x_knots, y_knots, x_axis)
        if idx == 0:
            plt.plot(x_axis, y_axis, **plot_kwargs, label=label)
        else:
            plt.plot(x_axis, y_axis, **plot_kwargs, label=None)
    plt.ylim(ylims)
    plt.xlim(
        config.getfloat("SIMULATOR", "min_x"), config.getfloat("SIMULATOR", "max_x")
    )
    plt.xlabel(r"Molecular extension x", fontsize=18)
    plt.ylabel(r"$G_0(x)$", fontsize=18)
    plt.grid(True)


def plot_spline_mean_with_error(
    posterior_samples: torch.Tensor,
    config: Union[str, ConfigParser],
    alpha: float = 0.05,
    ylims: tuple = (-10, 10),
    line_alpha: float = 1.0,
    **plot_kwargs,
) -> None:
    """Plot posterior mean and error for all parameters which build spline.

    Parameters
    ----------
    posterior_samples: torch.Tensor
        Samples from the posterior.
    config: str, ConfigParser
        Config file with entries for simualtion.
    alpha: float
        Confidence level for error bars.
    ylims: tuple
        Limits for y-axis.
    line_alpha: float
        Transparency of the spline function.
    plot_kwargs: dict
        Additional keyword arguments for matplotlib plt.plot function.
    """

    config = get_config_parser(config)
    if "Dx" in config["SIMULATOR"]:
        num_ind_var = 2
    else:
        num_ind_var = 3
    mean_posterior = torch.mean(posterior_samples.cpu(), dim=0)
    mean_posterior -= mean_posterior[num_ind_var:].mean(dim=0)
    x_axis = np.linspace(
        config.getfloat("SIMULATOR", "min_x"),
        config.getfloat("SIMULATOR", "max_x"),
        1000,
    )
    x_knots = np.linspace(
        config.getfloat("SIMULATOR", "min_x"),
        config.getfloat("SIMULATOR", "max_x"),
        config.getint("SIMULATOR", "num_knots"),
    )
    y_knots_err = np.zeros((2, config.getint("SIMULATOR", "num_knots")))
    y_knots_err[:, 2:-2] = np.abs(
        np.quantile(posterior_samples.cpu().numpy(), [alpha, 1 - alpha], axis=0)[
            :, num_ind_var:
        ]
        - mean_posterior[num_ind_var:].numpy()
    )
    spline_nodes = mean_posterior[num_ind_var:].numpy()
    y_knots = np.ones(config.getint("SIMULATOR", "num_knots"))
    y_knots[1] = spline_nodes[0] + config.getint("SIMULATOR", "max_G_1")
    y_knots[-2] = spline_nodes[-1] + config.getint("SIMULATOR", "max_G_1")
    y_knots[0] = spline_nodes[0] + config.getint("SIMULATOR", "max_G_0")
    y_knots[-1] = spline_nodes[-1] + config.getint("SIMULATOR", "max_G_0")
    y_knots[2:-2] = spline_nodes
    y_axis = c_spline(x_knots, y_knots, x_axis)

    plt.plot(x_axis, y_axis, alpha=line_alpha, color="blue", **plot_kwargs)
    plt.errorbar(
        x_knots,
        y_knots,
        yerr=y_knots_err,
        linestyle="",
        marker="o",
        color="blue",
        label="posterior mean",
        alpha=line_alpha,
    )
    plt.ylim(ylims)
    plt.xlim(
        config.getfloat("SIMULATOR", "min_x"), config.getfloat("SIMULATOR", "max_x")
    )
    plt.xlabel(r"Molecular extension x", fontsize=18)
    plt.ylabel(r"$G_0(x)$", fontsize=18)
    plt.grid(True)


def plot_spline(
    spline_nodes: torch.Tensor,
    config: Union[str, ConfigParser],
    ylims: tuple = (-10, 10),
    color: str = "red",
    line_alpha: float = 1.0,
    **plot_kwargs,
) -> None:
    """Construct and plot a spline function for representing a free-energy landscape.

    Parameters
    ----------
        spline_nodes: torch.Tensor
            Sample from the posterior.
        config: str, ConfigParser
            Config file with parameters.
        ylims: tuple
            Limits for y-axis.
        color: str
            Color of the spline.
        line_alpha: float
            Transparency of the spline.
        plot_kwargs: dict
            Additional keyword arguments for matplotlib plt.plot function.
    """
    config = get_config_parser(config)

    x_axis = np.linspace(
        config.getfloat("SIMULATOR", "min_x"),
        config.getfloat("SIMULATOR", "max_x"),
        1000,
    )
    x_knots = np.linspace(
        config.getfloat("SIMULATOR", "min_x"),
        config.getfloat("SIMULATOR", "max_x"),
        config.getint("SIMULATOR", "num_knots"),
    )

    y_knots = np.ones(config.getint("SIMULATOR", "num_knots"))
    y_knots[1] = spline_nodes[0] + config.getint("SIMULATOR", "max_G_1")
    y_knots[-2] = spline_nodes[-1] + config.getint("SIMULATOR", "max_G_1")
    y_knots[0] = spline_nodes[0] + config.getint("SIMULATOR", "max_G_0")
    y_knots[-1] = spline_nodes[-1] + config.getint("SIMULATOR", "max_G_0")

    if isinstance(spline_nodes, torch.Tensor):
        y_knots[2:-2] = spline_nodes.numpy()
    else:
        y_knots[2:-2] = spline_nodes
    y_axis = c_spline(x_knots, y_knots, x_axis)

    plt.plot(x_axis, y_axis, alpha=line_alpha, color=color, **plot_kwargs)
    plt.ylim(ylims)
    plt.xlim(
        config.getfloat("SIMULATOR", "min_x"), config.getfloat("SIMULATOR", "max_x")
    )
    plt.xlabel(r"Molecular extension x", fontsize=18)
    plt.ylabel(r"$G_0(x)$", fontsize=18)
    plt.grid(True)


def plot_inipendent_marginals(
    samples: torch.Tensor,
    config: Union[str, ConfigParser],
    figsize=(7, 3),
    fig_kwargs: dict = {},
    **hist_kwargs,
) -> tuple:
    """Plot independent marginals of the posterior samples.

    Parameters
    ----------
    samples: torch.Tensor
        Posterior samples.

    config: str, ConfigParser
        Config file with parameters.

    fig_size: tuple
        Figure size.

    hist_kwargs: dict
        Additional keyword arguments for matplotlib plt.hist function.

    Returns
    -------
    fig, axes: tuple
        Figure and axes objects.
    """

    config = get_config_parser(config)
    if "Dx" in config["SIMULATOR"]:
        num_ind_var = 2
        labels = [r"$log(D_q/D_x)$", r"$log(k_l)$"]
    else:
        num_ind_var = 3
        labels = [r"$log(D_x)$", r"$log(D_q)$", r"$log(k_l)$"]

    if isinstance(samples, torch.Tensor):
        samples = samples.cpu().numpy()

    fig, axes = plt.subplots(1, num_ind_var, figsize=figsize, **fig_kwargs)
    for i in range(num_ind_var):
        axes[i].hist(samples[:, i].flatten(), density=True, **hist_kwargs)
        axes[i].set_xlabel(labels[i])
        axes[i].set_yticks([])

    return fig, axes
