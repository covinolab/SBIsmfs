import torch
import numpy as np
import matplotlib.pyplot as plt
from sbi_smfs.utils.gsl_spline import c_spline
from sbi_smfs.utils.config_utils import get_config_parser


def plot_spline_ensemble(
    posterior_samples: torch.Tensor,
    num_splines: int,
    config: str,
    ylims: tuple = (-10, 10),
) -> None:
    """Plot a ensemble of splines from the posterior.

    Parameters
    ----------
    posterior_samples: torch.Tensor
        Samples from the posterior.
    num_splines: int
        Number of splines to plot.
    config: str, ConfigParser
        Config file with entries for simualtion.
    ylims: tuple
        Limits for y-axis.
    """
    config = get_config_parser(config)

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

    for sample in samples.cpu():
        y_knots = np.ones(config.getint("SIMULATOR", "num_knots"))
        y_knots[1] = y_knots[-2] = config.getint("SIMULATOR", "max_G_1")
        y_knots[0] = y_knots[-1] = config.getint("SIMULATOR", "max_G_0")
        if "Dx" in config["SIMULATOR"]:
            num_ind_var = 2
        else:
            num_ind_var = 3
        y_knots[2:-2] = sample[num_ind_var:].numpy()
        y_axis = c_spline(x_knots, y_knots, x_axis)
        plt.plot(x_axis, y_axis, alpha=0.02, color="blue")
    plt.ylim(ylims)
    plt.xlim(
        config.getfloat("SIMULATOR", "min_x"), config.getfloat("SIMULATOR", "max_x")
    )
    plt.xlabel(r"Molecular extension x", fontsize=18)
    plt.ylabel(r"$G_0(x)$", fontsize=18)
    plt.grid(True)


def plot_spline_mean_with_error(
    posterior_samples: torch.Tensor,
    config: str,
    alpha: float = 0.05,
    ylims: tuple = (-10, 10),
) -> None:
    """Plot the posterior mean of the spline nodes with error bars.

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
    """

    config = get_config_parser(config)
    if "Dx" in config["SIMULATOR"]:
        num_ind_var = 2
    else:
        num_ind_var = 3
    mean_posterior = torch.mean(posterior_samples.cpu(), dim=0)
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
    y_knots = np.ones(config.getint("SIMULATOR", "num_knots"))
    y_knots[1] = y_knots[-2] = config.getint("SIMULATOR", "max_G_1")
    y_knots[0] = y_knots[-1] = config.getint("SIMULATOR", "max_G_0")
    y_knots[2:-2] = mean_posterior[num_ind_var:].numpy()
    y_axis = c_spline(x_knots, y_knots, x_axis)

    plt.plot(x_axis, y_axis - np.min(y_axis), alpha=1, color="blue")
    plt.errorbar(
        x_knots,
        y_knots - np.min(y_axis),
        yerr=y_knots_err,
        linestyle="",
        marker="o",
        color="blue",
        label="posterior mean",
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
    config: str,
    ylims: tuple = (-10, 10),
    color: str = "red",
) -> None:
    """Plot a ensemble of splines from the posterior.

    Parameters
    ----------
        spline_nodes: torch.Tensor
            Sample from the posterior.
        config: str, ConfigParser
            Config file with entries for simualtion.
        ylims: tuple
            Limits for y-axis.
        color: str
            Color of the spline.
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
    y_knots[1] = y_knots[-2] = config.getint("SIMULATOR", "max_G_1")
    y_knots[0] = y_knots[-1] = config.getint("SIMULATOR", "max_G_0")
    if isinstance(spline_nodes, torch.Tensor):
        y_knots[2:-2] = spline_nodes.numpy()
    else:
        y_knots[2:-2] = spline_nodes
    y_axis = c_spline(x_knots, y_knots, x_axis)

    plt.plot(x_axis, y_axis, alpha=1, color=color)
    plt.ylim(ylims)
    plt.xlim(
        config.getfloat("SIMULATOR", "min_x"), config.getfloat("SIMULATOR", "max_x")
    )
    plt.xlabel(r"Molecular extension x", fontsize=18)
    plt.ylabel(r"$G_0(x)$", fontsize=18)
    plt.grid(True)


def plot_inipendent_marginals(
    samples: torch.Tensor, config: str, fig_kwars: dict = {}
) -> tuple:
    """Plot the posterior marginal distributions of the independent variables.

    Parameters
    ----------
    samples: torch.Tensor
        Samples from the posterior.
    config: str, ConfigParser
        Config file with entries for simualtion.

    Returns
    -------
    fig, axes: matplotlib.figure.Figure, matplotlib.axes.Axes
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

    fig, axes = plt.subplots(1, num_ind_var, **fig_kwars)
    for i in range(num_ind_var):
        axes[i].hist(samples[:, i].flatten(), bins=100, density=True, histtype="step")
        axes[i].set_xlabel(labels[i])
        axes[i].set_yticks([])

    return fig, axes
