import torch
import numpy as np
import matplotlib.pyplot as plt
from sbi_smfs.utils.gsl_spline import c_spline
from sbi_smfs.utils.config_utils import get_config_parser


def plot_spline_ensemble(posterior_samples, num_splines, config, ylims=(-10, 10)):
    """Plot a ensemble of splines from the posterior."""
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


def plot_spline_mean_with_error(posterior_samples, config, alpha=0.05, ylims=(-10, 10)):
    """Plot the posterior mean of the spline nodes with error bars."""

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


def plot_spline(spline_nodes, config, ylims=(-10, 10), color="red"):
    """Plot a ensemble of splines from the posterior."""
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