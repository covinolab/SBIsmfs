import torch
import numpy as np
from functools import partial
import configparser
from sbi_smfs.utils.summary_stats import build_transition_matricies
from sbi_smfs.simulator.brownian_integrator import brownian_integrator


def smfe_simulator_mm(
    parameters: torch.Tensor,
    dt,
    N,
    saving_freq,
    Dx,
    N_knots,
    min_x,
    max_x,
    max_G_0,
    max_G_1,
    init_xq_range,
    min_bin,
    max_bin,
    num_bins,
    lag_times,
):
    """
    Simulator for single-molecule force-spectroscopy experiments.
    The molecular free energy surface is described by a cubic spline

    Parameters
    ----------
    parameters : torch.Tensor
        Changeable_parameters of the simulator.
        parameters[0] : log10(Dq/Dx)
        parameters[1] : log10(kappa_l)
        parameters[2:]: G(x_i) of splines.
    dt : float
        Integration timestep
    N : int
        Totoal number of steps.
    saving_freq : int
        Saving frequency during integration.
    Dx : float
        Diffusion coefficient in x direction.
    N_knots : int
        Number of knots in spline potential.
    min_x : float
        Minimal x value of spline potenital.
    max_x : float
        Maximal x value of spline potenital.
    max_G_0 : float
        Additional barrier at the end of spline for first and last node.
    max_G_1 : float
        Additional barrier at the end of spline for seconf and second last node.
    init_xq_range: tuple[float, float]
        Range of inital positions.
    min_bin : float
        Outer left bin edge.
    max_bin:
        Outer right bin edge.
    num_bins: int
        Number of bins for transition matrix.
    lag_times: List[Int]:
        List of lag times for which a transition matrix is generated.

    Returns
    -------
    summary_stats : torch.Tensor
        Summary statistics of simulation perform with parameters.
    """

    # Select spline knots from parameters
    x_knots = np.linspace(min_x, max_x, N_knots)
    y_knots = np.zeros(N_knots)
    y_knots[0] = max_G_0 + parameters[2].numpy()
    y_knots[-1] = max_G_0 + parameters[-1].numpy()
    y_knots[1] = max_G_1 + parameters[2].numpy()
    y_knots[-2] = max_G_1 + parameters[-1].numpy()
    y_knots[2:-2] = parameters[2:].numpy()

    # Select random initial position for x and q
    x_init = np.random.uniform(low=init_xq_range[0], high=init_xq_range[1])
    q_init = np.random.uniform(low=init_xq_range[0], high=init_xq_range[1])

    # Select integration constants from parameters
    Dq = Dx * (10 ** parameters[0].item())
    k = 10 ** parameters[1].item()

    # Call integrator
    q = brownian_integrator(
        x0=x_init,
        q0=q_init,
        Dx=Dx,
        Dq=Dq,
        x_knots=x_knots,
        y_knots=y_knots,
        k=k,
        N=N,
        dt=dt,
        fs=saving_freq,
    )

    matrices = build_transition_matricies(q, lag_times, min_bin, max_bin, num_bins)
    return matrices


def get_simulator_from_config(config_file):
    """
    Initiates SMFS-Simulatr with integration constants from config file.

    Parameters
    ----------
     config_file: str
        Config file with entries for simualtion.

    Returns
    -------
    summary_stats : torch.Tensor
        Summary statistics of simulation perform with parameters.
    """

    config = configparser.ConfigParser(
        converters={
            "listint": lambda x: [int(i.strip()) for i in x.split(",")],
            "listfloat": lambda x: [float(i.strip()) for i in x.split(",")],
        }
    )
    config.read(config_file)

    return partial(
        smfe_simulator_mm,
        dt=config.getfloat("SIMULATOR", "dt"),
        N=config.getint("SIMULATOR", "num_steps"),
        saving_freq=config.getint("SIMULATOR", "saving_freq"),
        Dx=config.getfloat("SIMULATOR", "Dx"),
        N_knots=config.getint("SIMULATOR", "num_knots"),
        min_x=config.getfloat("SIMULATOR", "min_x"),
        max_x=config.getfloat("SIMULATOR", "max_x"),
        max_G_0=config.getfloat("SIMULATOR", "max_G_0"),
        max_G_1=config.getfloat("SIMULATOR", "max_G_1"),
        init_xq_range=config.getlistfloat("SIMULATOR", "init_xq_range"),
        min_bin=config.getfloat("SUMMARY_STATS", "min_bin"),
        max_bin=config.getfloat("SUMMARY_STATS", "max_bin"),
        num_bins=config.getint("SUMMARY_STATS", "num_bins"),
        lag_times=config.getlistint("SUMMARY_STATS", "lag_times"),
    )
