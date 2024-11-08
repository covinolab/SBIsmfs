from typing import Union, Tuple
import torch
import numpy as np
from functools import partial
import configparser
from sbi_smfs.utils.config_utils import get_config_parser
from sbi_smfs.utils.summary_stats import build_transition_matrices
from sbi_smfs.simulator.brownian_integrator import brownian_integrator


def smfe_simulator_mm(
    parameters: torch.Tensor,
    dt: float,
    N: int,
    saving_freq: int,
    Dx: Union[float, None],
    N_knots: int,
    min_x: float,
    max_x: float,
    max_G_0: float,
    max_G_1: float,
    init_xq_range: Tuple[float, float],
    min_bin: float,
    max_bin: float,
    num_bins: int,
    lag_times: list[int],
    return_q: bool = False,
) -> torch.Tensor:
    """
    Simulator for single-molecule force-spectroscopy experiments.
    The molecular free energy surface is described by a cubic spline

    Parameters
    ----------
    parameters : torch.Tensor
        Changeable_parameters of the simulator.
        (Dx, Dq, k, spline_nodes)
    dt : float
        Integration timestep
    N : int
        Total number of steps.
    saving_freq : int
        Saving frequency during integration.
    Dx : float
        Diffusion coefficient in x direction.
    N_knots : int
        Number of knots in spline potential.
    min_x : float
        Minimal x value of spline potential.
    max_x : float
        Maximal x value of spline potential.
    max_G_0 : float
        Additional barrier at the end of spline for first and last node.
    max_G_1 : float
        Additional barrier at the end of spline for second and second last node.
    init_xq_range: Tuple[float, float]
        Range of inital positions.
    min_bin : float
        Outer left bin edge.
    max_bin: float,
        Outer right bin edge.
    num_bins: int
        Number of bins for transition matrix.
    lag_times : list[int]
        List of lag times for which a transition matrix is generated.

    Returns
    -------
    summary_stats : torch.Tensor
        Summary statistics of simulation perform with parameters.
    """

    # Select integration constants from parameters
    if Dx is None:
        num_ind_params = 3
        Dx = 10 ** parameters[0].item()
        Dq = 10 ** parameters[1].item()
        k = 10 ** parameters[2].item()
    elif isinstance(Dx, (float, int)):
        num_ind_params = 2
        Dq = Dx * (10 ** parameters[0].item())
        k = 10 ** parameters[1].item()
    else:
        raise NotImplementedError("Dx should be either float or None")

    # Ensure parameters is a numpy array for further processing
    if isinstance(parameters, torch.Tensor):
        parameters = parameters.detach().cpu().numpy()

    # Select spline knots from parameters
    x_knots = np.linspace(min_x, max_x, N_knots)
    y_knots = np.zeros(N_knots)
    y_knots[0] = max_G_0 + parameters[num_ind_params]
    y_knots[-1] = max_G_0 + parameters[-1]
    y_knots[1] = max_G_1 + parameters[num_ind_params]
    y_knots[-2] = max_G_1 + parameters[-1]
    y_knots[2:-2] = parameters[num_ind_params:]

    # Select random initial position for x and q
    x_init = np.random.uniform(low=init_xq_range[0], high=init_xq_range[1])
    q_init = np.random.uniform(low=init_xq_range[0], high=init_xq_range[1])

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

    if q is None:
        raise ValueError("Simulation failed!")
    
    if return_q:
        return torch.from_numpy(q)

    matrices = build_transition_matrices(q, lag_times, min_bin, max_bin, num_bins)
    return matrices


def get_simulator_from_config(
    config_file: Union[str, configparser.ConfigParser], return_q: bool = False
) -> partial:
    """Get simulator function from config file.

    Parameters
    ----------
    config_file : str
        Path to config file.
    return_q : bool, optional
        If True, return the q trajectory from the simulator. Default is False.

    Returns
    -------
    simulator : function
        Simulator function.
    """

    config = get_config_parser(config_file, validate=True)
    if "Dx" in config["SIMULATOR"]:
        Dx = config.getfloat("SIMULATOR", "Dx")
    elif "type_Dx" in config["PRIORS"] and "parameters_Dx" in config["PRIORS"]:
        Dx = None
    else:
        raise NotImplementedError("Dx not properly specified in config file!")


    return partial(
        smfe_simulator_mm,
        dt=config.getfloat("SIMULATOR", "dt"),
        N=config.getint("SIMULATOR", "num_steps"),
        saving_freq=config.getint("SIMULATOR", "saving_freq"),
        Dx=Dx,
        N_knots=config.getint("SIMULATOR", "num_knots"),
        min_x=config.getfloat("SIMULATOR", "min_x"),
        max_x=config.getfloat("SIMULATOR", "max_x"),
        max_G_0=config.getfloat("SIMULATOR", "max_G_0"),
        max_G_1=config.getfloat("SIMULATOR", "max_G_1"),
        init_xq_range=config.gettuplefloat("SIMULATOR", "init_xq_range"),
        min_bin=config.getfloat("SUMMARY_STATS", "min_bin"),
        max_bin=config.getfloat("SUMMARY_STATS", "max_bin"),
        num_bins=config.getint("SUMMARY_STATS", "num_bins"),
        lag_times=config.getlistint("SUMMARY_STATS", "lag_times"),
        return_q=return_q,
    )
