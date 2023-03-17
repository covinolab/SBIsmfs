import torch
import numpy as np

from stats_utils import featurize_trajectory, bin_trajectory, build_transition_matrix
from brownian_integrator import brownian_integrator

import config_real_data as config


def smfe_simulator(parameters: torch.Tensor):
    '''
    Simulator for single-molecule force-spectroscopy experiments.
    The molecular free energy surface is described by a cubic spline

    Parameters
    ----------
    parameters : torch.Tensor
        Parameters of the simulator.

    Returns
    -------
    summary_stats : torch.Tensor
        Summary statistics of simulation perform with parameters.
    '''

    # Define integration constants
    dt = config.dt
    N = config.N
    saving_freq = config.saving_freq
    Dx = np.random.uniform(0.28, 0.48) #config.Dx
    N_knots = config.N_knots
    min_x = config.min_x
    max_x = config.max_x
    max_G = config.max_G_0
    init_xq_range = (-2, 2)

    # Select spline knots from parameters
    x_knots = np.linspace(min_x, max_x, N_knots)
    y_knots = np.zeros(N_knots)
    y_knots[0] = y_knots[-1] = config.max_G_0
    y_knots[1] = y_knots[-2] = config.max_G_1

    for idx, difference in enumerate(parameters[2:].numpy()):
        y_knots[2 + idx + 1] = y_knots[2 + idx] + difference
        y_knots[2: -2] -= y_knots[2: -2].mean()

    # Select random initial position for x and q
    x_init = np.random.uniform(low=init_xq_range[0], high=init_xq_range[1])
    q_init = np.random.uniform(low=init_xq_range[0], high=init_xq_range[1])

    # Select integration constants from parameters
    Dq = 10 ** parameters[0].item()
    k = parameters[1].item()

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
            fs=saving_freq
    )

    # Compute summary statistics
    lag_times = np.unique(np.logspace(0, 5, 20, dtype=int))
    summary_stats = featurize_trajectory(q, lag_times=lag_times)

    return torch.tensor(summary_stats)


def smfe_simulator_mm(parameters: torch.Tensor):
    '''
    Simulator for single-molecule force-spectroscopy experiments.
    The molecular free energy surface is described by a cubic spline

    Parameters
    ----------
    parameters : torch.Tensor
        Parameters of the simulator.

    Returns
    -------
    summary_stats : torch.Tensor
        Summary statistics of simulation perform with parameters.
    '''
    
    # Define integration constants
    dt = config.dt
    N = config.N
    saving_freq = config.saving_freq
    Dx = config.Dx
    N_knots = config.N_knots
    min_x = config.min_x
    max_x = config.max_x
    max_G = config.max_G_0
    init_xq_range = config.init_xq_range

    # Select spline knots from parameters
    x_knots = np.linspace(min_x, max_x, N_knots)
    y_knots = np.zeros(N_knots)
    y_knots[0] = config.max_G_0 + parameters[2].numpy()
    y_knots[-1] = config.max_G_0 + parameters[-1].numpy()
    y_knots[1] = config.max_G_1 + parameters[2].numpy()
    y_knots[-2] = config.max_G_1 + parameters[-1].numpy()
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
            fs=saving_freq
    )

    # Compute markov matricies
    bins = np.linspace(config.min_bin, config.max_bin, config.num_bins + 1)
    binned_q = bin_trajectory(q, bins)
    matricies = np.array([
        build_transition_matrix(binned_q, len(bins) - 1, t=lag_time)
        for lag_time in config.lag_times
    ])

    matricies = np.float32(matricies)

    return torch.from_numpy(np.nan_to_num(matricies, nan=0.0)).flatten()
