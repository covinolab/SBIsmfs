import numpy as np
import torch
from sbi_smfs.utils.stats_utils import prop_stats, moments, bin_trajectory, build_transition_matrix


def featurize_trajectory(q, lag_times):
    '''
    Computes summary statsitics for trajecotires.

    Parameters
    ----------
    q : np.Array
        Trajectory.
    lag_times : list
        The lag times used for calculation of the propagator.

    Returns
    -------
    features : list
        List with features computed from the trajectory.

    '''

    propagators_stats = []
    for lag_time in lag_times:
        for stat in prop_stats(q, t=lag_time):
            propagators_stats.append(stat)

    moments_q = moments(q)

    features = [*moments_q, *propagators_stats]

    return features


def build_transition_matricies(q, lag_times, min_bin, max_bin, num_bins):
    """
    Builds transition matricies from trajectory
    """
    
    bins = np.linspace(min_bin, max_bin, num_bins + 1)
    binned_q = bin_trajectory(q, bins)
    matricies = np.array([
        build_transition_matrix(binned_q, len(bins) - 1, t=lag_time)
        for lag_time in lag_times
    ])

    matricies = np.float32(matricies)

    return torch.from_numpy(np.nan_to_num(matricies, nan=0.0)).flatten()