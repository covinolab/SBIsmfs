from typing import Union
from configparser import ConfigParser
import numpy as np
import torch
from sbi_smfs.utils.stats_utils import (
    prop_stats,
    moments,
    bin_trajectory,
    build_transition_matrix,
)
from sbi_smfs.utils.config_utils import get_config_parser


def featurize_trajectory(q: np.ndarray, lag_times: list[int]) -> list[float]:
    """
    Featurizes trajectory by computing the first four moments of the stationary distribution and step sizes for specified lag times.

    Parameters
    ----------
    q : np.ndarray
        Trajectory
    lag_times : list[int]
        List of lag times to compute summary statistics

    Returns
    -------
    features : list
        List of summary statistics
    """

    propagators_stats = []
    for lag_time in lag_times:
        for stat in prop_stats(q, t=lag_time):
            propagators_stats.append(stat)

    moments_q = moments(q)

    features = [*moments_q, *propagators_stats]

    return features


def build_transition_matrices(
    q: np.ndarray, lag_times: list[int], min_bin: float, max_bin: float, num_bins: int
) -> torch.Tensor:
    """
    Builds transition matrices for given lag times.

    Parameters
    ----------
    q : np.ndarray or list of np.ndarray
        Single trajectory as a numpy array or a list of numpy arrays (each representing a trajectory)
    lag_times : list
        List of lag times to compute summary statistics
    min_bin : float
        Minimum value of the bins
    max_bin : float
        Maximum value of the bins
    num_bins : int
        Number of bins between min_bin and max_bin

    Returns
    -------
    matrices : torch.Tensor
        Transition matrices for given lag times
    """

    bins = np.linspace(min_bin, max_bin, num_bins + 1)
    if isinstance(q, np.ndarray):
        q = [q]
    binned_q = np.stack([bin_trajectory(_q, bins) for _q in q], axis=0)
    matricies = np.array(
        [
            build_transition_matrix(binned_q, len(bins) - 1, t=lag_time)
            for lag_time in lag_times
        ]
    )

    matricies = np.float32(matricies)

    return torch.from_numpy(np.nan_to_num(matricies, nan=0.0)).flatten()


def compute_stats(
    q: np.ndarray, config: Union[str, ConfigParser]
) -> torch.Tensor:
    
    config = get_config_parser(config)

    if isinstance(q, torch.Tensor):
        q = q.numpy()
    if isinstance(q, list):
        q = [qi.numpy() if isinstance(qi, torch.Tensor) else qi for qi in q]
        if len(q) > 1:
            print("Using multiple of trajectories for summary statistics computation")
        
    summary_stats = build_transition_matrices(
        q,
        config.getlistint("SUMMARY_STATS", "lag_times"),
        config.getfloat("SUMMARY_STATS", "min_bin"),
        config.getfloat("SUMMARY_STATS", "max_bin"),
        config.getint("SUMMARY_STATS", "num_bins"),
    )

    return summary_stats


def check_if_observation_contains_features(
    observation: Union[torch.Tensor, list], config: ConfigParser
) -> bool:
    """
    Checks if observation contains features.

    Parameters
    ----------
    observation : torch.Tensor
        Observation
    config : ConfigParser
        ConfigParser object

    Returns
    -------
    bool
        True if observation contains features, False otherwise
    """

    expected_shape = len(config.getlistint("SUMMARY_STATS", "lag_times")) * (
        config.getint("SUMMARY_STATS", "num_bins") ** 2
    )
    if isinstance(observation, list):
        return False
    elif expected_shape != observation.shape[-1]:
        return False
    else:
        return True
