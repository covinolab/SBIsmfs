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


def featurize_trajectory(q: np.ndarray, lag_times: list[int]) -> list:
    """
    Featurizes trajectory by computing summary statistics.

    Parameters
    ----------
    q : np.ndarray
        Trajectory
    lag_times : list
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


def build_transition_matricies(
    q: np.ndarray, lag_times: list[int], min_bin: float, max_bin: float, num_bins: int
) -> torch.tensor:
    """
    Builds transition matricies for given lag times.

    Parameters
    ----------
    q : np.ndarray
        Trajectory
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
    matricies : torch.tensor
        Transition matricies for given lag times
    """

    bins = np.linspace(min_bin, max_bin, num_bins + 1)
    binned_q = bin_trajectory(q, bins)
    matricies = np.array(
        [
            build_transition_matrix(binned_q, len(bins) - 1, t=lag_time)
            for lag_time in lag_times
        ]
    )

    matricies = np.float32(matricies)

    return torch.from_numpy(np.nan_to_num(matricies, nan=0.0)).flatten()


def compute_stats(
    trajectory: np.ndarray, config: Union[str, ConfigParser]
) -> torch.Tensor:
    """Computes summary statistics for given trajectory.

    Parameters
    ----------
    trajectory : np.ndarray
        Trajectory, shape (num_samples, num_dimensions) or (num_dimensions,)
    config : str, ConfigParser
        Path to config file or ConfigParser object

    Returns
    -------
    summary_stats : torch.Tensor
        Summary statistics
    """

    config = get_config_parser(config)

    if isinstance(trajectory, torch.Tensor):
        trajectory = trajectory.numpy()
    if len(trajectory.shape) > 2:
        raise ValueError("Trajectory should be 1D or 2D array.")
    elif len(trajectory.shape) == 2:
        summary_stats = [
            build_transition_matricies(
                trajectory[:, i],
                config.getlistint("SUMMARY_STATS", "lag_times"),
                config.getfloat("SUMMARY_STATS", "min_bin"),
                config.getfloat("SUMMARY_STATS", "max_bin"),
                config.getint("SUMMARY_STATS", "num_bins"),
            )
            for i in range(trajectory.shape[0])
        ]
        summary_stats = torch.stack(summary_stats, dim=0)
    elif len(trajectory.shape) == 1:
        summary_stats = build_transition_matricies(
            trajectory,
            config.getlistint("SUMMARY_STATS", "lag_times"),
            config.getfloat("SUMMARY_STATS", "min_bin"),
            config.getfloat("SUMMARY_STATS", "max_bin"),
            config.getint("SUMMARY_STATS", "num_bins"),
        )
        summary_stats = summary_stats

    return summary_stats


def check_if_observation_contains_features(
    observation: torch.tensor, config: Union[str, ConfigParser]
) -> bool:
    """Checks if observation contains features.

    Parameters
    ----------
    observation : torch.Tensor
        Observation
    config : str
        Path to config file

    Returns
    -------
    bool
        True if observation contains features, False otherwise
    """

    expected_shape = len(config.getlistint("SUMMARY_STATS", "lag_times")) * (
        config.getint("SUMMARY_STATS", "num_bins") ** 2
    )

    if expected_shape != observation.shape[-1]:
        return False
    else:
        return True
