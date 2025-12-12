from typing import Union
from configparser import ConfigParser
import numpy as np
import torch
from sbi_smfs.utils.stats_utils import (
    prop_stats,
    moments,
    bin_trajectory,
    build_transition_matrix,
    compute_normalized_fft_magnitudes,
    compute_psd_welch,
)
from sbi_smfs.utils.config_utils import get_config_parser


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


def featurize_trajectory(
    q: np.ndarray,
    lag_times: list[int],
    min_bin: float,
    max_bin: float,
    num_bins: int,
    num_freq: int = 50,
) -> torch.Tensor:
    """
    Build a feature vector from transition matrices and optional FFT magnitudes.

    Parameters
    ----------
    q : np.ndarray or list[np.ndarray]
        Trajectory (or list of trajectories).
    lag_times : list[int]
        Lag times for transition matrices.
    min_bin, max_bin : float
        Binning range.
    num_bins : int
        Number of bins.
    n_freq : int
        Number of FFT magnitudes to include. If <= 0, only matrices are returned.

    Returns
    -------
    torch.Tensor
        1-D tensor of features.
    """
    if isinstance(q, list):
        q = q[0]
    # Transition matrices are already returned as float32 1-D tensor from build_transition_matrices
    matrices = build_transition_matrices(q, lag_times, min_bin, max_bin, num_bins)

    if num_freq <= 0:
        return matrices

    fft_magnitudes = compute_psd_welch(q, num_freq=num_freq) #compute_normalized_fft_magnitudes(q, num_freq=num_freq)

    return torch.cat((matrices, fft_magnitudes))


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

    summary_stats = featurize_trajectory(
        q,
        config.getlistint("SUMMARY_STATS", "lag_times"),
        config.getfloat("SUMMARY_STATS", "min_bin"),
        config.getfloat("SUMMARY_STATS", "max_bin"),
        config.getint("SUMMARY_STATS", "num_bins"),
        config.getint("SUMMARY_STATS", "num_freq"),
    )
    print(f"Computed summary statistics of shape: {summary_stats.shape}")
    print(f"Using bins from {config.getfloat('SUMMARY_STATS', 'min_bin')} to {config.getfloat('SUMMARY_STATS', 'max_bin')} with {config.getint('SUMMARY_STATS', 'num_bins')} bins.")
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