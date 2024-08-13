from typing import Union
import numpy as np
import torch
from sbi_smfs.utils.stats_utils import (
    bin_trajectory,
    build_transition_matrix,
)


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


def compute_stats(trajectory: np.ndarray, lag_times, min_bin, max_bin, num_bins) -> torch.Tensor:
    """Computes summary statistics for given trajectory.

    Parameters
    ----------
    trajectory : np.ndarray
        Trajectory, shape (num_samples, num_dimensions) or (num_dimensions,)
    lag_times : list of int
        List of lag times for summary statistics
    min_bin : float
        Minimum bin value for histogram calculation
    max_bin : float
        Maximum bin value for histogram calculation
    num_bins : int
        Number of bins for histogram calculation

    Returns
    -------
    summary_stats : torch.Tensor
        Summary statistics
    """
    
    if isinstance(trajectory, torch.Tensor):
        trajectory = trajectory.numpy()
    
    # Ensure trajectory is at least 2D
    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(-1, 1)  # Reshape to (num_samples, 1)

    if trajectory.ndim != 2:
        raise ValueError("Trajectory should be 1D or 2D array.")

    # Compute summary statistics
    summary_stats = torch.stack([
        build_transition_matricies(
            trajectory[:, i],
            lag_times,
            min_bin,
            max_bin,
            num_bins,
        )
        for i in range(trajectory.shape[1])
    ], dim=1)

    return summary_stats