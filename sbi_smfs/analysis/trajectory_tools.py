from typing import Union
import numpy as np
import numba as nb
import bottleneck as bn
import scipy
import torch


@nb.jit(nopython=True)
def find_transitions(x: np.ndarray, turn_point: float = 0.0):
    """
    Find transition points in a trajectory.

    Parameters
    ----------
    x : np.ndarray
        Runnning average of a trajectory.
    turn_point : float, optional
        Value of x at which two basins are separated. The default is 0.0.

    Returns
    -------
    freq : int
        Number of observed turning point crossings.
    transition_pos : list
        List of positions of turning points.
    """

    freq = 0
    transition_pos = [0]
    for i in range(len(x) - 1):
        if (
            x[i] < turn_point
            and x[i + 1] > turn_point
            or x[i] > turn_point
            and x[i + 1] < turn_point
        ):
            freq += 1
            transition_pos.append(i)
    return freq, transition_pos


def split_trajectory(
    x: np.ndarray,
    turn_point: float = 0.0,
    window_size: int = 1,
    buffer_length: int = 100,
    min_length: int = 1000,
) -> list[np.ndarray]:
    """
    Split trajectory into individual trajectories in each basin.
    Uses a moving average to find the transition points.

    Parameters
    ----------
    x : np.ndarray
        Trajectory to be split.
    turn_point : float, optional
        Value of x at which to split trajectory. The default is 0.0.
    window_size : int, optional
        Size of moving average window. The default is 1.
    buffer_length : int, optional
        Number of points to exclude from the beginning and end of each trajectory. The default is 100.
    min_length : int, optional
        Minimum length of each trajectory segment. The default is 1000.

    Returns
    -------
        Tuple[list[np.ndarray], list[np.ndarray]]
            Tuple containing lists of trajectories in each basin.
    """
    above_turn_point = []
    below_turn_point = []

    num_transitions, transition_points = find_transitions(
        bn.move_mean(x, window=window_size), turn_point=turn_point
    )
    transition_points.append(len(x))  # Add last point to transition points

    for segment_idx in range(num_transitions + 1):  # +1 because of last transition
        lower_idx = transition_points[segment_idx] + buffer_length
        upper_idx = transition_points[segment_idx + 1] - buffer_length

        if upper_idx - lower_idx < min_length:
            continue

        trajectory_segment = x[lower_idx:upper_idx]
        if trajectory_segment.mean() > turn_point:
            above_turn_point.append(trajectory_segment.copy())
        elif trajectory_segment.mean() < turn_point:
            below_turn_point.append(trajectory_segment.copy())

    return above_turn_point, below_turn_point


def compare_pmfs(pmfs: list[np.ndarray], initial_perturbation: float = 0.1) -> list[np.ndarray]:
    """
    Minimize the difference between probability mass functions (PMFs) by shifting them along the y-axis.

    Parameters
    ----------
    pmfs : list[np.ndarray]
        List of PMFs to be compared.
    initial_perturbation : float, optional
        Standard deviation of the initial random offsets. Default is 0.1.

    Returns
    -------
    list[np.ndarray]
        List of aligned PMFs with minimized differences.
    """
    # Validate input PMFs have same shape
    if not all(pmf.shape == pmfs[0].shape for pmf in pmfs):
        raise ValueError("All PMFs must have the same shape")

    # Stack PMFs into 2D array for easier manipulation
    stacked_pmfs = np.stack(pmfs, axis=1)
    num_pmfs = stacked_pmfs.shape[1]

    def calculate_alignment_error(offsets: np.ndarray) -> float:
        """Calculate total standard deviation across aligned PMFs."""
        shifted_pmfs = stacked_pmfs + offsets.reshape((1, -1))
        return np.sum(shifted_pmfs.std(axis=1))

    # Optimize vertical alignment
    initial_offsets = np.random.normal(size=num_pmfs, scale=initial_perturbation)
    result = scipy.optimize.minimize(calculate_alignment_error, initial_offsets)

    # Apply optimal offsets and split back into list
    aligned_pmfs = stacked_pmfs + result.x.reshape((1, -1))
    return [aligned_pmfs[:, i] for i in range(num_pmfs)]


def align_spline_nodes(
    spline_nodes: torch.Tensor,
    initial_perturbation: float = 0.1,
    return_torch: bool = True,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Align multiple spline nodes by minimizing their vertical offsets.

    Parameters
    ----------
    spline_nodes : torch.Tensor
        Tensor containing multiple spline nodes to be aligned vertically.
    initial_perturbation : float, optional
        Standard deviation for initial random offset values. Default is 0.1.
    return_torch : bool, optional
        If True, returns torch.Tensor, otherwise returns numpy array. Default is True.

    Returns
    -------
    Union[torch.Tensor, np.ndarray]
        Aligned spline nodes with minimized vertical differences.
    """
    # Convert to numpy for optimization
    nodes_numpy = spline_nodes.cpu().numpy()
    num_splines = nodes_numpy.shape[0]
    
    def compute_alignment_error(offsets: np.ndarray) -> float:
        """Calculate sum of squared differences from reference spline."""
        shifted_splines = nodes_numpy + offsets.reshape((-1, 1))
        reference_spline = shifted_splines[0]
        return np.sum((shifted_splines - reference_spline) ** 2)
    
    # Optimize vertical offsets
    initial_offsets = np.random.normal(size=num_splines, scale=initial_perturbation)
    result = scipy.optimize.minimize(compute_alignment_error, initial_offsets)
    
    # Apply optimal offsets
    aligned_splines = nodes_numpy + result.x.reshape((-1, 1))
    
    return torch.from_numpy(aligned_splines) if return_torch else aligned_splines
