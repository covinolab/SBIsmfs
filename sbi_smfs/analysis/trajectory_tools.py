from typing import Union, Tuple, List
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
    transition_pos = []
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


import numpy as np
from numba import njit

@njit
def find_transitions_multi(x: np.ndarray, turn_points: np.ndarray):
    """
    Find transitions between multiple basins in a 1D trajectory.

    Parameters
    ----------
    x : np.ndarray
        Trajectory (e.g., running average or coordinate trace).
    turn_points : np.ndarray
        Sorted array of values that define basin boundaries.
        For example, for a 3-state system, you might have:
        turn_points = np.array([-0.5, 0.5])

    Returns
    -------
    freq : int
        Total number of transitions between basins.
    transition_pos : list
        List of positions (indices) where transitions occur.
    transitions : list of tuple
        List of (from_state, to_state) pairs indicating the state change.
    """

    n = len(x)
    freq = 0
    transition_pos = []
    transitions = []

    # Determine which basin each point belongs to
    # Basin indices: 0, 1, 2, ..., len(turn_points)
    state_prev = 0
    for j in range(len(turn_points)):
        if x[0] > turn_points[j]:
            state_prev = j + 1

    for i in range(1, n):
        state_curr = 0
        for j in range(len(turn_points)):
            if x[i] > turn_points[j]:
                state_curr = j + 1

        if state_curr != state_prev:
            freq += 1
            transition_pos.append(i)
            transitions.append((state_prev, state_curr))
            state_prev = state_curr

    return freq, transition_pos, transitions



def count_transitions(
    x: np.ndarray, turn_point: float = 0.0, window_size: int = 100
) -> int:
    num_transitions, transition_points = find_transitions(
        bn.move_mean(x, window=window_size), turn_point=turn_point
    )
    return num_transitions, transition_points


def count_transitions_multi(
    x: np.ndarray, turn_points: np.ndarray, window_size: int = 100
):
    """
    Count transitions between multiple states in a 1D trajectory.

    Parameters
    ----------
    x : np.ndarray
        Raw trajectory data.
    turn_points : np.ndarray
        Sorted array of values defining basin boundaries.
    window_size : int, optional
        Window size for running mean smoothing. Default is 100.

    Returns
    -------
    num_transitions : int
        Total number of transitions.
    transition_points : list
        Indices of transitions in the trajectory.
    transitions : list of tuple
        (from_state, to_state) pairs for each transition.
    """

    x_smooth = bn.move_mean(x, window=window_size, min_count=1)
    num_transitions, transition_points, transitions = find_transitions_multi(
        x_smooth, turn_points
    )
    return num_transitions, transition_points, transitions


def split_trajectory(
    x: np.ndarray,
    turn_point: float = 0.0,
    window_size: int = 1,
    buffer_length: int = 100,
    min_length: int = 1000,
) -> Tuple[list[np.ndarray], list[np.ndarray]]:
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

    num_transitions, transition_points = count_transitions(
        x, turn_point=turn_point, window_size=window_size
    )
    transition_points.insert(0, 0)  # Add first point to transition points
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


def split_trajectory_multi(
    x: np.ndarray,
    turn_points: np.ndarray,
    window_size: int = 1,
    buffer_length: int = 100,
    min_length: int = 1000,
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """
    Split trajectory into individual trajectory segments corresponding to each basin
    in a multi-state system defined by multiple turn points.

    Parameters
    ----------
    x : np.ndarray
        Trajectory to be split.
    turn_points : np.ndarray
        Sorted array of values defining basin boundaries.
        Example: np.array([-0.5, 0.5]) for a 3-state system.
    window_size : int, optional
        Window size for running mean smoothing. The default is 1 (no smoothing).
    buffer_length : int, optional
        Number of points to exclude from the start and end of each trajectory segment.
        Default is 100.
    min_length : int, optional
        Minimum segment length to keep. Default is 1000.

    Returns
    -------
    segments_by_state : list[list[np.ndarray]]
        List of trajectory segments for each state.
        segments_by_state[i] contains all trajectory segments assigned to basin i.
    transitions : list[tuple[int, int]]
        List of (from_state, to_state) transitions corresponding to boundaries
        between segments.
    """

    # --- Use your existing transition counter ---
    num_transitions, transition_points, transitions = count_transitions_multi(
        x, turn_points=turn_points, window_size=window_size
    )

    # --- Add first and last points for segmentation boundaries ---
    transition_points.insert(0, 0)
    transition_points.append(len(x))

    # --- Prepare storage for each basin ---
    n_states = len(turn_points) + 1
    segments_by_state = [[] for _ in range(n_states)]

    # --- Split the trajectory by transitions ---
    for seg_idx in range(num_transitions + 1):
        lower_idx = transition_points[seg_idx] + buffer_length
        upper_idx = transition_points[seg_idx + 1] - buffer_length

        if upper_idx - lower_idx < min_length:
            continue

        trajectory_segment = x[lower_idx:upper_idx]
        # Assign based on segment mean
        seg_mean = np.mean(trajectory_segment)
        seg_state = 0
        for j in range(len(turn_points)):
            if seg_mean > turn_points[j]:
                seg_state = j + 1
        segments_by_state[seg_state].append(trajectory_segment.copy())

    return segments_by_state, transitions


def compare_pmfs(
    pmfs: list[np.ndarray], initial_perturbation: float = 0.1
) -> list[np.ndarray]:
    """
    Minimize the difference between potentials of mean force (PMFs) by shifting them along the y-axis.

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
