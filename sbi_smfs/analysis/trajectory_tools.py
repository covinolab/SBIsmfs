import numpy as np
import numba as nb
import bottleneck as bn
import scipy


@nb.jit(nopython=True)
def find_transitions(x: np.ndarray, turn_point: float = 0.0):
    """
    Find transition points in a trajectory.

    Parameters
    ----------
    x : np.Array
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
    for i in range(len(x)):
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
        list[np.ndarray]
            List of trajectories in each basin.
    """

    above_turn_point = []
    below_turn_point = []

    num_transitions, transition_points = find_transitions(
        bn.move_mean(x, window=window_size), turn_point=turn_point
    )
    transition_points.append(len(x))  # Add last point to transition points
    print(transition_points, num_transitions)

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


def compare_pmfs(
    pmfs: list[np.ndarray], inital_perturbation: float = 0.1
) -> list[np.ndarray]:
    """
    Minimize the difference between pmfs by shifting them along the y-axis.

    Parameters
    ----------
    pmfs : list[np.ndarray]
        List of pmfs to be compared.
    inital_perturbation : float, optional
        Standard deviation of the initial perturbation. The default is 0.1.

    Returns
    -------
    list[np.ndarray]
        List of pmfs with minimized difference.
    """
    num_pmfs = len(pmfs)

    # check that all np.ndarrays in pmfs have the same shape
    assert all([pmfs[0].shape == pmf.shape for pmf in pmfs])

    # Concate pmfs along y-axis
    pmfs = np.stack(pmfs, axis=1)
    assert pmfs.shape[-1] == num_pmfs

    # Aligne pmfs by std along y
    def minimize(offsets):
        new_pmfs = pmfs + offsets.reshape((1, -1))
        return np.sum(new_pmfs.std(axis=1))

    opt_offsets = scipy.optimize.minimize(
        minimize, np.random.normal(size=(num_pmfs,), scale=inital_perturbation)
    )
    opt_pmfs = pmfs + opt_offsets.x
    opt_pmfs = [opt_pmfs[:, i] for i in range(num_pmfs)]

    return opt_pmfs
