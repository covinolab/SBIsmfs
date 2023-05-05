import numpy as np
import numba as nb
import bottleneck as bn


def split_trajectory(
    x: np.ndarray, turn_point: float = 0.0, window_size: int = 1
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

    Returns
    -------
        list[np.ndarray]
            List of trajectories in each basin.
    """
    pass


def autocoor_in_basins(x: np.ndarray) -> np.ndarray:
    pass


def compare_pmfs(pmfs: list[np.ndarray]) -> list[np.ndarray]:
    """
    Minimize the difference between pmfs by shifting them along the y-axis.

    Parameters
    ----------
    pmfs : list[np.ndarray]
        List of pmfs to be compared.

    Returns
    -------
    list[np.ndarray]
        List of pmfs with minimized difference.
    """
    pass
