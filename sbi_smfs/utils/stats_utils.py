import numpy as np
import scipy.stats as stats
import numba as nb


@nb.jit(nopython=True, fastmath=True)
def bin_trajectory(x: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """
    Bins an array accoring to bins.

    Parameters
    ----------
    x : np.Array
        Array to be binned.
    bins : np.Array
        Bin edges including the last one.

    Returns
    -------
    binned_x : np.Array
        Array with the bin indicies for each value.

    """

    binned_x = np.zeros(len(x), dtype=np.int64)
    for i in range(len(bins) - 1):
        for j in range(len(x)):
            if x[j] >= bins[i] and x[j] < bins[i + 1]:
                binned_x[j] = i
    return binned_x


@nb.jit(nopython=True, fastmath=True)
def build_transition_matrix(
    binned_x: np.ndarray, n_bins: np.ndarray, t: int = 1
) -> np.ndarray:
    """
    Calculates the markov transition matrix for a binned trajectory.

    Parameters
    ----------
    binned_x : np.ndarray
        Binned trajectory.
    n_bins : int
        Number of bins.
    t : int, optional
        Lag time for which the matrix is calculated. The default is 1.

    Returns
    -------
    np.ndarray
        Returns transition matrix of size (n_bins, n_bins).

    """

    matrix = np.zeros(shape=(n_bins, n_bins), dtype=np.int64)
    for i in range(len(binned_x) - t):
        column = binned_x[i]
        row = binned_x[i + t]
        matrix[row][column] += 1

    norm = np.sum(matrix, axis=0, dtype=np.int64)
    return matrix / norm
