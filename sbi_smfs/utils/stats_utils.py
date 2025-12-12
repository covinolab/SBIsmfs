import numpy as np
import torch
import scipy.stats as stats
from scipy.signal import welch
import numba as nb


@nb.jit(nopython=True)
def bin_trajectory(x: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """
    Bins an trajectory into discrete bins.

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


@nb.jit(nopython=True)
def build_transition_matrix(
    binned_x: np.ndarray, n_bins: int, t: int = 1
) -> np.ndarray:
    """
    Counts the transitions for a binned trajectory or a batch of binned trajectories.

    Parameters
    ----------
    binned_x : np.ndarray
        Binned trajectory (1D) or batched binned trajectories (2D: n_traj x length).
    n_bins : int
        Number of bins.
    t : int, optional
        Lag time for which the matrix is calculated. The default is 1.

    Returns
    -------
    np.ndarray
        Returns transition matrix of size (n_bins, n_bins) with columns normalized
        to sum to 1 (float64).
    """

    assert t >= 1, "Lag time t must be at least 1."
    assert n_bins >= 2, "Number of bins must be at least 2."

    matrix_int = np.zeros((n_bins, n_bins), dtype=np.int64)

    # handle 1D trajectory
    if binned_x.ndim == 1:
        L = binned_x.shape[0]
        for i in range(L - t):
            col = binned_x[i]
            row = binned_x[i + t]
            matrix_int[row, col] += 1
    # handle batched trajectories (2D: n_traj x length)
    else:
        n_traj = binned_x.shape[0]
        L = binned_x.shape[1]
        for traj in range(n_traj):
            for i in range(L - t):
                col = binned_x[traj, i]
                row = binned_x[traj, i + t]
                matrix_int[row, col] += 1

    # convert to float and normalize columns (avoid division by zero)
    matrix = np.zeros((n_bins, n_bins), dtype=np.float64)
    for i in range(n_bins):
        for j in range(n_bins):
            matrix[i, j] = matrix_int[i, j]

    for j in range(n_bins):
        s = 0.0
        for i in range(n_bins):
            s += matrix[i, j]
        if s > 0.0:
            for i in range(n_bins):
                matrix[i, j] = matrix[i, j] / s
        else:
            for i in range(n_bins):
                matrix[i, j] = 0.0

    return matrix


def moments(x: np.ndarray) -> tuple:
    """
    Calculated the first 4 moments of the random variable x.

    Parameters
    ----------
    x : np.ndarray
        Random variable..

    Returns
    -------
    First four moments of distribution.

    """

    m1 = np.mean(x)
    m2 = np.std(x)
    m3 = stats.skew(x)
    m4 = stats.kurtosis(x)

    return m1, m2, m3, m4


def propagator(x: np.ndarray, t: int = 1) -> np.ndarray:
    """
    Computes propagator for given lag time t.

    Parameters
    ----------
    x : np.Array
        Trajectory which is used to compute propagator.
    t : int, optional
        Lag time used to compute step size. The default is 1.

    Returns
    -------
    np.Array
        Propagator amples computed from imput trajectory x.

    """
    return x[t:] - x[:-t]


@nb.jit(nopython=True)
def transition_count(x: np.ndarray) -> int:
    """
    Counts observed zero crossings for imput trajectory.

    Parameters
    ----------
    x : np.Array
        Trajctory to compute zero crossings.

    Returns
    -------
    freq : int
        Number of observed zeros crossings.

    """

    freq = 0
    for i in range(len(x)):
        if np.sign(x[i]) == -np.sign(x[i + 1]):
            freq += 1
    return freq


def prop_stats(x: np.ndarray, t: int) -> tuple:
    """
    Computes the first four moments of the propagator for input trajectory.

    Parameters
    ----------
    x : np.Array
        Input trajctory to compute propagator stats from.
    t : int
        Lag time of the propagator.

    Returns
    -------
    tuple
        The first four moments of the propagator.

    """

    delta_x = propagator(x, t=t)
    _moments = moments(delta_x)

    return _moments


def compute_normalized_fft_magnitudes(traj, num_freq=50):
    """
    Computes the normalized frequency magnitudes of the FFT of the input trajectory.

    Parameters
    ----------
    traj : np.ndarray
        Input trajectory.
    n_freq : int, optional
        Number of frequency components to consider. The default is 50.
    
    Returns
    -------
    np.ndarray
        Normalized (by sum) magnitudes of the FFT of the trajectory.
    """

    fft_traj = np.abs(np.fft.fft(traj)[1:num_freq+1])
    fft_traj = fft_traj / fft_traj.sum()
    return torch.from_numpy(fft_traj)


def compute_psd_welch(traj, num_freq=15):
    freqs, psd = welch(traj, nperseg=10000, window='hann')
    psd = psd[0:num_freq]
    psd = psd / (psd.sum() + 1e-12) 
    return torch.from_numpy(psd.astype(np.float32))