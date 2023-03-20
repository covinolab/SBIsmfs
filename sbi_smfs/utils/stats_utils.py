import numpy as np
import scipy.stats as stats
import numba as nb


@nb.jit(nopython=True)
def bin_trajectory(x, bins):
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


@nb.jit(nopython=True)
def build_transition_matrix(binned_x, n_bins, t=1):
    """
    Calculates the markov transition matrix for a binned trajectory.

    Parameters
    ----------
    binned_x : np.Array
        Binned trajectory.
    n_bins : int
        Number of bins.
    t : int, optional
        Lag time for which the matrix is calculated. The default is 1.

    Returns
    -------
    np.Array
        Returns transition matrix of size (n_bins, n_bins).

    """

    matrix = np.zeros(shape=(n_bins, n_bins), dtype=np.int64)
    for i in range(len(binned_x) - t):
        column = binned_x[i]
        row = binned_x[i + t]
        matrix[row][column] += 1

    norm = np.sum(matrix, axis=0, dtype=np.int64)
    return matrix / norm


def moments(x):
    """
    Calculated the first 4 moments of the random variable x.

    Parameters
    ----------
    x : np.Array
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


def propagator(x, t=1):
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
def transition_count(x):
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
        if np.sign(x[i]) == -np.sign(x[i+1]):
            freq += 1
    return freq


def prop_stats(x, t):
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


def featurize_trajectory(q, lag_times):
    '''
    Computes summary statsitics for trajecotires.

    Parameters
    ----------
    q : np.Array
        Trajectory..
    lag_times : list
        The lag times used for calculation of the propagator.

    Returns
    -------
    features : list
        List with features computed from the trajectory.

    '''

    propagators_stats = []
    for lag_time in lag_times:
        for stat in prop_stats(q, t=lag_time):
            propagators_stats.append(stat)

    moments_q = moments(q)

    features = [*moments_q, *propagators_stats]

    return features
