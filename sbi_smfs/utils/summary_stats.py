import numpy as np


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


def build_transition_matricies(q, lag_times):
    pass
    # Build summary stats in form of transition matricies