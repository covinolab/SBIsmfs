import numpy as np
import scipy.stats as stats
import sbi_smfs.utils.stats_utils as tutils


def test_transition_matrix():
    binned_traj = np.array([0, 0, 1, 1, 1, 0, 0])
    bins = np.array([-2, 0, 2])
    transition_matrix = np.array([[2 / 3, 1 / 3], [1 / 3, 2 / 3]])
    test_matrix = tutils.build_transition_matrix(binned_traj, len(bins) - 1)
    assert np.isclose(test_matrix - transition_matrix, 0, atol=0.0001).all()


def test_bin_trajectory():
    trajectory = np.array([-1, -1.2, 0, 1, 1.3, 2, 10])
    bins = np.array([-2, 0, 2])
    binned_traj = np.array([0, 0, 1, 1, 1, 0, 0])
    test_binned_traj = tutils.bin_trajectory(trajectory, bins)
    assert (test_binned_traj == binned_traj).all()


def test_moments():
    distribution = np.random.standard_normal(size=(100000,))
    m1, m2 = distribution.mean(), distribution.std()
    m3, m4 = stats.skew(distribution), stats.kurtosis(distribution)
    test_m = tutils.moments(distribution)

    assert (m1, m2, m3, m4) == test_m
    assert np.isclose(test_m[0], 0, atol=0.1)
    assert np.isclose(test_m[1], 1, atol=0.1)
    assert np.isclose(test_m[2], 0, atol=0.1)
    assert np.isclose(test_m[3], 0, atol=0.1)


def test_propagator():
    trajectory = np.array([0, 1, 3, 6, 7, 10])
    step_size_l1 = np.array([1, 2, 3, 1, 3])
    step_size_l2 = np.array([3, 5, 4, 4])
    propagator_l1 = tutils.propagator(trajectory, 1)
    propagator_l2 = tutils.propagator(trajectory, 2)

    assert all(step_size_l1 == propagator_l1)
    assert all(step_size_l2 == propagator_l2)


def test_transition_count():
    trajectory = np.array([-1, -2, 2, 6, 7, 10])
    transitions = tutils.transition_count(trajectory,)
    assert transitions == 1


def test_prop_stats():
    trajectory = np.array([-1, -2, 2, 6, 7, 10])
    step_size_l1 = np.array([-1, 4, 4, 1, 3])
    m1, m2 = step_size_l1.mean(), step_size_l1.std()
    m3, m4 = stats.skew(step_size_l1), stats.kurtosis(step_size_l1)
    test_prop_m = tutils.prop_stats(trajectory, t=1)

    assert (m1, m2, m3, m4) == test_prop_m
