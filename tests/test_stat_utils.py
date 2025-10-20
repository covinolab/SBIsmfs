import pytest
import numpy as np
import bottleneck as bn
import scipy.stats as stats
import sbi_smfs.utils.stats_utils as stutils


def test_transition_matrix():
    binned_traj = np.array([0, 0, 1, 1, 1, 0, 0])
    bins = np.array([-2, 0, 2])
    transition_matrix = np.array([[2 / 3, 1 / 3], [1 / 3, 2 / 3]])
    test_matrix = stutils.build_transition_matrix(binned_traj, len(bins) - 1)
    assert np.isclose(test_matrix - transition_matrix, 0, atol=0.0001).all()


def test_transition_matrix_batched():
    binned_traj = np.array([[0, 0, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0, 0]])
    bins = np.array([-2, 0, 2])
    transition_matrix = np.array([[2 / 3, 1 / 3], [1 / 3, 2 / 3]])
    test_matrix = stutils.build_transition_matrix(binned_traj, len(bins) - 1)
    assert np.isclose(test_matrix - transition_matrix, 0, atol=0.0001).all()


def test_bin_trajectory():
    trajectory = np.array([-1, -1.2, 0, 1, 1.3, 2, 10])
    bins = np.array([-2, 0, 2])
    binned_traj = np.array([0, 0, 1, 1, 1, 0, 0])
    test_binned_traj = stutils.bin_trajectory(trajectory, bins)
    assert (test_binned_traj == binned_traj).all()


def test_moments():
    distribution = np.random.standard_normal(size=(100000,))
    m1, m2 = distribution.mean(), distribution.std()
    m3, m4 = stats.skew(distribution), stats.kurtosis(distribution)
    test_m = stutils.moments(distribution)

    assert (m1, m2, m3, m4) == test_m
    assert np.isclose(test_m[0], 0, atol=0.1)
    assert np.isclose(test_m[1], 1, atol=0.1)
    assert np.isclose(test_m[2], 0, atol=0.1)
    assert np.isclose(test_m[3], 0, atol=0.1)


def test_propagator():
    trajectory = np.array([0, 1, 3, 6, 7, 10])
    step_size_l1 = np.array([1, 2, 3, 1, 3])
    step_size_l2 = np.array([3, 5, 4, 4])
    propagator_l1 = stutils.propagator(trajectory, 1)
    propagator_l2 = stutils.propagator(trajectory, 2)

    assert all(step_size_l1 == propagator_l1)
    assert all(step_size_l2 == propagator_l2)


@pytest.mark.parametrize("num_transitions", [2, 10])
def test_transition_count_running_mean(num_transitions: int):
    rng = np.random.default_rng(seed=42)
    length = 10000 * num_transitions
    test_trajectory = np.zeros((length,))
    for i in range(num_transitions + 1):
        section_length = int(length / (num_transitions + 1))
        if i % 2 == 0:
            mean_val = 3
        else:
            mean_val = -3
        test_trajectory[i * section_length : (i + 1) * section_length] = (
            rng.standard_normal((section_length)) + mean_val
        )
    num_transitions_est = stutils.transition_count(
        bn.move_mean(test_trajectory, window=200)
    )
    assert num_transitions == num_transitions_est


def test_transition_count_raw():
    trajectory = np.array([-1, -2, 2, 6, 7, 10])
    transitions = stutils.transition_count(
        trajectory,
    )
    assert transitions == 1


def test_prop_stats():
    trajectory = np.array([-1, -2, 2, 6, 7, 10])
    step_size_l1 = np.array([-1, 4, 4, 1, 3])
    m1, m2 = step_size_l1.mean(), step_size_l1.std()
    m3, m4 = stats.skew(step_size_l1), stats.kurtosis(step_size_l1)
    test_prop_m = stutils.prop_stats(trajectory, t=1)

    assert (m1, m2, m3, m4) == test_prop_m
