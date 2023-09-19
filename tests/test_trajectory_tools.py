import numpy as np
import bottleneck as bn
import pytest
from sbi_smfs.analysis.trajectory_tools import (
    find_transitions,
    split_trajectory,
    compare_pmfs,
)


@pytest.mark.parametrize("num_transitions", [2, 10])
def test_find_transitions(num_transitions: int):
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

    transition_points = [
        i * length / (num_transitions + 1) for i in range(num_transitions + 1)
    ]
    num_transitions_est, transition_points_est = find_transitions(
        bn.move_mean(test_trajectory, window=100), turn_point=0
    )
    assert num_transitions == num_transitions_est
    assert len(transition_points) == (len(transition_points_est))
    for est, true in zip(
        transition_points_est, transition_points
    ):  # excludes last point which is not a transition
        assert np.isclose(est, true, atol=100)


@pytest.mark.parametrize("num_transitions", [1, 2, 3, 4])
def test_split_trajectory(num_transitions: int):
    rng = np.random.default_rng(seed=42)
    length = 10000 * num_transitions
    test_trajectory = np.zeros((length,))
    for i in range(num_transitions + 1):
        section_length = int(length / (num_transitions + 1))
        if i % 2 == 0:
            mean_val = 2
        else:
            mean_val = -2
        test_trajectory[i * section_length : (i + 1) * section_length] = (
            rng.standard_normal((section_length)) + mean_val
        )

    upper_sections, lower_sectrions = split_trajectory(
        test_trajectory,
        turn_point=0,
        min_length=1000,
        buffer_length=100,
        window_size=10,
    )
    assert len(upper_sections) == (num_transitions + 2) // 2
    assert len(lower_sectrions) == (num_transitions + 1) // 2
    assert all([len(section) > 1000 for section in upper_sections])
    assert all([len(section) > 1000 for section in lower_sectrions])


@pytest.mark.parametrize("inital_perturbation", [0.1, 1.0])
def test_compare_pmfs_quadratic_functions(inital_perturbation):
    pmfs = []
    for i in range(10):
        pmfs.append(np.linspace(-5, 5, 1000) ** 2 + np.random.rand() * 10)
    pmfs = compare_pmfs(pmfs, inital_perturbation=inital_perturbation)
    assert len(pmfs) == 10
    assert all(
        [np.isclose(pmfs[0], pmf).all() for pmf in pmfs]
    )  # all pmfs are the same
