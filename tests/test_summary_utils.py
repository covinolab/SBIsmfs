import numpy as np
import torch
from sbi_smfs.utils.summary_stats import (
    build_transition_matricies,
    featurize_trajectory,
)


def test_build_transition_matrices():
    traj = np.random.standard_normal(size=(1000,))
    lag_times = [1, 3, 10]
    matricies = build_transition_matricies(traj, lag_times, -2, 2, 4)
    assert matricies.shape[0] == 3 * (4 ** 2)


def test_featurize_trajectory():
    traj = np.random.standard_normal(size=(1000,))
    lag_times = [1, 3, 10]
    features = featurize_trajectory(traj, lag_times)
    assert len(features) == (4 + 3 * 4)
