import numpy as np
import torch
from sbi_smfs.utils.config_utils import get_config_parser
from sbi_smfs.utils.summary_stats import (
    build_transition_matrices,
    featurize_trajectory,
    compute_stats,
    check_if_observation_contains_features,
)


def test_build_transition_matrices():
    traj = np.random.standard_normal(size=(1000,))
    lag_times = [1, 3, 10]
    matricies = build_transition_matrices(traj, lag_times, -2, 2, 4)
    assert matricies.shape[0] == 3 * (4**2)


def test_featurize_trajectory():
    traj = np.random.standard_normal(size=(1000,))
    lag_times = [1, 3, 10]
    features = featurize_trajectory(traj, lag_times)
    assert len(features) == (4 + 3 * 4)


def test_compute_stats():
    config = get_config_parser("tests/config_files/test.config")
    traj = np.random.standard_normal(size=(100000,))
    features = compute_stats(traj, config)
    assert features.shape == torch.Size(
        [
            config.getint("SUMMARY_STATS", "num_bins") ** 2
            * len(config.getlistint("SUMMARY_STATS", "lag_times"))
        ]
    )


def test_compute_stats_batched_traj():
    config = get_config_parser("tests/config_files/test.config")
    traj = np.random.standard_normal(size=(10, 100000))
    features = compute_stats(traj, config)
    assert features.shape == torch.Size(
        [
            10,
            config.getint("SUMMARY_STATS", "num_bins") ** 2
            * len(config.getlistint("SUMMARY_STATS", "lag_times")),
        ]
    )


def test_check_if_observation_contains_features():
    config = get_config_parser("tests/config_files/test.config")
    features = torch.ones(
        (
            len(config.getlistint("SUMMARY_STATS", "lag_times"))
            * (config.getint("SUMMARY_STATS", "num_bins") ** 2),
        )
    )
    trajectory = torch.ones((100000,))
    assert check_if_observation_contains_features(features, config)
    assert not check_if_observation_contains_features(trajectory, config)
