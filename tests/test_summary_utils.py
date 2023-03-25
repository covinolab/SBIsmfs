import numpy as np
import torch


def test_build_transition_matrices():
    pass
'''    bins = np.linspace(min_bin, max_bin, num_bins + 1)
    binned_q = bin_trajectory(q, bins)
    matricies = np.array([
        build_transition_matrix(binned_q, len(bins) - 1, t=lag_time)
        for lag_time in lag_times
    ])

    matricies = np.float32(matricies)

    return torch.from_numpy(np.nan_to_num(matricies, nan=0.0)).flatten()'''