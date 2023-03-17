import torch
from sbi.utils import MultipleIndependent


class SplinePrior(MultipleIndependent):
    """Class which defines a prior for simulations on a spline potential"""

    def __init__(self, dists):
        super().__init__(dists, validate_args=None, arg_constraints={})

    def sample(self, sample_shape=torch.Size([])):
        samples = super().sample(sample_shape)
        if sample_shape == torch.Size():
            samples[2:] = samples[2:] - torch.mean(samples[2:]).reshape(-1, 1)
        else:
            samples[:, 2:] = samples[:, 2:] - torch.mean(samples[:, 2:], dim=1).reshape(-1, 1)
        return samples