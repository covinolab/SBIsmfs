import configparser
import torch
from sbi.utils import MultipleIndependent
import torch.distributions as dists

from sbi_smfs.utils.config_utils import get_config_parser


PRIORS = {"GAUSSIAN": dists.Normal, "UNIFORM": dists.Uniform}


class SplinePrior(MultipleIndependent):
    """Class which defines a prior for simulations on a spline potential"""

    def __init__(self, dists):
        super().__init__(dists, validate_args=False)

    def sample(self, sample_shape=torch.Size([])):
        samples = super().sample(sample_shape)
        if sample_shape == torch.Size():
            samples[2:] = samples[2:] - torch.mean(samples[2:]).reshape(-1, 1)
        else:
            samples[:, 2:] = samples[:, 2:] - torch.mean(samples[:, 2:], dim=1).reshape(
                -1, 1
            )
        return samples


def get_priors_from_config(config_file, device="cpu"):

    config = get_config_parser(config_file)

    dq_dist_params = config.getlistfloat("PRIORS", "parameters_Dq")
    prior_dq = PRIORS[config.get("PRIORS", "type_Dq")](
        torch.tensor([dq_dist_params[0]], device=device),
        torch.tensor([dq_dist_params[1]], device=device),
    )
    k_dist_params = config.getlistfloat("PRIORS", "parameters_k")
    prior_k = PRIORS[config.get("PRIORS", "type_k")](
        torch.tensor([k_dist_params[0]], device=device),
        torch.tensor([k_dist_params[1]], device=device),
    )
    spline_dist_params = config.getlistfloat("PRIORS", "parameters_spline")
    prior_splines = [
        PRIORS[config.get("PRIORS", "type_spline")](
            torch.tensor([spline_dist_params[0]], device=device),
            torch.tensor([spline_dist_params[1]], device=device),
        )
        for _ in range(config.getint("SIMULATOR", "num_knots"))
    ]

    priors = [prior_dq, prior_k, *prior_splines]

    if config.getboolean("PRIORS", "norm_spline_nodes"):
        return SplinePrior(priors)
    else:
        return MultipleIndependent(priors)
