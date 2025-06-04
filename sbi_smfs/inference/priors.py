import torch
from sbi.utils import MultipleIndependent
import torch.distributions as dists

from sbi_smfs.utils.config_utils import get_config_parser


PRIORS = {"GAUSSIAN": dists.Normal, "UNIFORM": dists.Uniform}


class SplinePrior(MultipleIndependent):
    """
    Class which defines a prior for simulations on a spline potential.

    Parameters
    ----------
        dists: torch.distributions
            Distribution to use for the prior.
        indipendent_vars: int
            Number of indipendent variables in the prior.

    Returns
    -------
        prior: torch.distributions
            Prior distribution.
    """

    def __init__(self, dists: dists, indipendent_vars: int = 2):
        super().__init__(dists, validate_args=False)
        self._ind_vars = indipendent_vars

    def sample(self, sample_shape=torch.Size([])):
        samples = super().sample(sample_shape)
        if sample_shape == torch.Size():
            samples[self._ind_vars :] = samples[self._ind_vars :] - torch.mean(
                samples[self._ind_vars :]
            ).reshape(-1, 1)
        else:
            samples[:, self._ind_vars :] = samples[:, self._ind_vars :] - torch.mean(
                samples[:, self._ind_vars :], dim=1
            ).reshape(-1, 1)
        return samples


def make_covariance_matrix(n, scale, correlation, device="cpu"):
    """
    Create a covariance matrix with specified scale and correlation.

    Args:
        n (int): Size of the covariance matrix (n x n).
        scale (float): Scaling factor for the covariance values.
        correlation (float): Correlation factor determining the decay of covariance.
        device (str, optional): Device to store the tensor ('cpu' or 'cuda'). Default is 'cpu'.

    Returns:
        torch.Tensor: Covariance matrix of size (n x n).
    """

    cov = torch.zeros(n, n, device=device)
    for i in range(n):
        for j in range(n):
            cov[i, j] = scale * torch.exp(-correlation * torch.abs(torch.tensor(i - j)))
    return cov


def make_gprior(mean, scale, correlation):
    """
    Creates a Gaussian prior distribution with the specified mean, scale, and correlation.

    Args:
        mean (torch.Tensor): The mean vector of the Gaussian distribution.
        scale (float): The scale factor for the covariance matrix.
        correlation (float): The correlation factor determining the decay of covariance values.

    Returns:
        torch.distributions.MultivariateNormal
    """

    cov = make_covariance_matrix(
        n=mean.shape[0], scale=scale, correlation=correlation, device=mean.device
    )
    return dists.MultivariateNormal(mean, cov)


def get_priors_from_config(config_file, device="cpu"):
    """
    Returns the prior distribution specified from the config file.

    Parameters
    ----------
    config_file: str
        Config file name.
    device: str
        Device of the prior.

    Returns
    -------
    prior: torch.distributions
        Prior distribution.
    """

    config = get_config_parser(config_file)
    indipendent_vars = 2

    dq_dist_params = config.getlistfloat("PRIORS", "parameters_Dq")
    prior_dq = PRIORS[config.get("PRIORS", "type_Dq")](
        torch.tensor([dq_dist_params[0][0]], device=device),
        torch.tensor([dq_dist_params[0][1]], device=device),
    )
    k_dist_params = config.getlistfloat("PRIORS", "parameters_k")
    prior_k = PRIORS[config.get("PRIORS", "type_k")](
        torch.tensor([k_dist_params[0][0]], device=device),
        torch.tensor([k_dist_params[0][1]], device=device),
    )
    spline_dist_params = config.getlistfloat("PRIORS", "parameters_spline")
    num_spline_knots = config.getint("SIMULATOR", "num_knots") - 4
    if len(spline_dist_params) == 1:
        prior_splines = [
            PRIORS[config.get("PRIORS", "type_spline")](
                torch.tensor([spline_dist_params[0][0]], device=device),
                torch.tensor([spline_dist_params[0][1]], device=device),
            )
            for _ in range(num_spline_knots)
        ]
    elif len(spline_dist_params) == 2:
        for i in range(2):
            assert (
                len(spline_dist_params[i]) == num_spline_knots
            ), "wrong spline configuration"
        prior_splines = [
            PRIORS[config.get("PRIORS", "type_spline")](
                torch.tensor([p1], device=device),
                torch.tensor([p2], device=device),
            )
            for p1, p2 in zip(spline_dist_params[0], spline_dist_params[1])
        ]
    elif (
        len(spline_dist_params) == 3
        and config.get("PRIORS", "type_spline") == "GAUSSIAN_PROCESS"
    ):
        prior_splines = [
            make_gprior(
                torch.tensor(spline_dist_params[0], device=device),
                torch.tensor(spline_dist_params[1], device=device),
                torch.tensor(spline_dist_params[2], device=device),
            )
        ]
    priors = [prior_dq, prior_k, *prior_splines]

    if "type_dx" in config["PRIORS"] and "parameters_dx" in config["PRIORS"]:
        dx_dist_params = config.getlistfloat("PRIORS", "parameters_Dx")
        prior_dx = PRIORS[config.get("PRIORS", "type_Dx")](
            torch.tensor([dx_dist_params[0][0]], device=device),
            torch.tensor([dx_dist_params[0][1]], device=device),
        )
        priors.insert(0, prior_dx)
        indipendent_vars += 1

    if config.getboolean("PRIORS", "norm_spline_nodes"):
        return SplinePrior(priors, indipendent_vars)
    else:
        return MultipleIndependent(priors)
