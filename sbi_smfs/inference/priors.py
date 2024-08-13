from typing import Union, List
import torch
from sbi.utils import MultipleIndependent
import torch.distributions as dist
from sbi_smfs.utils.configurations import load_config_yaml
from omegaconf import OmegaConf


class SplinePrior(MultipleIndependent):
    """
    Class which defines a prior for simulations on a spline potential.
    """

    def __init__(self, config: Union[str, OmegaConf], device: str = "cpu") -> None:
        self.config = self.load_config_yaml(config)
        super().__init__(
            self._create_priors(self.config, device),
            validate_args=False
        )

    def _create_priors(self, config: OmegaConf, device: str) -> List[dist.Distribution]:
        independent_vars = 2
        priors = []

        # Add Dx prior if specified
        if config.dx is not None:
            priors.append(self._create_prior(config.dx, device))
            independent_vars += 1

        # Add Dq and k priors
        priors.extend([
            self._create_prior(config.dq, device),
            self._create_prior(config.k, device)
        ])

        # Add spline priors
        spline_priors = self._create_spline_priors(config.spline_values, device)
        priors.extend(spline_priors)

        return priors
    
    def _create_spline_priors(self, spline_configs: OmegaConf, device: str) -> List[dist.Distribution]:
        return [self._create_prior(config, device) for config in spline_configs]

    @staticmethod
    def _create_prior(prior_config: OmegaConf, device: str) -> dist.Distribution:
        return prior_config.dist(**{k: torch.tensor([v], device=device) for k, v in prior_config.params.items()})