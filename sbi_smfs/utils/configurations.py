from dataclasses import dataclass, field
from typing import List, Union, Optional, Type
import torch.distributions as dist
from omegaconf import OmegaConf
import yaml


@dataclass
class PriorConfig:
    distribution: str
    params: dict

@dataclass
class SplinePriorConfig:
    distribution: str
    params: dict

@dataclass
class SplineConfig:
    num_knots: int
    min_x: float
    max_x: float
    max_G_0: float
    max_G_1: float

@dataclass
class SimulatorConfig:
    dt: float
    num_steps: int
    saving_freq: int
    spline: SplineConfig
    init_xq_range: List[float]
    log_dx: Union[None, float] = None

@dataclass
class TransitionMatrixConfig:
    min_bin: float
    max_bin: float
    num_bins: int
    lag_times: List[int]

@dataclass
class PriorsConfig:
    dx: PriorConfig
    dq: PriorConfig
    k: PriorConfig
    spline_values: SplinePriorConfig
    normalize_splines: bool = False

@dataclass
class TrainingConfig:
    validation_fraction: float = 0.1
    batch_size: int = 50
    learning_rate: float = 1e-3
    stop_after_epochs: int = 20

@dataclass
class NeuralNetworkConfig:
    embedding_net: str = "single_layer_cnn"
    model: str = "nsf"
    num_blocks: int = 2
    hidden_dim: int = 100
    num_transforms: int = 5
    num_bins: int = 10
    dropout_rate: float = 0.0
    use_batch_norm: bool = False

@dataclass
class NPEConfig:
    summary_statistics: TransitionMatrixConfig
    neural_network: NeuralNetworkConfig
    training: TrainingConfig

@dataclass
class SNPEConfig(NPEConfig):
    num_rounds: Union[None, int] = None
    num_sim_per_round: Union[None, int] = None

@dataclass
class Config:
    simulator: SimulatorConfig
    priors: PriorsConfig
    posterior: SNPEConfig

def initialize_distribution(prior_config: PriorConfig):
    return getattr(dist, prior_config.distribution)(**prior_config.params)

def initialize_distribution_spline(prior_config: SplinePriorConfig):
    distributions = []
    parameters = []
    for param, values in prior_config.params.items():
        for value in values:
            parameters.append({param: value})
    return print(parameters)
            


def load_config_yaml(config_file: str) -> Config:
    # Load the YAML file
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    
    # Create an OmegaConf object from the loaded dictionary
    config_omega = OmegaConf.create(config_dict)
    
    # Create a structured config from the Config dataclass
    structured_config = OmegaConf.structured(Config)
    
    # Merge the loaded config with the structured config
    merged_config = OmegaConf.merge(structured_config, config_omega)
    
    return merged_config