from typing import Union
from omegaconf import DictConfig
from sbi.utils.get_nn_models import posterior_nn
from sbi_smfs.inference.embedding_net import EMBEDDING_NETS
from sbi_smfs.utils.configurations import load_config_yaml

def build_npe_model(config: DictConfig):
    """Builds a neural posterior.

    Parameters
    ----------
    config: Union[str, Config]
        Either a path to the config file or a Config object.

    Returns
    -------
    neural_posterior: sbi.utils.posterior_nn.Posterior
        Neural posterior.
    """

    nn_config = config.posterior.neural_network
    summary_stats_config = config.posterior.summary_statistics

    if nn_config.embedding_net in EMBEDDING_NETS.keys():
        cnn_net = EMBEDDING_NETS[nn_config.embedding_net](
            summary_stats_config.num_bins,
            len(summary_stats_config.lag_times),
            nn_config.hidden_dim,
        )
        print(f"Using embedding net: {nn_config.embedding_net}")
    else:
        print(f"Only available embeddings are: {list(EMBEDDING_NETS.keys())}. Falling back to single_layer_cnn")
        cnn_net = EMBEDDING_NETS["single_layer_cnn"](
            len(summary_stats_config.lag_times),
            4,
            2,
            summary_stats_config.num_bins,
            len(summary_stats_config.lag_times),
        )

    kwargs_flow = {
        "num_blocks": nn_config.num_blocks,
        "dropout_probability": nn_config.dropout_rate,
        "use_batch_norm": nn_config.use_batch_norm,
    }

    neural_posterior = posterior_nn(
        model=nn_config.model,
        hidden_features=nn_config.hidden_dim,
        num_transforms=nn_config.num_transforms,
        num_bins=nn_config.num_bins,
        embedding_net=cnn_net,
        z_score_x="none",
        **kwargs_flow,
    )

    return neural_posterior