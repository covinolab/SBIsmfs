from sbi_smfs.utils.config_utils import get_config_parser
from sbi.utils.get_nn_models import posterior_nn
from sbi_smfs.simulator import get_simulator_from_config
from sbi_smfs.inference.priors import get_priors_from_config
from sbi_smfs.inference.embedding_net import SimpleCNN
from sbi_smfs.utils.config_utils import get_config_parser
from sbi_smfs.inference.embedding_net import EMBEDDING_NETS


def build_npe_model(config: str):
    """Builds a neural posterior.
    
    Parameters
    ----------
    config: str
        Config file name.
    
    Returns
    -------
    neural_posterior: sbi.utils.posterior_nn.Posterior
        Neural posterior.
    """

    config = get_config_parser(config)

    if "NEURAL_NETWORK" not in config.sections():
        print("No neural network specified in config file.")
        print("Using default neural network hyperparameters.")

        config["NEURAL_NETWORK"] = {
            "embedding_net": "single_layer_cnn",
            "num_blocks": 0,
            "dropout_probability": 0.0,
            "use_batch_norm": False,
            "model": "nsf",
            "hidden_features": 100,
            "num_transforms": 5,
            "num_bins": 10,
        }

    if config.get("NEURAL_NETWORK", "embedding_net") in EMBEDDING_NETS.keys():
        cnn_net = EMBEDDING_NETS[config.get("SUMMARY_STATS", "embedding_net")](
            config.getint("SUMMARY_STATS", "num_bins"),
            len(config.getlistint("SUMMARY_STATS", "lag_times")),
            100,
        )
        print("Using embedding net :", config.get("NEURAL_NETWORK", "embedding_net"))
    else:
        print(
            f"only available embeddings are : {[key for key in EMBEDDING_NETS.keys()]}. Falling back to single_layer_cnn"
        )
        cnn_net = EMBEDDING_NETS["single_layer_cnn"](
            len(config.getlistint("SUMMARY_STATS", "lag_times")),
            4,
            2,
            config.getint("SUMMARY_STATS", "num_bins"),
            len(config.getlistint("SUMMARY_STATS", "lag_times")),
        )

    kwargs_flow = {
        "num_blocks": config.getint("NEURAL_NETWORK", "num_blocks"),
        "dropout_probability": config.getfloat("NEURAL_NETWORK", "dropout_probability"),
        "use_batch_norm": config.getbool("NEURAL_NETWORK", "use_batch_norm"),
    }

    neural_posterior = posterior_nn(
        model=config.getstr("NEURAL_NETWORK", "model"),
        hidden_features=config.getint("NEURAL_NETWORK", "hidden_features"),
        num_transforms=config.getint("NEURAL_NETWORK", "num_transforms"),
        num_bins=config.getint("NEURAL_NETWORK", "num_bins"),
        embedding_net=cnn_net,
        z_score_x="none",
        **kwargs_flow,
    )

    return neural_posterior
