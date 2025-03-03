from sbi.utils.get_nn_models import posterior_nn
from sbi_smfs.utils.config_utils import get_config_parser
from sbi_smfs.utils.config_utils import get_config_parser
from sbi_smfs.inference.embedding_net import EMBEDDING_NETS


def build_npe_model(config: str):
    """
    Builds a neural posterior from a config file.

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
            "num_blocks": 2,
            "dropout_probability": 0.0,
            "use_batch_norm": False,
            "model": "nsf",
            "hidden_features": 100,
            "num_transforms": 5,
            "num_bins": 10,
        }

    if (
        config.get("NEURAL_NETWORK", "embedding_net") in EMBEDDING_NETS.keys()
        and config.get("NEURAL_NETWORK", "embedding_net") != "single_layer_cnn"
    ):
        cnn_net = EMBEDDING_NETS[config.get("NEURAL_NETWORK", "embedding_net")](
            config.getint("SUMMARY_STATS", "num_bins"),
            len(config.getlistint("SUMMARY_STATS", "lag_times")),
            100,
        )
        print("Using embedding net :", config.get("NEURAL_NETWORK", "embedding_net"))
    else:
        if config.get("NEURAL_NETWORK", "embedding_net") != "single_layer_cnn":
            print(
                f"embedding net {config.get('NEURAL_NETWORK', 'embedding_net')} not available."
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
        "use_batch_norm": config.getboolean("NEURAL_NETWORK", "use_batch_norm"),
    }
    
    neural_posterior = posterior_nn(
        model=config.get("NEURAL_NETWORK", "model"),
        hidden_features=config.getint("NEURAL_NETWORK", "hidden_features"),
        num_transforms=config.getint("NEURAL_NETWORK", "num_transforms"),
        num_bins=config.getint("NEURAL_NETWORK", "num_bins"),
        embedding_net=cnn_net,
        z_score_x="none",
        **kwargs_flow,
    )
    return neural_posterior


def get_train_parameter(config):
    """
    Get training parameters from config file.

    Parameters
    ----------
    config: str
        Config file name.

    Returns
    -------
    train_parameter: dict
        Training parameters.
    """

    config = get_config_parser(config)

    if "TRAINING_PARAMS" not in config.sections():
        print("No training parameters specified in config file.")
        print("Using default trainingparameters.")

        default_params = {
            "validation_fraction": 0.15,
            "training_batch_size": 50,
            "learning_rate": 0.0005,
            "stop_after_epochs": 20,
        }

        return default_params

    train_parameter = {
        "validation_fraction": config.getfloat(
            "TRAINING_PARAMS", "validation_fraction"
        ),
        "training_batch_size": config.getint("TRAINING_PARAMS", "training_batch_size"),
        "learning_rate": config.getfloat("TRAINING_PARAMS", "learning_rate"),
        "stop_after_epochs": config.getint("TRAINING_PARAMS", "stop_after_epochs"),
    }

    return train_parameter
