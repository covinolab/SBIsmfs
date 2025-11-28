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
            config.getint("NEURAL_NETWORK", "hidden_features"),
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

    default_params = {
        "validation_fraction": 0.15,
        "training_batch_size": 50,
        "learning_rate": 0.0005,
        "stop_after_epochs": 20,
        "retrain_from_scratch": False,
    }

    config = get_config_parser(config)

    if "TRAINING_PARAMS" not in config.sections():
        print("No training parameters specified in config file.")
        print("Using default training parameters.")
        return default_params

    train_parameter = {}

    for param, default in default_params.items():
        if param not in config["TRAINING_PARAMS"]:
            print(f"No {param} specified in config file.")
            print("Using default value :", default)
            train_parameter[param] = default
        else:
            try:
                if isinstance(default, bool):
                    value = config.getboolean("TRAINING_PARAMS", param)
                elif isinstance(default, int) and not isinstance(default, bool):
                    value = config.getint("TRAINING_PARAMS", param)
                elif isinstance(default, float):
                    value = config.getfloat("TRAINING_PARAMS", param)
                else:
                    value = config.get("TRAINING_PARAMS", param)
            except ValueError:
                print(f"Invalid value for {param} in config. Using default value :", default)
                value = default
            train_parameter[param] = value

    return train_parameter
