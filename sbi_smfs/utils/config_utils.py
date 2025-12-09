from typing import Union
from configparser import ConfigParser


def get_config_parser(
    config_file: Union[str, ConfigParser], validate: bool = False
) -> ConfigParser:
    """
    Reads config file and returns ConfigParser object.

    Parameters
    ----------
    config_file: str, ConfigParser
        Config file with entries for simualtion.
    validate: bool
        Whether to check that all parameters are correctly specified.

    Returns
    -------
    config: ConfigParser
        ConfigParser object with entries for simualtion.
    """

    if isinstance(config_file, ConfigParser):
        return config_file

    config = ConfigParser(
        converters={
            "listint": lambda x: [int(i.strip()) for i in x.split(",")],
            "listfloat": lambda x: [
                [float(i.strip()) for i in group.split(",")] for group in x.split(";")
            ],
            "tuplefloat": lambda x: tuple([float(i.strip()) for i in x.split(",")]),
        }
    )
    config.read(config_file)

    if validate:
        validate_config(config)

    return config


def validate_config(config_file: Union[str, ConfigParser]) -> None:
    """
    Checks that all parameters are specified in config file.

    Parameters
    ----------
     config: str, ConfigParser
        Config file with entries for simualtion.

    Returns
    -------
        None.
    """
    if isinstance(config_file, str):
        config = ConfigParser()
        config.read(config_file)
    elif isinstance(config_file, ConfigParser):
        config = config_file
    else:
        raise NotImplementedError("config not properly specified!")

    expected_sections = ["SIMULATOR", "PRIORS", "SUMMARY_STATS"]
    expected_keys = {
        "SIMULATOR": [
            "dt",
            "num_steps",
            "num_knots",
            "saving_freq",
            "min_x",
            "max_x",
            "max_G_0",
            "max_G_1",
            "init_xq_range",
        ],
        "PRIORS": [
            "type_spline",
            "parameters_spline",
            "norm_spline_nodes",
            "type_Dq",
            "parameters_Dq",
            "type_k",
            "parameters_k",
        ],
        "SUMMARY_STATS": ["min_bin", "max_bin", "num_bins", "lag_times", "num_freq"],
        "NEURAL_NETWORK": [
            "embedding_net",
            "num_blocks",
            "dropout_probability",
            "use_batch_norm",
            "model",
            "hidden_features",
            "num_transforms",
            "num_bins",
        ],
        "TRAINING_PARAMS": [
            "validation_fraction",
            "training_batch_size",
            "learning_rate",
            "stop_after_epochs",
        ],
    }

    # Check sections
    for sec in expected_sections:
        if (
            sec == "NEURAL_NETWORK" or sec == "TRAININ_PARAMS"
        ):  # optional, have defaults
            continue
        if sec not in config.sections():
            raise KeyError(f"Missing {sec} in config")

    for sec, keys in expected_keys.items():
        if sec not in config.sections():
            print(f"Skipping {sec} in config, falling back to default values")
            continue
        for key in keys:
            if key not in config[sec]:
                raise KeyError(f"Missing {key} in {sec}")

    if "Dx" in config["SIMULATOR"]:
        if "type_Dx" in config["PRIORS"] or "parameters_Dx" in config["PRIORS"]:
            raise KeyError("Dx specified in prior and simulator!")
    elif "Dx" not in config["SIMULATOR"]:
        if (
            "type_Dx" not in config["PRIORS"]
            and "parameters_Dx" not in config["PRIORS"]
        ):
            raise KeyError("Dx needs to be specified in simualtor or prior!")
    return True
