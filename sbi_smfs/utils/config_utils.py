import configparser


def get_config_parser(config_file):
    if isinstance(config_file, configparser.ConfigParser):
        return config_file

    config = configparser.ConfigParser(
        converters={
            "listint": lambda x: [int(i.strip()) for i in x.split(",")],
            "listfloat": lambda x: [float(i.strip()) for i in x.split(",")],
        }
    )
    config.read(config_file)

    return config


def validate_config(config_file):
    """
    Chacks that all parameters are contained in config file.

    Parameters
    ----------
     config: str, ConfigParser
        Config file with entries for simualtion.

    Returns
    -------
        None.
    """
    if isinstance(config_file, str):
        config = configparser.ConfigParser()
        config.read(config_file)
    elif isinstance(config_file, configparser.ConfigParser):
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
        "SUMMARY_STATS": ["min_bin", "max_bin", "num_bins", "lag_times"],
    }

    # Check sections
    for sec in expected_sections:
        if sec not in config.sections():
            raise KeyError(f"Missing {sec} in config")

    for sec, keys in expected_keys.items():
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
