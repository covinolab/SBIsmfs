import configparser

def validate_config(config):
    '''
    Chacks that all parameters are contained in config file.

    Parameters
    ----------
     config: str
        Config file with entries for simualtion.

    Returns
    -------
        None.
    '''

    config = configparser.ConfigParser(
        converters={
        'listint': lambda x: [int(i.strip()) for i in x.split(',')],
        'listfloat': lambda x: [float(i.strip()) for i in x.split(',')]
        }
    )
    config.read(config)
    # TODO: Write valication function
    # Should contain simualtion settings, summary stats settings, prior settings, nn settings
    # Test wether Dx is in prior or simulator