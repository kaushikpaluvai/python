import json
import os
import logging

DEFAULT_CONFIG_PATH = "configs/default_config.json"

def load_config(config_path: str = DEFAULT_CONFIG_PATH, strip_list_keys = False):
    """
    Loads the specified config file from disk, otherwise loads the default config.
    :return:
    """
    logging.info(f"Loading config with path: {config_path}")
    # If None was passed in for path value, use the default as well
    if not config_path:
        config_path = DEFAULT_CONFIG_PATH

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            # If required, grab the first item of each list-typed value in the config
            # This is used in testing to remove hyperparameter testing values from config.
            if strip_list_keys:
                config = {key: (value[0] if type(value)==list else value) for key, value in config.items()}
            return config
    else:
        raise FileNotFoundError(f"Config not found at path {config_path}")
