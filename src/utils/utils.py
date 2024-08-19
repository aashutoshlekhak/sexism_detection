import yaml
import os

def load_yaml(config_path:str)->dict:
    """Load config yaml file

    Args:
        config_path (str): path to the .yaml file

    Returns:
        config(dict): returns the yaml file as dict object
    """
    config_path = os.path.join(os.getcwd(), config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config
