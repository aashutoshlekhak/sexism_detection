import yaml
import os
from loguru import logger

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

def append_to_yaml(config_path:str, data_to_append:dict, yaml_key:str)->None:
    """Appends data to the config file

    Args:
        config_path (str): path to the config file (params.yaml)
        data_to_append (dict): data to be added to the config file in the form of dict
        yaml_key (str): the key to which value is to be appended.
    """
    old_data = load_yaml(config_path)
    
    old_data[yaml_key] = data_to_append.get(yaml_key, {})

    with open(config_path, 'w') as file:
        yaml.safe_dump(old_data, file)
        
    logger.info("Config added to the YAML file")
