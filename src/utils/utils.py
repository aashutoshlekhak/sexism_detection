import yaml
import os
import pandas as pd
import matplotlib.pyplot as plt
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



def visualize_trainer(log_path:str)-> None:
    """Generates the loss and accuracy plot from log history of trainer object

    Args:
        log_path (str): path to log history of trainer object (eg: fold_0.csv)
    """
    df = pd.read_csv(log_path)
    # Structure plot save path
    root_path = log_path.replace("logs", "reports").split("/")[:-1]
    root_path = "/".join(root_path)
    base_name = os.path.basename(log_path).split(".")[0]
    full_path = os.path.join(root_path, base_name)
    os.makedirs(full_path, exist_ok=True)


    # Plotting training and evaluation loss
    train_loss = df[df['loss'].notnull()][['epoch', 'loss']]
    eval_loss = df[df['eval_loss'].notnull()][['epoch', 'eval_loss']]
    eval_accuracy = df[df['eval_accuracy'].notnull()][['epoch', 'eval_accuracy']]

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss['epoch'], train_loss['loss'], label='Training Loss', marker='o')
    plt.plot(eval_loss['epoch'], eval_loss['eval_loss'], label='Evaluation Loss', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Evaluation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{full_path}/loss_plot.svg', format='svg')

    plt.figure(figsize=(10, 6))
    plt.plot(eval_accuracy['epoch'], eval_accuracy['eval_accuracy'], label='Evaluation Accuracy', marker='x', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Evaluation Accuracy Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{full_path}/accuracy_plot.svg', format='svg')

    logger.info(f"Accuracy and Loss Plot Saved at {full_path}")
    
