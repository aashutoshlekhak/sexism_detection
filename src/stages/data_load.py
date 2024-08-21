import os
import sys
import pandas as pd
from datasets import Dataset
import yaml
import argparse
from loguru import logger
from typing import Optional
sys.path.insert(0, os.getcwd())
from src.utils.utils import load_yaml

def load_data(config_path:str, data_name:str,data_type:str)->None:
    """Load csv or tsv files and saves those dataset in Dataset format of Huggingface

    Args:
        config_path (str): path to config file (eg:params.yaml)
        data_name (str): exist or csmb
        data_type (str): raw, processed or augmented
    """
    config = load_yaml(config_path)

    if data_name == "exist" and data_type == "raw":
        df = pd.read_csv(config["data"][data_name][data_type], delimiter="\t")
    else:
        df = pd.read_csv(config["data"][data_name][data_type])
    
    save_path = config["data"][data_name][data_type]

    data = Dataset.from_pandas(df)    
    logger.info(f"Data Loaded :\n {data}")
    dataset_path = os.path.join(os.path.dirname(save_path), os.path.basename(save_path).split(".")[0])
    data.save_to_disk(dataset_path)
    logger.info(f"Data Saved at {dataset_path}")

if __name__=="__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", dest="config", required=True)
    arg_parser.add_argument("--data-name",dest="data_name", required=True)
    arg_parser.add_argument("--data-type",dest="data_type", required=True)
    args = arg_parser.parse_args()
    load_data(args.config, args.data_name, args.data_type) 




