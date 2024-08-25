import os
import sys
import argparse
import pandas as pd
from datasets import DatasetDict, load_from_disk
from sklearn.model_selection import KFold
from typing import Optional
from loguru import logger
sys.path.insert(0, os.getcwd())
from src.utils.utils import load_yaml

def split_data(config_path:str, data_name:str,data_type:str)->None:
    """ Splits the data saved in Dataset format into 10 folds and saves it

    Args:
        config_path (str): path to config file (eg:params.yaml)
        data_name (str): exist or csmb
        data_type (str): raw, processed or augmented
    """

    config = load_yaml(config_path)
    kf = KFold(n_splits=10, shuffle=True, random_state=config["base"]["random_seed"])
    save_path = config["data"][data_name][data_type]
    dataset_path = os.path.join(os.path.dirname(save_path), os.path.basename(save_path).split(".")[0])
    fold_save_dir = os.path.join(dataset_path, "folds")
    dataset = load_from_disk(dataset_path)
    logger.info(f"DATA LOADED {dataset}")

    # Split data
    # Loop over the KFold splits
    for fold_number, (train_index, eval_index) in enumerate(kf.split(dataset)):
        # Split data into train and eval sets
        train_fold = dataset.select(train_index)
        eval_fold = dataset.select(eval_index)
        
        # Define paths to save the train and eval sets
        fold_dir = os.path.join(fold_save_dir, f'fold_{fold_number}')
        # Merge Dataset
        fold_dataset = DatasetDict({
            "train": train_fold,
            "eval": eval_fold
        })
        
        # Save the train and eval folds
        fold_dataset.save_to_disk(fold_dir)
    
    logger.info(f"Dataset Saved at {fold_save_dir}")


if __name__=="__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", dest="config", required=True)
    arg_parser.add_argument("--data-name",dest="data_name", required=True)
    arg_parser.add_argument("--data-type",dest="data_type", required=True)

    args = arg_parser.parse_args()
    split_data(args.config, args.data_name, args.data_type) 




