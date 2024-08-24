import os
import sys
import pandas as pd
from datasets import Dataset, ClassLabel
import argparse
from loguru import logger
sys.path.insert(0, os.getcwd())
from src.utils.utils import load_yaml, append_to_yaml

def load_data(config_path:str, data_name:str,data_type:str)->None:
    """Load csv or tsv files, encodes the class labels and saves those dataset in Dataset format of Huggingface
    along with class encoding to the config (params.yaml) file
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

    if data_name == "exist":
        class_label_one = config["class_label"]["task1"]
        class_label_two = config["class_label"]["task2"]
        data = data.cast_column('task1', ClassLabel(names=class_label_one))
        data = data.cast_column('task2', ClassLabel(names=class_label_two))
    else:
        data = data.class_encode_column("sexist")
        class_label = config["class_label"]["csmb"]
        data = data.cast_column('sexist', ClassLabel(names=class_label))

    logger.info("Dataset Encoded")

    config = load_yaml(config_path)
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




