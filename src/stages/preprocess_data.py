import os
import sys
import re
import argparse
import pandas as pd
from loguru import logger
sys.path.insert(0, os.getcwd())
from src.utils.utils import load_yaml

def preprocess_text(text:str)->str:
    """Preprocess text by replacing user name with [USER], links with [LINK] and removing hash tags (#)

    Args:
        text (str): the text from dataset

    Returns:
        text(str): text after performing cleaning 
    """
    # Replace user name with prefix '@' by [USER]
    text = re.sub('@\w+','[USER]' ,text)
    # Replace link by [LINK]
    url_regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    text = re.sub(url_regex, '[LINK]', text)
    # Remove hashtag
    text = re.sub('#', '', text)
    return text


def preprocess_data(config_path:str, data:str)->None:
    """Preprocess the dataset text column

    Args:
        config_path (str): path to config file (eg:params.yaml)
        data (str): the dataset i.e exist or csmb

    """
    config = load_yaml(config_path)
    raw_data_path = config["data"][data]["raw"]
    preprocess_data_path = config["data"][data]["preprocessed"]
    if raw_data_path.split(".")[-1] == "tsv":
        df = pd.read_csv(raw_data_path, delimiter="\t")
    else:
        df = pd.read_csv(raw_data_path)
    logger.info(f"READ {raw_data_path} \n with shape {df.shape}")
    df['text'] = df['text'].apply(preprocess_text)
    logger.info("Data Preprocessing Completed")
    df.to_csv(preprocess_data_path, index=False)
    logger.info(f"Saved preprocessed data at {preprocess_data_path}")

if __name__=="__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", dest="config", required=True)
    arg_parser.add_argument("--data",dest="data", required=True)
    args = arg_parser.parse_args()
    preprocess_data(args.config, args.data)