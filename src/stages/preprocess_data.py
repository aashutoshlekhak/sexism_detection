import os
import sys
import re
import pandas as pd
sys.path.insert(0, os.getcwd())
from src.utils.utils import load_yaml
from loguru import logger

def preprocess_text(text:str)->str:
    # Replace user name with prefix '@' by [USER]
    text = re.sub('@\w+','[USER]' ,text)
    # Replace link by [LINK]
    url_regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    text = re.sub(url_regex, '[LINK]', text)
    # Remove hashtag
    text = re.sub('#', '', text)
    return text


def preprocess_data(config_path:str)->None:
    config = load_yaml(config_path)
    raw_data_path = config["data"]["exist"]["raw"]
    preprocess_data_path = config["data"]["exist"]["preprocessed"]
    df = pd.read_csv(raw_data_path, delimiter="\t")
    logger.info(f"READ {raw_data_path} \n with shape {df.shape}")
    df['text'] = df['text'].apply(preprocess_text)
    logger.info("Data Preprocessing Completed")
    df.to_csv(preprocess_data_path, index=False)
    logger.info(f"Saved preprocessed data at {preprocess_data_path}")



if __name__=="__main__":
    preprocess_data("params.yaml")