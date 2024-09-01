import os
import sys
import argparse
from tqdm import tqdm
import pandas as pd
import translators as ts
from loguru import logger
sys.path.insert(0, os.getcwd())
from src.utils.utils import load_yaml


LANGUAGES = ["en", "es"] # English, Spanish

def translate(text:str, lang:str)->str:
    target_lang = LANGUAGES[0] if lang == "es" else LANGUAGES[1] 
    text = ts.translate_text(text, to_language=target_lang)
    return text

def augment_data(config_path:str)->None:
    """Augments the data by converting english text to spanish and vice-versa

    Args:
        config_path (str): path to config file (params.yaml)
    """
    config = load_yaml(config_path)
    raw_data_path = config["data"]["train"]["exist"]["raw"]
    augmented_data_path = config["data"]["train"]["exist"]["augmented"]

    df = pd.read_csv(raw_data_path, delimiter='\t')

    logger.info(f"READ {raw_data_path} \n with shape {df.shape}")

    translated_data = {
        "task1": list(),
        "task2": list(),
        "text": list()
    }

    for lang in LANGUAGES:
        lang_df = df[df["language"] == lang].copy()
        
        for _, row in tqdm(lang_df.iterrows()):
            task1 = row["task1"]
            task2 = row["task2"]
            text =  row["text"]
            translated_text = translate(text,lang)
            translated_data["task1"].append(task1)
            translated_data["task2"].append(task2)
            translated_data["text"].append(translated_text)

    augmented_df = pd.DataFrame(translated_data)
    # Merge original df and translated df
    final_df = pd.concat([df[["task1", "task2", "text"]], augmented_df])
    # Save data
    final_df.to_csv(augmented_data_path, index=False)

    logger.info(f"Saved {augmented_data_path} \n with shape {final_df.shape}")

if __name__=="__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", dest="config", required=True)
    args = arg_parser.parse_args()
    augment_data(args.config)
