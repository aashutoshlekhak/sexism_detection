import os
import sys
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd
import yaml
import argparse
from pathlib import Path
from loguru import logger
from typing import Optional
sys.path.insert(0, os.getcwd())
from src.utils.utils import load_yaml


## TODO: 2. Save loss, metrics and reports in proper format

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def train(config_path:str, data_name:str, data_type:str, model_name:str, fold:int, task:Optional[str])->None:
    """ Trains the model

    Args:
        config_path (str): path to config file (eg:params.yaml)
        data_name (str): exist or csmb
        data_type (str): raw, processed or augmented
        model_name (str): name of the model to fine-tune (eg: distil_bert)
        fold (int): the fold of dataset to be used (eg: 0,1,2,3,4,5,6,7,8,9)
        task (Optional[str]): "task1" or "task2" (only for exist dataset)
    """
    config = load_yaml(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_ckpt = config["base_model"][model_name]["model"]
    batch_size = int(config["base_model"][model_name]["hyper_params"]["batch_size"])
    epochs = int(config["base_model"][model_name]["hyper_params"]["epochs"])
    lr = float(config["base_model"][model_name]["hyper_params"]["lr"])
    wt_deacy = float(config["base_model"][model_name]["hyper_params"]["wt_deacy"])
    if data_name == "exist":
        if task:
            class_labels = config["class_label"][task]
            num_labels = len(class_labels)
            target_name = task
        else:
            print("Please provide task type as task 1 or task 2")
            return None
    else:
        class_labels = config["class_label"]["csmb"]
        num_labels = len(class_labels)
        target_name = "sexist"

    model = (AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels).to(device))
    tokenizer= AutoTokenizer.from_pretrained(model_ckpt)

    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True)
    

    data_path = config["data"][data_name][data_type]
    dataset_path = os.path.join(os.path.dirname(data_path), os.path.basename(data_path).split(".")[0])
    fold_data = load_from_disk(f"{dataset_path}/folds/fold_{fold}")
    fold_data = fold_data.select_columns(['text', target_name])
    fold_data = fold_data.rename_column(target_name, "labels")
    logger.info("Dataset Loaded")

    data_encoded = fold_data.map(tokenize, batched=True, batch_size=None)

    logging_steps = len(data_encoded) // batch_size

    model_name = f"models/{model_ckpt}-finetuned-exist21-{fold}"

    training_args = TrainingArguments(output_dir=model_name,
                                num_train_epochs=epochs,
                                learning_rate=lr,
                                per_device_train_batch_size=batch_size,
                                per_device_eval_batch_size=batch_size,
                                weight_decay=wt_deacy,
                                eval_strategy="epoch",
                                disable_tqdm=False,
                                logging_steps=logging_steps,
                                push_to_hub=False,
                                log_level="error")

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=data_encoded["train"],
        eval_dataset=data_encoded["eval"],
        tokenizer=tokenizer
    ) 
    # Train the model
    trainer.train()

    log_history = pd.DataFrame(trainer.state.log_history)

    log_history.to_csv(f"src/logs/{model_name}.csv", index=False)
    logger.info(f"Logs saved at src/logs/{model_name}.csv")

if __name__=="__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", dest="config", required=True)
    arg_parser.add_argument("--data-name",dest="data_name", required=True)
    arg_parser.add_argument("--data-type",dest="data_type", required=True)
    arg_parser.add_argument("--model",dest="model", required=True)
    arg_parser.add_argument("--fold",dest="fold", required=True)
    arg_parser.add_argument("--task",dest="task", required=False)

    args = arg_parser.parse_args()
    train(args.config, args.data_name, args.data_type,args.model, int(args.fold),args.task) 



