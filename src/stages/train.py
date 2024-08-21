import os
import sys
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import yaml
import argparse
from loguru import logger
from typing import Optional
sys.path.insert(0, os.getcwd())
from src.utils.utils import load_yaml


## TODO: Save best model and simplify the workings on each loop

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def train(config_path:str, data_name:str, data_type:str, task:Optional[str])->None:
    """_summary_

    Args:
        config_path (str): _description_
        data_name (str): _description_
        data_type (str): _description_
        task (Optional[str]): _description_
    """
    config = load_yaml(config_path)
    num_labels = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_ckpt = config["base_model"]["distil_bert"]
    batch_size = config["base_model"]["hyper_params"]["batch_size"]
    epochs = config["base_model"]["hyper_params"]["epochs"]
    lr = config["base_model"]["hyper_params"]["lr"]
    wt_deacy = config["base_model"]["hyper_params"]["wt_deacy"]
    model = (AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels).to(device))
    tokenizer= AutoTokenizer.from_pretrained(model_ckpt)

    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True)
    
    eval_results = []
    models = []

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


    data_path = config["data"][data_name][data_type]
    dataset_path = os.path.join(os.path.dirname(data_path), os.path.basename(data_path).split(".")[0])
    for i in range(10):
        train_fold = load_from_disk(f"{dataset_path}/train_fold_{i}")
        eval_fold = load_from_disk(f"{dataset_path}/eval_fold_{i}")
        train_fold = train_fold.select_columns(['text', task])
        eval_fold = eval_fold.select_columns(['text', task])
        train_dataset = train_fold.rename_column(task, "labels")
        eval_dataset = eval_fold.rename_column(task, "labels")
        train_dataset= train_dataset.class_encode_column("labels")
        class_label_feature = train_dataset.features["labels"]
        eval_dataset =eval_dataset.cast_column("labels", class_label_feature)

        logger.info(class_label_feature)

        train_encoded = train_dataset.map(tokenize, batched=True, batch_size=None)
        eval_encoded = eval_dataset.map(tokenize, batched=True, batch_size=None)

        logging_steps = len(train_encoded) // batch_size
        model_name = f"{model_ckpt}-finetuned-exist21"

        # Create Trainer instance
        trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_fold,
            eval_dataset=eval_fold,
            tokenizer=tokenizer
        )
        
        # Train the model
        trainer.train()
        
        # Evaluate the model
        eval_result = trainer.evaluate(eval_fold)
        eval_results.append(eval_result)
        
        # Store the trained model
        models.append(trainer.model)

        eval_result_f1=  [result['eval_f1'] for result in eval_results]
        best_model_index = np.argmax(eval_result_f1) # choosing the best model with the help of the f1-score 
        best_model_eval_result = eval_results[best_model_index]

        best_model = models[best_model_index]

        print("Best model evaluation results:", best_model_eval_result)
        print("Best model:", best_model)


if __name__=="__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", dest="config", required=True)
    arg_parser.add_argument("--data-name",dest="data_name", required=True)
    arg_parser.add_argument("--data-type",dest="data_type", required=True)
    arg_parser.add_argument("--task",dest="task", required=False)

    args = arg_parser.parse_args()
    train(args.config, args.data_name, args.data_type, args.task) 




