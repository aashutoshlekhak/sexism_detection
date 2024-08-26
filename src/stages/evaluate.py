import sys
import os
import argparse
import torch
import pandas as pd
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm
sys.path.insert(0, os.getcwd())
from src.utils.utils import load_yaml



def load_model(model_path: str, num_labels:int):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels = num_labels
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Initialize the pipeline and set the device
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

    return pipe

def evaluate(config_path:str, data_name:str, data_type:str, model_ckpt:str,model_name:str,fold:int, task:Optional[str])->None:
    config = load_yaml(config_path)
    save_path = config["data"]["test"][data_name][data_type]
    dataset_path = os.path.join(os.path.dirname(save_path), os.path.basename(save_path).split(".")[0])

    # Loading pretrained model
    if data_name == "exist":
        if task:
            class_labels = config["class_label"][task]
            num_labels = len(class_labels)
            target_name = task
        else:
            print("Please provide task type as task1 or task2")
    else:
        class_labels = config["class_label"]["csmb"]
        num_labels = len(class_labels)
        target_name = "sexist"
        
    pipe = load_model(model_ckpt, num_labels)

    # Load test data
    test_data = load_from_disk(dataset_path)
    test_data = test_data.select_columns(['text', target_name])
    test_data = test_data.rename_column(target_name, "labels")

    # Extract the texts and labels
    texts = test_data['text']
    labels = test_data['labels']

    predictions = []
    # Use KeyDataset and tqdm for generating predictions
    for out in tqdm(pipe(KeyDataset(test_data, "text"), batch_size=16, truncation=True), total=len(test_data)):
        predicted_label = out['label']
        predicted_label_index = int(predicted_label.split("_")[-1])
        predictions.append(predicted_label_index)
    

    if task:
        metric_dir = os.path.join("src/metrics", model_name, data_type,data_name,task, f"fold_{fold}")
    else:
        metric_dir = os.path.join("src/metrics", model_name, data_type,data_name, f"fold_{fold}")

    os.makedirs(metric_dir, exist_ok=True)

    # Generate classification report
    report = classification_report(labels, predictions,target_names=class_labels)

    with open(f"{metric_dir}/classification_report.txt", 'w') as f:
        f.write(report)

    # Generate confusion matrix
    cm = confusion_matrix(labels, predictions)

    # Plot confusion matrix using seaborn
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()

    # Save confusion matrix as SVG
    plt.savefig(f"{metric_dir}/confusion_matrix.svg", format='svg')
    plt.show()

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", dest="config", required=True)
    arg_parser.add_argument("--data-name",dest="data_name", required=True)
    arg_parser.add_argument("--data-type",dest="data_type", required=True)
    arg_parser.add_argument("--model-path",dest="model_path", required=True)
    arg_parser.add_argument("--model-name",dest="model", required=True)
    arg_parser.add_argument("--fold",dest="fold", required=True)
    arg_parser.add_argument("--task",dest="task", required=False)

    args = arg_parser.parse_args()
    evaluate(args.config, args.data_name, args.data_type,args.model_path,args.model,int(args.fold),args.task) 