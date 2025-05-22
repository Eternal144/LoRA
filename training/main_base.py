import sys
sys.path.insert(0, '../src')

# standard library imports
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import math

# related third party imports
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import glob

# local library specific imports
# from bert_from_scratch import BertForSequenceClassification as MyBertForSequenceClassification
from transformers import BertForSequenceClassification as HFBertForSequenceClassification
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.lora_from_scratch import (
    LinearLoRA,
    create_lora,
    add_lora_layers,
    freeze_model,
    unfreeze_model,
    create_linear,
    merge_lora_layers,
)
from train_utile import (
    data_load,
    BertTrainer,
)

def main():
    train_dataloader, val_dataloader, test_dataloader = data_load()
    
    model_type = '/net/scratch/lss/bert-base-uncased'

    tokenizer_base = BertTokenizer.from_pretrained(model_type) # 'bert-base-uncased'

    bert_base = HFBertForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=model_type,
        num_labels=2,
    )

    #bert base
    trainer_bert_base = BertTrainer(
        bert_base,
        tokenizer_base,
        lr=5e-06,
        epochs=5,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        output_dir='/net/scratch/lss/models_tys/bert_base_fine_tuned',
        output_filename='bert_base',
        save=True,
    )

    trainer_bert_base.train(evaluate=True)

    # copy weights from the saved fine-tuned model
    model_dir = "/net/scratch/lss/models_tys/bert_base_fine_tuned"
    # Get all .pt files in the folder
    model_files = glob.glob(os.path.join(model_dir, "*.pt"))

    state_dict = torch.load(model_files[0])  # replace with .pt file from models dir
    bert_base.load_state_dict(state_dict["model_state_dict"])

    # trainer
    trainer_bert_base = BertTrainer(
        bert_base,
        tokenizer_base,
        lr=5e-06,
        epochs=5,
        train_dataloader=train_dataloader,
        eval_dataloader=test_dataloader,
        output_dir='/net/scratch/lss/models_tys/bert_base_fine_tuned',
        output_filename='bert_base',
        save=False,
    )

    # evaluate on test set
    trainer_bert_base.evaluate()

    # LoRA BERT-base rank = 8
    add_lora_layers(bert_base, r=1, lora_alpha=16, ignore_layers=[0,1,2,3,4,5,6,7])  # inject the LoRA layers into the model
    freeze_model(bert_base)  # freeze the non-LoRA parameters

    n_params = 0
    n_trainable_params = 0

    # count the number of trainable parameters
    for n, p in bert_base.named_parameters():
        n_params += p.numel()
        if p.requires_grad:
            n_trainable_params += p.numel()

    print(f"Total parameters: {n_params}")
    print(f"Trainable parameters: {n_trainable_params}")
    print(f"Percentage trainable: {round(n_trainable_params / n_params * 100, 2)}%")

    # bert base lora all r = 8
    trainer_bert_base_lora = BertTrainer(
        bert_base,
        tokenizer_base,
        lr=5e-04,
        epochs=5,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        output_dir='/net/scratch/lss/models_tys/bert_base_fine_tuned_lora_r8',
        output_filename='bert_base_lora_r8',
        save=True,
    )

    trainer_bert_base_lora.train(evaluate=True)

    model_dir = "/net/scratch/lss/models_tys/bert_base_fine_tuned_lora_r8"

    # Get all .pt files in the folder
    model_files = glob.glob(os.path.join(model_dir, "*.pt"))

    # Load the model
    state_dict = torch.load(model_files[0])
    # state_dict = torch.load("/net/scratch/lss/models_tys/bert_base_fine_tuned_lora_r8/bert_base_lora_r8_epoch_0.pt")
    bert_base.load_state_dict(state_dict["model_state_dict"])

    # merge weights
    merge_lora_layers(bert_base) 
    unfreeze_model(bert_base)

    # create directory and filepaths
    dir_path = Path("/net/scratch/lss/models_tys/bert_base_fine_tuned_lora_r8/merged")
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / "bert_base_lora_r8_epoch_best_merged.pt"

    # save model
    torch.save({
        "model_state_dict": bert_base.state_dict(),
    }, file_path)

    trainer_bert_base_lora_r8 = BertTrainer(
        bert_base,
        tokenizer_base,
        lr=5e-06,
        epochs=1,
        train_dataloader=train_dataloader,
        eval_dataloader=test_dataloader,
        output_dir='/net/scratch/lss/models_tys/bert_base_fine_tuned_lora_r8',
        output_filename='bert_base_lora_r8',
        save=False,
    )

    trainer_bert_base_lora_r8.evaluate()
    

if __name__ == "__main__":
    main()

