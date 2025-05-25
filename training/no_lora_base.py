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
import numpy as np

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
from train_utils import (
    data_load,
    BertTrainer,
)


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)

    train_dataloader, val_dataloader, test_dataloader = data_load()
    
    model_type = '/net/scratch/lss/bert-base-uncased'

    tokenizer_base = BertTokenizer.from_pretrained(model_type) # 'bert-base-uncased'

    bert_base = HFBertForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=model_type,
        num_labels=2,
    )

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

    #bert base
    trainer_bert_base = BertTrainer(
        bert_base,
        tokenizer_base,
        lr=5e-06,
        epochs=5,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        output_dir='/net/scratch/lss/VE/bert_base_fine_tuned',
        output_filename='bert_base',
        save=True,
    )

    trainer_bert_base.train(evaluate=True)

    model_dir = f"/net/scratch/lss/VE/bert_base_fine_tuned"
    model_files = glob.glob(os.path.join(model_dir, "*.pt"))
    state_dict = torch.load(model_files[0])
    bert_base.load_state_dict(state_dict["model_state_dict"])

    # trainer
    trainer_bert_base = BertTrainer(
        bert_base,
        tokenizer_base,
        lr=5e-06,
        epochs=5,
        train_dataloader=train_dataloader,
        eval_dataloader=test_dataloader,
        output_dir='/net/scratch/lss/VE/bert_base_fine_tuned',
        output_filename='bert_base',
        save=False,
    )

    trainer_bert_base.evaluate()

if __name__ == "__main__":
    main()