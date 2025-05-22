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
    data_path = "/net/scratch/lss/train-sample.csv"
    train_dataloader, val_dataloader, test_dataloader = data_load(data_path)

    model_type = '/net/scratch/lss/BERTOverflow_stackoverflow_github'

    tokenizer_overflow = BertTokenizer.from_pretrained(model_type) # 'lanwuwei/BERTOverflow_stackoverflow_github'
    bert_overflow = HFBertForSequenceClassification.from_pretrained(
        model_type=model_type, 
        num_labels=2,
        # config_args={"vocab_size": 82000, "n_classes": 2, "pad_token_id": 0}
    )

    # bert overflow
    trainer_bert_overflow = BertTrainer(
        bert_overflow,
        tokenizer_overflow,
        lr=5e-06,
        epochs=5,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        output_dir='../models/bert_overflow_fine_tuned',
        output_filename='bert_overflow',
        save=True
    )

    trainer_bert_overflow.train(evaluate=True)

    state_dict = torch.load('../models/bert_overflow_fine_tuned/bert_overflow_epoch_3.pt')
    bert_overflow.load_state_dict(state_dict["model_state_dict"])

    trainer_bert_overflow = BertTrainer(
        bert_overflow,
        tokenizer_overflow,
        lr=5e-06,
        epochs=5,
        train_dataloader=train_dataloader,
        eval_dataloader=test_dataloader,
        output_dir='../models/bert_overflow_fine_tuned',
        output_filename='bert_overflow',
        save=False,
    )

    trainer_bert_overflow.evaluate()

if __name__ == "__main__":
    main()

