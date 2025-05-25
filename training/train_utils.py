# add python path to include src directory
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
from datasets import load_dataset, load_from_disk
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


device = "cuda" if torch.cuda.is_available() else "cpu"


def collate_fn(batch):
    """ Instructs how the DataLoader should process the data into a batch"""
    text = [item['sentence'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch])
    return {'text': text, 'label': labels}


def data_load():
    # ds = load_dataset("stanfordnlp/sst2")
    ds= load_from_disk("/net/scratch/lss/sst2")

    train_val_split = ds['train'].train_test_split(test_size=0.2, seed=42)
    train_dataset = train_val_split['train']
    test_dataset = train_val_split['test']
    val_dataset = ds['validation']
    # breakpoint()

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    return train_dataloader, val_dataloader, test_dataloader

class BertTrainer:
    """ A training and evaluation loop for PyTorch models with a BERT like architecture. """
    
    
    def __init__(
        self, 
        model,
        tokenizer,
        train_dataloader,
        eval_dataloader=None,
        epochs=1,
        lr=5e-04,
        output_dir='./',
        output_filename='model_state_dict.pt',
        save=False,
        tabular=False,
    ):
        """
        Args:
            model: torch.nn.Module: = A PyTorch model with a BERT like architecture,
            tokenizer: = A BERT tokenizer for tokenizing text input,
            train_dataloader: torch.utils.data.DataLoader = 
                A dataloader containing the training data with "text" and "label" keys (optionally a "tabular" key),
            eval_dataloader: torch.utils.data.DataLoader = 
                A dataloader containing the evaluation data with "text" and "label" keys (optionally a "tabular" key),
            epochs: int = An integer representing the number epochs to train,
            lr: float = A float representing the learning rate for the optimizer,
            output_dir: str = A string representing the directory path to save the model,
            output_filename: string = A string representing the name of the file to save in the output directory,
            save: bool = A boolean representing whether or not to save the model,
            tabular: bool = A boolean representing whether or not the BERT model is modified to accept tabular data,
        """
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cpu'
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()
        self.output_dir = output_dir
        self.output_filename = output_filename
        self.save = save
        self.eval_loss = float('inf')  # tracks the lowest loss so as to only save the best model  
        self.epochs = epochs
        self.epoch_best_model = 0  # tracks which epoch the lowest loss is in so as to only save the best model
        self.tabular = tabular
    
    def train(self, evaluate=False):
        """ Calls the batch iterator to train and optionally evaluate the model."""
        for epoch in range(self.epochs):
            self.iteration(epoch, self.train_dataloader)
            if evaluate and self.eval_dataloader is not None:
                self.iteration(epoch, self.eval_dataloader, train=False)

    def evaluate(self):
        """ Calls the batch iterator to evaluate the model."""
        epoch=0
        self.iteration(epoch, self.eval_dataloader, train=False)
    
    def iteration(self, epoch, data_loader, train=True):
        """ Iterates through one epoch of training or evaluation"""
        
        # initialize variables
        loss_accumulated = 0.
        correct_accumulated = 0
        samples_accumulated = 0
        preds_all = []
        labels_all = []
        
        self.model.train() if train else self.model.eval()
        
        # progress bar
        mode = "train" if train else "eval"
        batch_iter = tqdm.tqdm(
            enumerate(data_loader),
            desc=f"EP ({mode}) {epoch}",
            total=len(data_loader),
            bar_format="{l_bar}{r_bar}"
        )
        
        # iterate through batches of the dataset
        for i, batch in batch_iter:

            # tokenize data
            batch_t = self.tokenizer(
                batch['text'],
                padding='max_length', 
                max_length=512, 
                truncation=True,
                return_tensors='pt', 
            )
            batch_t = {key: value.to(self.device) for key, value in batch_t.items()}
            batch_t["input_labels"] = batch["label"].to(self.device)
            # batch_t["tabular_vectors"] = batch["tabular"].to(self.device)

            # forward pass - include tabular data if it is a tabular model
            if self.tabular:
                logits = self.model(
                    input_ids=batch_t["input_ids"], 
                    token_type_ids=batch_t["token_type_ids"], 
                    attention_mask=batch_t["attention_mask"],
                    tabular_vectors=batch_t["tabular_vectors"],
                )   
            
            else:
                logits = self.model(
                    input_ids=batch_t["input_ids"], 
                    token_type_ids=batch_t["token_type_ids"], 
                    attention_mask=batch_t["attention_mask"],
                )

            logits = logits.logits
            # breakpoint()
            # calculate loss
            loss = self.loss_fn(logits, batch_t["input_labels"])
    
            # compute gradient and and update weights
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # calculate the number of correct predictions
            preds = logits.argmax(dim=-1)
            correct = preds.eq(batch_t["input_labels"]).sum().item()
            
            # accumulate batch metrics and outputs
            loss_accumulated += loss.item()
            correct_accumulated += correct
            samples_accumulated += len(batch_t["input_labels"])
            preds_all.append(preds.detach())
            labels_all.append(batch_t['input_labels'].detach())
        
        # concatenate all batch tensors into one tensor and move to cpu for compatibility with sklearn metrics
        preds_all = torch.cat(preds_all, dim=0).cpu()
        labels_all = torch.cat(labels_all, dim=0).cpu()
        
        # metrics
        accuracy = accuracy_score(labels_all, preds_all)
        precision = precision_score(labels_all, preds_all, average='macro')
        recall = recall_score(labels_all, preds_all, average='macro')
        f1 = f1_score(labels_all, preds_all, average='macro')
        avg_loss_epoch = loss_accumulated / len(data_loader)
        
        # print metrics to console
        print(
            f"samples={samples_accumulated}, \
            correct={correct_accumulated}, \
            acc={round(accuracy, 4)}, \
            recall={round(recall, 4)}, \
            prec={round(precision,4)}, \
            f1={round(f1, 4)}, \
            loss={round(avg_loss_epoch, 4)}"
        )    
        
        print(f"epoch {epoch} with best epoch as {self.epoch_best_model}")
        print(f"avg loss: {avg_loss_epoch}, best eval loss: {self.eval_loss}")
        print(f"save: {self.save}")
        print(f"train: {train}")
        print(avg_loss_epoch < self.eval_loss)
        # save the model if the evaluation loss is lower than the previous best epoch 
        if self.save and not train and avg_loss_epoch < self.eval_loss:
            
            # create directory and filepaths
            dir_path = Path(self.output_dir)
            dir_path.mkdir(parents=True, exist_ok=True)
            file_path = dir_path / f"{self.output_filename}_epoch_{epoch}.pt"
            

            # delete previous best model from hard drive
            if self.epoch_best_model != epoch:
                file_path_best_model = dir_path / f"{self.output_filename}_epoch_{self.epoch_best_model}.pt"
                # !rm -f $file_path_best_model
                file_path_best_model.unlink(missing_ok=True)
            
            # save model
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, file_path)
            
            # update the new best loss and epoch
            self.eval_loss = avg_loss_epoch
            self.epoch_best_model = epoch
            print(f"updated best epoch {self.epoch_best_model} with loss {self.eval_loss}")
