# LoRA for Parameter-Efficient Fine-Tuning in Text Classification

This repository contains our implementation and experimental results for evaluating the effectiveness of **Low-Rank Adaptation (LoRA)** in fine-tuning large language models, specifically applied to **sentiment classification using BERT** on the SST-2 dataset.

## Overview

LoRA enables efficient fine-tuning of pre-trained models by injecting **trainable low-rank matrices** into selected layers while **freezing the original model weights**. This drastically reduces the number of trainable parameters and memory footprint, making fine-tuning accessible under resource constraints.

Our project investigates:
- How LoRA compares to full fine-tuning in terms of accuracy and efficiency
- The effect of different **ranks** $ r \in \{1, 4, 8, 16, 32\} $
- Layer-wise adaptation strategies to identify critical transformer components

## Key Results

- **LoRA (r=16)** outperforms full fine-tuning on SST-2 with **94.34% accuracy**, using **<1% of trainable parameters** (0.54M vs. 110M)
- Training time was reduced by up to **26%**
- Bottom layers (6–11) are most important for LoRA-based adaptation

| Method            | Accuracy | Trainable Params | Time/Epoch |
|-------------------|----------|------------------|-------------|
| Full Fine-Tuning  | 93.58%   | 100%             | 17:05       |
| LoRA (r=16)       | 94.34%   | 0.54%            | 12:36       |
| LoRA (Bottom 6)   | 93.15%   | 0.14%            | 12:15       |
| LoRA (r=4)        | 93.99%   | 0.14%            | 12:36       |

## Implementation

- Base model: `bert-base-uncased`
- Dataset: [SST-2](https://huggingface.co/datasets/stanfordnlp/sst2)
- Optimizer: AdamW
- LoRA applied to: query/value projections
- Layer selection and rank configuration are fully customizable

## Structure

- `src/lora_from_scratch.py`: Custom LoRA module
- `training/train_utils.py`: Trainer and data loading utilities
- `training/no_lora_base.py`: Full fine-tuning script without LoRA
- `training/main_base.py`: LoRA experiments

## Requirements

- Python 3.8+
- PyTorch
- Transformers (HuggingFace)