import os, time, argparse, torch, wandb, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_from_disk
from sklearn.metrics import roc_auc_score
from transformers import (
    DataCollatorWithFlattening,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    ModernBertForMaskedLM,
    ModernBertConfig,
)
from torch.distributed import is_initialized, get_rank, barrier, all_gather_object


class TokenizerLoader:
    def __init__(self, tokenizer_path):
        self.tokenizer_path = tokenizer_path

    def load(self):
        return PreTrainedTokenizerFast.from_pretrained(self.tokenizer_path)

tokenizer = TokenizerLoader("char_tokenizer").load()


train_ds = load_from_disk("/gpfs/data/brandeslab/Data/processed_datasets/uniref90_tokenized_8192/train_only/train_representative")
print(train_ds)

data_collator = DataCollatorWithFlattening(
    return_position_ids=True, 
    separator_id=-100
)

for i in range(2):
    print(train_ds[i]["length"]) # 157, 303 

print("\n=== Testing Data Collator ===")
sample_examples = [train_ds[i] for i in range(2)]
flattened_batch = data_collator(sample_examples)

print(flattened_batch)
print(f"Packed batch shape: {flattened_batch['input_ids'].shape}")          # [1, 460]
print(f"Labels shape: {flattened_batch['labels'].shape}")                   # [1, 460]
print(f"Position IDs shape: {flattened_batch['position_ids'].shape}")       # [1, 460]


class CustomPackedMLMCollator:
    def __init__(self, tokenizer, mlm_probability=0.15, separator_id=-100):
        self.tokenizer = tokenizer
        self.mlm_prob = mlm_probability
        self.separator_id = separator_id

    def mask_tokens(self, input_ids):
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_prob)
        special_tokens_mask = (input_ids == self.separator_id)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens

        # 80% [MASK], 10% random, 10% original
        rand = torch.rand(input_ids.shape)
        mask_token_id = self.tokenizer.mask_token_id or 103  # fallback

        input_ids[masked_indices & (rand < 0.8)] = mask_token_id
        input_ids[masked_indices & (rand >= 0.8) & (rand < 0.9)] = torch.randint(
            low=0, high=self.tokenizer.vocab_size, size=(1,)
        )

        return input_ids, labels

    def __call__(self, examples):
        packed_input_ids, packed_labels, position_ids = [], [], []
        attention_mask = []
        pos_counter = 0

        for ex in examples:
            seq = torch.tensor(ex["input_ids"])
            length = ex["length"]

            seq_input_ids = seq[:length]
            masked_input_ids, labels = self.mask_tokens(seq_input_ids)

            packed_input_ids.append(masked_input_ids)
            packed_labels.append(labels)
            position_ids.append(torch.arange(length))
            attention_mask.append(torch.ones(length))

            # Separator
            if self.separator_id != -100:
                packed_input_ids.append(torch.tensor([self.separator_id]))
                packed_labels.append(torch.tensor([-100]))
                position_ids.append(torch.tensor([0]))
                attention_mask.append(torch.tensor([0]))

        # Flatten
        return {
            "input_ids": torch.cat(packed_input_ids).unsqueeze(0),
            "labels": torch.cat(packed_labels).unsqueeze(0),
            "position_ids": torch.cat(position_ids).unsqueeze(0),
            "attention_mask": torch.cat(attention_mask).unsqueeze(0)
        }


