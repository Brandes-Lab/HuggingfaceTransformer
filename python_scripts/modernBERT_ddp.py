import os, time, argparse, torch, wandb, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_from_disk
from sklearn.metrics import roc_auc_score
from transformers import (
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
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

class ProteinBertModel:
    def __init__(self, vocab_size, tokenizer):
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer

    def build(self):
        config = ModernBertConfig(
            vocab_size=self.vocab_size,
            max_position_embeddings=8192,
            num_hidden_layers=8,
            num_attention_heads=8,
            hidden_size=512,
            intermediate_size=2048,
            type_vocab_size=1,
            hidden_activation="gelu",
            global_attn_every_n_layers=3,
            local_attention=512,
            deterministic_flash_attn=False,
            global_rope_theta=160000.0,
            local_rope_theta=10000.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            cls_token_id=self.tokenizer.cls_token_id,
            sep_token_id=self.tokenizer.sep_token_id,
        )
        return ModernBertForMaskedLM(config)



class ElapsedTimeLoggerCallback(TrainerCallback):
    def __init__(self):
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        elapsed_hours = (time.time() - self.start_time) / 3600
        if logs is not None:
            logs["elapsed_hours"] = elapsed_hours
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                wandb.log(logs, step=state.global_step)



class LossPrintCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Print the Trainer-logged loss (usually after accumulation)
        if logs is not None and "loss" in logs:
            print(f"\n[LOG STEP {state.global_step}] Trainer-logged loss: {logs['loss']}")
            print(f"  gradient_accumulation_steps: {args.gradient_accumulation_steps}")

def print_rank0(*args, **kwargs):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(*args, **kwargs)



def main():
    rank = int(os.environ.get("LOCAL_RANK", 0))

    print(f"[Rank {rank}] MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
    print(f"[Rank {rank}] MASTER_PORT: {os.environ.get('MASTER_PORT')}")
    print(f"[Rank {rank}] NCCL_TIMEOUT: {os.environ.get('NCCL_TIMEOUT')}")

    tokenizer = TokenizerLoader("char_tokenizer").load()
    pad_id = tokenizer.pad_token_id
    print_rank0("Tokenizer vocab size:", tokenizer.vocab_size)

    model = ProteinBertModel(tokenizer.vocab_size, tokenizer).build()
    model.gradient_checkpointing_enable()
    model = model.to(rank)

    # Calculate and round model parameter count
    param_count = sum(p.numel() for p in model.parameters())
    rounded_param_count = round(param_count / 1e6)  # e.g. 33M
    run_name = f"modernBERT_{rounded_param_count}M"


    print_rank0(f"[Rank {rank}] Model parameters: {param_count:,}", flush=True)

    if rank == 0:
        wandb.init(project="modernBERT_training", name=run_name, entity="sinha-anushka12-na")

    train_ds = load_from_disk("/gpfs/data/brandeslab/Data/processed_datasets/uniref90_tokenized_8192/train_only/train")
    val_ds = load_from_disk("/gpfs/data/brandeslab/Data/processed_datasets/uniref90_tokenized_8192/val_only/validation")

    # print_rank0("Max train length:", max(train_ds["length"]))
    # print_rank0("99th percentile:", np.percentile(train_ds["length"], 99))
    # print_rank0("95th percentile:", np.percentile(train_ds["length"], 95))

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=f"/gpfs/data/brandeslab/model_checkpts/{run_name}",
        max_steps=2_000_000,
        
        per_device_train_batch_size=8,
        gradient_accumulation_steps=32,
        per_device_eval_batch_size=4,
        
        bf16=True,
        fp16=False,
        
        dataloader_num_workers=16,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=2,
        
        eval_strategy="no",
        logging_strategy="steps",
        logging_steps=5000,
        save_strategy="steps",
        save_steps=10000,

        
        report_to="wandb",
        run_name=run_name,
        learning_rate=3e-4,
        
        remove_unused_columns=False,
        group_by_length=True,
        length_column_name="length"
        
        local_rank=rank,
        ddp_backend="nccl", 
        ddp_timeout=3000, 
        ddp_find_unused_parameters=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )


    trainer.add_callback(ElapsedTimeLoggerCallback())
    trainer.add_callback(LossPrintCallback())

    trainer.train()

if __name__ == "__main__":
    main()
