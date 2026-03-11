# type: ignore
import os
import time
from dataclasses import dataclass, field
from typing import Literal

import torch
import wandb
from datasets import load_from_disk
from transformers import HfArgumentParser, Trainer, TrainingArguments

from gLM.callbacks import ElapsedTimeLoggerCallback, PercentIdentityLoggingCallback
from gLM.collator import PhyloCollator, create_mlm_collator
from gLM.dataset import SeqPairIterableDataset, SeqPairMapDataset, Uniref90ArrowDataset
from gLM.models import ProteinBertModel, ProteinT5Model, ProteinT5GemmaModel
from gLM.tokenizers import PhyloTokenizerLoader
from gLM.train_utils import CustomBatchSizeTrainer, PhyloTrainer


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def print_rank0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)


@dataclass
class ModelArguments:
    model_type: Literal["ModernBERT", "T5", "T5Gemma"] = field(
        default="ModernBERT", metadata={"help": "Type of model to use"}
    )
    tokenizer_path: str = field(
        default="char_tokenizer", metadata={"help": "Path to tokenizer dir"}
    )
    max_position_embeddings: int = field(
        default=1024, metadata={"help": "Max sequence length"}
    )
    attn_implementation: Literal["flash_attention_2", "sdpa"] = field(
        default="flash_attention_2", metadata={"help": "Attention implementation"}
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    run_name: str = field(default="run")
    vep_eval_steps: int = field(default=10000)
    mlm_probability: float = field(default=0.15)
    batch_sampler: Literal["default", "length_adaptive", "token_budget", "phylo_default"] = field(
        default="phylo_default"
    )
    training_type: Literal["MLM", "phylo_encoder_only", "phylo_encoder_decoder"] = field(
        default="MLM"
    )
    max_tokens_per_batch: int = field(default=50_000)
    shuffle_batches: bool = field(default=True)
    base_batch_size: int = field(default=8)

    # Safer defaults for PyTorch DataLoader
    dataloader_prefetch_factor: int = field(default=2)
    dataloader_persistent_workers: bool = field(default=True)

    # DDP knobs (do NOT define local_rank yourself)
    ddp_find_unused_parameters: bool = field(default=False)
    ddp_timeout: int = field(default=6000)

    # Common stable defaults (override via CLI if needed)
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)
    group_by_length: bool = field(default=False)
    eval_strategy: str = field(default="no")
    logging_strategy: str = field(default="steps")
    save_strategy: str = field(default="steps")
    report_to: str = field(default="wandb")


@dataclass
class DataArguments:
    train_dataset_type: Literal["iterable", "tokenized_map", "seq_pair_map", "uniref90_arrow"] = field(
        default="uniref90_arrow"
    )
    train_dataset_path: str = field(default="")
    val_dataset_path: str = field(default="")
    vep_input_csv: str = field(default="")
    fasta_path: str = field(default="/gpfs/data/brandeslab/Data/uniref/uniref100.fasta")
    index_db_path: str = field(default="/gpfs/data/brandeslab/User/as12267/uniref100.idx")


@dataclass
class WandbArguments:
    wandb_project: str = field(default="phylo-llm")
    wandb_entity: str = field(default="sinha-anushka12-na")
    disable_wandb: bool = field(default=False)


def main():
    print(
        "RANK", os.environ.get("RANK"),
        "LOCAL_RANK", os.environ.get("LOCAL_RANK"),
        "WORLD_SIZE", os.environ.get("WORLD_SIZE"),
        flush=True
    )

    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments, WandbArguments))
    model_args, data_args, training_args, wandb_args = parser.parse_args_into_dataclasses()

    # W&B: only rank0 logs
    if not wandb_args.disable_wandb and get_rank() == 0:
        wandb.init(project=wandb_args.wandb_project, name=training_args.run_name, entity=wandb_args.wandb_entity)
    else:
        wandb.init(mode="disabled")

    # Tokenizer
    tokenizer = PhyloTokenizerLoader(model_args.tokenizer_path)
    print_rank0(f"Using tokenizer from: {model_args.tokenizer_path}")
    print_rank0(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print_rank0("Mask ID:", tokenizer.mask_token_id, "Pad ID:", tokenizer.pad_token_id)

    # Model (do NOT manually move to device; Trainer/Accelerate will handle)
    if model_args.model_type == "ModernBERT":
        print_rank0("Using ModernBERT model...")
        model = ProteinBertModel(
            vocab_size=tokenizer.vocab_size,
            tokenizer=tokenizer,
            attn_implementation=model_args.attn_implementation,
        ).build()
        model.gradient_checkpointing_enable()
        hidden = getattr(model.config, "hidden_size", None)

    elif model_args.model_type == "T5":
        print_rank0("Using T5 model...")
        model = ProteinT5Model(vocab_size=tokenizer.vocab_size, tokenizer=tokenizer).build()
        model.config.decoder_start_token_id = tokenizer.pad_token_id
        model.gradient_checkpointing_enable()
        hidden = getattr(model.config, "d_model", getattr(model.config, "hidden_size", None))

    elif model_args.model_type == "T5Gemma":
        print_rank0("Using T5Gemma model...")
        model = ProteinT5GemmaModel(
            vocab_size=tokenizer.vocab_size,
            tokenizer=tokenizer,
            attn_implementation=model_args.attn_implementation,
        ).build()
        model.config.decoder_start_token_id = tokenizer.pad_token_id
        model.gradient_checkpointing_enable()
        hidden = getattr(model.config, "d_model", getattr(model.config, "hidden_size", None))

    else:
        raise ValueError(f"Unknown model_type: {model_args.model_type}")

    print_rank0(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print_rank0("Hidden size used:", hidden)
    print_rank0("max_position_embeddings:", model_args.max_position_embeddings)

    # Dataset
    if data_args.train_dataset_type == "tokenized_map":
        print_rank0("Using pre-tokenized dataset")
        train_ds = load_from_disk(data_args.train_dataset_path)
        val_ds = load_from_disk(data_args.val_dataset_path).shuffle(seed=42) if data_args.val_dataset_path else None

    elif data_args.train_dataset_type == "seq_pair_map":
        print_rank0("Using SeqPairMapDataset")
        train_ds = SeqPairMapDataset(dataset_path=data_args.train_dataset_path, training_type=training_args.training_type)
        val_ds = None

    elif data_args.train_dataset_type == "uniref90_arrow":
        print_rank0("Using Uniref90ArrowDataset (FASTA index_db)")
        train_ds = Uniref90ArrowDataset(
            dataset_path=data_args.train_dataset_path,
            training_type=training_args.training_type,
            fasta_path=data_args.fasta_path,
            idx_db_path=data_args.index_db_path,
        )
        val_ds = None

    elif data_args.train_dataset_type == "iterable":
        print_rank0("Using SeqPairIterableDataset")
        train_ds = SeqPairIterableDataset(
            dataset_path=data_args.train_dataset_path,
            tokenizer=tokenizer,
            training_type=training_args.training_type,
            shuffle_buffer=100000,
        )
        val_ds = None

    else:
        raise ValueError(f"Unknown train_dataset_type: {data_args.train_dataset_type}")

    # Collator
    if training_args.training_type == "MLM":
        print_rank0(f"Using MLM collator (p={training_args.mlm_probability})")
        data_collator = create_mlm_collator(
            tokenizer,
            max_seq_len=model_args.max_position_embeddings,
            mlm_probability=training_args.mlm_probability,
        )
    else:
        print_rank0(f"Using Phylo collator for {training_args.training_type}")
        data_collator = PhyloCollator(
            tokenizer=tokenizer,
            training_type=training_args.training_type,
            max_seq_len=model_args.max_position_embeddings,
        )

    # Trainer
    if training_args.batch_sampler == "phylo_default":
        print_rank0("Using PhyloTrainer")
        trainer = PhyloTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
    elif training_args.batch_sampler != "default":
        print_rank0("Using CustomBatchSizeTrainer")
        trainer = CustomBatchSizeTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
    else:
        print_rank0("Using default Trainer")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

    # Callbacks
    trainer.add_callback(ElapsedTimeLoggerCallback())
    if training_args.training_type == "phylo_encoder_only":
        trainer.add_callback(PercentIdentityLoggingCallback())

    # Train
    trainer.train()


if __name__ == "__main__":
    main()