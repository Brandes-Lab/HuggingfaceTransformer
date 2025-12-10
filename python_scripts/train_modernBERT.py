# type: ignore
import os
from dataclasses import dataclass, field  # type: ignore
from typing import Literal
import pyarrow.parquet as pq
import torch  # type: ignore
from datasets import load_from_disk, Dataset
from transformers import (
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

import wandb
from gLM.callbacks import (
    ElapsedTimeLoggerCallback,
    LossPrintCallback,
    ZeroShotVEPEvaluationCallback,
    PercentIdentityLoggingCallback
)
from gLM.data_utils import TruncatingDataCollatorForMLM
from gLM.models import ProteinBertModel
from gLM.tokenizers import TokenizerLoader, PhyloTokenizerLoader
from gLM.train_utils import CustomBatchSizeTrainer
from gLM.collator import create_mlm_collator, SequencePairCollator
from gLM.dataset import UniRefClusterIterableDataset
from gLM.train_utils import PhyloTrainer

if torch.cuda.is_available():
    print("Using CUDA")
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    print("Using MPS")
    DEVICE = torch.device("mps")
else:
    print("Using CPU")
    DEVICE = torch.device("cpu")


def print_rank0(*args, **kwargs):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(*args, **kwargs)


@dataclass
class ModelArguments:
    """Arguments for model configuration."""

    tokenizer_path: str = field(
        default="char_tokenizer", metadata={"help": "Path to the tokenizer directory"}
    )
    max_position_embeddings: int = field(
        default=8192, metadata={"help": "Maximum sequence length for the model"}
    )
    attn_implementation: Literal["flash_attention_2", "sdpa"] = field(
        default="flash_attention_2",
        metadata={"help": "Attention implementation to use"},
    )

@dataclass
class CustomTrainingArguments(TrainingArguments):
    run_name: str = field(
        default="modernBERT_1B",
        metadata={"help": "Name for the experiment run"},
    )
    output_dir: str = field(
        default="/gpfs/data/brandeslab/model_checkpts",
        metadata={"help": "Directory to save model checkpoints"},
    )
    max_steps: int = field(
        default=3_000_000, metadata={"help": "Maximum number of training steps"}
    )
    per_device_train_batch_size: int = field(
        default=1, metadata={"help": "Training batch size per device"}
    )
    gradient_accumulation_steps: int = field(
        default=32, metadata={"help": "Number of gradient accumulation steps"}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Evaluation batch size per device"}
    )
    learning_rate: float = field(
        default=1e-3, metadata={"help": "Learning rate for training"}
    )
    logging_steps: int = field(
        default=32, metadata={"help": "Number of steps between logging"}
    )
    vep_eval_steps: int = field(
        default=10000, metadata={"help": "Number of steps between VEP evaluations"}
    )
    dataloader_num_workers: int = field(
        default=6, metadata={"help": "Number of dataloader workers"}
    )
    dataloader_persistent_workers: bool = field(default=True, 
        metadata={"help": "Number of dataloader_persistent_workers"}
    )
    dataloader_prefetch_factor: int = field(default=0, 
        metadata={"help": "Number of dataloader_prefetch_factor"}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Probability for masking tokens in MLM"}
    )
    batch_sampler: Literal["default", "length_adaptive", "token_budget", "phylo_default"] = field(
        default="default", metadata={"help": "Batch sampler to use"}
    )
    max_tokens_per_batch: int = field(
        default=50_000, metadata={"help": "Maximum number of tokens per batch"}
    )
    shuffle_batches: bool = field(
        default=True,
        metadata={"help": "Whether to shuffle batches after bucketing by length"},
    )
    training_type: Literal["MLM", "phylo"] = field(
        default="MLM", metadata={"help": "Type of training to perform"}
    )
    ## DDP arguments
    local_rank = (int(os.environ.get("LOCAL_RANK", 0)),)
    ddp_backend = ("nccl",)
    ddp_timeout = (6000,)
    ddp_find_unused_parameters = False

    # Arguments that shouldn't be changed really
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
    eval_strategy: str = field(default="no")  # not running eval
    # eval_steps: int = field(default=50000)
    logging_strategy: str = field(default="steps")
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=1_000_000)
    report_to: str = field(default="wandb")
    remove_unused_columns: bool = field(default=False)
    group_by_length: bool = field(default=False)
    length_column_name: str = field(default="length")
    include_num_input_tokens_seen: str = field(default="non_padding")

    base_batch_size: int = field(default=8, metadata={"help": "Base batch size for dynamic batching"})

@dataclass
class DataArguments:
    """Arguments for data configuration."""
    train_dataset_path: str = field(
        default="/gpfs/data/brandeslab/Data/processed_datasets/uniref90_tokenized_8192/train_only/train",
        metadata={"help": "Path to the training dataset"},
    )
    val_dataset_path: str = field(
        default="/gpfs/data/brandeslab/Data/processed_datasets/uniref90_tokenized_8192/val_only/validation",
        metadata={"help": "Path to the validation dataset"},
    )
    vep_input_csv: str = field(
        default="/gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv",
        metadata={"help": "Path to the VEP evaluation CSV file"},
    )
    fasta_path: str = field(
        default="/gpfs/data/brandeslab/Data/uniref/uniref100.fasta",
        metadata={"help": "Path to the FASTA file for phylogenetic modeling"},
    )
    index_db_path: str = field(
        default="/gpfs/data/brandeslab/User/as12267/uniref100.idx",
        metadata={"help": "Path to the index DB file for FASTA"},
    )


@dataclass
class WandbArguments:
    """Arguments for Weights & Bias initialization."""

    wandb_project: str = field(
        default="long_runs",
        metadata={"help": "Weights & Biases project name"},
    )
    wandb_entity: str = field(
        default="sinha-anushka12-na", metadata={"help": "Weights & Biases entity name"}
    )
    disable_wandb: bool = field(
        default=False,
        metadata={"help": "Whether to disable WandB logging. Defaults to False."},
    )


def main():
    # Parse arguments
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, CustomTrainingArguments, WandbArguments)
    )
    model_args, data_args, training_args, wandb_args = (
        parser.parse_args_into_dataclasses()
    )

    print(
        f"[Rank {training_args.local_rank}] MASTER_ADDR: {os.environ.get('MASTER_ADDR')}"
    )
    print(
        f"[Rank {training_args.local_rank}] MASTER_PORT: {os.environ.get('MASTER_PORT')}"
    )
    print(
        f"[Rank {training_args.local_rank}] NCCL_TIMEOUT: {os.environ.get('NCCL_TIMEOUT')}"
    )

    # Initialize wandb
    if not wandb_args.disable_wandb and training_args.local_rank == 0:
        wandb.init(
            project=wandb_args.wandb_project,
            name=training_args.run_name,
            entity=wandb_args.wandb_entity,
        )
    else:
        wandb.init(mode="disabled")

    # Load tokenizer
    if training_args.training_type == "MLM":
        tokenizer = TokenizerLoader(model_args.tokenizer_path).load()
        pad_id = tokenizer.pad_token_id
        gap_id = None
        print("MLM Tokenizer loaded.")

    elif training_args.training_type == "phylo":
        tokenizer = PhyloTokenizerLoader(model_args.tokenizer_path).load()
        pad_id = tokenizer.pad_token_id
        gap_id = tokenizer.convert_tokens_to_ids("-")
        print("Phylo Tokenizer loaded. GAP ID:", gap_id)

    
    print_rank0("Tokenizer vocab size:", tokenizer.vocab_size)

    # Build model
    model = ProteinBertModel(
        vocab_size=tokenizer.vocab_size, 
        tokenizer=tokenizer, 
        attn_implementation=model_args.attn_implementation
    ).build()
    model.gradient_checkpointing_enable()
    model.to(training_args.local_rank)
    print_rank0(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Hidden size actually used:", model.config.hidden_size)

    # Load datasets
    if training_args.training_type == "MLM":
        train_ds = load_from_disk(data_args.train_dataset_path)
        # train_ds = train_ds.select(range(500))  # for testing
        val_ds = load_from_disk(data_args.val_dataset_path)
        val_ds = val_ds.shuffle(seed=42)
        
        # Select first 500k after shuffling
        val_subset = val_ds.select(range(500_000))

        # Use MLM collator
        data_collator = create_mlm_collator(
        tokenizer,
        mlm_probability=training_args.mlm_probability
        )
    elif training_args.training_type == "phylo":
        train_ds = UniRefClusterIterableDataset(
            parquet_path=data_args.train_dataset_path,
            index_db_path=data_args.index_db_path,
            fasta_path=data_args.fasta_path, 
            tokenizer=PhyloTokenizerLoader(model_args.tokenizer_path),
            max_seq_len=model_args.max_position_embeddings,
        )
        val_ds = None
        data_collator = SequencePairCollator(
            pad_id = tokenizer.pad_token_id,
        )

    # print("Max train length:", max(train_ds["length"]))
    # print(f"Number of seqs of length 8192: {(np.array(train_ds['length'])==8192).sum()}")
    # print("99th percentile:", np.percentile(train_ds["length"], 99))
    # print("95th percentile:", np.percentile(train_ds["length"], 95))


    # Update training arguments with parsed values
    training_args.output_dir = f"{training_args.output_dir}/{training_args.run_name}"

    
    if training_args.batch_sampler == "phylo_default":
        print("using phylo_default Trainer")
        trainer = PhyloTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
    elif training_args.batch_sampler != "default":
        print("using CustomBatchSizeTrainer")
        trainer = CustomBatchSizeTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
    else:
        print("using default Trainer")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )


    if training_args.training_type == "phylo":
        trainer.add_callback(PercentIdentityLoggingCallback())

    trainer.add_callback(
        ZeroShotVEPEvaluationCallback(
            tokenizer=tokenizer,
            input_csv=data_args.vep_input_csv,
            trainer=trainer,
            max_len=model_args.max_position_embeddings,
            batch_size=8,
            eval_every_n_steps=training_args.vep_eval_steps,
            training_type=training_args.training_type, 
        )
    )

    trainer.add_callback(ElapsedTimeLoggerCallback())
    trainer.add_callback(LossPrintCallback())
    
    trainer.train()


if __name__ == "__main__":
    main()
