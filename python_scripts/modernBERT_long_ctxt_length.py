from dataclasses import dataclass, field

import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import DataLoader

import wandb
from gLM.callbacks import ElapsedTimeLoggerCallback, ZeroShotVEPEvaluationCallback
from gLM.data_utils import DynamicBatchSampler, DynamicDataCollator
from gLM.models import ProteinBertModel
from gLM.tokenizers import TokenizerLoader

if torch.cuda.is_available():
    print("Using CUDA")
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    print("Using MPS")
    DEVICE = torch.device("mps")
else:
    print("Using CPU")
    DEVICE = torch.device("cpu")


class DynamicBatchTrainer(Trainer):
    """Custom Trainer that supports dynamic batching."""

    def __init__(self, max_tokens_per_batch=32768, use_dynamic_batching=True, **kwargs):
        super().__init__(**kwargs)
        self.max_tokens_per_batch = max_tokens_per_batch
        self.use_dynamic_batching = use_dynamic_batching

    def get_train_dataloader(self):
        """Create train dataloader with dynamic batching if enabled."""
        if not self.use_dynamic_batching:
            return super().get_train_dataloader()

        train_dataset = self.train_dataset

        # Create dynamic batch sampler
        batch_sampler = DynamicBatchSampler(
            dataset=train_dataset,
            max_tokens_per_batch=self.max_tokens_per_batch,
            shuffle=True,
            drop_last=self.args.dataloader_drop_last,
        )

        # Create dataloader with custom batch sampler
        dataloader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
            prefetch_factor=self.args.dataloader_prefetch_factor,
        )

        return dataloader

    def get_eval_dataloader(self, eval_dataset=None):
        """Create eval dataloader with dynamic batching if enabled."""
        if not self.use_dynamic_batching:
            return super().get_eval_dataloader(eval_dataset)

        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        # Create dynamic batch sampler for evaluation
        batch_sampler = DynamicBatchSampler(
            dataset=eval_dataset,
            max_tokens_per_batch=self.max_tokens_per_batch,
            shuffle=False,  # Don't shuffle for evaluation
            drop_last=False,
        )

        # Create dataloader with custom batch sampler
        dataloader = DataLoader(
            eval_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
            prefetch_factor=self.args.dataloader_prefetch_factor,
        )

        return dataloader


@dataclass
class ModelArguments:
    """Arguments for model configuration."""

    tokenizer_path: str = field(
        default="char_tokenizer", metadata={"help": "Path to the tokenizer directory"}
    )
    max_position_embeddings: int = field(
        default=8192, metadata={"help": "Maximum sequence length for the model"}
    )


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


class CustomTrainingArguments(TrainingArguments):
    run_name: str = field(
        default="modernBERT_uniref_tokenized8192",
        metadata={"help": "Name for the experiment run"},
    )
    output_dir: str = field(
        default="/gpfs/data/brandeslab/model_checkpts",
        metadata={"help": "Directory to save model checkpoints"},
    )
    max_steps: int = field(
        default=2_000_000, metadata={"help": "Maximum number of training steps"}
    )
    per_device_train_batch_size: int = field(
        default=16, metadata={"help": "Training batch size per device"}
    )
    gradient_accumulation_steps: int = field(
        default=16, metadata={"help": "Number of gradient accumulation steps"}
    )
    per_device_eval_batch_size: int = field(
        default=4, metadata={"help": "Evaluation batch size per device"}
    )
    learning_rate: float = field(
        default=3e-4, metadata={"help": "Learning rate for training"}
    )
    logging_steps: int = field(
        default=1000, metadata={"help": "Number of steps between logging"}
    )
    vep_eval_steps: int = field(
        default=50000, metadata={"help": "Number of steps between VEP evaluations"}
    )
    dataloader_num_workers: int = field(
        default=16, metadata={"help": "Number of dataloader workers"}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Probability for masking tokens in MLM"}
    )
    max_tokens_per_batch: int = field(
        default=32768,
        metadata={"help": "Maximum total tokens per batch for dynamic batching"},
    )
    use_dynamic_batching: bool = field(
        default=True,
        metadata={"help": "Whether to use dynamic batching based on sequence lengths"},
    )

    # Arguments that shouldn't be changed really
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
    dataloader_persistent_workers: bool = field(default=True)
    dataloader_prefetch_factor: int = field(default=2)
    eval_strategy: str = field(default="no")  # not running eval
    logging_strategy: str = field(default="steps")
    save_strategy: str = field(default="no")
    report_to: str = field(default="wandb")
    remove_unused_columns: bool = field(default=False)
    group_by_length: bool = field(default=True)
    length_column_name: str = field(default="length")


@dataclass
class WandbArguments:
    """Arguments for Weights & Bias initialization."""

    wandb_project: str = field(
        default="huggingface_bert_sweep",
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

    # Initialize wandb
    if not wandb_args.disable_wandb:
        wandb.init(
            project=wandb_args.wandb_project,
            name=training_args.run_name,
            entity=wandb_args.wandb_entity,
        )

    # Load tokenizer
    tokenizer = TokenizerLoader(model_args.tokenizer_path).load()
    print("Tokenizer vocab size:", tokenizer.vocab_size)

    # Build model
    model = ProteinBertModel(tokenizer.vocab_size, tokenizer).build()
    model.to(device=DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load pre-tokenized datasets
    train_ds = load_from_disk(data_args.train_dataset_path)
    val_ds = load_from_disk(data_args.val_dataset_path)

    print("Max train length:", max(train_ds["length"]))
    print("99th percentile:", np.percentile(train_ds["length"], 99))
    print("95th percentile:", np.percentile(train_ds["length"], 95))

    print(training_args.mlm_probability.default)

    # Choose data collator based on dynamic batching setting
    if training_args.use_dynamic_batching:
        data_collator = DynamicDataCollator(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=training_args.mlm_probability.default,
        )
        print(
            f"Using dynamic batching with max_tokens_per_batch={training_args.max_tokens_per_batch}"
        )
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=training_args.mlm_probability.default,
        )
        print("Using standard batching")

    # Update training arguments with parsed values
    training_args.output_dir = f"{training_args.output_dir}/{training_args.run_name}"

    # Use custom trainer for dynamic batching
    trainer = DynamicBatchTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        max_tokens_per_batch=training_args.max_tokens_per_batch,
        use_dynamic_batching=training_args.use_dynamic_batching,
    )
    # Inspect first few batches to verify dynamic batching
    dl = trainer.get_train_dataloader()
    print("\n=== Dynamic Batching Statistics ===")
    for i, batch in enumerate(dl):
        batch_size = batch["input_ids"].shape[0]
        max_length = batch["input_ids"].shape[1]
        total_tokens = batch_size * max_length

        if "batch_stats" in batch:
            stats = batch["batch_stats"]
            print(
                f"Batch {i}: size={batch_size}, max_len={max_length}, "
                f"total_tokens={total_tokens}, avg_len={stats['avg_length']:.1f}, "
                f"std={stats['length_std']:.1f}"
            )
        else:
            print(
                f"Batch {i}: size={batch_size}, max_len={max_length}, total_tokens={total_tokens}"
            )

        if i >= 10:  # Show more batches to see the variation
            break
    print("===================================\n")

    trainer.add_callback(
        ZeroShotVEPEvaluationCallback(
            tokenizer=tokenizer,
            input_csv=data_args.vep_input_csv,
            trainer=trainer,
            eval_every_n_steps=training_args.vep_eval_steps,
        )
    )
    trainer.add_callback(ElapsedTimeLoggerCallback())
    trainer.train()


if __name__ == "__main__":
    main()
