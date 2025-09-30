from dataclasses import dataclass, field

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_from_disk
from transformers import (
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

import wandb
from gLM.callbacks import (
    ElapsedTimeLoggerCallback,
    ZeroShotVEPEvaluationCallback,
    LossPrintCallback,
)
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
        default=8, metadata={"help": "Training batch size per device"}
    )
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Number of gradient accumulation steps"}
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

    # Arguments that shouldn't be changed really
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
    dataloader_persistent_workers: bool = field(default=True)
    dataloader_prefetch_factor: int = field(default=8)
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
    pad_id = tokenizer.pad_token_id
    print("Tokenizer vocab size:", tokenizer.vocab_size)

    # Build model
    model = ProteinBertModel(tokenizer.vocab_size, tokenizer).build()
    model.to(device=DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load pre-tokenized datasets
    train_ds = load_from_disk(data_args.train_dataset_path)
    val_ds = load_from_disk(data_args.val_dataset_path)

    # print("Max train length:", max(train_ds["length"]))
    # print("99th percentile:", np.percentile(train_ds["length"], 99))
    # print("95th percentile:", np.percentile(train_ds["length"], 95))

    print(training_args.mlm_probability)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=training_args.mlm_probability
    )

    # Update training arguments with parsed values
    training_args.output_dir = f"{training_args.output_dir}/{training_args.run_name}"

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # # === Inspect batches before training ===
    # dl = trainer.get_train_dataloader()
    # for batch in dl:
    #     print("input_ids", batch["input_ids"][0])
    #     print("labels", batch["labels"][0])
    #     print("PAD token id:", pad_id)
    #     # Check loss calculation
    #     outputs = model(input_ids=batch["input_ids"].cuda(), labels=batch["labels"].cuda())
    #     print("One batch loss:", outputs.loss.item())
    #     break  # Only check the first batch

    # batch_lens = []

    # print("\nInspecting first 20 batches...\n")
    # for i, batch in enumerate(dl):
    #     input_ids = batch["input_ids"]
    #     if isinstance(input_ids, torch.Tensor):
    #         lengths = (input_ids != pad_id).sum(dim=1).tolist()
    #         min_len = min(lengths)
    #         max_len = max(lengths)
    #         batch_lens.append((i, min_len, max_len))

    #         print(f"Batch {i:03d} → min length = {min_len}, max length = {max_len}, batch size = {len(lengths)}")

    #     if i >= 100:
    #         break

    # # === Optional: Plot Length Spread ===
    # if batch_lens:
    #     batch_ids, min_lens, max_lens = zip(*batch_lens)
    #     plt.figure(figsize=(10, 4))
    #     plt.plot(batch_ids, max_lens, label='Max length')
    #     plt.plot(batch_ids, min_lens, label='Min length')
    #     plt.fill_between(batch_ids, min_lens, max_lens, alpha=0.2, label='Length spread')
    #     plt.xlabel("Batch Index")
    #     plt.ylabel("Sequence Length")
    #     plt.title("Min/Max Token Length per Batch (after group_by_length)")
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig("group_by_len.png")
    #     plt.show()

    trainer.add_callback(
        ZeroShotVEPEvaluationCallback(
            tokenizer=tokenizer,
            input_csv=data_args.vep_input_csv,
            trainer=trainer,
            eval_every_n_steps=training_args.vep_eval_steps,
        )
    )
    trainer.add_callback(ElapsedTimeLoggerCallback())
    trainer.add_callback(LossPrintCallback())

    trainer.train()


if __name__ == "__main__":
    main()
