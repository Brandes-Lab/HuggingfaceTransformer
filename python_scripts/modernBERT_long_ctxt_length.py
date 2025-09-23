from dataclasses import dataclass, field

import numpy as np
import wandb
from datasets import load_from_disk
from transformers import (
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

from gLM.callbacks import ElapsedTimeLoggerCallback, ZeroShotVEPEvaluationCallback
from gLM.models import ProteinBertModel
from gLM.tokenizers import TokenizerLoader


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


@dataclass
class ExperimentArguments:
    """Arguments for experiment configuration."""

    run_name: str = field(
        default="modernBERT_uniref_tokenized8192",
        metadata={"help": "Name for the experiment run"},
    )
    output_dir: str = field(
        default="/gpfs/data/brandeslab/model_checkpts",
        metadata={"help": "Directory to save model checkpoints"},
    )
    wandb_project: str = field(
        default="huggingface_bert_sweep",
        metadata={"help": "Weights & Biases project name"},
    )
    wandb_entity: str = field(
        default="sinha-anushka12-na", metadata={"help": "Weights & Biases entity name"}
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


def main():
    # Parse arguments
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, ExperimentArguments, TrainingArguments)
    )
    model_args, data_args, exp_args, training_args = (
        parser.parse_args_into_dataclasses()
    )

    # Initialize wandb
    wandb.init(
        project=exp_args.wandb_project,
        name=exp_args.run_name,
        entity=exp_args.wandb_entity,
    )

    # Load tokenizer
    tokenizer = TokenizerLoader(model_args.tokenizer_path).load()
    print("Tokenizer vocab size:", tokenizer.vocab_size)

    # Build model
    model = ProteinBertModel(tokenizer.vocab_size, tokenizer).build()
    model.cuda()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load pre-tokenized datasets
    train_ds = load_from_disk(data_args.train_dataset_path)
    val_ds = load_from_disk(data_args.val_dataset_path)

    print("Max train length:", max(train_ds["length"]))
    print("99th percentile:", np.percentile(train_ds["length"], 99))
    print("95th percentile:", np.percentile(train_ds["length"], 95))

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=exp_args.mlm_probability
    )

    # Update training arguments with parsed values
    training_args.output_dir = f"{exp_args.output_dir}/{exp_args.run_name}"
    training_args.max_steps = exp_args.max_steps
    training_args.per_device_train_batch_size = exp_args.per_device_train_batch_size
    training_args.gradient_accumulation_steps = exp_args.gradient_accumulation_steps
    training_args.per_device_eval_batch_size = exp_args.per_device_eval_batch_size
    training_args.dataloader_num_workers = exp_args.dataloader_num_workers
    training_args.logging_steps = exp_args.logging_steps
    training_args.learning_rate = exp_args.learning_rate
    training_args.run_name = exp_args.run_name

    # Set other training arguments that weren't parameterized
    training_args.bf16 = True
    training_args.fp16 = False
    training_args.dataloader_persistent_workers = True
    training_args.dataloader_prefetch_factor = 2
    training_args.eval_strategy = "no"  # not running eval
    training_args.logging_strategy = "steps"
    training_args.save_strategy = "no"
    training_args.report_to = "wandb"
    training_args.remove_unused_columns = False
    training_args.group_by_length = True
    training_args.length_column_name = "length"

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    dl = trainer.get_train_dataloader()
    for i, batch in enumerate(dl):
        print(f"Batch {i} max length: {batch['input_ids'].shape[1]}")
        if i > 5:
            break

    trainer.add_callback(
        ZeroShotVEPEvaluationCallback(
            tokenizer=tokenizer,
            input_csv=data_args.vep_input_csv,
            trainer=trainer,
            eval_every_n_steps=exp_args.vep_eval_steps,
        )
    )
    trainer.add_callback(ElapsedTimeLoggerCallback())
    trainer.train()


if __name__ == "__main__":
    main()
