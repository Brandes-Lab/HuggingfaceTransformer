import argparse

import numpy as np
import wandb
from datasets import load_from_disk
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from .evaluation import ElapsedTimeLoggerCallback, ZeroShotVEPEvaluationCallback
from .models import ProteinBertModel
from .tokenization import TokenizerLoader


def main():
    parser = argparse.ArgumentParser(
        description="Train ModernBERT with long context length"
    )
    parser.add_argument(
        "--train_dataset",
        type=str,
        default="/gpfs/data/brandeslab/Data/processed_datasets/uniref90_tokenized_8192/train_only/train",
        help="Path to training dataset",
    )
    parser.add_argument(
        "--val_dataset",
        type=str,
        default="/gpfs/data/brandeslab/Data/processed_datasets/uniref90_tokenized_8192/val_only/validation",
        help="Path to validation dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/gpfs/data/brandeslab/model_checkpts",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="char_tokenizer",
        help="Path to tokenizer directory",
    )
    parser.add_argument(
        "--vep_csv",
        type=str,
        default="/gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv",
        help="Path to ClinVar CSV for zero-shot VEP evaluation",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="modernBERT_uniref_tokenized8192",
        help="Name for this training run",
    )
    parser.add_argument(
        "--max_steps", type=int, default=2_000_000, help="Maximum training steps"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Training batch size per device",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--eval_every_n_steps",
        type=int,
        default=50000,
        help="Run VEP evaluation every N steps",
    )

    args = parser.parse_args()

    wandb.init(
        project="huggingface_bert_sweep",
        name=args.run_name,
        entity="sinha-anushka12-na",
    )

    tokenizer = TokenizerLoader(args.tokenizer_path).load()
    print("Tokenizer vocab size:", tokenizer.vocab_size)

    model = ProteinBertModel.create_long_context_modern(
        vocab_size=tokenizer.vocab_size,
        tokenizer=tokenizer,
        max_position_embeddings=8192,
    ).build()
    model.cuda()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load pre-tokenized datasets
    train_ds = load_from_disk(args.train_dataset)
    val_ds = load_from_disk(args.val_dataset)

    print("Max train length:", max(train_ds["length"]))
    print("99th percentile:", np.percentile(train_ds["length"], 99))
    print("95th percentile:", np.percentile(train_ds["length"], 95))

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}/{args.run_name}",
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=4,
        bf16=True,
        fp16=False,
        dataloader_num_workers=16,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=2,
        eval_strategy="no",  # not running eval
        logging_strategy="steps",
        logging_steps=1000,
        save_strategy="no",
        report_to="wandb",
        run_name=args.run_name,
        learning_rate=args.learning_rate,
        remove_unused_columns=False,
        group_by_length=True,
        length_column_name="length",
    )

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
        ZeroShotVEPEvaluationCallback.create_long_context(
            tokenizer=tokenizer,
            input_csv=args.vep_csv,
            trainer=trainer,
            max_len=8192,
            eval_every_n_steps=args.eval_every_n_steps,
        )
    )
    trainer.add_callback(ElapsedTimeLoggerCallback())
    trainer.train()


if __name__ == "__main__":
    main()
