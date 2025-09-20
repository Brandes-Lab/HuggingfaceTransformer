import argparse

import wandb
from torch.utils.data import DataLoader
from transformers import (
    Trainer,
    TrainingArguments,
)

from .callbacks import ElapsedTimeLoggerCallback, ZeroShotVEPEvaluationCallback
from .data import MLMDataCollator, ProteinDataset
from .models import ProteinBertModel
from .utils import TokenizerLoader


def main():
    parser = argparse.ArgumentParser(description="Train ModernBERT on single GPU")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/gpfs/data/brandeslab/Data/tokenized_datasets/uniref90_tokenized_single_char_512",
        help="Path to tokenized dataset",
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
        default="modernBERT_medium_512_old",
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
        default=64,
        help="Training batch size per device",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--eval_every_n_steps",
        type=int,
        default=20000,
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

    dataset = ProteinDataset(args.dataset_path).load()
    dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(10_000))
    dataset["test"] = dataset["test"].shuffle(seed=42).select(range(10_000))

    model = ProteinBertModel.create_single_gpu_modern(
        vocab_size=tokenizer.vocab_size,
        tokenizer=tokenizer,
        max_position_embeddings=512,
    ).build()
    model.cuda()

    print(
        f"Model parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True
    )
    data_collator = MLMDataCollator(tokenizer).get()

    dl = DataLoader(dataset["train"], batch_size=4, collate_fn=data_collator)

    batch = next(iter(dl))
    labels = batch["labels"]

    # print("Labels shape:", labels.shape)
    # print("Example label row:", labels[0])
    # print("Number of masked tokens in sample 0:", (labels[0] != -100).sum().item())

    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}/{args.run_name}",
        # num_train_epochs=100,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=4,
        dataloader_num_workers=16,
        eval_strategy="steps",
        eval_steps=args.eval_every_n_steps,
        save_steps=500000,
        logging_strategy="steps",
        logging_steps=args.eval_every_n_steps,
        report_to="wandb",
        run_name=args.run_name,
        fp16=True,
        learning_rate=args.learning_rate,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.add_callback(
        ZeroShotVEPEvaluationCallback.create_single_gpu(
            tokenizer=tokenizer,
            input_csv=args.vep_csv,
            trainer=trainer,
            max_len=512,
            eval_every_n_steps=args.eval_every_n_steps,
        )
    )

    trainer.add_callback(ElapsedTimeLoggerCallback())
    trainer.train()


if __name__ == "__main__":
    main()
