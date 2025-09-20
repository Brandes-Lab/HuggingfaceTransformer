import argparse
import os

import wandb
from transformers import (
    Trainer,
    TrainingArguments,
)

from .callbacks import ZeroShotVEPEvaluationCallback
from .data import MLMDataCollator, ProteinDataset
from .models import ProteinBertModel
from .utils import TokenizerLoader


def main():
    parser = argparse.ArgumentParser(description="Train BERT with multi-GPU support")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/gpfs/data/brandeslab/Data/tokenized_datasets/uniref90_tokenized_single_char_2048",
        help="Path to tokenized dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/gpfs/data/brandeslab/model_checkpts/bert_2048_uniref_2GPU",
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
        default="bert_2048_2GPU",
        help="Name for this training run",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32,
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
        default=100000,
        help="Run evaluation every N steps",
    )

    args = parser.parse_args()

    rank = int(os.environ["LOCAL_RANK"])
    tokenizer = TokenizerLoader(args.tokenizer_path).load()
    dataset = ProteinDataset(args.dataset_path).load()

    model = ProteinBertModel.create_multi_gpu_modern(
        vocab_size=tokenizer.vocab_size,
        tokenizer=tokenizer,
        max_position_embeddings=2048,
    ).build()
    model = model.to(rank)
    print(
        f"[Rank {rank}] Model parameters: {sum(p.numel() for p in model.parameters()):,}",
        flush=True,
    )

    data_collator = MLMDataCollator(tokenizer).get()

    training_args = TrainingArguments(
        ddp_find_unused_parameters=False,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=4,
        eval_strategy="steps",
        eval_steps=args.eval_every_n_steps,
        save_steps=args.eval_every_n_steps,
        logging_strategy="steps",
        logging_steps=args.eval_every_n_steps,
        report_to="wandb" if rank == 0 else None,
        run_name=args.run_name,
        fp16=True,
        local_rank=rank,
        ddp_backend="nccl",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print("Initializing wandb")
        wandb.init(project="huggingface_bert_multiGPU", name=args.run_name)

    trainer.add_callback(
        ZeroShotVEPEvaluationCallback.create_multi_gpu(
            tokenizer=tokenizer,
            input_csv=args.vep_csv,
            trainer=trainer,
            max_len=2048,
            eval_every_n_steps=args.eval_every_n_steps,
        )
    )

    trainer.train()


if __name__ == "__main__":
    main()
