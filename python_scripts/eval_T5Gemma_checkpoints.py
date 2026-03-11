# type: ignore
import os
import re
import math
import random
from dataclasses import dataclass, field
from typing import Literal, List

import pandas as pd
import torch
import wandb
from torch.utils.data import Dataset
from datasets import load_from_disk
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    T5GemmaForConditionalGeneration,
)

from gLM.sequences.pairwise_align import align_pair, percent_identity
from gLM.sequences.seq_fetcher import SequenceFetcher
from gLM.tokenizers import PhyloTokenizerLoader
from gLM.collator import create_mlm_collator, PhyloCollator
from gLM.dataset import Uniref90ArrowEvalDataset
from gLM.train_utils import PhyloTrainer


def print0(*args, **kwargs):
    print(*args, **kwargs)



@dataclass
class ModelArguments:
    tokenizer_path: str = field(default="./phylo_char_tokenizer_updated")
    max_position_embeddings: int = field(default=1024)


@dataclass
class EvalArguments:
    checkpoint_root: str = field(
        default="/gpfs/data/brandeslab/model_checkpts/T5Gemma_97M_phylo_bs_4096_arrow_dataset_index_file"
    )
    val_dataset_path: str = field(
        default="/gpfs/data/brandeslab/Data/uniref/uniref90_clusters_arrow/test"
    )
    fasta_path: str = field(
        default="/gpfs/data/brandeslab/Data/uniref/uniref100.fasta"
    )
    index_db_path: str = field(
        default="/gpfs/data/brandeslab/User/as12267/uniref100.idx"
    )
    output_csv: str = field(
        default="/gpfs/data/brandeslab/model_checkpts/T5Gemma_97M_phylo_bs_4096_arrow_dataset_index_file/validation_metrics.csv"
    )
    training_type: Literal["MLM", "phylo_encoder_only", "phylo_encoder_decoder"] = field(
        default="phylo_encoder_decoder"
    )
    per_device_eval_batch_size: int = field(default=8)
    dataloader_num_workers: int = field(default=4)
    dataloader_persistent_workers: bool = field(default=True)
    dataloader_prefetch_factor: int = field(default=2)
    max_eval_checkpoints: int = field(default=-1)
    deterministic_pair: bool = field(default=True)

    wandb_project: str = field(default="phylo-llm")
    wandb_entity: str = field(default="sinha-anushka12-na")
    wandb_run_name: str = field(default="T5Gemma_checkpoint_eval")
    disable_wandb: bool = field(default=False)


def get_checkpoint_step(path: str) -> int:
    m = re.search(r"checkpoint-(\d+)$", os.path.basename(path.rstrip("/")))
    if not m:
        raise ValueError(f"Could not parse checkpoint step from: {path}")
    return int(m.group(1))


def list_checkpoints(checkpoint_root: str) -> List[str]:
    ckpts = []
    for name in os.listdir(checkpoint_root):
        full = os.path.join(checkpoint_root, name)
        if os.path.isdir(full) and re.match(r"checkpoint-\d+$", name):
            ckpts.append(full)
    return sorted(ckpts, key=get_checkpoint_step)


def main():
    parser = HfArgumentParser((ModelArguments, EvalArguments))
    model_args, eval_args = parser.parse_args_into_dataclasses()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print0("Using device:", device)

    if eval_args.disable_wandb:
        wandb.init(mode="disabled")
    else:
        wandb.init(
            project=eval_args.wandb_project,
            entity=eval_args.wandb_entity,
            name=eval_args.wandb_run_name,
            config={
                "checkpoint_root": eval_args.checkpoint_root,
                "val_dataset_path": eval_args.val_dataset_path,
                "training_type": eval_args.training_type,
                "per_device_eval_batch_size": eval_args.per_device_eval_batch_size,
                "max_position_embeddings": model_args.max_position_embeddings,
            },
        )

    print0("Loading tokenizer...")
    tokenizer = PhyloTokenizerLoader(model_args.tokenizer_path)

    print0("Building eval dataset...")
    val_ds = Uniref90ArrowEvalDataset(
        dataset_path=eval_args.val_dataset_path,
        training_type=eval_args.training_type,
        fasta_path=eval_args.fasta_path,
        idx_db_path=eval_args.index_db_path,
        deterministic_pair=eval_args.deterministic_pair,
    )
    print0(f"Validation dataset size: {len(val_ds):,}")

    if eval_args.training_type == "MLM":
        data_collator = create_mlm_collator(
            tokenizer,
            max_seq_len=model_args.max_position_embeddings,
            mlm_probability=0.15,
        )
        trainer_cls = Trainer
    elif eval_args.training_type in ["phylo_encoder_only", "phylo_encoder_decoder"]:
        data_collator = PhyloCollator(
            tokenizer=tokenizer,
            training_type=eval_args.training_type,
            max_seq_len=model_args.max_position_embeddings,
        )
        trainer_cls = PhyloTrainer
    else:
        raise ValueError(f"Unknown training_type: {eval_args.training_type}")

    checkpoints = list_checkpoints(eval_args.checkpoint_root)
    if eval_args.max_eval_checkpoints > 0:
        checkpoints = checkpoints[:eval_args.max_eval_checkpoints]

    print0(f"Found {len(checkpoints)} checkpoints")
    for ckpt in checkpoints:
        print0(" ", ckpt)

    hf_eval_args = TrainingArguments(
        output_dir=os.path.join(eval_args.checkpoint_root, "tmp_eval_outputs"),
        per_device_eval_batch_size=eval_args.per_device_eval_batch_size,
        dataloader_num_workers=eval_args.dataloader_num_workers,
        dataloader_persistent_workers=eval_args.dataloader_persistent_workers,
        dataloader_prefetch_factor=eval_args.dataloader_prefetch_factor,
        remove_unused_columns=False,
        report_to=["wandb"] if not eval_args.disable_wandb else [],
        run_name=eval_args.wandb_run_name,
        bf16=torch.cuda.is_available(),
        fp16=False,
    )

    rows = []

    for ckpt in checkpoints:
        step = get_checkpoint_step(ckpt)
        print0(f"\nEvaluating checkpoint {step} ...")

        model = T5GemmaForConditionalGeneration.from_pretrained(ckpt).to(device)
        model.config.decoder_start_token_id = tokenizer.pad_token_id
        model.eval()

        trainer = trainer_cls(
            model=model,
            args=hf_eval_args,
            train_dataset=None,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        metrics = trainer.evaluate(metric_key_prefix="eval")

        eval_loss = metrics.get("eval_loss", float("nan"))
        ppl = math.exp(eval_loss) if (eval_loss == eval_loss and eval_loss < 20) else float("nan")

        row = {
            "checkpoint": ckpt,
            "step": step,
            "eval_loss": eval_loss,
            "perplexity": ppl,
            **metrics,
        }
        rows.append(row)

        if not eval_args.disable_wandb and wandb.run is not None:
            wandb.log(
                {
                    "checkpoint_step": step,
                    "eval_loss_manual": eval_loss,
                    "perplexity": ppl,
                },
                step=step,
            )

        print0(f"step={step} eval_loss={eval_loss:.6f}")

        del trainer
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows).sort_values("step")
    os.makedirs(os.path.dirname(eval_args.output_csv), exist_ok=True)
    df.to_csv(eval_args.output_csv, index=False)

    print0(f"\nSaved results to: {eval_args.output_csv}")
    print0(df[["step", "eval_loss", "perplexity"]].to_string(index=False))

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()