# type: ignore
import os
import random
from dataclasses import dataclass, field  # type: ignore
from typing import Literal

import numpy as np
import torch  # type: ignore
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

import wandb
from gLM.callbacks import (
    ZeroShotVEPEvaluationCallback,
    PercentIdentityLoggingCallback,
)
from gLM.models import ProteinBertModel
from gLM.models import ProteinT5Model
from gLM.models import ProteinBARTModel
from gLM.models import ProteinT5GemmaModel
from gLM.tokenizers import PhyloTokenizerLoader
from gLM.train_utils import CustomBatchSizeTrainer
from gLM.collator import create_mlm_collator, PhyloCollator
from gLM.dataset import (
    Uniref90ArrowDatasetForFASTA,
    Uniref90ArrowEvalDatasetForFASTA,
    Uniref90ArrowDatasetForLMDB,
    Uniref90ArrowEvalDatasetForLMDB,
)


# -------------------------
# Device
# -------------------------
if torch.cuda.is_available():
    DEVICE = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
    print(f"✅ CUDA available, using device: {DEVICE}")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✅ MPS available, using MPS")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ CUDA/MPS not available. Using CPU")


def print_rank0(*args, **kwargs):
    if int(os.environ.get("RANK", "0")) == 0:
        print(*args, **kwargs)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


# -------------------------
# Arguments
# -------------------------
@dataclass
class ModelArguments:
    model_type: Literal["ModernBERT", "T5", "BART", "T5Gemma"] = field(
        default="ModernBERT", metadata={"help": "Type of model to use"}
    )
    tokenizer_path: str = field(
        default="char_tokenizer", metadata={"help": "Path to tokenizer directory"}
    )
    max_position_embeddings: int = field(
        default=8192, metadata={"help": "Maximum sequence length"}
    )
    attn_implementation: Literal["flash_attention_2", "sdpa"] = field(
        default="flash_attention_2",
        metadata={"help": "Attention implementation to use"},
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    run_name: str = field(
        default="modernBERT_debug",
        metadata={"help": "Experiment run name"},
    )
    output_dir: str = field(
        default="/gpfs/data/brandeslab/model_checkpts",
        metadata={"help": "Directory to save model checkpoints"},
    )
    max_steps: int = field(default=-1)
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=32)
    per_device_eval_batch_size: int = field(default=8)
    learning_rate: float = field(default=1e-3)
    logging_steps: int = field(default=32)
    vep_eval_steps: int = field(default=10000)
    dataloader_num_workers: int = field(default=6)
    dataloader_persistent_workers: bool = field(default=True)
    dataloader_prefetch_factor: int | None = field(default=2)
    mlm_probability: float = field(default=0.15)
    batch_sampler: Literal["default", "length_adaptive", "token_budget", "phylo_default"] = field(
        default="default"
    )
    trainer_mode: Literal["old", "new"] = field(
        default="new",
        metadata={"help": "Which PhyloTrainer implementation to use"}
    )
    max_tokens_per_batch: int = field(default=50_000)
    shuffle_batches: bool = field(default=True)
    training_type: Literal["MLM", "phylo_encoder_only", "phylo_encoder_decoder"] = field(
        default="MLM"
    )

    # Mostly fixed args
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
    eval_strategy: str = field(default="no")
    eval_steps: int = field(default=50000)
    logging_strategy: str = field(default="steps")
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=1_000_000)
    report_to: str = field(default="wandb")
    remove_unused_columns: bool = field(default=False)
    group_by_length: bool = field(default=False)
    length_column_name: str = field(default="length")
    include_num_input_tokens_seen: str = field(default="non_padding")
    lr_scheduler_type: str = field(default="linear")
    warmup_steps: int = field(default=0)
    base_batch_size: int = field(default=8)

    seed: int = field(default=42)
    data_seed: int = field(default=42)


@dataclass
class DataArguments:
    train_dataset_type: Literal["tokenized_map", "uniref90_arrow_fasta", "uniref90_arrow_lmdb"] = field(
        default="uniref90_arrow_fasta"
    )
    train_dataset_path: str = field(
        default="/gpfs/data/brandeslab/Data/processed_datasets/uniref90_tokenized_8192/train_only/train"
    )
    val_dataset_path: str = field(
        default="/gpfs/data/brandeslab/Data/processed_datasets/uniref90_tokenized_8192/val_only/validation"
    )
    vep_input_csv: str = field(
        default="/gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv"
    )
    fasta_path: str = field(
        default="/gpfs/data/brandeslab/Data/uniref/uniref100.fasta"
    )
    index_db_path: str = field(
        default="/gpfs/data/brandeslab/User/as12267/uniref100.idx"
    )
    lmdb_path: str = field(
        default="/gpfs/data/brandeslab/Data/uniref/uniref100_bk.lmdb"
    )


@dataclass
class WandbArguments:
    wandb_project: str = field(default="long_runs")
    wandb_entity: str = field(default="sinha-anushka12-na")
    disable_wandb: bool = field(default=False)


# -------------------------
# Debug helpers
# -------------------------
def build_debug_dataloader_kwargs(training_args, data_collator):
    kwargs = dict(
        batch_size=training_args.train_batch_size,
        shuffle=True,
        num_workers=training_args.dataloader_num_workers,
        pin_memory=training_args.dataloader_pin_memory,
        persistent_workers=training_args.dataloader_persistent_workers,
        worker_init_fn=seed_worker,
        collate_fn=data_collator,
        drop_last=training_args.dataloader_drop_last,
    )
    if training_args.dataloader_num_workers > 0 and training_args.dataloader_prefetch_factor is not None:
        kwargs["prefetch_factor"] = training_args.dataloader_prefetch_factor
    return kwargs


class PhyloTrainerNew(Trainer):
    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            **build_debug_dataloader_kwargs(self.args, self.data_collator),
        )


class PhyloTrainerOld(Trainer):
    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            **build_debug_dataloader_kwargs(self.args, self.data_collator),
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        result = super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            **kwargs,
        )

        if return_outputs:
            loss, outputs = result
        else:
            loss = result

        pid = inputs.get("percent_identity", None)
        if pid is not None and self.state.is_world_process_zero:
            avg_pid = pid.detach().float().mean().item()
            self.log({"percent_identity": avg_pid})

        if return_outputs:
            return loss, outputs
        return loss


class LossDebugCallback(TrainerCallback):
    """
    Sample a fresh train batch and print:
      - raw model(**batch).loss
      - active label count/fraction
      - input==label rate on active positions
    """
    def __init__(self, trainer, every_n_steps=32, max_batches=5):
        self.trainer = trainer
        self.every_n_steps = every_n_steps
        self.max_batches = max_batches
        self.n_printed = 0

    def on_step_end(self, args, state, control, **kwargs):
        if self.n_printed >= self.max_batches:
            return
        if state.global_step == 0:
            return
        if state.global_step % self.every_n_steps != 0:
            return
        if not state.is_world_process_zero:
            return

        trainer = self.trainer
        model = kwargs["model"]

        model_was_training = model.training
        model.train()

        try:
            batch = next(iter(trainer.get_train_dataloader()))
            batch = {
                k: (v.to(model.device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }

            with torch.no_grad():
                outputs = model(**batch)

            labels = batch.get("labels", None)
            if labels is None:
                print(f"[DEBUG step {state.global_step}] No labels found in batch.")
                return

            active = labels != -100
            n_active = active.sum().item()
            frac_active = active.float().mean().item()

            same_on_active = None
            if n_active > 0 and "input_ids" in batch:
                same_on_active = (
                    batch["input_ids"][active] == labels[active]
                ).float().mean().item()

            print("=" * 100)
            print(f"[DEBUG step {state.global_step}] raw outputs.loss = {outputs.loss.item():.6f}")
            print(f"[DEBUG step {state.global_step}] input_ids shape = {tuple(batch['input_ids'].shape)}")
            print(f"[DEBUG step {state.global_step}] labels shape    = {tuple(labels.shape)}")
            print(f"[DEBUG step {state.global_step}] active labels   = {n_active}")
            print(f"[DEBUG step {state.global_step}] active fraction = {frac_active:.6f}")
            if same_on_active is not None:
                print(f"[DEBUG step {state.global_step}] input==label on active positions = {same_on_active:.6f}")
            print("=" * 100)

            self.n_printed += 1

        finally:
            if not model_was_training:
                model.eval()


class TrainerLossPathDebugCallback(TrainerCallback):
    """
    Compare:
      - raw model(**batch).loss
      - trainer.compute_loss(model, batch)
    """
    def __init__(self, trainer, every_n_steps=32, max_batches=5):
        self.trainer = trainer
        self.every_n_steps = every_n_steps
        self.max_batches = max_batches
        self.n_printed = 0

    def on_step_end(self, args, state, control, **kwargs):
        if self.n_printed >= self.max_batches:
            return
        if state.global_step == 0:
            return
        if state.global_step % self.every_n_steps != 0:
            return
        if not state.is_world_process_zero:
            return

        trainer = self.trainer
        model = kwargs["model"]

        batch = next(iter(trainer.get_train_dataloader()))
        batch = {
            k: (v.to(model.device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

        model_was_training = model.training
        model.train()

        try:
            with torch.no_grad():
                raw_model_loss = model(**batch).loss

            loss_from_trainer = trainer.compute_loss(
                model,
                batch,
                return_outputs=False,
            )

            if isinstance(loss_from_trainer, tuple):
                loss_from_trainer = loss_from_trainer[0]

            print(f"[DEBUG step {state.global_step}] raw_model_loss       = {raw_model_loss.item():.6f}")
            print(f"[DEBUG step {state.global_step}] trainer.compute_loss  = {loss_from_trainer.item():.6f}")
            print(f"[DEBUG step {state.global_step}] difference           = {(loss_from_trainer.item() - raw_model_loss.item()):.6f}")

            self.n_printed += 1
        finally:
            if not model_was_training:
                model.eval()


class LogValueDebugCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs is not None:
            print(f"[LOG step {state.global_step}] {logs}")


def one_time_batch_debug(trainer, model, rank):
    if rank != 0:
        return

    model_was_training = model.training
    model.eval()

    try:
        train_dataloader = trainer.get_train_dataloader()
        batch = next(iter(train_dataloader))
        batch = {
            k: (v.to(model.device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

        with torch.no_grad():
            outputs = model(**batch)

        labels = batch.get("labels", None)

        print("\n" + "=" * 100)
        print("ONE-TIME TRAIN BATCH DEBUG")
        print(f"raw model loss: {outputs.loss.item():.6f}")

        if "input_ids" in batch:
            print(f"input_ids shape: {tuple(batch['input_ids'].shape)}")

        if labels is not None:
            active = labels != -100
            print(f"labels shape:    {tuple(labels.shape)}")
            print(f"active labels:   {active.sum().item()}")
            print(f"active fraction: {active.float().mean().item():.6f}")

            if active.any() and "input_ids" in batch:
                same = (batch["input_ids"][active] == labels[active]).float().mean().item()
                print(f"input==label on active positions: {same:.6f}")

            predicted_tokens_per_sample = (labels != -100).sum(dim=-1).float()
            print(f"mean predicted tokens/sample: {predicted_tokens_per_sample.mean().item():.3f}")
            print(f"loss x mean_tokens/sample: {outputs.loss.item() * predicted_tokens_per_sample.mean().item():.6f}")

        print("=" * 100 + "\n")

    finally:
        if model_was_training:
            model.train()


# -------------------------
# Main
# -------------------------
def main():
    print(
        "RANK", os.environ.get("RANK"),
        "LOCAL_RANK", os.environ.get("LOCAL_RANK"),
        "WORLD_SIZE", os.environ.get("WORLD_SIZE"),
    )

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, CustomTrainingArguments, WandbArguments)
    )
    model_args, data_args, training_args, wandb_args = parser.parse_args_into_dataclasses()

    random.seed(training_args.seed)
    np.random.seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_args.seed)

    print(f"[Rank {training_args.local_rank}] MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
    print(f"[Rank {training_args.local_rank}] MASTER_PORT: {os.environ.get('MASTER_PORT')}")
    print(f"[Rank {training_args.local_rank}] NCCL_TIMEOUT: {os.environ.get('NCCL_TIMEOUT')}")

    rank = int(os.environ.get("RANK", 0))

    if not wandb_args.disable_wandb and rank == 0:
        wandb.init(
            project=wandb_args.wandb_project,
            name=training_args.run_name,
            entity=wandb_args.wandb_entity,
        )
    else:
        wandb.init(mode="disabled")

    tokenizer = PhyloTokenizerLoader(model_args.tokenizer_path)
    print_rank0(f"Using tokenizer from: {model_args.tokenizer_path}")
    print_rank0("Mask ID:", tokenizer.mask_token_id)
    print_rank0("Pad ID:", tokenizer.pad_token_id)
    print_rank0("Non-GAP ID:", tokenizer.convert_tokens_to_ids("-"))
    print_rank0("GAP ID:", tokenizer.convert_tokens_to_ids("[GAP]"))
    print_rank0("Tokenizer vocab size:", tokenizer.vocab_size)

    # -------------------------
    # Model
    # -------------------------
    if model_args.model_type == "ModernBERT":
        print_rank0("Using ModernBERT model...")
        model = ProteinBertModel(
            vocab_size=tokenizer.vocab_size,
            tokenizer=tokenizer,
            attn_implementation=model_args.attn_implementation,
        ).build()
        model.gradient_checkpointing_enable()
        model.to(training_args.local_rank)
        print_rank0(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print_rank0("Hidden size used:", model.config.hidden_size)

    elif model_args.model_type == "T5":
        print_rank0("Using T5 model...")
        model = ProteinT5Model(
            vocab_size=tokenizer.vocab_size,
            tokenizer=tokenizer,
        ).build()
        model.config.decoder_start_token_id = tokenizer.pad_token_id
        print_rank0("decoder_start_token_id =", model.config.decoder_start_token_id)
        model.gradient_checkpointing_enable()
        device = torch.device(f"cuda:{training_args.local_rank}" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print_rank0(f"Moving model to device: {device}")
        print_rank0(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print_rank0("Hidden size used:", model.config.d_model)

    elif model_args.model_type == "T5Gemma":
        print_rank0("Using T5Gemma model...")
        model = ProteinT5GemmaModel(
            vocab_size=tokenizer.vocab_size,
            tokenizer=tokenizer,
            attn_implementation=model_args.attn_implementation,
        ).build()
        model.config.decoder_start_token_id = tokenizer.pad_token_id
        print_rank0("decoder_start_token_id =", model.config.decoder_start_token_id)
        model.gradient_checkpointing_enable()
        device = torch.device(f"cuda:{training_args.local_rank}" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print_rank0(f"Moving model to device: {device}")
        print_rank0(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    elif model_args.model_type == "BART":
        print_rank0("Using BART model...")
        model = ProteinBARTModel(
            vocab_size=tokenizer.vocab_size,
            tokenizer=tokenizer,
        ).build()
        model.gradient_checkpointing_enable()
        device = torch.device(f"cuda:{training_args.local_rank}" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print_rank0(f"Moving model to device: {device}")
        print_rank0(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    else:
        raise ValueError(f"Unknown model_type: {model_args.model_type}")

    training_args.output_dir = f"{training_args.output_dir}/{training_args.run_name}"
    print_rank0(f"max_position_embeddings: {model_args.max_position_embeddings}")

    # -------------------------
    # Dataset
    # -------------------------
    if data_args.train_dataset_type == "tokenized_map":
        print_rank0("using pre-tokenized dataset")
        train_ds = load_from_disk(data_args.train_dataset_path)
        val_ds = load_from_disk(data_args.val_dataset_path)
        val_ds = val_ds.shuffle(seed=42)

    elif data_args.train_dataset_type == "uniref90_arrow_fasta":
        print_rank0("using Uniref90 Arrow and Index File dataset")
        train_ds = Uniref90ArrowDatasetForFASTA(
            dataset_path=data_args.train_dataset_path,
            training_type=training_args.training_type,
            fasta_path=data_args.fasta_path,
            idx_db_path=data_args.index_db_path,
        )
        val_ds = Uniref90ArrowEvalDatasetForFASTA(
            dataset_path=data_args.val_dataset_path,
            training_type=training_args.training_type,
            fasta_path=data_args.fasta_path,
            idx_db_path=data_args.index_db_path,
        )
        print_rank0("Validation dataset size:", len(val_ds))

    elif data_args.train_dataset_type == "uniref90_arrow_lmdb":
        print_rank0("using Uniref90 Arrow and LMDB dataset")
        train_ds = Uniref90ArrowDatasetForLMDB(
            dataset_path=data_args.train_dataset_path,
            training_type=training_args.training_type,
            lmdb_path=data_args.lmdb_path,
        )
        val_ds = Uniref90ArrowEvalDatasetForLMDB(
            dataset_path=data_args.val_dataset_path,
            training_type=training_args.training_type,
            lmdb_path=data_args.lmdb_path,
        )
        print_rank0("Validation dataset size:", len(val_ds))

    else:
        raise ValueError(f"Unknown train_dataset_type: {data_args.train_dataset_type}")

    # -------------------------
    # Collator
    # -------------------------
    if training_args.training_type == "MLM":
        print_rank0(f"Using MLM collator for training type: {training_args.training_type}")
        print_rank0(f"Using {training_args.mlm_probability} masking probability")
        data_collator = create_mlm_collator(
            tokenizer,
            max_seq_len=model_args.max_position_embeddings,
            mlm_probability=training_args.mlm_probability,
        )

    elif training_args.training_type in ["phylo_encoder_only", "phylo_encoder_decoder"]:
        print_rank0(f"Using Phylo collator for training type: {training_args.training_type}")
        data_collator = PhyloCollator(
            tokenizer=tokenizer,
            training_type=training_args.training_type,
            max_seq_len=model_args.max_position_embeddings,
        )

    else:
        raise ValueError(f"Unknown training_type: {training_args.training_type}")

    # -------------------------
    # Trainer
    # -------------------------
    if training_args.batch_sampler == "phylo_default":
        print_rank0(f"using phylo_default Trainer ({training_args.trainer_mode})")

        trainer_cls = PhyloTrainerOld if training_args.trainer_mode == "old" else PhyloTrainerNew

        trainer = trainer_cls(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

    elif training_args.batch_sampler != "default":
        print_rank0("using CustomBatchSizeTrainer")
        trainer = CustomBatchSizeTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
    else:
        print_rank0("using default Trainer")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

    if training_args.training_type == "phylo_encoder_only":
        trainer.add_callback(PercentIdentityLoggingCallback())

    # trainer.add_callback(
    #     ZeroShotVEPEvaluationCallback(
    #         tokenizer=tokenizer,
    #         input_csv=data_args.vep_input_csv,
    #         trainer=trainer,
    #         max_len=model_args.max_position_embeddings,
    #         batch_size=256,
    #         eval_every_n_steps=training_args.vep_eval_steps,
    #         training_type=training_args.training_type,
    #     )
    # )

    # -------------------------
    # Debug callbacks
    # -------------------------
    trainer.add_callback(LogValueDebugCallback())
    trainer.add_callback(
        LossDebugCallback(
            trainer=trainer,
            every_n_steps=training_args.logging_steps,
            max_batches=5,
        )
    )
    trainer.add_callback(
        TrainerLossPathDebugCallback(
            trainer=trainer,
            every_n_steps=training_args.logging_steps,
            max_batches=5,
        )
    )

    # One-time debug before training
    one_time_batch_debug(trainer, model, rank)

    trainer.train()


if __name__ == "__main__":
    main()