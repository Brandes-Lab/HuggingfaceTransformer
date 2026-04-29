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
import time
from gLM.callbacks import (
    ZeroShotVEPEvaluationCallback,
    PercentIdentityLoggingCallback
)
from gLM.data_utils import TruncatingDataCollatorForMLM
from gLM.models import ProteinBertModel
from gLM.models import ProteinT5Model
from gLM.models import ProteinBARTModel
from gLM.models import ProteinT5GemmaModel
from gLM.models.protein_modernbert_phylo import ProteinModernBertPrefixLM, run_sanity_checks
from gLM.attention_mask import run_encoder_flash, prefixlm_forward_flash
from gLM.tokenizers import TokenizerLoader, PhyloTokenizerLoader
from gLM.train_utils import CustomBatchSizeTrainer
from gLM.collator import create_mlm_collator, PhyloCollator
from gLM.collator.prefixlm_collator import PrefixLMCollator
from gLM.dataset import Uniref90ArrowDatasetForFASTA, Uniref90ArrowEvalDatasetForFASTA, Uniref90ArrowDatasetForLMDB, Uniref90ArrowEvalDatasetForLMDB
from gLM.train_utils import PhyloTrainer
from gLM.train_utils.prefixlm_trainer import PrefixLMTrainer
from typing import Literal, Optional

# Define device globally
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


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    model_type: Literal["ModernBERT", "T5", "BART", "T5Gemma"] = field(
        default="ModernBERT", metadata={"help": "Type of model to use"}
    )

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
        default=-1, metadata={"help": "Maximum number of training steps"}
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
    dataloader_persistent_workers: bool = field(
        default=True,
        metadata={"help": "Number of dataloader_persistent_workers"}
    )
    dataloader_prefetch_factor: Optional[int] = field(
        default=None,
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
    training_type: Literal["MLM", "phylo_encoder_only", "phylo_encoder_decoder", "prefixlm_modernbert"] = field(
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
    base_batch_size: int = field(default=8, metadata={"help": "Base batch size for dynamic batching"})

@dataclass
class DataArguments:
    """Arguments for data configuration."""
    train_dataset_type: Literal["tokenized_map", "uniref90_arrow_fasta", "uniref90_arrow_lmdb"] = field(
        default="iterable", metadata={"help": "Type of training dataset"}
    )
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
    lmdb_path: str = field(
        default="/gpfs/data/brandeslab/Data/uniref/uniref100_bk.lmdb",
        metadata={"help": "Path to the LMDB file for FASTA sequences"},
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
    print("RANK", os.environ.get("RANK"),
      "LOCAL_RANK", os.environ.get("LOCAL_RANK"),
      "WORLD_SIZE", os.environ.get("WORLD_SIZE"))

    # Parse arguments
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, CustomTrainingArguments, WandbArguments)
    )

    model_args, data_args, training_args, wandb_args = (
        parser.parse_args_into_dataclasses()
    )

    if training_args.dataloader_num_workers == 0:
        training_args.dataloader_prefetch_factor = None
        training_args.dataloader_persistent_workers = False

    print(
        f"[Rank {training_args.local_rank}] MASTER_ADDR: {os.environ.get('MASTER_ADDR')}"
    )
    print(
        f"[Rank {training_args.local_rank}] MASTER_PORT: {os.environ.get('MASTER_PORT')}"
    )
    print(
        f"[Rank {training_args.local_rank}] NCCL_TIMEOUT: {os.environ.get('NCCL_TIMEOUT')}"
    )

    rank = int(os.environ.get("RANK", 0))
    if not wandb_args.disable_wandb and rank == 0:
        wandb.init(
            project=wandb_args.wandb_project,
            name=training_args.run_name,
            entity=wandb_args.wandb_entity,
        )
    else:
        wandb.init(mode="disabled")

    # Load tokenizer
    tokenizer = PhyloTokenizerLoader(model_args.tokenizer_path)
    print_rank0(f"Using tokenizer from: {model_args.tokenizer_path}")
    pad_id = tokenizer.pad_token_id
    mask_id = tokenizer.mask_token_id
    non_gap_id = tokenizer.convert_tokens_to_ids("-")
    gap_id = tokenizer.convert_tokens_to_ids("[GAP]")
    print_rank0("Phylo Tokenizer loaded. GAP ID:", gap_id)
    print_rank0("Mask ID:", mask_id)
    print_rank0("Non-GAP ID:", non_gap_id)
    print_rank0("GAP ID:", gap_id)
    print_rank0("Tokenizer vocab size:", tokenizer.vocab_size)

    # =========================================================================
    # Build model
    # =========================================================================
    if model_args.model_type == "ModernBERT":

        if training_args.training_type == "prefixlm_modernbert":
            print_rank0(f"Using ModernBERT model with PrefixLM attention ({model_args.attn_implementation})...")
            model = ProteinModernBertPrefixLM(
                vocab_size=tokenizer.vocab_size,
                tokenizer=tokenizer
            ).build()

        else:
            print_rank0("Using ModernBERT model...")
            model = ProteinBertModel(
                vocab_size=tokenizer.vocab_size,
                tokenizer=tokenizer,
                attn_implementation=model_args.attn_implementation
            ).build()

        # CHANGE 1: gradient_checkpointing_enable() before .to(DEVICE).
        # gradient checkpointing installs hooks on the model graph; doing it
        # after .to() works too but before is the conventional safe order.
        model.gradient_checkpointing_enable()

        # CHANGE 2: do NOT cast to bfloat16 here.
        # training_args.bf16=True makes the HF Trainer wrap every forward pass
        # in torch.autocast(dtype=bfloat16) automatically. Parameters stay in
        # fp32 (standard mixed-precision training). Casting the model to bf16
        # here would make parameters bf16 too, which loses the fp32 master
        # copy and destabilises training.
        # The "Flash Attention 2 without specifying a torch dtype" warning is
        # silenced by setting config.torch_dtype inside ProteinModernBertPrefixLM.build()
        # (see that file) — no change needed here.
        model.to(DEVICE)

        print_rank0(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print_rank0("Hidden size used:", model.config.hidden_size)
        print_rank0(f"max_position_embeddings: {model.config.max_position_embeddings}")

    elif model_args.model_type == "T5":
        print_rank0("Using T5 model...")
        model = ProteinT5Model(
            vocab_size=tokenizer.vocab_size,
            tokenizer=tokenizer
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
            attn_implementation=model_args.attn_implementation
        ).build()
        model.config.decoder_start_token_id = tokenizer.pad_token_id
        print_rank0("decoder_start_token_id =", model.config.decoder_start_token_id)
        model.gradient_checkpointing_enable()
        device = torch.device(f"cuda:{training_args.local_rank}" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print_rank0(f"Moving model to device: {device}")
        print_rank0(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # CHANGE 3: moved max_position_embeddings log into the ModernBERT block
    # above (printed from model.config, not model_args, so it reflects what
    # was actually built). Removed the stale print here.
    training_args.output_dir = f"{training_args.output_dir}/{training_args.run_name}"

    # =========================================================================
    # Load dataset
    # =========================================================================
    if data_args.train_dataset_type == "tokenized_map":
        print_rank0(f"using pre-tokenized dataset")
        train_ds = load_from_disk(data_args.train_dataset_path)
        val_ds = load_from_disk(data_args.val_dataset_path)
        val_ds = val_ds.shuffle(seed=42)

    elif data_args.train_dataset_type == "uniref90_arrow_fasta":
        print_rank0(f"using Uniref90 Arrow and Index File dataset")

        ds_training_type = (
            "phylo_encoder_decoder"
            if training_args.training_type == "prefixlm_modernbert"
            else training_args.training_type
        )

        train_ds = Uniref90ArrowDatasetForFASTA(
            dataset_path=data_args.train_dataset_path,
            training_type=ds_training_type,
            fasta_path=data_args.fasta_path,
            idx_db_path=data_args.index_db_path
        )
        val_ds = Uniref90ArrowEvalDatasetForFASTA(
            dataset_path=data_args.val_dataset_path,
            training_type=ds_training_type,
            fasta_path=data_args.fasta_path,
            idx_db_path=data_args.index_db_path
        )
        print_rank0("Validation dataset size:", len(val_ds))

    elif data_args.train_dataset_type == "uniref90_arrow_lmdb":
        print_rank0(f"using Uniref90 Arrow and LMDB dataset")

        ds_training_type = (
            "phylo_encoder_decoder"
            if training_args.training_type == "prefixlm_modernbert"
            else training_args.training_type
        )

        train_ds = Uniref90ArrowDatasetForLMDB(
            dataset_path=data_args.train_dataset_path,
            training_type=ds_training_type,
            lmdb_path=data_args.lmdb_path
        )
        val_ds = Uniref90ArrowEvalDatasetForLMDB(
            dataset_path=data_args.val_dataset_path,
            training_type=ds_training_type,
            lmdb_path=data_args.lmdb_path
        )
        print_rank0("Validation dataset size:", len(val_ds))

    # =========================================================================
    # Create collator
    # =========================================================================
    if training_args.training_type == "MLM":
        print_rank0(f"Using MLM collator for training type: {training_args.training_type}")
        print_rank0(f"Using {training_args.mlm_probability} masking probability")
        data_collator = create_mlm_collator(
            tokenizer,
            max_seq_len=model_args.max_position_embeddings,
            mlm_probability=training_args.mlm_probability
        )

    elif training_args.training_type in ["phylo_encoder_only", "phylo_encoder_decoder"]:
        print_rank0(f"Using Phylo collator for training type: {training_args.training_type}")
        data_collator = PhyloCollator(
            tokenizer=tokenizer,
            training_type=training_args.training_type,
            max_seq_len=model_args.max_position_embeddings
        )

    elif training_args.training_type == "prefixlm_modernbert":
        print_rank0(f"Using PrefixLM collator for training type: {training_args.training_type}")
        print_rank0(f"Packing format: [CLS] seq2 [SEP] seq1 [SEP]")
        data_collator = PrefixLMCollator(
            tokenizer=tokenizer,
            max_seq_len=model_args.max_position_embeddings,
            has_pid=False,
        )

    # =========================================================================
    # Create trainer
    # =========================================================================
    if training_args.training_type == "prefixlm_modernbert":
        print_rank0("Using PrefixLMTrainer (custom compute_loss with layer-by-layer encoder bypass)")
        trainer = PrefixLMTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        # CHANGE 4: sanity checks only on rank 0, followed by a barrier so
        # all other ranks wait before proceeding to trainer.train().
        # Without the barrier, rank 0 is busy with sanity checks while other
        # ranks have already entered trainer.train() and are blocked on the
        # DDP collective init — this causes an NCCL timeout/hang.
        if rank == 0:
            print_rank0("\n--- PrefixLM sanity checks ---")
            batch_raw = [train_ds[i] for i in range(4)]
            batch_out = data_collator(batch_raw)
            model.eval()
            loss, logits = prefixlm_forward_flash(model, batch_out, DEVICE)
            print_rank0(f"Sanity forward pass — loss: {loss.item():.4f}")
            model.zero_grad()
            run_sanity_checks(model, batch_out, DEVICE)
            model.train()

        # All ranks synchronise here before training starts.
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    elif training_args.batch_sampler == "phylo_default":
        print_rank0("using phylo_default Trainer")
        trainer = PhyloTrainer(
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
            data_collator=data_collator
        )

    if training_args.training_type == "prefixlm_modernbert":
        trainer.add_callback(
            ZeroShotVEPEvaluationCallback(
                tokenizer=tokenizer,
                input_csv=data_args.vep_input_csv,
                trainer=trainer,
                max_len=model_args.max_position_embeddings,
                batch_size=training_args.per_device_eval_batch_size,
                eval_every_n_steps=training_args.vep_eval_steps,
                training_type="prefixlm_modernbert",
            )
        )

    trainer.train()


if __name__ == "__main__":
    main()