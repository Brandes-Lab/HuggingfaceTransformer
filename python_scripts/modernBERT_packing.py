import os, time, argparse, torch, wandb, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_from_disk
from sklearn.metrics import roc_auc_score
from transformers import (
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    ModernBertForMaskedLM,
    ModernBertConfig,
)
from torch.distributed import is_initialized, get_rank, barrier, all_gather_object


class TokenizerLoader:
    def __init__(self, tokenizer_path):
        self.tokenizer_path = tokenizer_path

    def load(self):
        return PreTrainedTokenizerFast.from_pretrained(self.tokenizer_path)


class ProteinBertModel:
    def __init__(self, vocab_size, tokenizer):
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer

    def build(self):
        config = ModernBertConfig(
            vocab_size=self.vocab_size,
            max_position_embeddings=8192,
            num_hidden_layers=24,
            num_attention_heads=20,
            hidden_size=1600,
            intermediate_size=6656,
            type_vocab_size=1,
            hidden_activation="gelu",
            global_attn_every_n_layers=3,
            local_attention=512,
            deterministic_flash_attn=False,
            global_rope_theta=160000.0,
            local_rope_theta=10000.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            cls_token_id=self.tokenizer.cls_token_id,
            sep_token_id=self.tokenizer.sep_token_id,
        )
        return ModernBertForMaskedLM(config)


def get_attention_mask_for_packed_sequence(x, token_id, eos=True):
    """
    Creates attention mask that prevents attention across packed sequence boundaries.
    Based on: https://huggingface.co/blog/sirluk/llm-sequence-packing
    
    Args:
        x: Input tensor of shape (B, T) containing token IDs
        token_id: ID of the separator token (e.g., SEP or EOS)
        eos: Whether the token_id marks end of sequence
    """
    B, T = x.shape
    eos_idx = (x.view(-1) == token_id).nonzero(as_tuple=True)[0] + eos
    eos_idx_expanded = torch.cat([eos_idx, torch.arange(0, B*T+1, T)]).unique().sort()[0]
    normalized_idx = eos_idx_expanded - (eos_idx_expanded // T) * T
    normalized_idx = torch.where(normalized_idx == 0, T, normalized_idx)
    reps = normalized_idx[1:] - normalized_idx[:-1]
    reps = torch.where(reps < 1, normalized_idx[1:], reps)
    repeated_idx = torch.repeat_interleave(normalized_idx[1:], reps).view(B, 1, T).expand(-1, T, -1)
    mask_indices = torch.arange(T).view(1, -1, 1).expand(B, -1, T)
    mask = torch.ones(T, T, dtype=torch.bool).tril().expand(B, -1, -1)
    mask = mask.masked_fill(mask_indices >= repeated_idx, False)
    return mask


def get_position_ids_for_packed_sequence(x, token_id, eos=True):
    """
    Creates position IDs that reset at each sequence boundary.
    
    Args:
        x: Input tensor of shape (B, T) containing token IDs
        token_id: ID of the separator token
        eos: Whether the token_id marks end of sequence
    """
    B, T = x.shape
    eos_idx = (x.view(-1) == token_id).nonzero(as_tuple=True)[0] + eos
    eos_idx_expanded = torch.cat([eos_idx, torch.arange(0, B*T+1, T)]).unique().sort()[0]
    normalized_idx = eos_idx_expanded - (eos_idx_expanded // T) * T
    normalized_idx = torch.where(normalized_idx == 0, T, normalized_idx)
    reps = normalized_idx[1:] - normalized_idx[:-1]
    reps = torch.where(reps < 1, normalized_idx[1:], reps)
    pos_ids = (torch.arange(B*T) - torch.repeat_interleave(eos_idx_expanded[:-1], reps)).view(B, T)
    return pos_ids


class DataCollatorForPackedMLM:
    """
    Data collator that packs multiple sequences together and applies MLM masking.
    Combines sequence packing with masked language modeling for efficient training.
    """
    def __init__(self, tokenizer, mlm_probability=0.15, max_length=8192, separator_id=None):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.max_length = max_length
        # Use SEP token as separator, fallback to EOS if not available
        self.separator_id = separator_id if separator_id is not None else (
            tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
        )
        self.pad_token_id = tokenizer.pad_token_id
        
    def pack_sequences(self, examples):
        """Pack multiple sequences into one until max_length is reached."""
        packed_batches = []
        current_pack = []
        current_length = 0
        
        for ex in examples:
            input_ids = ex['input_ids'] if isinstance(ex['input_ids'], list) else ex['input_ids'].tolist()
            
            # Add separator token
            seq_with_sep = input_ids + [self.separator_id]
            seq_len = len(seq_with_sep)
            
            # If adding this sequence exceeds max_length, start a new pack
            if current_length + seq_len > self.max_length:
                if current_pack:
                    packed_batches.append(current_pack)
                current_pack = seq_with_sep
                current_length = seq_len
            else:
                current_pack.extend(seq_with_sep)
                current_length += seq_len
        
        # Don't forget the last pack
        if current_pack:
            packed_batches.append(current_pack)
        
        return packed_batches
    
    def mask_tokens(self, inputs):
        """Apply MLM masking to input tokens."""
        labels = inputs.clone()
        
        # Create probability matrix for masking
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        
        # Don't mask special tokens (pad, sep, cls, etc.)
        special_tokens_mask = torch.zeros_like(labels, dtype=torch.bool)
        special_tokens_mask |= (labels == self.pad_token_id)
        special_tokens_mask |= (labels == self.separator_id)
        if self.tokenizer.cls_token_id is not None:
            special_tokens_mask |= (labels == self.tokenizer.cls_token_id)
        if self.tokenizer.bos_token_id is not None:
            special_tokens_mask |= (labels == self.tokenizer.bos_token_id)
        
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens
        
        # 80% of the time, replace with [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.mask_token_id
        
        # 10% of the time, replace with random token
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        
        # 10% of the time, keep original token (remaining masked_indices)
        
        return inputs, labels
    
    def __call__(self, examples):
        # Pack sequences
        packed_sequences = self.pack_sequences(examples)
        
        # Pad packed sequences to same length in batch
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(seq, dtype=torch.long) for seq in packed_sequences],
            batch_first=True,
            padding_value=self.pad_token_id
        )
        
        # Create attention masks for packed sequences
        attention_mask = get_attention_mask_for_packed_sequence(
            input_ids, 
            self.separator_id,
            eos=True
        )
        
        # Create position IDs that reset at sequence boundaries
        position_ids = get_position_ids_for_packed_sequence(
            input_ids,
            self.separator_id,
            eos=True
        )
        
        # Apply MLM masking
        input_ids, labels = self.mask_tokens(input_ids)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'labels': labels
        }


class ZeroShotVEPEvaluationCallback(TrainerCallback):
    def __init__(self, tokenizer, input_csv, trainer, max_len=8192, eval_every_n_steps=50000, batch_size=8):
        self.tokenizer = tokenizer
        self.input_csv = input_csv
        self.max_len = max_len
        self.eval_every_n_steps = eval_every_n_steps
        self.trainer = trainer
        self.batch_size = batch_size
        self.start_time = time.time()

        self.df = pd.read_csv(
            input_csv,
            usecols=["sequence", "pos", "ref", "alt", "label"],
            dtype={"pos": np.int32, "label": np.int8},
        )

    def compute_log_odds_batch(self, model, seqs, poses, refs, alts):
        masked_seqs = []
        valid_indices = []

        for i, (seq, pos, ref, alt) in enumerate(zip(seqs, poses, refs, alts)):
            if len(seq) > self.max_len:
                continue 
            if pos >= len(seq) or seq[pos] != ref:
                continue
            
            masked_seq = list(seq)
            masked_seq[pos] = self.tokenizer.mask_token
            masked_seqs.append("".join(masked_seq))
            valid_indices.append(i)

        if not masked_seqs:
            return [None] * len(seqs)
        
        inputs = self.tokenizer(
            masked_seqs, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=self.max_len,
        )

        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
        
        mask_token_id = self.tokenizer.mask_token_id
        mask_indices = (inputs["input_ids"] == mask_token_id).nonzero(as_tuple=False)

        results = [None] * len(seqs)
        for idx_in_batch, (batch_idx, token_idx) in enumerate(mask_indices):
            input_idx = valid_indices[batch_idx]
            ref_id = self.tokenizer.convert_tokens_to_ids(refs[input_idx])
            alt_id = self.tokenizer.convert_tokens_to_ids(alts[input_idx])
            if ref_id is None or alt_id is None:
                continue 
                
            prob = torch.nn.functional.softmax(logits[batch_idx, token_idx], dim=0)
            log_odds = (torch.log(prob[alt_id]) - torch.log(prob[ref_id])).item()
            results[input_idx] = log_odds
        
        return results

    def run_vep_eval(self, model, step_id):
        rank = get_rank() if is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        print(f"[Rank {rank}] World Size: {world_size}", flush=True)

        elapsed_hours = (time.time() - self.start_time) / 3600
        start_time = time.time()

        print(f"[Rank {rank}] Running zero-shot VEP evaluation at step {step_id}", flush=True)

        seqs = self.df["sequence"].values
        poses = self.df["pos"].values
        refs = self.df["ref"].values
        alts = self.df["alt"].values
        labels = self.df["label"].to_numpy(dtype=np.int8)

        n = len(labels)
        indices = np.arange(n)
        shard_indices = indices[rank::world_size]
        preds_shard = np.full(len(shard_indices), np.nan, dtype=np.float32)

        was_training = model.training
        model.eval()
        try:
            for i in range(0, len(shard_indices), self.batch_size):
                batch_ids = shard_indices[i:i+self.batch_size]
                batch_seqs = seqs[batch_ids]
                batch_poses = poses[batch_ids]
                batch_refs = refs[batch_ids]
                batch_alts = alts[batch_ids]

                batch_scores = self.compute_log_odds_batch(
                    model, batch_seqs, batch_poses, batch_refs, batch_alts
                )

                for j, score in enumerate(batch_scores):
                    if score is not None:
                        preds_shard[i + j] = -float(score)

                if (i + self.batch_size) % 5000 < self.batch_size:
                    print(f"[Rank {rank}] Evaluation progress: {i + self.batch_size}/{len(shard_indices)}", flush=True)
        finally:
            if was_training:
                model.train()

        # DDP all-gather logic
        gathered_preds = [None for _ in range(world_size)]
        gathered_labels = [None for _ in range(world_size)]
        gathered_indices = [None for _ in range(world_size)]

        all_gather_object(gathered_preds, preds_shard.tolist())
        all_gather_object(gathered_labels, labels[shard_indices].tolist())
        all_gather_object(gathered_indices, shard_indices.tolist())

        if rank == 0:
            flat_preds = np.full(n, np.nan, dtype=np.float32)
            for preds, idxs in zip(gathered_preds, gathered_indices):
                flat_preds[np.array(idxs)] = np.array(preds)

            mask = ~np.isnan(flat_preds)
            if mask.sum() >= 10 and (labels[mask].min() != labels[mask].max()):
                auc = roc_auc_score(labels[mask], flat_preds[mask])
                print(f"AUC at step {step_id}: {auc:.4f}", flush=True)
                wandb.log({"zero_shot_vep_auc": auc, "step": step_id, "elapsed_hours": elapsed_hours})
            else:
                print(f"Skipping AUC at step {step_id} due to insufficient data", flush=True)

            elapsed = time.time() - start_time
            print(f"[TIMER] Zero-shot VEP took {elapsed:.2f} seconds", flush=True)

        if is_initialized():
            barrier()

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0:
            self.run_vep_eval(model, step_id=state.global_step)
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_every_n_steps == 0 and state.global_step > 0:
            self.run_vep_eval(model, step_id=state.global_step)
        return control


class ElapsedTimeLoggerCallback(TrainerCallback):
    def __init__(self):
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        elapsed_hours = (time.time() - self.start_time) / 3600
        print(f"[Step {state.global_step}] Elapsed time: {elapsed_hours:.2f} hours")
        if logs is not None:
            logs["elapsed_hours"] = elapsed_hours
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                wandb.log(logs, step=state.global_step)


def print_rank0(*args, **kwargs):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(*args, **kwargs)


def main():
    rank = int(os.environ.get("LOCAL_RANK", 0))

    print(f"[Rank {rank}] MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
    print(f"[Rank {rank}] MASTER_PORT: {os.environ.get('MASTER_PORT')}")
    print(f"[Rank {rank}] NCCL_TIMEOUT: {os.environ.get('NCCL_TIMEOUT')}")

    tokenizer = TokenizerLoader("char_tokenizer").load()
    pad_id = tokenizer.pad_token_id
    print_rank0("Tokenizer vocab size:", tokenizer.vocab_size)

    model = ProteinBertModel(tokenizer.vocab_size, tokenizer).build()
    model.gradient_checkpointing_enable()
    model = model.to(rank)

    param_count = sum(p.numel() for p in model.parameters())
    rounded_param_count = round(param_count / 1e6)
    run_name = f"modernBERT_{rounded_param_count}M_packed"

    print_rank0(f"[Rank {rank}] Model parameters: {param_count:,}", flush=True)

    if rank == 0:
        wandb.init(project="modernBERT_training", name=run_name, entity="sinha-anushka12-na")
    else:
        wandb.init(mode="disabled")

    checkpoint_root = f"/gpfs/data/brandeslab/fake_model_checkpts/{run_name}"

    train_ds = load_from_disk("/gpfs/data/brandeslab/Data/processed_datasets/uniref90_tokenized_8192/train_only/train_representative")
    val_ds = load_from_disk("/gpfs/data/brandeslab/Data/processed_datasets/uniref90_tokenized_8192/val_only/validation")

    # Initialize data collator with sequence packing
    data_collator = DataCollatorForPackedMLM(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        max_length=8192,
        separator_id=tokenizer.sep_token_id,  # Use SEP token as boundary marker
    )

    # Test the data collator
    print_rank0("\n=== Testing Data Collator ===")
    sample_examples = [train_ds[i] for i in range(4)]
    batch = data_collator(sample_examples)
    
    print_rank0(f"Packed batch shape: {batch['input_ids'].shape}")
    print_rank0(f"Position IDs shape: {batch['position_ids'].shape}")
    print_rank0(f"Attention mask shape: {batch['attention_mask'].shape}")
    print_rank0(f"Sample tokens (first 100): {tokenizer.decode(batch['input_ids'][0][:100])}")
    print_rank0("=== Data Collator Test Complete ===\n")

    training_args = TrainingArguments(
        output_dir=checkpoint_root,
        max_steps=2_000_000,

        per_device_train_batch_size=512,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=4,

        bf16=True,
        fp16=False,

        dataloader_num_workers=16,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=2,

        eval_strategy="steps",
        eval_steps=50000,

        logging_strategy="steps",
        logging_steps=100,

        save_strategy="steps",
        save_steps=50000,

        report_to="wandb",
        run_name=run_name,
        learning_rate=3e-4,

        remove_unused_columns=False,  # Important for custom collator
        group_by_length=False,  # Disable since we're doing custom packing

        local_rank=rank,
        ddp_backend="nccl",
        ddp_timeout=3000,
        ddp_find_unused_parameters=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Uncomment to enable zero-shot VEP evaluation
    # trainer.add_callback(ZeroShotVEPEvaluationCallback(
    #     tokenizer=tokenizer,
    #     input_csv="/gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv",
    #     trainer=trainer,
    #     eval_every_n_steps=50000
    # ))

    trainer.add_callback(ElapsedTimeLoggerCallback())

    trainer.train()

if __name__ == "__main__":
    main()