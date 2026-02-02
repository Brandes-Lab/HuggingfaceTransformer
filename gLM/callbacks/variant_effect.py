# type: ignore
import time
import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.metrics import roc_auc_score
from transformers import TrainerCallback
from torch.distributed import is_initialized, get_rank, barrier, all_gather_object


class ZeroShotVEPEvaluationCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        input_csv,
        trainer,
        max_len=512,
        eval_every_n_steps=20000,
        batch_size=8,
        training_type="phylo_encoder_decoder",
    ):
        self.tokenizer = tokenizer
        self.input_csv = input_csv
        self.max_len = max_len
        self.eval_every_n_steps = eval_every_n_steps
        self.trainer = trainer
        self.batch_size = batch_size
        self.start_time = time.time()
        self.training_type = training_type

        self.df = pd.read_csv(
            input_csv,
            usecols=["sequence", "pos", "ref", "alt", "label"],
            dtype={"pos": np.int32, "label": np.int8},
        )

    def compute_log_odds_batch(self, model, seqs, poses, refs, alts):
        if self.training_type == "MLM":
            return self.compute_log_odds_MLM(model, seqs, poses, refs, alts)
        elif self.training_type == "phylo_encoder_only":
            return self.compute_log_odds_phylo_encoder_only(model, seqs, poses, refs, alts)
        elif self.training_type == "phylo_encoder_decoder":
            return self.compute_log_odds_encoder_decoder(model, seqs, poses, refs, alts)
        else:
            raise ValueError(f"Unknown training type: {self.training_type}")

    def compute_log_odds_MLM(self, model, seqs, poses, refs, alts):
        """
        Computes zero-shot variant effect scores for encoder-only masked language models (e.g., BERT, ESM-1b).
        For each variant:
            • Mask the position of interest in the sequence
            • Run the model to predict the masked token
            • Compare log-probabilities of alt vs ref at that position

            log_odds = log P(alt | masked_seq) - log P(ref | masked_seq)
        """

        # Step 1: Filter valid examples from the batch
        valid_data = []
        for i, (seq, pos, ref, alt) in enumerate(zip(seqs, poses, refs, alts)):
            if len(seq) <= self.max_len and pos < len(seq) and seq[pos] == ref:
                masked_seq = seq[:pos] + self.tokenizer.mask_token + seq[pos + 1:]
                valid_data.append((i, masked_seq, ref, alt))

        if not valid_data:
            return [None] * len(seqs)

        # Unpack batch
        indices, masked_seqs, valid_refs, valid_alts = zip(*valid_data)

        # Step 2: Tokenize batch of masked sequences
        inputs = self.tokenizer(
            list(masked_seqs),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_len,
        )
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Step 3: Forward pass
        with torch.no_grad():
            logits = model(**inputs).logits

        # Step 4: Find masked position in each sequence
        mask_token_id = self.tokenizer.mask_token_id
        mask_indices = (inputs["input_ids"] == mask_token_id).nonzero(as_tuple=False)

        # Step 5 — Compute log-odds for ref vs alt
        results = [None] * len(seqs)
        for (batch_idx, token_idx), input_idx in zip(mask_indices, indices):
            ref_token = valid_refs[batch_idx]
            alt_token = valid_alts[batch_idx]

            ref_id = self.tokenizer.convert_tokens_to_ids(ref_token)
            alt_id = self.tokenizer.convert_tokens_to_ids(alt_token)
            if ref_id is None or alt_id is None:
                continue

            prob = torch.nn.functional.softmax(logits[batch_idx, token_idx], dim=0)
            log_odds = (torch.log(prob[alt_id]) - torch.log(prob[ref_id])).item()
            results[input_idx] = log_odds

        return results

    
    def compute_log_odds_phylo_encoder_only(self, model, seqs, poses, refs, alts):
        """
        Computes zero-shot variant effect scores for phylo-style encoder-only models (e.g., ModernBERT trained on aligned sequence pairs).

        For each sequence:
            • Encode the full (unaltered) reference sequence
            • Extract the model logits at the variant position
            • Convert to probabilities and compute:

                log_odds = log P(alt | seq) - log P(ref | seq)
        """

        results = [None] * len(seqs)
        device = next(model.parameters()).device

        # Step 1 — Tokenize unmodified reference sequences (no masking)
        # Prepare batch 
        inputs = self.tokenizer(
            list(seqs),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_len,
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Step 2 — Forward pass
        with torch.no_grad():
            logits = model(**inputs).logits        # shape: (B, L, Vocab_size) 

        # Step 3 — Compute log-odds for each valid variant
        for batch_idx, (seq, pos, ref, alt) in enumerate(zip(seqs, poses, refs, alts)):
            if len(seq) <= self.max_len and pos < len(seq) and seq[pos] == ref:
                ref_id = self.tokenizer.convert_tokens_to_ids(ref)
                alt_id = self.tokenizer.convert_tokens_to_ids(alt)
                if ref_id is None or alt_id is None:
                    continue
                
                # Extract probability distribution at position
                prob = torch.nn.functional.softmax(logits[batch_idx, pos], dim=0)
                
                # Compute log odds of alt vs ref
                log_odds = (torch.log(prob[alt_id]) - torch.log(prob[ref_id])).item()
                results[batch_idx] = log_odds
        
        return results
    
    def compute_log_odds_encoder_decoder(self, model, seqs, poses, refs, alts):
        """
        Computes zero-shot variant effect scores for encoder-decoder models (e.g., T5).
        For each variant:
            log_odds = log P(alt | reference_sequence) - log P(ref | reference_sequence)

        The encoder always receives the *reference sequence*.
        The decoder is prompted with a single token (ref or alt), and we compare
        the probabilities assigned by the model.

        • One encoder forward pass per batch
        • Two decoder passes (alt + ref)
        """
        results = [None] * len(seqs)
        device = next(model.parameters()).device

        # Step 1: Filter valid examples from the batch
        valid_data = []
        for i, (seq, pos, ref, alt) in enumerate(zip(seqs, poses, refs, alts)):
            if len(seq) <= self.max_len and pos < len(seq) and seq[pos] == ref:

                ref_id = self.tokenizer.convert_tokens_to_ids(ref)
                alt_id = self.tokenizer.convert_tokens_to_ids(alt)

                if ref_id is None or alt_id is None:
                    continue

                valid_data.append((i, seq, pos, ref_id, alt_id))

        
        if not valid_data:
            return results # all None 

        indices, valid_seqs, valid_poses, ref_ids, alt_ids = zip(*valid_data)

        # Step 2: Tokenize encoder input batch 
        enc_inputs = self.tokenizer(
            list(valid_seqs),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=False,
        ).to(device)

        ref_ids_tensor = torch.tensor(ref_ids, dtype=torch.long, device=device)
        alt_ids_tensor = torch.tensor(alt_ids, dtype=torch.long, device=device)

        decoder_input_refs = ref_ids_tensor.unsqueeze(1)   # shape (B, 1)
        decoder_input_alts = alt_ids_tensor.unsqueeze(1) # shape (B, 1)

        with torch.no_grad():
            # Step 3: Forward pass for alt 
            out_alt = model(**enc_inputs, decoder_input_ids=decoder_input_alts)
            logprobs_alt = torch.softmax(out_alt.logits[:, 0, :], dim=1) # shape (B, Vocab_size)
            logp_alt = logprobs_alt.gather(1, alt_ids_tensor.unsqueeze(1)).squeeze(1) # shape (B,)

            # Step 4: Forward pass for ref
            out_ref = model(**enc_inputs, decoder_input_ids=decoder_input_refs)
            logprobs_ref = torch.softmax(out_ref.logits[:, 0, :], dim=1) # shape (B, Vocab_size)
            logp_ref = logprobs_ref.gather(1, ref_ids_tensor.unsqueeze(1)).squeeze(1) # shape (B,)

            log_odds = (logp_alt - logp_ref).tolist()

        
        # Step 5: Assing log odds to correct position 
        for i, log_odd_value in zip(indices, log_odds):
            results[i] = log_odd_value
        
        return results

        
    
    def run_vep_eval(self, model, step_id):
        rank = get_rank() if is_initialized() else 0
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        print(f"[Rank {rank}] Starting zero-shot VEP eval @ step {step_id}", flush=True)

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
        start_time = time.time()

        with torch.no_grad():
            for i in range(0, len(shard_indices), self.batch_size):
                batch_ids = shard_indices[i : i + self.batch_size]
                batch_seqs = seqs[batch_ids]
                batch_poses = poses[batch_ids]
                batch_refs = refs[batch_ids]
                batch_alts = alts[batch_ids]

                batch_scores = self.compute_log_odds_batch(
                    model, batch_seqs, batch_poses, batch_refs, batch_alts
                )

                for j, score in enumerate(batch_scores):
                    if score is not None:
                        preds_shard[i + j] = -float(score) # negate the score, higher = more pathogenic, lower = benign

                if i % 20000 == 0:
                    print(f"[Rank {rank}] Progress: {i}/{len(shard_indices)}", flush=True)

        if was_training:
            model.train()

        # All-gather combined structure
        gathered_data = [None for _ in range(world_size)]
        local_data = list(
            zip(shard_indices.tolist(), preds_shard.tolist(), labels[shard_indices].tolist())
        )
        all_gather_object(gathered_data, local_data)

        if rank == 0:
            flat_preds = np.full(n, np.nan, dtype=np.float32)
            for data in gathered_data:
                for idx, pred, _ in data:
                    flat_preds[idx] = pred

            mask = ~np.isnan(flat_preds)
            if mask.sum() >= 10 and (labels[mask].min() != labels[mask].max()):
                auc = roc_auc_score(labels[mask], flat_preds[mask])
                print(f"AUC at step {step_id}: {auc:.4f}", flush=True)
                wandb.log(
                    {
                        "zero_shot_vep_auc": auc,
                        "step": step_id,
                        "elapsed_hours": (time.time() - self.start_time) / 3600,
                    }
                )
            else:
                print(f"Skipping AUC at step {step_id} due to insufficient data", flush=True)

            print(f"[TIMER] VEP eval took {time.time() - start_time:.2f} seconds", flush=True)

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

