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
        max_len=4096,
        eval_every_n_steps=20000,
        batch_size=2,
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
        results = [None] * len(seqs)
        device = next(model.parameters()).device
        valid_data = []

        for i, (seq, pos, ref, alt) in enumerate(zip(seqs, poses, refs, alts)):
            if len(seq) <= self.max_len and pos < len(seq) and seq[pos] == ref:
                ref_id = self.tokenizer.convert_tokens_to_ids(ref)
                alt_id = self.tokenizer.convert_tokens_to_ids(alt)
                if ref_id is None or alt_id is None:
                    continue
                valid_data.append((i, seq, pos, ref_id, alt_id))

        if not valid_data:
            return results

        indices, valid_seqs, valid_poses, ref_ids, alt_ids = zip(*valid_data)

        enc_inputs = self.tokenizer(list(valid_seqs), return_tensors="pt", padding="longest",
                                    truncation=True, max_length=self.max_len).to(device)
        
        decoder_prefixes = [seq[:pos] if pos > 0 else "" for seq, pos in zip(valid_seqs, valid_poses)]
        
        decoder_inputs = self.tokenizer(decoder_prefixes, return_tensors="pt", padding="longest",
                                        truncation=True, max_length=self.max_len, add_special_tokens=False).to(device)

        batch_size = decoder_inputs.input_ids.shape[0]
        start_tokens = torch.full((batch_size, 1), self.tokenizer.pad_token_id, device=device)
        decoder_input_ids = torch.cat([start_tokens, decoder_inputs.input_ids], dim=1)
        start_mask = torch.ones((batch_size, 1), dtype=torch.long, device=device)
        decoder_attention_mask = torch.cat([start_mask, decoder_inputs.attention_mask], dim=1)


        with torch.no_grad():
            outputs = model(
                input_ids=enc_inputs.input_ids,
                attention_mask=enc_inputs.attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask
            )

            logits = outputs.logits
            
            seq_lengths = decoder_attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(len(seq_lengths), device=device)
            logits_i = logits[batch_indices, seq_lengths, :]  # logits at mutation position
            
            probs = torch.nn.functional.softmax(logits_i, dim=-1)

            ref_ids_tensor = torch.tensor(ref_ids, device=device)
            alt_ids_tensor = torch.tensor(alt_ids, device=device)

            p_ref = probs[batch_indices, ref_ids_tensor]
            p_alt = probs[batch_indices, alt_ids_tensor]

            log_odds = torch.log(p_alt) - torch.log(p_ref)
            log_odds = log_odds.tolist()

        for i, log_odd_value in zip(indices, log_odds):
            results[i] = log_odd_value

        return results



    # def compute_log_odds_encoder_decoder(self, model, seqs, poses, refs, alts):
    #     """
    #     Which token (ref or alt) is more probable at position i, given the unmutated sequence upto position i as context.
    #     Computes zero-shot variant effect scores for encoder-decoder models (e.g., T5).
    #     For each variant:
    #         log_odds = log P(alt | decoder_prefix, reference_sequence) - log P(ref | decoder_prefix, reference_sequence)

    #     Where:
    #         • The encoder always receives the full *reference sequence* (unmutated).
    #         • The decoder is given the prefix of the reference sequence *up to but not including* the variant position.
    #         • The model is asked to predict the next token (at the variant position), and we compare the probabilities it assigns to the ref and alt tokens.


    #     • One encoder forward pass per batch
    #     • Two decoder passes per variant (one for ref, one for alt)
    #     """
    #     results = [None] * len(seqs)
    #     device = next(model.parameters()).device

    #     # Step 1: Filter valid examples from the batch
    #     valid_data = []
    #     for i, (seq, pos, ref, alt) in enumerate(zip(seqs, poses, refs, alts)):
    #         # Skip variants at position 0 (no prefix to condition on)
    #         if pos == 0:
    #             continue
    #         if len(seq) <= self.max_len and pos < len(seq) and seq[pos] == ref:

    #             ref_id = self.tokenizer.convert_tokens_to_ids(ref)
    #             alt_id = self.tokenizer.convert_tokens_to_ids(alt)

    #             if ref_id is None or alt_id is None:
    #                 continue

    #             valid_data.append((i, seq, pos, ref_id, alt_id))

        
    #     if not valid_data:
    #         return results # all None 

    #     indices, valid_seqs, valid_poses, ref_ids, alt_ids = zip(*valid_data)

    #     # Step 2: Tokenize encoder input sequences
    #     # enc_inputs.input_ids: [B, L_enc]
    #     enc_inputs = self.tokenizer(
    #         list(valid_seqs),
    #         return_tensors="pt",
    #         padding="longest",
    #         truncation=True,
    #         max_length=self.max_len
    #     ).to(device)

    #     # Step 3: Prepare decoder inputs (prefixes up to mutation site)
    #     decoder_prefixes = [seq[:pos] for seq, pos in zip(valid_seqs, valid_poses)]
    #     # print(f"Decoder prefixes (first 5): {decoder_prefixes[:5]}")
    #    # decoder_inputs.input_ids: [B, L_dec] (L_dec varies but is padded)
    #     decoder_inputs = self.tokenizer(
    #         decoder_prefixes,
    #         return_tensors="pt",
    #         padding="longest",
    #         truncation=True,
    #         max_length=self.max_len
    #     ).to(device)

    #     with torch.no_grad():
    #         # Step 4: Forward pass
    #         outputs = model(
    #             input_ids=enc_inputs.input_ids,                # [B, L_enc]
    #             attention_mask=enc_inputs.attention_mask,      # [B, L_enc]
    #             decoder_input_ids=decoder_inputs.input_ids,    # [B, L_dec]
    #             decoder_attention_mask=decoder_inputs.attention_mask,  # [B, L_dec]
    #         )

    #         # outputs.logits: [B, L_dec, V] (V = vocab size)
    #         logits = outputs.logits

    #         # Step 5: Get logits for the next token at variant position (last in decoder sequence)
    #         # logits_i: [B, V]
    #         logits_i = logits[:, -1, :]

    #         # probs: [B, V] — probability distribution over vocab for each sample
    #         probs = torch.nn.functional.softmax(logits_i, dim=-1)

    #         # ref_ids_tensor / alt_ids_tensor: [B]
    #         ref_ids_tensor = torch.tensor(ref_ids, device=device) # Shape: [B]
    #         alt_ids_tensor = torch.tensor(alt_ids, device=device) # Shape: [B]

    #         # p_ref / p_alt: [B]
    #         # grab the model's predicted probability for the specific token (ref or alt) for each sample
    #         p_ref = probs[torch.arange(len(ref_ids_tensor)), ref_ids_tensor]  # p_ref[i] = P(ref_token_i | decoder_prefix_i)
    #         p_alt = probs[torch.arange(len(alt_ids_tensor)), alt_ids_tensor]  # p_alt[i] = P(alt_token_i | decoder_prefix_i)

    #         # log_odds: [B]
    #         log_odds = torch.log(p_alt) - torch.log(p_ref)
    #         log_odds = log_odds.tolist()


    #     # Step 6: Assign results to original positions
    #     for i, log_odd_value in zip(indices, log_odds):
    #         results[i] = log_odd_value
        
    #     return results

        
    
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

