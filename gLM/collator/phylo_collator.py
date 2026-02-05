from transformers import DataCollatorForTokenClassification
import torch
from gLM.sequences.pairwise_align import align_pair, percent_identity

# class SequencePairCollator(DataCollatorForTokenClassification):
#     def __call__(self, features):
#         # Extract and remove percent_identity from features
#         percent_identities = [feature.pop("percent_identity") for feature in features]

#         # Call the parent class to handle padding
#         batch = super().__call__(features)

#         # Add percent_identity back to the batch
#         batch["percent_identity"] = torch.tensor(percent_identities, dtype=torch.float32)
#         return batch



class SequencePairCollator(DataCollatorForTokenClassification):
    def __init__(self, tokenizer, training_type, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.training_type = training_type

    def __call__(self, features):
        if self.training_type == "phylo_encoder":
            percent_identities = [feature.pop("percent_identity") for feature in features]
            batch = super().__call__(features)
            batch["percent_identity"] = torch.tensor(percent_identities, dtype=torch.float32)
            return batch

        elif self.training_type == "phylo_encoder_decoder":
            return super().__call__(features)

        else:
            raise ValueError(f"Unsupported training_type: {self.training_type}")



class PhyloCollator:
    def __init__(self, tokenizer, training_type: str, max_seq_len: int):
        self.tokenizer = tokenizer
        self.training_type = training_type
        self.max_seq_len = max_seq_len

    def __call__(self, batch):
        # batch is a list of (seq1 and seq2)

        if self.training_type == "MLM":
            sequences = [s1[:self.max_seq_len] for s1, _ in batch]
            return self.tokenizer(
                sequences, 
                padding="longest", 
                truncation=True,
                return_tensors="pt")

        elif self.training_type == "phylo_encoder_only":
            # P(Seq1 | Seq2)
            aligned_pairs = []
            pids = []
            for s1, s2 in batch:
                # per pair alignment
                a1, a2 = align_pair(s1, s2)

                if len(a1) == len(a2):
                    aligned_pairs.append((a1, a2))
                    pid = percent_identity(a1, a2)
                    pids.append(pid)

            # Tokenize aligned pairs (batch)    
            batch_out = self.tokenizer.batch_encode_aligned(
                aligned_pairs, max_length=self.max_seq_len)
            
            return batch_out
        
        elif self.training_type == "phylo_encoder_decoder":
            # P(Seq1 | Seq2)
            inputs = [s2[:self.max_seq_len] for s2, _ in batch]
            targets = [s1[:self.max_seq_len] for _, s1 in batch]
            enc = self.tokenizer(
                inputs,
                padding="longest",
                truncation=True,
                max_length=self.max_seq_len,
                return_tensors="pt",
            )

            dec = self.tokenizer(
                targets,
                padding="longest",
                truncation=True,
                max_length=self.max_seq_len,
                return_tensors="pt",
            )

            # Conver {PAD} tokens in labels to -100
            labels = dec["input_ids"].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100

            assert(labels == -100).sum() == (dec["input_ids"] == self.tokenizer.pad_token_id).sum()

            return {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": labels,
            }

        else:
            raise ValueError(f"Unsupported training_type: {self.training_type}")

        


               
