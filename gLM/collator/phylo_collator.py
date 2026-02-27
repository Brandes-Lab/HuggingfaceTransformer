from transformers import DataCollatorForTokenClassification
import torch
from gLM.sequences.pairwise_align import align_pair, percent_identity



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


        if self.training_type == "phylo_encoder_only":
            # P(Seq1 | Seq2), input_ids from Seq2, labels from Seq1
            # batch: List[Tuple[a1, a2, pid]]
            a1s, a2s, pids = zip(*batch)

            # Replace gaps with [GAP] token and join back to string
            tokens1 = ["".join(["[GAP]" if c == "-" else c for c in seq]) for seq in a1s]
            tokens2 = ["".join(["[GAP]" if c == "-" else c for c in seq]) for seq in a2s]
            
            # Convert to tokens, truncate to max length if needed, pad to the max in the batch
            input_ids = self.tokenizer(
                    tokens2,
                    padding="longest",
                    truncation=True,
                    max_length = self.max_seq_len,
                    return_tensors="pt",
                )

            dec = self.tokenizer(
                    tokens1,
                    padding="longest",
                    truncation=True,
                    max_length = self.max_seq_len,
                    return_tensors="pt",
                )

            # Conver {PAD} tokens in labels to -100
            labels = dec["input_ids"].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            
        
            assert(labels == -100).sum() == (dec["input_ids"] == self.tokenizer.pad_token_id).sum()
            assert torch.all((labels == -100) | ((labels >= 0) & (labels < self.tokenizer.vocab_size)))
            
            batch_out = {
                "input_ids": input_ids["input_ids"],
                "attention_mask": input_ids["attention_mask"],
                "labels": labels,
            }
            
            return batch_out

        
        elif self.training_type == "phylo_encoder_decoder":
            # P(Seq1 | Seq2), input_ids from Seq2, targets from Seq1
            # batch: List[Tuple[s1, s2]]
            inputs, targets = zip(*batch)
            # print("Batch size:", len(inputs))
            # print("len targets", len(targets))

            # truncate to max length if needed, pad to the max in the batch
            enc = self.tokenizer(
                list(inputs), 
                padding="longest",
                truncation=True,
                max_length=self.max_seq_len,
                return_tensors="pt"
            )

            dec = self.tokenizer(
                list(targets),
                padding="longest",
                truncation=True,
                max_length=self.max_seq_len,
                return_tensors="pt"
            )
            
            # for i, seq in enumerate(targets):
            #     decoded = self.tokenizer.decode(dec["input_ids"][i], skip_special_tokens=False)
            #     print(f"[{i}] Target raw: {repr(seq)}")
            #     print(f"[{i}] Tokenized length: {(dec['input_ids'][i] != self.tokenizer.pad_token_id).sum().item()}")
            #     print(f"[{i}] Decoded: {repr(decoded)}")
            #     print(f"[{i}] All pad: {(dec['input_ids'][i] == self.tokenizer.pad_token_id).all().item()}")

            # Conver {PAD} tokens in labels to -100
            labels = dec["input_ids"].clone()
            labels[labels == self.tokenizer.pad_token_id] = -100

            assert(labels == -100).sum() == (dec["input_ids"] == self.tokenizer.pad_token_id).sum()
            
            # Verify attention_mask matches padding
            assert (dec["attention_mask"] == 0).sum() == (dec["input_ids"] == self.tokenizer.pad_token_id).sum()

            batch_out = {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "decoder_attention_mask": dec["attention_mask"],
                "labels": labels,
            }

            # print("input_ids shape:", batch_out["input_ids"].shape)
            # print("attention_mask shape:", batch_out["attention_mask"].shape)
            # print("attention_mask sum:", batch_out["attention_mask"].sum(dim=1))
            # print("seq lengths in labels", dec["attention_mask"].sum(dim=1))
            # print("labels shape:", batch_out["labels"].shape)

            return batch_out

        else:
            raise ValueError(f"Unsupported training_type: {self.training_type}")

