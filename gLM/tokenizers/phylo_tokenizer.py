from transformers import (
    PreTrainedTokenizerFast,
)
import torch

class PhyloTokenizerLoader:
    def __init__(self, tokenizer_path):
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    def batch_encode_aligned(self, aligned_pairs, max_length: int = None):
        """"
        aligned_pairs: list of (a1, a2) aligned strings
        returns dict with input_ids, labels, attention_mask
        """
        # P(Seq1 | Seq2), input_ids from Seq2, labels from Seq1
        # Split into two lists: a1s and a2s
        a1s, a2s = zip(*aligned_pairs)

        # print("example of aligned input:", a1s[0], a2s[0])

        # Replace gaps with [GAP] token and join back to string
        tokens1 = ["".join(["[GAP]" if c == "-" else c for c in seq]) for seq in a1s]
        tokens2 = ["".join(["[GAP]" if c == "-" else c for c in seq]) for seq in a2s]

        # print("example of tokenized input:", tokens1[0], tokens2[0])
              
        # Truncate if needed
        # if max_length:
        #     tokens1 = [seq[:max_length] for seq in tokens1]
        #     tokens2 = [seq[:max_length] for seq in tokens2]
        
        # Convert tokens to ids 
        input_ids = self.tokenizer(
                tokens2,
                padding="longest",
                truncation=True,
                max_length = max_length,
                return_tensors="pt",
            )

        dec = self.tokenizer(
                tokens1,
                padding="longest",
                truncation=True,
                max_length = max_length,
                return_tensors="pt",
            )

        # Conver {PAD} tokens in labels to -100
        labels = dec["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
    
        assert(labels == -100).sum() == (dec["input_ids"] == self.tokenizer.pad_token_id).sum()
        assert torch.all((labels == -100) | ((labels >= 0) & (labels < self.tokenizer.vocab_size)))
        
        
        return {
            "input_ids": input_ids["input_ids"],
            "attention_mask": input_ids["attention_mask"],
            "labels": labels,
        }

        # self.tokenizer.encode_aligned = encode_aligned

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def __len__(self):
        return len(self.tokenizer)

