from transformers import (
    PreTrainedTokenizerFast,
)


class PhyloTokenizerLoader:
    def __init__(self, tokenizer_path):
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

        # Attach custom encode to HF tokenizer
        self.tokenizer.encode_aligned = self.encode_aligned

    def encode_aligned(self, aligned_seq1, aligned_seq2):
        inputs = self.tokenizer(aligned_seq2, add_special_tokens=False)
        labels = self.tokenizer(aligned_seq1, add_special_tokens=False)
        return {
            "input_ids": inputs["input_ids"],
            "labels": labels["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

    def load(self):
        # Return the HF tokenizer directly
        return self.tokenizer


# # Set breakpoint inside encode() method
# # Then run this:
# phylo_tokenizer = PhyloTokenizerLoader("phylo_char_tokenizer")

# # When you call encode, debugger will pause at your breakpoint
# result = phylo_tokenizer.encode("ATCG", "ATCG")