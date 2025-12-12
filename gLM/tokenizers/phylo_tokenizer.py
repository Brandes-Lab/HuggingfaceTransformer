from transformers import (
    PreTrainedTokenizerFast,
)


# class PhyloTokenizerLoader:
#     def __init__(self, tokenizer_path):
#         self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

#         # Attach custom encode to HF tokenizer
#         self.tokenizer.encode_aligned = self.encode_aligned
    
#     def __getattr__(self, name):
#         # If attribute not found on this object, get it from self.tokenizer
#         return getattr(self.tokenizer, name)

#     def encode_aligned(self, aligned_seq1, aligned_seq2):
#         inputs = self.tokenizer(aligned_seq2, add_special_tokens=False)
#         labels = self.tokenizer(aligned_seq1, add_special_tokens=False)
#         return {
#             "input_ids": inputs["input_ids"],
#             "labels": labels["input_ids"],
#             "attention_mask": inputs["attention_mask"],
#         }

#     def load(self):
#         # Return the HF tokenizer directly
#         return self.tokenizer


# # Set breakpoint inside encode() method
# # Then run this:
# phylo_tokenizer = PhyloTokenizerLoader("phylo_char_tokenizer")

# # When you call encode, debugger will pause at your breakpoint
# result = phylo_tokenizer.encode("ATCG", "ATCG")


from transformers import PreTrainedTokenizerFast

class PhyloTokenizerLoader:
    def __init__(self, tokenizer_path):
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

        # Attach aligned encoding
        def encode_aligned(a1, a2):
            inputs = self.tokenizer(a2, add_special_tokens=False)
            labels = self.tokenizer(a1, add_special_tokens=False)
            return {
                "input_ids": inputs["input_ids"],
                "labels": labels["input_ids"],
                "attention_mask": inputs["attention_mask"],
            }

        self.tokenizer.encode_aligned = encode_aligned

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def __len__(self):
        return len(self.tokenizer)

