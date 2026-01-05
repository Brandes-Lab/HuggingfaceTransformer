from transformers import (
    PreTrainedTokenizerFast,
)

class PhyloTokenizerLoader:
    def __init__(self, tokenizer_path):
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

        # # Attach aligned encoding
        # def encode_aligned(a1, a2):
        #     inputs = self.tokenizer(a2, add_special_tokens=False)
        #     labels = self.tokenizer(a1, add_special_tokens=False)
        #     return {
        #         "input_ids": inputs["input_ids"],
        #         "labels": labels["input_ids"],
        #         "attention_mask": inputs["attention_mask"],
        #     }


    def batch_encode_aligned(self, aligned_pairs, max_length: int = None):
        """"
        aligned_pairs: list of (a1, a2) aligned strings
        returns list[dict] with input_ids, labels, attention_mask
        """

        # Split into two lists: a1s and a2s
        a1s, a2s = zip(*aligned_pairs)

        # Replace gaps with [GAP] token
        tokens1 = [["[GAP]" if c == "-" else c for c in seq] for seq in a1s]
        tokens2 = [["[GAP]" if c == "-" else c for c in seq] for seq in a2s]

        # Truncate if needed
        if max_length:
            tokens1 = [seq[:max_length] for seq in tokens1]
            tokens2 = [seq[:max_length] for seq in tokens2]
        
        # Convert tokens to ids 
        input_ids = [self.tokenizer.convert_tokens_to_ids(seq) for seq in tokens2]
        labels = [self.tokenizer.convert_tokens_to_ids(seq) for seq in tokens1]
        attention_masks = [[1] * len(seq) for seq in input_ids]

        # Build output list of dicts
        return [
            {
                "input_ids": input_ids[i],
                "labels": labels[i],
                "attention_mask": attention_masks[i],
            }
            for i in range(len(input_ids))
        ]


    def encode_aligned(self, a1: str, a2: str, max_length: int = None):
        """
        Tokenize aligned sequences, preserve 1:1 alignment.
        Relaces - with [GAP].
        Returns unpadded input_ids, labels, attention_mask.
        Truncates to max_length if provided.
        """

        assert len(a1) == len(a2), "Aligned sequences must be of equal length"

        tokens1 = []
        tokens2 = []

        for c1, c2, in zip(a1, a2):
            t1 = "[GAP]" if c1 == "-" else c1
            t2 = "[GAP]" if c2 == "-" else c2
            tokens1.append(t1)
            tokens2.append(t2)
        print(f"Tokens1: {len(tokens1)}")
        print(f"Tokens2: {len(tokens2)}")

        if max_length is not None:
            tokens1 = tokens1[:max_length]
            tokens2 = tokens2[:max_length]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens2)
        labels = self.tokenizer.convert_tokens_to_ids(tokens1)
        attention_mask = [1] * len(input_ids)


        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

        # self.tokenizer.encode_aligned = encode_aligned

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def __len__(self):
        return len(self.tokenizer)

