# modules/dataset/mlm_collator.py

from transformers import DataCollatorForLanguageModeling
import torch


class MLMCollator:
    def __init__(self, tokenizer, max_seq_len: int, mlm_probability: float = 0.15):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mlm_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=mlm_probability
        )

    def __call__(self, batch):
        # batch: List[str] — raw sequences, just like PhyloCollator
        tokenized = self.tokenizer(
            batch,
            padding="longest",
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt"
        )
        # DataCollatorForLanguageModeling expects a list of dicts
        # Convert the batch encoding into a list of individual examples
        examples = [
            {key: tokenized[key][i] for key in tokenized}
            for i in range(len(batch))
        ]
        # This applies random masking and returns input_ids, attention_mask, labels
        return self.mlm_collator(examples)


def create_mlm_collator(tokenizer, max_seq_len: int, mlm_probability: float = 0.15):
    return MLMCollator(tokenizer, max_seq_len, mlm_probability)