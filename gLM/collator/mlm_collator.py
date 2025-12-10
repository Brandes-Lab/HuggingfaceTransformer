# modules/dataset/mlm_collator.py

from transformers import DataCollatorForLanguageModeling

def create_mlm_collator(tokenizer, mlm_probability: float = 0.15):

    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
    )
