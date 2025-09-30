import logging
from typing import Any, Dict, List, Optional
import torch
from transformers import DataCollatorForLanguageModeling


logger = logging.getLogger(__name__)


class TruncatingDataCollatorForMLM(DataCollatorForLanguageModeling):
    """
    Data collator that truncates sequences to a maximum length before applying MLM.

    This is useful when using dynamic batch sampling where you want to ensure that
    no sequence exceeds a certain length.
    """

    def __init__(
        self,
        tokenizer,
        mlm: bool = True,
        mlm_probability: float = 0.15,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            mlm=mlm,
            mlm_probability=mlm_probability,
            pad_to_multiple_of=pad_to_multiple_of,
            **kwargs,
        )
        self.max_length = max_length
        self._truncation_warning_shown = False

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Truncate sequences if needed, then apply the standard MLM collation.
        """
        if self.max_length is not None:
            processed_examples = []
            for example in examples:
                # Only create a copy if truncation is necessary
                if "input_ids" in example:
                    input_ids = example["input_ids"]
                    original_length = len(input_ids)

                    if original_length > self.max_length:
                        # Create a copy only when truncation is needed
                        truncated_example = example.copy()
                        truncated_example["input_ids"] = input_ids[: self.max_length]

                        # Also truncate other sequence-related fields if they exist
                        if "attention_mask" in truncated_example:
                            truncated_example["attention_mask"] = truncated_example[
                                "attention_mask"
                            ][: self.max_length]
                        if "token_type_ids" in truncated_example:
                            truncated_example["token_type_ids"] = truncated_example[
                                "token_type_ids"
                            ][: self.max_length]

                        # Show warning once
                        if not self._truncation_warning_shown:
                            logger.warning(
                                f"Truncating sequence from length {original_length} to {self.max_length}. "
                                "This warning will only be shown once."
                            )
                            self._truncation_warning_shown = True

                        processed_examples.append(truncated_example)
                    else:
                        # No truncation needed, use original example
                        processed_examples.append(example)
                else:
                    # No input_ids field, use original example
                    processed_examples.append(example)

            examples = processed_examples

        # Call the parent class's __call__ method to apply MLM
        return super().__call__(examples)
