from transformers import DataCollatorForTokenClassification
import torch

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
