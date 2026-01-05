from transformers import DataCollatorForTokenClassification
import torch

class SequencePairCollator(DataCollatorForTokenClassification):
    def __call__(self, features):
        # Extract and remove percent_identity from features
        percent_identities = [feature.pop("percent_identity") for feature in features]

        # Call the parent class to handle padding
        batch = super().__call__(features)

        # Add percent_identity back to the batch
        batch["percent_identity"] = torch.tensor(percent_identities, dtype=torch.float32)
        return batch