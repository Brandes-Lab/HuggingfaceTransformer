import torch 

class SequencePairCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id
    
    def __call__(self, batch):
        # print("------BATCH START------")
        # pad seqeunces 
        max_len = max(len(x["input_ids"]) for x in batch)

        input_ids = []
        labels = []
        attention_masks = []

        print(f"[Collator] Batch max input length: {max_len}")
        for i, item in enumerate(batch):
            print(f"  Sample {i} input length: {len(item['input_ids'])}")

            # print(f"SEQ LEN: {len(item['input_ids'])}")

            n = len(item["input_ids"])
            pad_len = max_len - n

            input_ids.append(
                item["input_ids"] + [self.pad_id] * pad_len
            )
            labels.append(
                item["labels"] + [-100] * pad_len
            )
            attention_masks.append(
                item["attention_mask"] + [0] * pad_len
            )

        # print("-------BATCH END-------")
        pids = [x["percent_identity"] for x in batch]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "percent_identity": torch.tensor(pids, dtype=torch.float32)
        }