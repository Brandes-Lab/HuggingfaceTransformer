import os, time, argparse, torch, wandb, random
import numpy as np
import pandas as pd
from itertools import islice
from datasets import load_dataset, load_from_disk
import numpy as np

from datasets import load_dataset, load_from_disk, Dataset, Value
from sklearn.metrics import roc_auc_score
from transformers import (
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    ModernBertForMaskedLM,
    ModernBertConfig,
)


class TokenizerLoader:
    def __init__(self, tokenizer_path):
        self.tokenizer_path = tokenizer_path

    def load(self):
        return PreTrainedTokenizerFast.from_pretrained(self.tokenizer_path)


class ProteinBertModel:
    def __init__(self, vocab_size, tokenizer):
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer

    def build(self):
        config = ModernBertConfig(
            vocab_size=self.vocab_size,
            max_position_embeddings=8192,
            num_hidden_layers=8,
            num_attention_heads=8,
            hidden_size=512,
            intermediate_size=2048,
            type_vocab_size=1,
            hidden_activation="gelu",
            global_attn_every_n_layers=3,
            local_attention=512,
            deterministic_flash_attn=False,
            global_rope_theta=160000.0,
            local_rope_theta=10000.0,
            pad_token_id=getattr(self.tokenizer, 'pad_token_id', None),
            eos_token_id=getattr(self.tokenizer, 'eos_token_id', None),
            bos_token_id=getattr(self.tokenizer, 'bos_token_id', None),
            cls_token_id=getattr(self.tokenizer, 'cls_token_id', None),
            sep_token_id=getattr(self.tokenizer, 'sep_token_id', None),
        )
        return ModernBertForMaskedLM(config)

class OnTheFlyMLMCollator:
    """
    Collator that:
      1) Tokenizes raw strings from the dataset (field: "text")
      2) Applies MLM masking
    """
    def __init__(self, tokenizer, mlm_probability=0.15, max_length=8192):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=mlm_probability
        )

    def __call__(self, features):
        texts = [f["text"].upper().strip() for f in features if f.get("text")]
        if not texts:
            raise ValueError("Collator got an empty/whitespace-only batch")
        encodings = self.tokenizer(
            texts,
            padding="longest",      # Pad to the longest sequence in the batch
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False
        )
        # Convert encodings to list-of-dicts, one dict per example

        features = [
            {k: v[i] for k, v in encodings.items()}
            for i in range(len(texts))
        ]
        return self.mlm(features)


class ZeroShotVEPEvaluationCallback(TrainerCallback):
    def __init__(self, tokenizer, input_csv, trainer, max_len=8192, eval_every_n_steps=20000):
        self.tokenizer = tokenizer
        self.input_csv = input_csv
        self.max_len = max_len
        self.eval_every_n_steps = eval_every_n_steps
        self.trainer = trainer
        self.start_time = time.time()

        self.df = pd.read_csv(
            input_csv,
            usecols=["sequence", "pos", "ref", "alt", "label"],
            dtype={"pos": np.int32, "label": np.int8},
        )

    def compute_log_odds(self, model, seq, pos, ref, alt):
    # skip if > max_len or ref mismatch
        if len(seq) > self.max_len or pos >= len(seq) or seq[pos] != ref:
            return None

        masked_seq = list(seq)
        masked_seq[pos] = self.tokenizer.mask_token
        masked_seq = "".join(masked_seq)

        inputs = self.tokenizer(
            masked_seq, return_tensors="pt", truncation=True, max_length=self.max_len
        )
        inputs = {k: v.cuda(non_blocking=True) for k, v in inputs.items()}

        with torch.inference_mode():
            logits = model(**inputs).logits

        mask_index = (inputs["input_ids"][0] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0].item()
        probs = torch.nn.functional.softmax(logits[0, mask_index], dim=0)

        ref_id = self.tokenizer.convert_tokens_to_ids(ref)
        alt_id = self.tokenizer.convert_tokens_to_ids(alt)
        if ref_id is None or alt_id is None:
            return None

        return (torch.log(probs[alt_id]) - torch.log(probs[ref_id])).item()

    def run_vep_eval(self, model, step_id):
        if not self.trainer.is_world_process_zero():
            return
        elapsed_hours = (time.time() - self.start_time) / 3600
        print(f"Running zero-shot VEP evaluation at step {step_id}", flush=True)

        seqs   = self.df["sequence"].values
        poses  = self.df["pos"].values
        refs   = self.df["ref"].values
        alts   = self.df["alt"].values
        labels = self.df["label"].to_numpy(dtype=np.int8)

        n = len(labels)
        preds = np.full(n, np.nan, dtype=np.float32)  


        was_training = model.training
        model.eval()
        try:
            for i in range(n):
                s = self.compute_log_odds(model, seqs[i], int(poses[i]), refs[i], alts[i])
                if s is not None:
                    preds[i] = -float(s)  
        finally:
            if was_training:
                model.train()

        mask = ~np.isnan(preds)
        if mask.sum() >= 10 and (labels[mask].min() != labels[mask].max()):
            auc = roc_auc_score(labels[mask], preds[mask])
            print(f"AUC at step {step_id}: {auc:.4f}")
            wandb.log({"zero_shot_vep_auc": auc, "step": step_id, "elapsed_hours": elapsed_hours})
        else:
            print(f"Skipping AUC at step {step_id} due to insufficient data", flush=True)

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0:
            self.run_vep_eval(model, step_id=state.global_step)
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_every_n_steps == 0 and state.global_step > 0:
            self.run_vep_eval(model, step_id=state.global_step)
        return control



class ElapsedTimeLoggerCallback(TrainerCallback):
    def __init__(self):
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        elapsed_hours = (time.time() - self.start_time) / 3600
        if logs is not None:
            logs["elapsed_hours"] = elapsed_hours
            wandb.log(logs, step=state.global_step)


def main():
    run_name = "modernBERT_all_uniref"
    wandb.init(project="huggingface_bert_sweep", name=run_name, entity="sinha-anushka12-na")
    
    tokenizer = TokenizerLoader("char_tokenizer").load()
    print("Tokenizer vocab size:", tokenizer.vocab_size)

    
    # Build model
    model = ProteinBertModel(tokenizer.vocab_size, tokenizer).build()
    model.cuda()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    # Map-style train dataset (required for group_by_length)

    # 1) Load
    ds = load_dataset(
        "text",
        data_files={
            "train": "/gpfs/data/brandeslab/Data/raw_data/Uniref90/train/train_shards/train_part_*_shuf.txt",
            "validation": "/gpfs/data/brandeslab/Data/raw_data/Uniref90/val/val.txt",
        },
        streaming=False,
    )

    # 2) Filter out empty / whitespace-only lines
    def keep_nonempty(ex):
        t = ex["text"]
        return isinstance(t, str) and (t.strip() != "")

    train_ds = ds["train"].filter(keep_nonempty, num_proc=16)
    val_ds   = ds["validation"].filter(keep_nonempty, num_proc=16).select(range(10_000))


    # 3) Map length 
    def compute_len(batch):
        ids = tokenizer(
            batch["text"],
            truncation=True, max_length=8192, add_special_tokens=False,
            padding=False, return_attention_mask=False, return_token_type_ids=False
        )["input_ids"]
        return {"length": [len(x) for x in ids]}
    
    train_ds = train_ds.map(compute_len, batched=True, batch_size=50_000, num_proc=16)
    
    print("train_df after mapping", train_ds)  # should list columns: ['text', 'length']
    print(train_ds[0].keys())  # must contain 'text'

    # 4) Make 'length' compact so global bucketing doesn’t OOM
    train_ds = train_ds.cast_column("length", Value("int32"))
    # train_ds = train_ds.with_format(type="numpy", columns=["length"])
    print("train_df making length compact", train_ds) 
    print(train_ds[0].keys())  # must contain 'text'

    # (optional sanity)
    print("Any blanks left? ",
          any(len(s.strip()) == 0 for s in train_ds.select(range(1000))["text"]))


    data_collator = OnTheFlyMLMCollator(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        max_length=8192
    )

    training_args = TrainingArguments(
        output_dir=f"/gpfs/data/brandeslab/model_checkpts/{run_name}",
        max_steps=2_000_000,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=32,
        per_device_eval_batch_size=1,
        
        dataloader_num_workers=8,
        dataloader_persistent_workers= False, 
        dataloader_prefetch_factor=1,
        
        eval_strategy="steps",
        eval_steps=20000,
    
        logging_strategy="steps",
        logging_steps=1000,

        save_strategy="no",
        report_to="wandb",
        run_name=run_name,
        fp16=True,
        learning_rate=3e-4,
        remove_unused_columns=False,

        group_by_length=True,
        length_column_name="length"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    batch = next(iter(trainer.get_train_dataloader()))
    labels = batch["labels"]
    n_mask = (labels != -100).sum().item()

    model.eval()
    with torch.inference_mode():
        out = model(**{k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)})
    loss_sum = out.loss.item()
    loss_per_mask = loss_sum / max(n_mask, 1)
    print(f"masked_tokens={n_mask}  loss_sum≈{loss_sum:.2f}  loss_per_mask≈{loss_per_mask:.4f}")
    model.train()

    trainer.add_callback(
        ZeroShotVEPEvaluationCallback(
            tokenizer=tokenizer,
            input_csv="/gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv",
            trainer=trainer,
            eval_every_n_steps=20000     
        )
    )
    trainer.add_callback(ElapsedTimeLoggerCallback())
    trainer.train()

if __name__ == "__main__":
    main()
