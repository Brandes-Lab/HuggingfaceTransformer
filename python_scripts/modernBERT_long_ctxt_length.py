import os, time, argparse, torch, wandb, random
import numpy as np
import pandas as pd
from datasets import load_from_disk
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
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            cls_token_id=self.tokenizer.cls_token_id,
            sep_token_id=self.tokenizer.sep_token_id,
        )
        return ModernBertForMaskedLM(config)


class ZeroShotVEPEvaluationCallback(TrainerCallback):
    def __init__(self, tokenizer, input_csv, trainer, max_len=8192, eval_every_n_steps=50000):
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

        # seed = int(getattr(trainer.args, "seed", 42))  
        # df = pd.read_csv(
        #     input_csv,
        #     usecols=["sequence", "pos", "ref", "alt", "label"],
        #     dtype={"pos": np.int32, "label": np.int8},
        # )
        # # keep exactly 5k random rows 
        # n = min(5000, len(df))
        # self.df = df.sample(n=n, random_state=seed).reset_index(drop=True)

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
    run_name = "modernBERT_uniref_tokenized8192"
    wandb.init(project="huggingface_bert_sweep", name=run_name, entity="sinha-anushka12-na")

    tokenizer = TokenizerLoader("char_tokenizer").load()
    print("Tokenizer vocab size:", tokenizer.vocab_size)

    model = ProteinBertModel(tokenizer.vocab_size, tokenizer).build()
    model.cuda()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load pre-tokenized datasets
    train_ds = load_from_disk("/gpfs/data/brandeslab/Data/processed_datasets/uniref90_tokenized_8192/train_only/train")
    val_ds = load_from_disk("/gpfs/data/brandeslab/Data/processed_datasets/uniref90_tokenized_8192/val_only/validation")

    print("Max train length:", max(train_ds["length"]))
    print("99th percentile:", np.percentile(train_ds["length"], 99))
    print("95th percentile:", np.percentile(train_ds["length"], 95))

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=f"/gpfs/data/brandeslab/model_checkpts/{run_name}",
        max_steps=2_000_000,
        
        per_device_train_batch_size=16,
        gradient_accumulation_steps=16,
        per_device_eval_batch_size=4,
        
        bf16=True,
        fp16=False,
        
        dataloader_num_workers=16,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=2,
        
        eval_strategy="no",             # not running eval
        logging_strategy="steps",
        logging_steps=1000,
        save_strategy="no",
        report_to="wandb",
        run_name=run_name,
        
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
    dl = trainer.get_train_dataloader() 
    for i, batch in enumerate(dl): 
        print(f"Batch {i} max length: {batch['input_ids'].shape[1]}") 
        if i > 5: 
            break

    trainer.add_callback(ZeroShotVEPEvaluationCallback(
        tokenizer=tokenizer,
        input_csv="/gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv",
        trainer=trainer,
        eval_every_n_steps=50000
    ))
    trainer.add_callback(ElapsedTimeLoggerCallback())
    trainer.train()

if __name__ == "__main__":
    main()