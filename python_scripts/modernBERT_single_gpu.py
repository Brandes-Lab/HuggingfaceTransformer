import os, time, torch, wandb
import pandas as pd
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
from datasets import load_from_disk

class TokenizerLoader:
    def __init__(self, tokenizer_path):
        self.tokenizer_path = tokenizer_path

    def load(self):
        return PreTrainedTokenizerFast.from_pretrained(self.tokenizer_path)


class ProteinDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load(self):
        dataset = load_from_disk(self.data_dir)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        return dataset


class ProteinBertModel:
    def __init__(self, vocab_size, tokenizer):
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer

    def build(self):
        config = ModernBertConfig(
            vocab_size=self.vocab_size,
            max_position_embeddings=512,
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


class MLMDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def get(self):
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=0.15
        )


class ZeroShotVEPEvaluationCallback(TrainerCallback):
    def __init__(self, tokenizer, input_csv, trainer, max_len=512, eval_every_n_steps=20000):
        self.tokenizer = tokenizer
        self.input_csv = input_csv
        self.max_len = max_len
        self.eval_every_n_steps = eval_every_n_steps
        self.trainer = trainer
        self.start_time = time.time()
        self.df = pd.read_csv(input_csv)

    def compute_log_odds(self, model, seq, pos, ref, alt):
        if len(seq) > self.max_len or pos >= len(seq) or seq[pos] != ref:
            return None

        masked_seq = list(seq)
        masked_seq[pos] = self.tokenizer.mask_token
        masked_seq = "".join(masked_seq)

        inputs = self.tokenizer(masked_seq, return_tensors="pt", truncation=True, max_length=self.max_len)
        inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        mask_index = (inputs["input_ids"][0] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0].item()
        probs = torch.nn.functional.softmax(logits[0, mask_index], dim=0)

        ref_id = self.tokenizer.convert_tokens_to_ids(ref)
        alt_id = self.tokenizer.convert_tokens_to_ids(alt)

        if ref_id is None or alt_id is None:
            return None

        return (torch.log(probs[alt_id]) - torch.log(probs[ref_id])).item()

    def run_vep_eval(self, model, step_id):
        elapsed_hours = (time.time() - self.start_time) / 3600
        if not self.trainer.is_world_process_zero():
            return

        print(f"Running zero-shot VEP evaluation at step {step_id}", flush=True)
        log_odds_scores = []
        labels = []

        for _, row in self.df.iterrows():
            score = self.compute_log_odds(model, row["sequence"], int(row["pos"]), row["ref"], row["alt"])
            log_odds_scores.append(score)
            labels.append(int(row["label"]))

        df_out = self.df.copy()
        df_out["log_odds"] = log_odds_scores

        valid_mask = df_out["log_odds"].notnull()
        if valid_mask.sum() >= 10 and len(set(df_out["label"])) > 1:
            auc = roc_auc_score(df_out.loc[valid_mask, "label"], -df_out.loc[valid_mask, "log_odds"])
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
    run_name = "modernBERT_medium_512_old"
    wandb.init(project="huggingface_bert_sweep", name=run_name, entity="sinha-anushka12-na")
    
    tokenizer = TokenizerLoader("char_tokenizer").load()
    print("Tokenizer vocab size:", tokenizer.vocab_size)

    dataset = ProteinDataset("/gpfs/data/brandeslab/Data/tokenized_datasets/uniref90_tokenized_single_char_512").load()
    dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(10_000))
    dataset["test"] = dataset["test"].shuffle(seed=42).select(range(10_000))

    model = ProteinBertModel(tokenizer.vocab_size, tokenizer).build()
    model.cuda()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    data_collator = MLMDataCollator(tokenizer).get()

    from torch.utils.data import DataLoader

    dl = DataLoader(dataset["train"], batch_size=4, collate_fn=data_collator)

    batch = next(iter(dl))
    labels = batch["labels"]

    # print("Labels shape:", labels.shape)
    # print("Example label row:", labels[0])
    # print("Number of masked tokens in sample 0:", (labels[0] != -100).sum().item())

    training_args = TrainingArguments(
        output_dir=f"/gpfs/data/brandeslab/model_checkpts/{run_name}",
        #num_train_epochs=100,
        max_steps=2_000_000,
        per_device_train_batch_size=64,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=4,
        dataloader_num_workers=16,
        eval_strategy="steps",
        eval_steps=20000,
        save_steps=500000,
        logging_strategy="steps",
        logging_steps=20000,
        report_to="wandb",
        run_name=run_name,
        fp16=True,
        learning_rate=3e-4,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.add_callback(
        ZeroShotVEPEvaluationCallback(
            tokenizer=tokenizer,
            input_csv="/gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv",
            trainer=trainer,
            eval_every_n_steps=20000,
        )
    )

    trainer.add_callback(ElapsedTimeLoggerCallback())
    trainer.train()


if __name__ == "__main__":
    main()
