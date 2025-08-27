import os, torch, wandb
from sklearn.metrics import roc_auc_score
from transformers import (
    BertConfig, BertForMaskedLM, PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer, TrainingArguments, TrainerCallback
)
from datasets import load_from_disk
import pandas as pd
from transformers.trainer_utils import is_main_process

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
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def build(self):
        config = BertConfig(
            vocab_size=self.vocab_size,
            max_position_embeddings=2048,
            num_hidden_layers=12,
            num_attention_heads=12,
            hidden_size=768,
            type_vocab_size=1,
        )
        return BertForMaskedLM(config)


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
    def __init__(self, tokenizer, input_csv, trainer, max_len=2048, eval_every_n_steps=100000):
        self.tokenizer = tokenizer
        self.input_csv = input_csv
        self.max_len = max_len
        self.eval_every_n_steps = eval_every_n_steps
        self.skipped_long_seqs = 0
        self.trainer = trainer 
        self.df = pd.read_csv(input_csv)
        self.df = self.df

    def compute_log_odds(self, model, seq, pos, ref, alt):
        if len(seq) > self.max_len:
            self.skipped_long_seqs += 1
            return None
        if pos >= len(seq) or seq[pos] != ref:
            return None

        masked_seq = list(seq)
        masked_seq[pos] = self.tokenizer.mask_token
        masked_seq = "".join(masked_seq)

        inputs = self.tokenizer(masked_seq, return_tensors="pt", truncation=True, max_length=self.max_len)
        
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

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
        # Skip evaluation on non-zero ranks; only rank 0 runs eval
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
            # print(f"AUC at step {step_id}: {auc:.4f}", flush=True)
            # self.trainer.log({"zero_shot_vep_auc": auc})
            wandb.log({"zero_shot_vep_auc": auc}, step=step_id)
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
    

def main():
    rank = int(os.environ["LOCAL_RANK"])
    tokenizer = TokenizerLoader("char_tokenizer").load()
    dataset   = ProteinDataset("/gpfs/data/brandeslab/Data/tokenized_datasets/uniref90_tokenized_single_char_2048").load()

    model = ProteinBertModel(tokenizer.vocab_size).build()
    model = model.to(rank)
    print(f"[Rank {rank}] Model parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    data_collator = MLMDataCollator(tokenizer).get()

    training_args = TrainingArguments(
        ddp_find_unused_parameters=False,
        output_dir="/gpfs/data/brandeslab/model_checkpts/bert_2048_uniref_2GPU",
        num_train_epochs=100,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=4,
        eval_strategy="steps",
        eval_steps=100000,
        save_steps=100000,
        logging_strategy="steps",
        logging_steps=100000,
        report_to="wandb" if rank == 0 else None,
        run_name="bert_2048_2GPU",
        fp16=True,
        local_rank=rank,
        ddp_backend="nccl"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    if int(os.environ.get("LOCAL_RANK", 0)) == 0: 
        print("Initializing wandb")
        wandb.init(project="huggingface_bert_multiGPU", name="bert_2048_2GPU")

    trainer.add_callback(  
        ZeroShotVEPEvaluationCallback(
            tokenizer=tokenizer,
            input_csv="/gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv",
            trainer=trainer,
            eval_every_n_steps=100000
        )
)


    trainer.train()


if __name__ == "__main__":
    main()
