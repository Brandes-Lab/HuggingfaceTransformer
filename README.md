# ProteinBERT-Based Genomic Foundation Model for Zero-Shot Variant Effect Prediction via PrefixLM Conditional Sequence Modeling

This repository implements a PrefixLM-based protein language model that learns the conditional distribution P(seq1 | seq2) over pairs of evolutionarily related protein sequences, and evaluates it on zero-shot missense variant effect prediction using ClinVar.

---

## Quick Start

To launch a training job on the cluster:

```bash
sbatch slurm_scripts/train_phylo.sh
```

To score saved checkpoints offline:

```bash
python python_scripts/score_checkpoints.py --help
```


## Directory Structure

```
HuggingfaceTransformer/
│
├── gLM/                                  # Core library
│   ├── attention_mask/
│   │   ├── prefixlm_flash.py             # ✅ Main PrefixLM attention (use this)
│   │   ├── prefixlm_flash1.py            # Earlier version (deprecated)
│   │   ├── prefixlm_flash2.py            # Earlier version (deprecated)
│   │   └── prefixlm_flash_old_w...       # Earlier version (deprecated)
│   │
│   ├── callbacks/
│   │   ├── variant_effect.py             # Zero-shot VEP callback (runs during training)
│   │   └── percent_identity.py           # Logs percent identity of training pairs
│   │
│   ├── collator/
│   │   ├── prefixlm_collator.py          # Collator for PrefixLM — packs [CLS] seq2 [SEP] seq1 [SEP]
│   │   ├── phylo_collator.py             # Collator for aligned phylo training
│   │   └── mlm_collator.py              # Collator for MLM baseline
│   │
│   ├── data_utils/
│   │   ├── dynamic_batch.py              # Dynamic batching utilities
│   │   └── truncating_collator.py        # Truncation logic
│   │
│   ├── dataset/
│   │   ├── uniref90_pair_arrow_lmdb.py   # ✅ Main dataset — loads UniRef90 pairs from LMDB
│   │   └── uniref90_pair_arrow_fasta.py  # Alternative dataset using FASTA + index file
│   │
│   ├── models/
│   │   ├── protein_modernbert_phylo.py   # ✅ PrefixLM model builder (ModernBERT + PrefixLM head)
│   │   ├── protein_bert.py              # ModernBERT MLM baseline
│   │   ├── protein_BART.py              # BART encoder-decoder (not used in this project)
│   │   ├── protein_T5.py               # T5 model (not used in this project)
│   │   └── protein_T5Gemma.py          # T5Gemma model (not used in this project)
│   │
│   ├── sequences/
│   │   ├── pairwise_align.py            # Needleman-Wunsch alignment via parasail
│   │   └── seq_fetcher.py              # Sequence fetching utilities
│   │
│   ├── tokenizers/
│   │   ├── phylo_tokenizer.py           # Character-level protein tokenizer
│   │   └── loader.py                   # Tokenizer loader
│   │
│   └── train_utils/                     # Trainer subclasses
│       └── prefixlm_trainer.py          # ✅ PrefixLMTrainer — overrides compute_loss and dataloader
│
├── python_scripts/
│   ├── train_modernBERT.py              # ✅ Main training entry point
│   ├── score_checkpoints.py             # ✅ Offline VEP scoring across all checkpoints
│   └── score_two_checkpoints_full_dist.py # ✅ Full amino acid distribution at two checkpoints
│
├── slurm_scripts/
│   └── train_phylo.sh                  # ✅ SLURM job script for cluster training
│
├── phylo_char_tokenizer_with_bos/       # ✅ Tokenizer with [BOS] token (use for PrefixLM)
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
│
├── phylo_char_tokenizer_updated/        # Tokenizer without [BOS] (use for MLM baseline)
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
│
├── notebooks/                          # Jupyter notebooks for analysis
│   ├── pll_vep_auc.ipynb               # VEP AUC analysis
│   ├── vep_auc_comparison.ipynb        # Aligned vs unaligned comparison
│   └── modernbert_phylo.ipynb          # Model exploration
│
├── environment.yml                     # Conda environment specification
├── requirements.txt                    # Python dependencies
└── README.md
```

---

## Setup

### 1. Create conda environment

```bash
conda env create -f environment.yml
conda activate huggingface_bert_cu126
```

Or install dependencies manually:

```bash
pip install torch transformers datasets lmdb safetensors scikit-learn pandas numpy wandb
pip install flash-attn --no-build-isolation
pip install parasail
```

### 2. Install the gLM package

```bash
pip install -e .
```

### 3. Set PYTHONPATH

```bash
export PYTHONPATH=/path/to/HuggingfaceTransformer:$PYTHONPATH
```

---

## Data

| Data | Description | Path (cluster) |
|------|-------------|----------------|
| UniRef90 clusters (Arrow) | Cluster membership for sequence pairing | `/gpfs/data/brandeslab/Data/uniref/uniref90_clusters_arrow/` |
| UniRef100 sequences (LMDB) | Full sequence database for random access | `/gpfs/data/brandeslab/Data/uniref/uniref100_merged.lmdb` |
| ClinVar VEP benchmark | 118,961 missense variants with pathogenicity labels | `/gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv` |

The ClinVar CSV must have columns: `sequence`, `pos`, `ref`, `alt`, `label`.

---

## Training

### PrefixLM (unaligned) — main model

```bash
torchrun \
  --nproc-per-node=1 \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  python_scripts/train_modernBERT.py \
  --run-name prefixlm_unaligned \
  --model_type "ModernBERT" \
  --training_type "prefixlm_modernbert" \
  --tokenizer-path ./phylo_char_tokenizer_with_bos \
  --train_dataset_type "uniref90_arrow_lmdb" \
  --train_dataset_path /path/to/uniref90_clusters_arrow/train \
  --val_dataset_path /path/to/uniref90_clusters_arrow/test \
  --lmdb_path /path/to/uniref100_merged.lmdb \
  --vep-input-csv /path/to/clinvar_AA_zero_shot_input.csv \
  --output-dir /path/to/checkpoints \
  --max_position_embeddings 2048 \
  --max_steps 200000 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 32 \
  --learning_rate 1e-4 \
  --dataloader_num_workers 8 \
  --dataloader_persistent_workers True \
  --dataloader_prefetch_factor 4 \
  --vep_eval_steps 500 \
  --logging_steps 100 \
  --save_steps 200 \
  --save_strategy "steps" \
  --eval_strategy "no"
```

### PrefixLM (aligned) — use phylo_encoder_decoder dataset type

Change `--training_type` to `"phylo_encoder_decoder"` and `--train_dataset_type` accordingly — alignment is performed on-the-fly during collation via `phylo_collator.py`.

### MLM baseline

```bash
  --model_type "ModernBERT" \
  --training_type "MLM" \
  --tokenizer-path ./phylo_char_tokenizer_updated \
```

### SLURM

```bash
sbatch slurm_scripts/train_phylo.sh
```

---

## Evaluation: Offline VEP Scoring

### Score all checkpoints (ref/alt log-odds only)

```bash
python python_scripts/score_checkpoints.py \
    --checkpoint_dir /path/to/checkpoints/run_name \
    --tokenizer_path ./phylo_char_tokenizer_with_bos \
    --vep_csv /path/to/clinvar_AA_zero_shot_input.csv \
    --output_csv ./clinvar_scores_all_steps.csv \
    --step_start 200 \
    --step_end 10000 \
    --step_size 200 \
    --batch_size 8 \
    --max_len 2048
```

Output CSV columns: `step, variant_idx, pos, ref, alt, label, scored, ref_log_prob, alt_log_prob, log_odds, model_score`

### Full amino acid distribution at two checkpoints

```bash
python python_scripts/score_two_checkpoints_full_dist.py \
    --checkpoint_dir /path/to/checkpoints/run_name \
    --tokenizer_path ./phylo_char_tokenizer_with_bos \
    --vep_csv /path/to/clinvar_AA_zero_shot_input.csv \
    --output_csv ./clinvar_full_dist.csv \
    --steps 3400 6000 \
    --batch_size 8 \
    --max_len 2048
```

Output CSV adds `prob_A, prob_C, prob_D, ...` columns (one per amino acid token) for analyzing the full softmax distribution at the mutation site.

---

## VEP Scoring Formula

For a missense variant at position `p` in sequence `s`, the model scores:

```
log_odds = log P(alt | s[0:p], s) - log P(ref | s[0:p], s)
model_score = -log_odds        # higher = more pathogenic
```

The full wildtype sequence `s` is the prefix (evolutionary context); `s[0:p]` is the suffix. The logit at the last suffix position predicts the amino acid at position `p`.

---

## Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Attention implementation | Two-call Flash Attention | O(T) memory, no 4D mask materialization |
| Tokenizer | `phylo_char_tokenizer_with_bos` | Has [BOS] token needed for PrefixLM |
| Dataset | LMDB | Fast random access across 250M sequences |
| Alignment | Needleman-Wunsch (parasail, BLOSUM62) | Global alignment for aligned variant |
| Loss | Cross-entropy on suffix tokens only | Prefix tokens masked with ignore index |

---

## Notes

- Always use `phylo_char_tokenizer_with_bos` for PrefixLM training and scoring. The `phylo_char_tokenizer_updated` is for the MLM baseline only.
- Set `remove_unused_columns=False` in TrainingArguments — the Trainer strips `prefix_lengths` otherwise.
- `prefixlm_flash.py` is the current version. The `prefixlm_flash1.py`, `prefixlm_flash2.py`, and `prefixlm_flash_old_w...` files are earlier iterations and should not be used.
