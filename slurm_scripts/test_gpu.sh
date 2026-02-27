#!/bin/bash
#SBATCH --job-name=modernBERT_113M_mlm_6_lr3e-4_bs4096_ctxt_1024
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --output=modernBERT_113M_mlm_6_%j.out
#SBATCH --error=modernBERT_113M_mlm_6_%j.err

set -euo pipefail

# --- environment ---
module purge
module load cuda/12.6

source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
conda activate /gpfs/data/brandeslab/User/as12267/.conda/envs/huggingface_bert_cu126

# If you previously hit cuDNN mismatches, keep this; otherwise you can remove it.
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' \
  | grep -v '^/gpfs/share/apps/cuda/12.6' \
  | paste -sd: -)

# --- distributed basics (single node, single proc) ---
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$((12000 + RANDOM % 20000))

# --- caches / logging ---
export HF_HOME=/gpfs/data/brandeslab/User/as12267/cache/huggingface
export TOKENIZERS_PARALLELISM=false

# --- run ---
cd /gpfs/data/brandeslab/Project/HuggingfaceTransformer/

torchrun \
  --nproc-per-node=1 \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  python_scripts/train_modernBERT.py \
  --run-name modernBERT_113M_mlm_6_lr3e-4_bs4096_ctxt_1024 \
  --model_type "ModernBERT" \
  --training_type "MLM" \
  --mlm_probability 0.06 \
  --wandb_project "phylo-llm" \
  --tokenizer-path ./phylo_char_tokenizer_updated \
  --train_dataset_type "seq_pair_map" \
  --max_position_embeddings 1024 \
  --train_dataset_path /gpfs/home/as12267/train.jsonl \
  --index_db_path /gpfs/data/brandeslab/User/as12267/uniref100.idx \
  --fasta_path /gpfs/data/brandeslab/Data/uniref/uniref100.fasta \
  --vep-input-csv /gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv \
  --output-dir /gpfs/data/brandeslab/model_checkpts \
  --num_train_epochs 100 \
  --vep_eval_steps 3742 \
  --logging_steps 4 \
  --batch_sampler "phylo_default" \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 32 \
  --learning_rate 3e-4 \
  --dataloader_num_workers 16 \
  --dataloader_persistent_workers True \
  --dataloader_prefetch_factor 8 \
  --eval_strategy "no" \
  --save_strategy "epoch"
