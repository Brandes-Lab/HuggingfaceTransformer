#!/bin/bash
#SBATCH --job-name=modernBERT_113M_prefixlm_bs512_ctxt_2048_correct
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=03-00:00:00
#SBATCH --output=modernBERT_113M_prefixlm_bs512_ctxt_2048_correct_%j.out
#SBATCH --error=modernBERT_113M_prefixlm_bs512_ctxt_2048_correct_%j.err

set -euo pipefail

# --- environment ---
module purge
module load cuda/12.6

source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
conda activate /gpfs/data/brandeslab/User/as12267/.conda/envs/huggingface_bert_cu126

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH=/gpfs/home/rm7569/HuggingfaceTransformer:${PYTHONPATH:-}


echo "Python executable: $(which python)"
echo "Torch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

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

# --- WANDB ---
export WANDB_API_KEY="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"
echo "WANDB_API_KEY is: $WANDB_API_KEY"
export PYTORCH_ALLOC_CONF=expandable_segments:True
# torchrun \
#   --nproc-per-node=1 \
#   --master_addr="${MASTER_ADDR}" \
#   --master_port="${MASTER_PORT}" \
#   python_scripts/train_modernBERT.py \
#   --run-name modernBERT_113M_prefixlm_bs512_ctxt_4096 \
#   --model_type "ModernBERT" \
#   --training_type "prefixlm_modernbert" \
#   --wandb_project "phylo-llm" \
#   --tokenizer-path ./phylo_char_tokenizer_with_bos \
#   --train_dataset_type "uniref90_arrow_fasta" \
#   --max_position_embeddings 4096 \
#   --train_dataset_path /gpfs/data/brandeslab/Data/uniref/uniref90_clusters_arrow/train \
#   --val_dataset_path /gpfs/data/brandeslab/Data/uniref/uniref90_clusters_arrow/test \
#   --index_db_path /gpfs/data/brandeslab/User/as12267/uniref100.idx \
#   --fasta_path /gpfs/data/brandeslab/Data/uniref/uniref100.fasta \
#   --vep-input-csv /gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv \
#   --output-dir /gpfs/data/brandeslab/model_checkpts \
#   --attn_implementation sdpa \
#   --num_train_epochs 100 \
#   --vep_eval_steps 500 \
#   --logging_steps 4 \
#   --per_device_train_batch_size 128 \
#   --gradient_accumulation_steps 16 \
#   --learning_rate 3e-4 \
#   --dataloader_num_workers 16 \
#   --dataloader_persistent_workers True \
#   --dataloader_prefetch_factor 8 \
#   --eval_strategy "no" \
#   --save_steps 500 \
#   --save_strategy "steps"



# torchrun \
#   --nproc-per-node=1 \
#   --master_addr="${MASTER_ADDR}" \
#   --master_port="${MASTER_PORT}" \
#   python_scripts/train_modernBERT.py \
#   --run-name modernBERT_113M_prefixlm_bs512_ctxt_2048 \
#   --model_type "ModernBERT" \
#   --training_type "prefixlm_modernbert" \
#   --wandb_project "phylo-llm" \
#   --tokenizer-path ./phylo_char_tokenizer_with_bos \
#   --train_dataset_type "uniref90_arrow_lmdb" \
#   --max_position_embeddings 2048 \
#   --train_dataset_path /gpfs/data/brandeslab/Data/uniref/uniref90_clusters_arrow/train \
#   --lmdb_path /gpfs/data/brandeslab/Data/uniref/uniref100_merged.lmdb \
#   --val_dataset_path /gpfs/data/brandeslab/Data/uniref/uniref90_clusters_arrow/test \
#   --vep-input-csv /gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv \
#   --output-dir /gpfs/data/brandeslab/model_checkpts \
#   --attn_implementation flash_attention_2 \
#   --num_train_epochs 100 \
#   --vep_eval_steps 500 \
#   --logging_steps 4 \
#   --per_device_train_batch_size 32 \
#   --gradient_accumulation_steps 16 \
#   --learning_rate 3e-4 \
#   --dataloader_num_workers 0 \
#   --dataloader_persistent_workers False \
#   --eval_strategy "no" \
#   --save_steps 500 \
#   --save_strategy "steps"

CUDA_LAUNCH_BLOCKING=1 torchrun \
  --nproc-per-node=1 \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  python_scripts/train_modernBERT.py \
  --run-name modernBERT_113M_prefixlm_bs512_ctxt_2048_correct \
  --model_type "ModernBERT" \
  --training_type "prefixlm_modernbert" \
  --wandb_project "phylo-llm" \
  --tokenizer-path ./phylo_char_tokenizer_with_bos \
  --train_dataset_type "uniref90_arrow_lmdb" \
  --max_position_embeddings 2048 \
  --train_dataset_path /gpfs/data/brandeslab/Data/uniref/uniref90_clusters_arrow/train \
  --lmdb_path /gpfs/data/brandeslab/Data/uniref/uniref100_merged.lmdb \
  --val_dataset_path /gpfs/data/brandeslab/Data/uniref/uniref90_clusters_arrow/test \
  --vep-input-csv /gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv \
  --output-dir /gpfs/data/brandeslab/phylo_llm_checkpts \
  --attn_implementation flash_attention_2 \
  --max_steps 200000 \
  --vep_eval_steps 10000000000 \
  --logging_steps 4 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 32 \
  --learning_rate 1e-4 \
  --dataloader_num_workers 8 \
  --dataloader_persistent_workers True \
  --dataloader_prefetch_factor 4 \
  --eval_strategy "no" \
  --save_steps 500 \
  --save_strategy "steps"