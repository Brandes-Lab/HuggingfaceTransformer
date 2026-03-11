#!/bin/bash
#SBATCH --job-name=T5Gemma_97M_phylo_bs_4096_arrow_dataset_index_file_validation_checkpoints
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=03-00:00:00
#SBATCH --output=T5Gemma_97M_phylo_bs_4096_arrow_dataset_index_file_validation_checkpoints%j.out
#SBATCH --error=T5Gemma_97M_phylo_bs_4096_arrow_dataset_index_file_validation_checkpoints%j.err

set -euo pipefail

# --- environment ---
module purge
module load cuda/12.6

source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
conda activate /gpfs/data/brandeslab/User/as12267/.conda/envs/huggingface_bert_cu126

echo "which python: $(which python)"
echo "which torchrun: $(which torchrun)"
python -c "import sys; print(sys.executable)"

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



/gpfs/data/brandeslab/User/as12267/.conda/envs/huggingface_bert_cu126/bin/python \
python_scripts/eval_T5Gemma_checkpoints.py \
  --tokenizer_path ./phylo_char_tokenizer_updated \
  --max_position_embeddings 1024 \
  --checkpoint_root /gpfs/data/brandeslab/model_checkpts/T5Gemma_97M_phylo_bs_4096_arrow_dataset_index_file \
  --val_dataset_path /gpfs/data/brandeslab/Data/uniref/uniref90_clusters_arrow/test \
  --fasta_path /gpfs/data/brandeslab/Data/uniref/uniref100.fasta \
  --index_db_path /gpfs/data/brandeslab/User/as12267/uniref100.idx \
  --output_csv /gpfs/data/brandeslab/model_checkpts/T5Gemma_97M_phylo_bs_4096_arrow_dataset_index_file/validation_metrics.csv \
  --training_type phylo_encoder_decoder \
  --per_device_eval_batch_size 256 \
  --dataloader_num_workers 4 \
  --dataloader_persistent_workers True \
  --dataloader_prefetch_factor 2 \
  --wandb_project phylo-llm \
  --wandb_entity sinha-anushka12-na \
  --wandb_run_name T5Gemma_97M_phylo_bs_4096_arrow_dataset_index_file_validation_checkpoints