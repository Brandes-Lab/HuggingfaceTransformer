#!/bin/bash
#SBATCH --job-name=dms_control_modernBERT_113M
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03-00:00:00
#SBATCH --output=/gpfs/data/brandeslab/Project/slurm_logs/%x_%j.out
#SBATCH --error=/gpfs/data/brandeslab/Project/slurm_logs/%x_%j.err

# Load Anaconda
module load anaconda3
source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
conda activate /gpfs/data/brandeslab/User/as12267/.conda/envs/huggingface_bert

# Set Hugging Face cache location to non-home directory
export HF_HOME=/gpfs/data/brandeslab/User/as12267/cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
echo "Caching to: $HF_HOME"

# === Weights & Biases config ===
export WANDB_PROJECT=modernBERT_benchmarks
export WANDB_API_KEY=ae9049d442db2ba3fa77f7928c1dae68353b3762

export MASTER_PORT=$((29500 + RANDOM % 1000))
echo "Using MASTER_PORT=$MASTER_PORT"

# Move to project directory
cd /gpfs/data/brandeslab/Project/HuggingfaceTransformer/

# -------------------------------
# Run pretrained fine-tuning benchmark
# -------------------------------
# Pass any number of checkpoints as CLI arguments to this script.
# Example:
# sbatch slurm_scripts/dms.sh \
#   /gpfs/data/brandeslab/model_checkpts/modernBERT_113M/checkpoint-20000 \
#   /gpfs/data/brandeslab/model_checkpts/pre_trained_modernBERT_113M/checkpoint-50000 \
#   /gpfs/data/brandeslab/model_checkpts/pre_trained_modernBERT_113M/checkpoint-170000 \
#   /gpfs/data/brandeslab/model_checkpts/pre_trained_modernBERT_113M_2/checkpoint-60000

# torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
#   python_scripts/dms_benchmark.py \
#   --mode pre_trained_bert \
#   --checkpoints "$@"

torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
  python_scripts/dms_benchmark.py \
  --mode control


